"""EvalRunner — runs an agent over a dataset and produces an EvalReport."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from harness.core.context import AgentResult
from harness.eval.datasets import EvalCase, EvalDataset, MultiAgentEvalDataset
from harness.eval.diagnostics import EvalDiagnostics, build_diagnostics
from harness.eval.scorers import ScoreResult, score_exact_match

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Aggregated evaluation results for a dataset run.

    Attributes:
        dataset_name:        Name of the EvalDataset.
        agent_type:          The agent type evaluated.
        total_cases:         Total number of cases attempted.
        passed:              Cases that scored >= 0.5 (configurable threshold).
        failed:              Cases that scored < 0.5 or raised an exception.
        success_rate:        passed / total_cases.
        avg_steps:           Mean step count across all runs.
        avg_tokens:          Mean token count across all runs.
        avg_cost_usd:        Mean USD cost across all runs.
        avg_latency_seconds: Mean wall-clock seconds per run.
        scores:              Per-case float scores in [0, 1].
        errors:              Per-case error messages for failed cases.
        prompt_version:      Active prompt version ID at time of eval.
        run_at:              UTC timestamp when this eval was executed.
    """

    dataset_name: str
    agent_type: str
    total_cases: int
    passed: int
    failed: int
    success_rate: float
    avg_steps: float
    avg_tokens: float
    avg_cost_usd: float
    avg_latency_seconds: float
    scores: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    prompt_version: str = "unknown"
    run_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    diagnostics: EvalDiagnostics | None = None

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the report as a Markdown-formatted string."""
        lines = [
            f"# Eval Report: {self.dataset_name}",
            "",
            f"**Agent type:** `{self.agent_type}`  ",
            f"**Prompt version:** `{self.prompt_version}`  ",
            f"**Run at:** {self.run_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total cases | {self.total_cases} |",
            f"| Passed | {self.passed} |",
            f"| Failed | {self.failed} |",
            f"| Success rate | {self.success_rate:.1%} |",
            f"| Avg steps | {self.avg_steps:.1f} |",
            f"| Avg tokens | {self.avg_tokens:.0f} |",
            f"| Avg cost (USD) | ${self.avg_cost_usd:.4f} |",
            f"| Avg latency (s) | {self.avg_latency_seconds:.2f} |",
            "",
        ]

        if self.scores:
            lines += [
                "## Per-Case Scores",
                "",
                "| Case ID | Score | Error |",
                "|---------|-------|-------|",
            ]
            for case_id, score in sorted(self.scores.items()):
                error = self.errors.get(case_id, "")
                error_short = error[:60] + "..." if len(error) > 60 else error
                lines.append(f"| `{case_id}` | {score:.2f} | {error_short} |")
            lines.append("")

        if self.diagnostics is not None:
            lines += ["", self.diagnostics.to_markdown(), ""]

        return "\n".join(lines)

    def compare(self, other: EvalReport) -> str:
        """Produce a diff summary comparing this report to *other*.

        Args:
            other: The baseline EvalReport to compare against.

        Returns:
            A human-readable Markdown comparison string.
        """
        delta_sr = self.success_rate - other.success_rate
        delta_steps = self.avg_steps - other.avg_steps
        delta_tokens = self.avg_tokens - other.avg_tokens
        delta_cost = self.avg_cost_usd - other.avg_cost_usd
        delta_latency = self.avg_latency_seconds - other.avg_latency_seconds

        def fmt_delta(
            v: float,
            unit: str = "",
            higher_is_better: bool = True,
            percent: bool = False,
        ) -> str:
            sign = "+" if v >= 0 else ""
            direction = ""
            if v > 0:
                direction = " (better)" if higher_is_better else " (worse)"
            elif v < 0:
                direction = " (worse)" if higher_is_better else " (better)"
            if percent:
                # v is a fraction (e.g. 0.05) — render as a percentage point delta.
                return f"{sign}{v * 100:.1f}{unit or '%'}{direction}"
            return f"{sign}{v:.4f}{unit}{direction}"

        lines = [
            f"## Comparison: {self.dataset_name} vs baseline",
            "",
            "| Metric | Baseline | New | Delta |",
            "|--------|----------|-----|-------|",
            f"| Success rate | {other.success_rate:.1%} | {self.success_rate:.1%} | {fmt_delta(delta_sr, '%', True, percent=True)} |",
            f"| Avg steps | {other.avg_steps:.1f} | {self.avg_steps:.1f} | {fmt_delta(delta_steps, '', False)} |",
            f"| Avg tokens | {other.avg_tokens:.0f} | {self.avg_tokens:.0f} | {fmt_delta(delta_tokens, '', False)} |",
            f"| Avg cost (USD) | ${other.avg_cost_usd:.4f} | ${self.avg_cost_usd:.4f} | {fmt_delta(delta_cost, '', False)} |",
            f"| Avg latency (s) | {other.avg_latency_seconds:.2f} | {self.avg_latency_seconds:.2f} | {fmt_delta(delta_latency, '', False)} |",
            "",
        ]

        # Per-case diffs
        all_ids = sorted(set(self.scores) | set(other.scores))
        if all_ids:
            lines += [
                "## Per-Case Score Differences",
                "",
                "| Case ID | Baseline | New | Delta |",
                "|---------|----------|-----|-------|",
            ]
            for cid in all_ids:
                b_score = other.scores.get(cid, float("nan"))
                n_score = self.scores.get(cid, float("nan"))
                if b_score != b_score or n_score != n_score:  # NaN check
                    diff = "N/A"
                else:
                    d = n_score - b_score
                    diff = f"{'+' if d >= 0 else ''}{d:.2f}"
                lines.append(
                    f"| `{cid}` | {b_score:.2f} | {n_score:.2f} | {diff} |"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise report to a plain dict."""
        return {
            "dataset_name": self.dataset_name,
            "agent_type": self.agent_type,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_tokens": self.avg_tokens,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_latency_seconds": self.avg_latency_seconds,
            "scores": self.scores,
            "errors": self.errors,
            "prompt_version": self.prompt_version,
            "run_at": self.run_at.isoformat(),
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

_PASS_THRESHOLD = 0.5  # Score at or above this is considered a pass


class EvalRunner:
    """Runs an agent over an EvalDataset and produces an EvalReport.

    Args:
        agent_runner: Object with an ``execute_run(run_id)`` or
                      ``run(context)`` async method, or a callable that
                      takes (tenant_id, agent_type, task) and returns an
                      AgentResult-like object.
        llm_provider: Optional LLMProvider used by ``score_llm_judge``.
    """

    def __init__(
        self,
        agent_runner: Any,
        llm_provider: Any | None = None,
    ) -> None:
        self._runner = agent_runner
        self._llm = llm_provider

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        dataset: EvalDataset,
        tenant_id: str = "eval",
        concurrency: int = 3,
        scorer: Callable | None = None,
        pass_threshold: float = _PASS_THRESHOLD,
        prompt_version: str = "unknown",
    ) -> EvalReport:
        """Execute all cases in the dataset and return a consolidated EvalReport.

        Args:
            dataset:        The EvalDataset to evaluate.
            tenant_id:      Tenant ID to use for all eval runs.
            concurrency:    Maximum number of concurrent agent runs.
            scorer:         Scoring function to use.  If None: uses
                            ``score_exact_match`` when expected is set,
                            otherwise checks ``result.success``.
            pass_threshold: Score >= this value counts as "passed".
            prompt_version: Label for the prompt version in the report.

        Returns:
            EvalReport with aggregated and per-case metrics.
        """
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._run_case(
                case=case,
                tenant_id=tenant_id,
                scorer=scorer,
                semaphore=semaphore,
            )
            for case in dataset.cases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate metrics
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        total_steps: list[float] = []
        total_tokens: list[float] = []
        total_cost: list[float] = []
        total_latency: list[float] = []
        diagnostic_records: list[dict[str, Any]] = []

        for case, result in zip(dataset.cases, results, strict=True):
            if isinstance(result, Exception):
                scores[case.case_id] = 0.0
                errors[case.case_id] = str(result)
                diagnostic_records.append(
                    {
                        "case_id": case.case_id,
                        "agent_type": case.agent_type,
                        "score": 0.0,
                        "result": None,
                        "error": str(result),
                    }
                )
                logger.warning("Case %s raised exception: %s", case.case_id, result)
            else:
                score_val, agent_result = result
                scores[case.case_id] = score_val
                diagnostic_records.append(
                    {
                        "case_id": case.case_id,
                        "agent_type": case.agent_type,
                        "score": score_val,
                        "result": agent_result,
                        "error": "",
                    }
                )
                if agent_result is not None:
                    total_steps.append(float(getattr(agent_result, "steps", 0)))
                    total_tokens.append(float(getattr(agent_result, "tokens", 0)))
                    total_cost.append(float(getattr(agent_result, "cost_usd", 0.0)))
                    total_latency.append(float(getattr(agent_result, "elapsed_seconds", 0.0)))
                if score_val < pass_threshold:
                    error_msg = getattr(agent_result, "error_message", "") or ""
                    if error_msg:
                        errors[case.case_id] = error_msg

        passed = sum(1 for s in scores.values() if s >= pass_threshold)
        failed = len(dataset.cases) - passed

        def _safe_avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        diagnostics = build_diagnostics(
            dataset.name,
            diagnostic_records,
            pass_threshold=pass_threshold,
        )

        return EvalReport(
            dataset_name=dataset.name,
            agent_type=dataset.agent_type,
            total_cases=len(dataset.cases),
            passed=passed,
            failed=failed,
            success_rate=passed / max(len(dataset.cases), 1),
            avg_steps=_safe_avg(total_steps),
            avg_tokens=_safe_avg(total_tokens),
            avg_cost_usd=_safe_avg(total_cost),
            avg_latency_seconds=_safe_avg(total_latency),
            scores=scores,
            errors=errors,
            prompt_version=prompt_version,
            run_at=datetime.now(UTC),
            diagnostics=diagnostics,
        )

    async def run_multi_agent(
        self,
        dataset: MultiAgentEvalDataset,
        tenant_id: str = "eval",
        concurrency: int = 2,
        scorer: Callable | None = None,
        pass_threshold: float = _PASS_THRESHOLD,
        prompt_version: str = "unknown",
        timeout: float = 300.0,
    ) -> EvalReport:
        """Execute multi-agent DAG eval cases through the Scheduler.

        Each case is converted into a TaskPlan and run through the production
        scheduler. The case-level score evaluates the combined final outputs;
        diagnostics also include each sub-agent's resource use and failure stage.
        """
        from harness.orchestrator.scheduler import Scheduler

        semaphore = asyncio.Semaphore(concurrency)
        scheduler = Scheduler(agent_runner=self._runner)

        async def _run_plan_case(case):
            async with semaphore:
                plan = case.to_task_plan()
                results = await scheduler.execute_plan(
                    plan,
                    tenant_id=tenant_id,
                    timeout=float(case.metadata.get("timeout", timeout)),
                )
                output = "\n\n".join(
                    getattr(result, "output", "")
                    for result in results.values()
                    if getattr(result, "output", "")
                )
                success = bool(results) and all(
                    getattr(result, "success", False)
                    for result in results.values()
                )
                synthetic = AgentResult(
                    run_id=plan.plan_id,
                    output=output,
                    steps=sum(int(getattr(result, "steps", 0)) for result in results.values()),
                    tokens=sum(int(getattr(result, "tokens", 0)) for result in results.values()),
                    success=success,
                    error_message="" if success else "One or more subtasks failed",
                    elapsed_seconds=sum(
                        float(getattr(result, "elapsed_seconds", 0.0))
                        for result in results.values()
                    ),
                    cost_usd=sum(
                        float(getattr(result, "cost_usd", 0.0))
                        for result in results.values()
                    ),
                    tool_calls=sum(
                        int(getattr(result, "tool_calls", 0))
                        for result in results.values()
                    ),
                    tool_errors=sum(
                        int(getattr(result, "tool_errors", 0))
                        for result in results.values()
                    ),
                    guardrail_hits=sum(
                        int(getattr(result, "guardrail_hits", 0))
                        for result in results.values()
                    ),
                    handoff_count=sum(
                        int(getattr(result, "handoff_count", 0))
                        for result in results.values()
                    ),
                    cache_hits=sum(
                        int(getattr(result, "cache_hits", 0))
                        for result in results.values()
                    ),
                    cache_read_tokens=sum(
                        int(getattr(result, "cache_read_tokens", 0))
                        for result in results.values()
                    ),
                )
                score = await self._score(case, output, synthetic, scorer)
                subtask_records = []
                for subtask in plan.subtasks:
                    result = results.get(subtask.id)
                    subtask_records.append(
                        {
                            "case_id": f"{case.case_id}/{subtask.id}",
                            "agent_type": subtask.agent_type,
                            "score": 1.0 if getattr(result, "success", False) else 0.0,
                            "result": result,
                            "error": getattr(result, "error_message", "") if result else "missing result",
                        }
                    )
                return score, synthetic, subtask_records

        raw_results = await asyncio.gather(
            *[_run_plan_case(case) for case in dataset.cases],
            return_exceptions=True,
        )

        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        total_steps: list[float] = []
        total_tokens: list[float] = []
        total_cost: list[float] = []
        total_latency: list[float] = []
        diagnostic_records: list[dict[str, Any]] = []

        for case, raw in zip(dataset.cases, raw_results, strict=True):
            if isinstance(raw, Exception):
                scores[case.case_id] = 0.0
                errors[case.case_id] = str(raw)
                diagnostic_records.append(
                    {
                        "case_id": case.case_id,
                        "agent_type": "multi",
                        "score": 0.0,
                        "result": None,
                        "error": str(raw),
                    }
                )
                continue

            score_val, agent_result, subtask_records = raw
            scores[case.case_id] = score_val
            if score_val < pass_threshold:
                errors[case.case_id] = getattr(agent_result, "error_message", "") or "score below threshold"
            total_steps.append(float(agent_result.steps))
            total_tokens.append(float(agent_result.tokens))
            total_cost.append(float(agent_result.cost_usd))
            total_latency.append(float(agent_result.elapsed_seconds))
            diagnostic_records.append(
                {
                    "case_id": case.case_id,
                    "agent_type": "multi",
                    "score": score_val,
                    "result": agent_result,
                    "error": "",
                }
            )
            diagnostic_records.extend(subtask_records)

        passed = sum(1 for score in scores.values() if score >= pass_threshold)

        def _safe_avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        diagnostics = build_diagnostics(
            dataset.name,
            diagnostic_records,
            pass_threshold=pass_threshold,
        )
        return EvalReport(
            dataset_name=dataset.name,
            agent_type="multi",
            total_cases=len(dataset.cases),
            passed=passed,
            failed=len(dataset.cases) - passed,
            success_rate=passed / max(len(dataset.cases), 1),
            avg_steps=_safe_avg(total_steps),
            avg_tokens=_safe_avg(total_tokens),
            avg_cost_usd=_safe_avg(total_cost),
            avg_latency_seconds=_safe_avg(total_latency),
            scores=scores,
            errors=errors,
            prompt_version=prompt_version,
            run_at=datetime.now(UTC),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Patch comparison
    # ------------------------------------------------------------------

    async def compare_patches(
        self,
        dataset: EvalDataset,
        baseline_config: dict,
        patched_config: dict,
        tenant_id: str = "eval",
    ) -> tuple[EvalReport, EvalReport]:
        """Run the dataset twice with different configs and return both reports.

        The runner's config is temporarily swapped to *baseline_config* then
        *patched_config*.  If the runner exposes a ``configure(dict)`` method
        it will be called; otherwise the configs are passed as metadata.

        Args:
            dataset:         Dataset to evaluate.
            baseline_config: Config dict for the baseline run.
            patched_config:  Config dict for the patched run.
            tenant_id:       Tenant ID to use for all runs.

        Returns:
            Tuple of (baseline_report, patched_report).
        """
        logger.info(
            "compare_patches: running baseline for dataset=%s", dataset.name
        )
        baseline_report = await self._run_with_config(
            dataset, baseline_config, tenant_id, label="baseline"
        )

        logger.info(
            "compare_patches: running patched for dataset=%s", dataset.name
        )
        patched_report = await self._run_with_config(
            dataset, patched_config, tenant_id, label="patched"
        )

        return baseline_report, patched_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_case(
        self,
        case: EvalCase,
        tenant_id: str,
        scorer: Callable | None,
        semaphore: asyncio.Semaphore,
    ) -> tuple[float, Any]:
        """Execute a single EvalCase and return (score, agent_result)."""
        async with semaphore:
            try:
                agent_result = await self._invoke_runner(
                    tenant_id=tenant_id,
                    agent_type=case.agent_type,
                    task=case.task,
                    metadata=case.metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Runner raised exception for case %s: %s", case.case_id, exc
                )
                raise

            output = getattr(agent_result, "output", "") or ""
            score = await self._score(case, output, agent_result, scorer)
            return score, agent_result

    async def _invoke_runner(
        self,
        tenant_id: str,
        agent_type: str,
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke the agent runner using the most appropriate interface."""
        metadata = metadata or {}

        # Protocol 1: production AgentRunner lifecycle.
        if hasattr(self._runner, "create_run") and hasattr(self._runner, "execute_run"):
            record = await self._maybe_await(
                self._runner.create_run(
                    tenant_id=tenant_id,
                    agent_type=agent_type,
                    task=task,
                    metadata=metadata,
                )
            )
            updated = await self._maybe_await(self._runner.execute_run(record.run_id))
            return _record_to_agent_result(updated)

        # Protocol 2: runner.run(tenant_id, agent_type, task[, metadata])
        if hasattr(self._runner, "run") and callable(self._runner.run):
            params = inspect.signature(self._runner.run).parameters
            if "metadata" in params:
                result = self._runner.run(tenant_id, agent_type, task, metadata=metadata)
            else:
                result = self._runner.run(tenant_id, agent_type, task)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Protocol 3: runner is a coroutine function itself
        if asyncio.iscoroutinefunction(self._runner):
            kwargs = _runner_kwargs(self._runner, tenant_id, agent_type, task, metadata)
            return await self._runner(**kwargs)

        # Protocol 4: synchronous callable
        if callable(self._runner):
            kwargs = _runner_kwargs(self._runner, tenant_id, agent_type, task, metadata)
            return self._runner(**kwargs)

        raise TypeError(
            f"agent_runner does not expose a callable interface: {type(self._runner)}"
        )

    async def _maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value):
            return await value
        return value

    async def _score(
        self,
        case: EvalCase,
        output: str,
        agent_result: Any,
        scorer: Callable | None,
    ) -> float:
        """Compute a score for the given case and output."""
        if scorer is not None:
            result = _invoke_scorer(scorer, output, case.expected_output, case)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, ScoreResult):
                return result.score
            return float(result)

        # Default scoring strategy
        if case.expected_output:
            return score_exact_match(output, case.expected_output)

        # No ground truth: use success flag
        return 1.0 if getattr(agent_result, "success", False) else 0.0

    async def _run_with_config(
        self,
        dataset: EvalDataset,
        config: dict,
        tenant_id: str,
        label: str,
    ) -> EvalReport:
        """Apply config to runner, run eval, restore prior state."""
        # Attempt to apply config if runner supports it
        if hasattr(self._runner, "configure") and callable(self._runner.configure):
            try:
                cfg_result = self._runner.configure(config)
                if asyncio.iscoroutine(cfg_result):
                    await cfg_result
            except Exception as exc:
                logger.warning("Runner.configure() raised: %s", exc)

        report = await self.run(
            dataset=dataset,
            tenant_id=tenant_id,
            prompt_version=config.get("prompt_version", label),
        )
        return report


def _invoke_scorer(scorer: Callable[..., Any], output: str, expected: Any, case: EvalCase) -> Any:
    """Call ``scorer`` with the case when it accepts it, else the (output, expected) form.

    Case-aware scorers (e.g. code-execution pass@1, which needs the test harness in
    ``case.metadata``) declare a 3rd positional ``case`` param or a ``case`` keyword.
    Plain ``scorer(output, expected)`` callables are unaffected.
    """
    try:
        params = inspect.signature(scorer).parameters
        positional = sum(
            1
            for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params.values())
        if "case" in params or positional >= 3 or has_varargs:
            return scorer(output, expected, case)
    except (TypeError, ValueError):
        pass
    return scorer(output, expected)


def _runner_kwargs(
    fn: Any,
    tenant_id: str,
    agent_type: str,
    task: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs for a runner callable without forcing metadata support."""
    kwargs: dict[str, Any] = {
        "tenant_id": tenant_id,
        "agent_type": agent_type,
        "task": task,
    }
    try:
        params = inspect.signature(fn).parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
        if accepts_kwargs or "metadata" in params:
            kwargs["metadata"] = metadata
    except (TypeError, ValueError):
        kwargs["metadata"] = metadata
    return kwargs


def _record_to_agent_result(record: Any) -> AgentResult:
    """Convert a RunRecord-like object to AgentResult for eval aggregation."""
    result_data = getattr(record, "result", None) or {}
    return AgentResult(
        run_id=result_data.get("run_id", getattr(record, "run_id", "")),
        output=result_data.get("output", ""),
        steps=int(result_data.get("steps", 0) or 0),
        tokens=int(result_data.get("tokens", 0) or 0),
        success=bool(result_data.get("success", getattr(record, "status", "") == "completed")),
        failure_class=result_data.get("failure_class"),
        error_message=result_data.get("error_message"),
        elapsed_seconds=float(result_data.get("elapsed_seconds", 0.0) or 0.0),
        cost_usd=float(result_data.get("cost_usd", 0.0) or 0.0),
        mlflow_run_id=result_data.get("mlflow_run_id"),
        tool_calls=int(result_data.get("tool_calls", 0) or 0),
        tool_errors=int(result_data.get("tool_errors", 0) or 0),
        guardrail_hits=int(result_data.get("guardrail_hits", 0) or 0),
        handoff_count=int(result_data.get("handoff_count", 0) or 0),
        cache_hits=int(result_data.get("cache_hits", 0) or 0),
        cache_read_tokens=int(result_data.get("cache_read_tokens", 0) or 0),
    )
