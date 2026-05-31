"""GEPA adapter that scores prompt candidates against a gold-labeled EvalDataset.

Unlike the error-replay metric in ``adapter.py`` (which rewards self-reported
success on previously-failing tasks), this adapter optimizes against *correctness*:
each candidate is run through the existing :class:`EvalRunner`, whose scorers
compare agent output to gold (`expected_output` / `gold_actions`) — exact match,
SQL equivalence, execution match, LLM-judge, etc.

It is **multi-component**: a GEPA candidate is a dict mapping component name ->
text (e.g. ``{"system_prompt": "...", "handoff_prompt": "...", "context_summary": "..."}``).
The whole dict is injected into each case's ``metadata[OVERRIDES_KEY]``; every
prompt-construction site that calls ``gepa_override(ctx, name, fallback)`` then
picks up its component. Today ``system_prompt`` is wired in BaseAgent; adding a
component is a one-line ``gepa_override`` read at its construction site.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from harness.core.prompt_overrides import OVERRIDES_KEY
from harness.eval.datasets import EvalDataset
from harness.improvement.gepa.reflection import CoroRunner

logger = logging.getLogger(__name__)


class EvalDatasetGepaAdapter:
    """Implements GEPA's ``GEPAAdapter`` protocol over the EvalRunner + gold scorers.

    Args:
        eval_runner:    An ``EvalRunner`` (``async run(dataset, ...) -> EvalReport``).
        run_coro:       Blocking coroutine runner (drives the async EvalRunner from
            GEPA's synchronous worker thread).
        tenant_id:      Tenant id for eval runs.
        scorer:         Optional custom scorer passed through to ``EvalRunner.run``.
        pass_threshold: Score >= this counts as a pass (for feedback wording).
        concurrency:    Max concurrent cases per evaluation.
    """

    # Use GEPA's built-in instruction proposer (calls the reflection LM).
    propose_new_texts = None

    def __init__(
        self,
        eval_runner: Any,
        run_coro: CoroRunner,
        *,
        tenant_id: str = "gepa-eval",
        scorer: Any = None,
        pass_threshold: float = 0.5,
        concurrency: int = 3,
    ) -> None:
        self._runner = eval_runner
        self._run_coro = run_coro
        self._tenant_id = tenant_id
        self._scorer = scorer
        self._pass_threshold = pass_threshold
        self._concurrency = concurrency

    # ------------------------------------------------------------------
    # GEPAAdapter.evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Run ``candidate`` (a component-overrides dict) over ``batch`` of EvalCases."""
        from gepa.core.adapter import EvaluationBatch  # local import: optional dep

        # Inject the candidate as per-case overrides without mutating the originals.
        cases = [
            replace(case, metadata={**(case.metadata or {}), OVERRIDES_KEY: candidate})
            for case in batch
        ]
        agent_type = cases[0].agent_type if cases else "eval"
        dataset = EvalDataset(name="gepa-candidate", agent_type=agent_type, cases=cases)

        async def _run() -> Any:
            return await self._runner.run(
                dataset,
                tenant_id=self._tenant_id,
                concurrency=self._concurrency,
                scorer=self._scorer,
                pass_threshold=self._pass_threshold,
                prompt_version="gepa-candidate",
            )

        report = self._run_coro(_run())

        report_scores: dict[str, float] = getattr(report, "scores", {}) or {}
        report_errors: dict[str, str] = getattr(report, "errors", {}) or {}
        diag_by_case = _diagnostics_by_case(report)

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for case in batch:
            score = float(report_scores.get(case.case_id, 0.0) or 0.0)
            passed = score >= self._pass_threshold
            scores.append(score)
            outputs.append({"case_id": case.case_id, "score": score, "passed": passed})

            if trajectories is not None:
                trajectories.append(
                    {
                        "task": getattr(case, "task", ""),
                        "expected": getattr(case, "expected_output", None),
                        "score": score,
                        "passed": passed,
                        "error": report_errors.get(case.case_id, ""),
                        "diagnostic": diag_by_case.get(case.case_id, ""),
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # ------------------------------------------------------------------
    # GEPAAdapter.make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build per-component feedback from gold scores + failure diagnostics."""
        trajectories = getattr(eval_batch, "trajectories", None) or []

        records: list[dict[str, Any]] = []
        for traj in trajectories:
            records.append(
                {
                    "Inputs": {"task": traj.get("task", "")},
                    "Generated Outputs": (
                        "Passed the gold check."
                        if traj.get("passed")
                        else "Did not match the expected/gold output."
                    ),
                    "Feedback": _build_feedback(traj),
                }
            )

        if not records:
            records.append(
                {
                    "Inputs": {"task": ""},
                    "Generated Outputs": "No cases were evaluated.",
                    "Feedback": "No evaluation feedback available for this minibatch.",
                }
            )

        return dict.fromkeys(components_to_update, records)


def _diagnostics_by_case(report: Any) -> dict[str, str]:
    """Map case_id -> a short diagnostic string from the report's diagnostics."""
    out: dict[str, str] = {}
    diagnostics = getattr(report, "diagnostics", None)
    cases = getattr(diagnostics, "cases", None) or []
    for case_diag in cases:
        case_id = getattr(case_diag, "case_id", None)
        if case_id is None:
            continue
        stage = getattr(case_diag, "failure_stage", "")
        recs = getattr(case_diag, "recommendations", None) or []
        hint = recs[0] if recs else ""
        out[case_id] = f"{stage}: {hint}".strip(": ").strip()
    return out


def _build_feedback(traj: dict[str, Any]) -> str:
    """Compose natural-language feedback for one evaluated case."""
    if traj.get("passed"):
        return (
            f"This task passed the gold check (score={traj.get('score', 0):.2f}). "
            "Preserve the guidance that produces correct output."
        )

    parts = [f"This task FAILED the gold check (score={traj.get('score', 0):.2f})."]
    expected = traj.get("expected")
    if expected:
        parts.append(f"Expected output resembled: {str(expected)[:200]}")
    error = str(traj.get("error", "")).strip()
    if error:
        parts.append(f"Error/reason: {error[:200]}")
    diagnostic = str(traj.get("diagnostic", "")).strip()
    if diagnostic:
        parts.append(f"Diagnosis: {diagnostic[:200]}")
    parts.append(
        "Revise the prompt component(s) to produce output that matches the expected "
        "result for this kind of task — address the root cause, stay concise."
    )
    return " ".join(parts)
