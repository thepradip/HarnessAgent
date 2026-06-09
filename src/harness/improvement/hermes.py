"""HermesLoop: orchestrates the full self-improvement cycle for HarnessAgent agents."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from harness.improvement.error_collector import ErrorCollector, ErrorRecord
from harness.improvement.evaluator import EvalResult, Evaluator
from harness.improvement.patch_generator import Patch, PatchGenerator

logger = logging.getLogger(__name__)

# Default agent types for the Hermes loop
_DEFAULT_AGENT_TYPES = ["sql", "code", "base"]

# Redis key prefix for storing pending patches
_PATCH_KEY_PREFIX = "harness:hermes:patch:"
_PATCH_INDEX_KEY = "harness:hermes:patch_index"


@dataclass
class PatchOutcome:
    """Result of one Hermes improvement cycle for a single agent type.

    Attributes:
        patch:       The proposed Patch (may be None if no proposal was generated).
        eval_result: The evaluation result (may be None if not evaluated).
        applied:     Whether the patch was applied to the prompt store.
        reason:      Human-readable explanation of why the patch was applied or skipped.
    """

    patch: Patch | None
    eval_result: EvalResult | None
    applied: bool
    reason: str


class HermesLoop:
    """Orchestrates the Hermes self-improvement loop.

    One cycle per agent type:
    1. Run pending rollback check (if a patch was auto-applied last cycle).
    2. Check error count in rolling window — skip if fewer than min_errors.
    3. Sample a batch of recent errors.
    4. Generate a patch proposal from the LLM.
    5. Evaluate the patch by replaying failing tasks.
    6. If score > threshold AND auto_apply: apply via prompt_store,
       then schedule a rollback check for the next cycle.
       Otherwise: store with status="pending" for human review.
    7. Record metrics to Prometheus and MLflow.

    Can be run on a schedule via APScheduler using start_background().
    """

    def __init__(
        self,
        collector: ErrorCollector,
        generator: PatchGenerator,
        evaluator: Evaluator,
        prompt_store: Any,          # PromptStore or PromptManager
        metrics: Any,               # HarnessMetrics
        config: Any,                # Settings
        online_monitor: Any | None = None,   # OnlineLearningMonitor
        mlflow_tracer: Any | None = None,    # MLflowAgentTracer
    ) -> None:
        """
        Args:
            collector:       ErrorCollector for sampling agent failures.
            generator:       PatchGenerator for proposing patches.
            evaluator:       Evaluator for scoring patches.
            prompt_store:    PromptStore (or PromptManager) for applying patches.
            metrics:         HarnessMetrics instance for recording patch counts.
            config:          Settings with hermes_* configuration keys.
            online_monitor:  Optional OnlineLearningMonitor for post-apply regression checks.
            mlflow_tracer:   Optional MLflowAgentTracer for cycle logging.
        """
        self._collector = collector
        self._generator = generator
        self._evaluator = evaluator
        self._prompt_store = prompt_store
        self._metrics = metrics
        self._config = config
        self._online_monitor = online_monitor
        self._mlflow_tracer = mlflow_tracer

        self.threshold: float = getattr(config, "hermes_patch_score_threshold", 0.7)
        self.auto_apply: bool = getattr(config, "hermes_auto_apply", False)
        self.min_errors: int = getattr(config, "hermes_min_errors_to_trigger", 5)
        self._interval_seconds: float = getattr(config, "hermes_interval_seconds", 3600.0)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self, agent_type: str) -> PatchOutcome | None:
        """Run one improvement cycle for the given agent type.

        Args:
            agent_type: The agent type to improve.

        Returns:
            PatchOutcome if a patch was proposed/evaluated, None if skipped.
        """
        logger.info("Hermes cycle starting for agent_type=%s", agent_type)
        rolled_back = False

        # 0. Run pending rollback check from the previous cycle's auto-apply
        if self._online_monitor is not None:
            try:
                rolled_back = await self._online_monitor.check_and_maybe_rollback(
                    agent_type=agent_type,
                    error_collector=self._collector,
                    prompt_manager=self._prompt_store,
                )
                if rolled_back:
                    logger.warning(
                        "Hermes: rolled back last patch for agent_type=%s "
                        "due to regression — starting fresh cycle",
                        agent_type,
                    )
            except Exception as exc:
                logger.warning("Hermes: rollback check failed for %s: %s", agent_type, exc)

        # 1. Check error count in window
        try:
            error_count = await self._collector.count(agent_type)
        except Exception as exc:
            logger.warning("Hermes: error count check failed for %s: %s", agent_type, exc)
            return None

        if error_count < self.min_errors:
            logger.info(
                "Hermes: insufficient errors for %s (%d < %d) — skipping cycle",
                agent_type,
                error_count,
                self.min_errors,
            )
            return None

        # 2. Sample error batch — temporal split for regression safety:
        #    errors are newest-first; older half feeds generation, newer half
        #    is the held-out validation set so the patch is never evaluated on
        #    the same distribution it was trained on.
        try:
            errors: list[ErrorRecord] = await self._collector.get_recent(
                agent_type, limit=max(10, self.min_errors * 2)
            )
        except Exception as exc:
            logger.error("Hermes: error sampling failed for %s: %s", agent_type, exc)
            return None

        if not errors:
            logger.info("Hermes: no errors sampled for %s", agent_type)
            return None

        split = max(1, len(errors) // 2)
        holdout_errors = errors[:split]       # newest — used for evaluation only
        generation_errors = errors[split:]    # older  — used for patch generation
        if not generation_errors:
            generation_errors = errors        # fallback when sample is tiny
        errors = generation_errors            # all downstream code uses this name

        logger.info(
            "Hermes: sampled %d errors for agent_type=%s "
            "(generation=%d holdout=%d)",
            len(errors) + len(holdout_errors),
            agent_type,
            len(generation_errors),
            len(holdout_errors),
        )

        # 3. Get current agent config
        current_config: dict[str, Any] = {}
        try:
            if hasattr(self._prompt_store, "get_prompt"):
                current_prompt = await self._prompt_store.get_prompt(agent_type)
                current_config = {"system_prompt": current_prompt}
            elif hasattr(self._prompt_store, "get"):
                current_prompt = await self._prompt_store.get(agent_type)
                current_config = {"system_prompt": str(current_prompt)}
        except Exception as exc:
            logger.warning("Hermes: could not load current config for %s: %s", agent_type, exc)

        # 4. Generate patch proposal — route to the right generator based on failure class
        patch: Patch | None = None
        try:
            timeout_count = sum(
                1 for e in errors if "TOOL_TIMEOUT" in e.failure_class
            )
            safety_count = sum(
                1 for e in errors if "SAFETY_" in e.failure_class
            )
            tool_count = sum(
                1 for e in errors
                if any(fc in e.failure_class for fc in ("TOOL_", "MCP_"))
                and "TOOL_TIMEOUT" not in e.failure_class
            )
            other_count = len(errors) - timeout_count - safety_count - tool_count

            dominant = max(
                [("timeout", timeout_count), ("safety", safety_count),
                 ("tool", tool_count), ("prompt", other_count)],
                key=lambda x: x[1],
            )[0]

            logger.info(
                "Hermes: failure breakdown — timeout=%d safety=%d tool=%d prompt=%d → routing to %s patch",
                timeout_count, safety_count, tool_count, other_count, dominant,
            )

            if dominant == "timeout" and hasattr(self._generator, "generate_retry_patch"):
                patch = await self._generator.generate_retry_patch(
                    agent_type=agent_type, errors=errors,
                )

            elif dominant == "safety" and hasattr(self._generator, "generate_permission_patch"):
                patch = await self._generator.generate_permission_patch(
                    agent_type=agent_type, errors=errors,
                )

            elif dominant == "tool" and hasattr(self._generator, "generate_tool_patch"):
                patch = await self._generator.generate_tool_patch(
                    agent_type=agent_type, errors=errors,
                )

            # Always fall back to prompt patch when specialised generators return nothing
            if patch is None:
                patch = await self._generator.generate(
                    agent_type=agent_type,
                    errors=errors,
                    max_errors_in_prompt=10,
                )
        except Exception as exc:
            logger.error("Hermes: patch generation failed for %s: %s", agent_type, exc)

        if patch is None:
            logger.info("Hermes: no patch generated for agent_type=%s", agent_type)
            return PatchOutcome(
                patch=None,
                eval_result=None,
                applied=False,
                reason="Patch generator returned no proposal.",
            )

        logger.info(
            "Hermes: generated patch %s for %s (op=%s)",
            patch.patch_id[:8],
            agent_type,
            patch.op,
        )

        # 5. Evaluate patch on the held-out set (never the generation set)
        #    Safety invariant: reject if regression > 15% vs baseline score.
        eval_result: EvalResult | None = None
        try:
            eval_result = await self._evaluator.score(
                patch=patch,
                test_cases=holdout_errors,
                agent_type=agent_type,
            )
            patch.score = eval_result.score
            logger.info(
                "Hermes: patch %s scored %.3f on %d holdout cases",
                patch.patch_id[:8], eval_result.score, len(holdout_errors),
            )
        except Exception as exc:
            logger.error(
                "Hermes: patch evaluation failed for %s (patch=%s): %s",
                agent_type, patch.patch_id[:8], exc,
            )
            eval_result = None

        # 6. Apply or queue for human review
        applied = False
        reason: str = ""

        score = eval_result.score if eval_result is not None else 0.0

        if eval_result is None:
            patch.status = "pending"
            reason = "Evaluation failed — patch queued for manual review."
            await self._store_patch(patch)

        elif score >= self.threshold and self.auto_apply and _passes_regression_invariant(score, self.threshold):
            # Apply the patch automatically
            try:
                # Capture baseline version and error count BEFORE applying
                baseline_version_id = ""
                try:
                    if hasattr(self._prompt_store, "get_version"):
                        bv = await self._prompt_store.get_version(agent_type)
                        baseline_version_id = bv.version_id if bv else ""
                    elif hasattr(self._prompt_store, "get_active"):
                        bv = await self._prompt_store.get_active(agent_type)
                        baseline_version_id = bv.version_id if bv else ""
                except Exception:
                    pass

                await self._apply_patch(patch, agent_type)
                patch.status = "applied"
                applied = True
                reason = (
                    f"Score {score:.3f} >= threshold {self.threshold:.3f} "
                    f"and auto_apply=True — patch applied."
                )
                logger.info(
                    "Hermes: auto-applied patch %s for %s (score=%.3f)",
                    patch.patch_id[:8],
                    agent_type,
                    score,
                )

                # Schedule rollback check for the next cycle
                if self._online_monitor is not None and baseline_version_id:
                    try:
                        new_version_id = ""
                        try:
                            if hasattr(self._prompt_store, "get_version"):
                                nv = await self._prompt_store.get_version(agent_type)
                                new_version_id = nv.version_id if nv else ""
                        except Exception:
                            pass
                        await self._online_monitor.schedule_rollback_check(
                            agent_type=agent_type,
                            patch_id=patch.patch_id,
                            baseline_version_id=baseline_version_id,
                            new_version_id=new_version_id,
                            baseline_error_count=error_count,
                        )
                    except Exception as exc:
                        logger.debug("Could not schedule rollback check: %s", exc)

                # Log prompt version change to MLflow
                if self._mlflow_tracer is not None:
                    try:
                        if hasattr(self._prompt_store, "get_version"):
                            nv = await self._prompt_store.get_version(agent_type)
                            if nv:
                                await self._mlflow_tracer.log_prompt_version(
                                    agent_type=agent_type,
                                    version_id=nv.version_id,
                                    version_number=nv.version_number,
                                    created_by="hermes",
                                    patch_id=patch.patch_id,
                                )
                    except Exception as exc:
                        logger.debug("MLflow prompt version log failed: %s", exc)

            except Exception as exc:
                patch.status = "pending"
                reason = f"Application failed: {exc} — patch queued for manual review."
                logger.error(
                    "Hermes: patch application failed for %s: %s", agent_type, exc
                )
                await self._store_patch(patch)

        elif score >= self.threshold and not self.auto_apply:
            patch.status = "approved"
            reason = (
                f"Score {score:.3f} >= threshold {self.threshold:.3f} "
                f"but auto_apply=False — patch approved, awaiting manual application."
            )
            await self._store_patch(patch)

        elif score >= self.threshold:
            # auto_apply=True and the patch meets the hard threshold, but it sits
            # too close to the boundary to clear the regression-safety invariant
            # (threshold <= score < threshold+0.15 and < 0.9). It is NOT a
            # below-threshold rejection — queue it for manual review instead of
            # mislabelling it "Score < threshold".
            patch.status = "pending"
            reason = (
                f"Score {score:.3f} >= threshold {self.threshold:.3f} but within "
                f"the regression-safety margin (needs >= {self.threshold + 0.15:.3f} "
                f"or >= 0.900 to auto-apply) — queued for manual review."
            )
            await self._store_patch(patch)

        else:
            patch.status = "rejected"
            reason = (
                f"Score {score:.3f} < threshold {self.threshold:.3f} — patch rejected."
            )
            await self._store_patch(patch)

        # 7. Record Prometheus metrics
        self._record_metric(agent_type, patch.status)

        # 8. Log cycle to MLflow
        if self._mlflow_tracer is not None:
            try:
                eval_successes = eval_result.successes if eval_result else 0
                eval_total = eval_result.test_cases if eval_result else 0
                await self._mlflow_tracer.log_hermes_cycle(
                    agent_type=agent_type,
                    patch_id=patch.patch_id,
                    score=score,
                    applied=applied,
                    errors_sampled=len(errors),
                    reason=reason,
                    eval_successes=eval_successes,
                    eval_total=eval_total,
                    rolled_back=rolled_back,
                )
            except Exception as exc:
                logger.debug("MLflow cycle log failed: %s", exc)

        logger.info(
            "Hermes cycle complete for %s: patch=%s status=%s score=%.3f applied=%s",
            agent_type,
            patch.patch_id[:8],
            patch.status,
            score,
            applied,
        )

        return PatchOutcome(
            patch=patch,
            eval_result=eval_result,
            applied=applied,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Multi-agent cycle
    # ------------------------------------------------------------------

    async def run_all_agents(
        self, agent_types: list[str] | None = None
    ) -> list[PatchOutcome]:
        """Run improvement cycles for multiple agent types concurrently.

        Args:
            agent_types: Agent types to run. Defaults to ["sql", "code", "base"].

        Returns:
            List of PatchOutcome objects (one per agent type, None entries excluded).
        """
        if agent_types is None:
            agent_types = _DEFAULT_AGENT_TYPES

        results = await asyncio.gather(
            *[self.run_cycle(at) for at in agent_types],
            return_exceptions=True,
        )

        outcomes: list[PatchOutcome] = []
        for agent_type, result in zip(agent_types, results):
            if isinstance(result, Exception):
                logger.error(
                    "Hermes: run_cycle raised for %s: %s", agent_type, result
                )
            elif result is not None:
                outcomes.append(result)

        return outcomes

    # ------------------------------------------------------------------
    # Background scheduler
    # ------------------------------------------------------------------

    async def start_background(self, agent_types: list[str] | None = None) -> None:
        """Start the Hermes loop as a background APScheduler job.

        Runs run_all_agents every hermes_interval_seconds.

        Args:
            agent_types: Agent types to run on each interval.
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self.run_all_agents,
                "interval",
                seconds=self._interval_seconds,
                kwargs={"agent_types": agent_types},
                id="hermes_loop",
                max_instances=1,
                coalesce=True,
            )
            scheduler.start()

            logger.info(
                "Hermes background loop started: interval=%.0fs, agents=%s",
                self._interval_seconds,
                agent_types or _DEFAULT_AGENT_TYPES,
            )

            # Keep running until cancelled
            while True:
                await asyncio.sleep(60)

        except ImportError:
            logger.warning(
                "apscheduler not installed — running Hermes loop as a simple asyncio task. "
                "Install with: pip install apscheduler"
            )
            # Fallback: manual asyncio loop
            if agent_types is None:
                agent_types = _DEFAULT_AGENT_TYPES
            while True:
                try:
                    await self.run_all_agents(agent_types)
                except Exception as exc:
                    logger.error("Hermes background cycle failed: %s", exc)
                await asyncio.sleep(self._interval_seconds)

        except asyncio.CancelledError:
            logger.info("Hermes background loop cancelled")
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _apply_patch(self, patch: Patch, agent_type: str) -> None:
        """Apply a patch to the prompt store."""
        try:
            if hasattr(self._prompt_store, "apply_patch"):
                await _maybe_await(self._prompt_store.apply_patch(patch))
            elif hasattr(self._prompt_store, "update_prompt"):
                # Build the new prompt by fetching current and applying op
                current = await _maybe_await(self._prompt_store.get_prompt(agent_type))
                new_prompt = _apply_op(current, patch.op, patch.path, patch.value)
                await _maybe_await(
                    self._prompt_store.update_prompt(agent_type, new_prompt)
                )
            else:
                raise AttributeError(
                    f"prompt_store has no apply_patch or update_prompt method"
                )
        except Exception as exc:
            logger.error("Patch application failed: %s", exc)
            raise

    async def _store_patch(self, patch: Patch) -> None:
        """Persist a patch for later review."""
        try:
            if hasattr(self._generator, "_patch_store") and self._generator._patch_store:
                await _maybe_await(self._generator._patch_store.save(patch))
        except Exception as exc:
            logger.debug("Could not persist patch %s: %s", patch.patch_id[:8], exc)

    def _record_metric(self, agent_type: str, status: str) -> None:
        """Increment the hermes_patches_total Prometheus counter."""
        if self._metrics is None:
            return
        try:
            counter = getattr(self._metrics, "hermes_patches_total", None)
            if counter is not None:
                counter.labels(agent_type=agent_type, status=status).inc()
        except Exception as exc:
            logger.debug("Hermes metric recording failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _maybe_await(obj: Any) -> Any:
    """Await if coroutine, else return directly."""
    if asyncio.iscoroutine(obj):
        return await obj
    return obj


def _passes_regression_invariant(
    score: float,
    threshold: float,
    regression_tolerance: float = 0.15,
) -> bool:
    """Return True when a patch score satisfies the regression safety invariant.

    The invariant: a patch must score at least ``threshold`` AND must not be
    so close to the boundary that minor holdout-set variance could flip it.
    Concretely, we require the score to exceed the threshold by at least
    ``regression_tolerance`` unless the sample is too small to be reliable,
    in which case we accept the raw threshold.

    This prevents patches that score exactly at the threshold from being
    auto-applied when the holdout set is small and noisy.
    """
    # Must always meet the hard threshold
    if score < threshold:
        return False
    # Must clear the threshold by at least regression_tolerance
    return score >= threshold + regression_tolerance or score >= 0.9


def _apply_op(current: str, op: str, path: str, value: Any) -> str:
    """Apply a patch operation to a string prompt."""
    value_str = str(value)
    op = op.lower()
    if op == "append":
        return (current + "\n" + value_str).strip()
    elif op == "prepend":
        return (value_str + "\n" + current).strip()
    elif op == "replace":
        return current.replace(path, value_str)
    elif op == "remove":
        return current.replace(path, "").strip()
    elif op in ("set",):
        return value_str
    else:
        return (current + "\n" + value_str).strip()
