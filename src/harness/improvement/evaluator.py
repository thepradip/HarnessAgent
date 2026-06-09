"""Evaluator — scores proposed patches by replaying failing tasks."""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalResult dataclass (as specified in the module design)
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Quantitative result of evaluating a patch against test cases.

    Attributes:
        patch_id:         The patch that was evaluated.
        test_cases:       Total number of test cases.
        successes:        Number of test cases that succeeded after patching.
        failures:         Number of test cases that still failed.
        avg_steps_delta:  Mean change in step count vs baseline (negative = fewer).
        avg_tokens_delta: Mean change in token count vs baseline (negative = fewer).
        score:            Composite score in [0, 1]:
                          success_rate - 0.01 * avg_steps_delta - 0.001 * avg_tokens_delta
    """

    patch_id: str
    test_cases: int
    successes: int
    failures: int
    avg_steps_delta: float = 0.0
    avg_tokens_delta: float = 0.0
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.score == 0.0 and self.test_cases > 0:
            self.score = max(0.0, min(1.0,
                self.success_rate
                - 0.01 * self.avg_steps_delta
                - 0.001 * self.avg_tokens_delta
            ))

    @property
    def success_rate(self) -> float:
        if self.test_cases == 0:
            return 0.0
        return self.successes / self.test_cases


class Evaluator:
    """Evaluates a patch by replaying failing tasks with the patch applied.

    For each test case (an ErrorRecord), the evaluator:
    1. Builds a patched config by applying the patch to the base config.
    2. Creates and runs an agent on the original task with the patched config.
    3. Compares success/failure and resource usage against the original.

    The composite score rewards success, brevity (fewer steps), and
    token efficiency (fewer tokens).
    """

    def __init__(self, agent_runner: Any, error_collector: Any) -> None:
        """
        Args:
            agent_runner:    AgentRunner instance used to create/execute test runs.
            error_collector: ErrorCollector for sampling and resolving records.
        """
        self._runner = agent_runner
        self._error_collector = error_collector

    async def score(
        self,
        patch: Any,  # Patch
        test_cases: list[Any],  # list[ErrorRecord]
        agent_type: str,
    ) -> EvalResult:
        """Evaluate a patch across a set of test cases.

        Args:
            patch:      The Patch to evaluate.
            test_cases: ErrorRecord objects whose tasks will be replayed.
            agent_type: The agent type for these test cases.

        Returns:
            An EvalResult with composite score.
        """
        if not test_cases:
            logger.info("No test cases for patch %s — returning zero score", getattr(patch, "patch_id", "?"))
            return EvalResult(
                patch_id=getattr(patch, "patch_id", ""),
                test_cases=0,
                successes=0,
                failures=0,
            )

        # Get the base config
        base_config = await self._get_base_config(agent_type)
        patched_config = await self._apply_patch_to_config(base_config, patch)

        successes = 0
        failures = 0
        step_deltas: list[float] = []
        token_deltas: list[float] = []

        tasks = [
            self._run_test_case(tc, patched_config, agent_type)
            for tc in test_cases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            tc = test_cases[i]
            if isinstance(result, Exception):
                logger.warning("Test case %d failed with exception: %s", i, result)
                failures += 1
                continue

            run_result, baseline_steps, baseline_tokens = result

            if run_result is None or not getattr(run_result, "success", False):
                failures += 1
            else:
                successes += 1
                # Compare resource usage to baseline
                new_steps = getattr(run_result, "steps", 0)
                new_tokens = getattr(run_result, "tokens", 0)
                orig_steps = getattr(tc, "context_snapshot", {}).get("step_count", baseline_steps)
                orig_tokens = getattr(tc, "context_snapshot", {}).get("token_count", baseline_tokens)

                if orig_steps > 0:
                    step_deltas.append(new_steps - orig_steps)
                if orig_tokens > 0:
                    token_deltas.append(new_tokens - orig_tokens)

        total = len(test_cases)
        avg_steps_delta = sum(step_deltas) / len(step_deltas) if step_deltas else 0.0
        avg_tokens_delta = sum(token_deltas) / len(token_deltas) if token_deltas else 0.0
        success_rate = successes / total if total > 0 else 0.0

        # Composite score: higher is better
        # Penalty for more steps and more tokens
        raw_score = success_rate - 0.01 * avg_steps_delta - 0.001 * avg_tokens_delta
        score = max(0.0, min(1.0, raw_score))

        patch_id = getattr(patch, "patch_id", "")
        logger.info(
            "Patch %s scored %.3f: %d/%d success, steps_delta=%.1f, tokens_delta=%.1f",
            patch_id[:8],
            score,
            successes,
            total,
            avg_steps_delta,
            avg_tokens_delta,
        )

        return EvalResult(
            patch_id=patch_id,
            test_cases=total,
            successes=successes,
            failures=failures,
            avg_steps_delta=avg_steps_delta,
            avg_tokens_delta=avg_tokens_delta,
            score=score,
        )

    async def _run_test_case(
        self,
        test_case: Any,
        patched_config: dict[str, Any],
        agent_type: str,
    ) -> tuple[Any, int, int]:
        """Run a single test case and return (result, baseline_steps, baseline_tokens)."""
        task = getattr(test_case, "task", "")
        ctx_snapshot = getattr(test_case, "context_snapshot", {})
        baseline_steps = ctx_snapshot.get("step_count", 0)
        baseline_tokens = ctx_snapshot.get("token_count", 0)

        metadata = {
            "hermes_eval": True,
            "patched_config": patched_config,
            "original_error_id": getattr(test_case, "record_id", ""),
        }

        try:
            record = await self._runner.create_run(
                tenant_id="hermes-eval",
                agent_type=agent_type,
                task=task,
                metadata=metadata,
            )
            updated = await self._runner.execute_run(record.run_id)

            # Build a simple result object from the record
            result_data = updated.result or {}

            class _R:
                success = result_data.get("success", False)
                steps = result_data.get("steps", 0)
                tokens = result_data.get("tokens", 0)

            return _R(), baseline_steps, baseline_tokens

        except Exception as exc:
            logger.warning("Test case execution raised: %s", exc)
            raise

    async def _get_base_config(self, agent_type: str) -> dict[str, Any]:
        """Return the current base configuration for the agent type."""
        return {"agent_type": agent_type}

    async def _apply_patch_to_config(
        self, base_config: dict[str, Any], patch: Any
    ) -> dict[str, Any]:
        """Deep copy base_config and apply the patch operations.

        Supports ops: append, prepend, replace, remove, set, add_example.
        """
        config = copy.deepcopy(base_config)

        op = getattr(patch, "op", "append")
        path = getattr(patch, "path", "system_prompt")
        value = getattr(patch, "value", "")
        target = getattr(patch, "target", "system_prompt")

        # Navigate path (dotted notation)
        parts = path.split(".") if path and "." in path else [path or target]
        current = config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        key = parts[-1] if parts else "system_prompt"

        if op == "append":
            existing = current.get(key, "")
            current[key] = (existing + "\n" + str(value)).strip()
        elif op == "prepend":
            existing = current.get(key, "")
            current[key] = (str(value) + "\n" + existing).strip()
        elif op == "replace":
            existing = current.get(key, "")
            if isinstance(value, dict) and "old" in value and "new" in value:
                current[key] = existing.replace(value["old"], value["new"])
            else:
                current[key] = str(value)
        elif op == "remove":
            existing = current.get(key, "")
            current[key] = existing.replace(str(value), "").strip()
        elif op in ("set", "add_example"):
            current[key] = value
        else:
            logger.warning("Unknown patch op '%s' — skipping", op)

        return config


class PatchEvaluator:
    """Evaluates a patch by running before/after evals on a dataset.

    Args:
        eval_runner:    An EvalRunner instance.
        prompt_manager: PromptManager for reading/applying prompts.
    """

    def __init__(self, eval_runner: Any, prompt_manager: Any) -> None:
        self._runner = eval_runner
        self._pm = prompt_manager

    async def score_patch(
        self,
        patch: Any,
        dataset: Any,
        tenant_id: str = "hermes-eval",
    ) -> float:
        """Score a patch by comparing pre/post eval reports.

        Args:
            patch:     The Patch to evaluate.
            dataset:   EvalDataset to run against.
            tenant_id: Tenant for eval runs.

        Returns:
            Score in [0, 1] representing improvement.
            0.5 = no change, >0.5 = improvement, <0.5 = regression.
        """
        agent_type = getattr(patch, "agent_type", "")

        # Capture the EXACT active version BEFORE touching anything, so we can
        # restore precisely it on failure or regression. rollback(steps=1) is
        # unsafe here: if apply_patch fails before promoting a new version, a
        # blind step-back demotes the pre-existing good version.
        baseline_version_id = ""
        try:
            active = await self._pm.get_version(agent_type)
            baseline_version_id = active.version_id if active else ""
        except Exception as exc:
            logger.debug("Could not capture baseline version: %s", exc)

        async def _restore_baseline(context: str) -> None:
            if not baseline_version_id:
                return
            try:
                await self._pm.promote(baseline_version_id)
                logger.info("Restored baseline version %s (%s)",
                            baseline_version_id[:8], context)
            except Exception as exc:
                logger.warning("Failed to restore baseline version (%s): %s",
                               context, exc)

        # Baseline: current active prompt
        baseline_report = await self._runner.run(
            dataset=dataset,
            tenant_id=tenant_id,
            prompt_version="baseline",
        )

        # Apply patch temporarily and eval
        promoted = False
        try:
            new_version = await self._pm.apply_patch(patch)
            promoted = True
            patched_report = await self._runner.run(
                dataset=dataset,
                tenant_id=tenant_id,
                prompt_version=new_version.version_id,
            )
        except Exception as exc:
            logger.error("Error applying patch for eval: %s", exc)
            # Only restore if the patch was actually promoted; otherwise the
            # active version is untouched and must be left alone.
            if promoted:
                await _restore_baseline("apply/eval failed after promotion")
            return 0.0

        baseline_sr = baseline_report.success_rate
        patched_sr = patched_report.success_rate

        # Normalise improvement to [0, 1] score
        # 0.5 baseline → improvement maps 0..+1 to 0.5..1.0, regression to 0..0.5
        delta = patched_sr - baseline_sr
        score = 0.5 + delta / 2.0  # maps [-1, +1] -> [0, 1]
        score = max(0.0, min(1.0, score))

        logger.info(
            "Patch %s: baseline_sr=%.3f patched_sr=%.3f score=%.3f",
            getattr(patch, "patch_id", "?")[:8],
            baseline_sr,
            patched_sr,
            score,
        )

        # Keep the patch promoted ONLY on strict improvement. A tie
        # (patched_sr == baseline_sr) adds prompt complexity for no gain, so we
        # restore the exact baseline version rather than leaving it promoted.
        if patched_sr <= baseline_sr:
            await _restore_baseline(
                "regression" if patched_sr < baseline_sr else "no improvement (tie)"
            )

        return score
