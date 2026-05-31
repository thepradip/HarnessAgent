"""GEPA adapter bridging the optimizer to HarnessAgent's Evaluator.

The adapter is GEPA's single integration point. It:

1. ``evaluate`` — instantiates the agent's system prompt with a candidate's text,
   replays each previously-failing task through the existing :class:`Evaluator`,
   and returns *per-example* scores (GEPA needs one score per item; the Evaluator
   natively returns an aggregate, so we score one record at a time, concurrently).
2. ``make_reflective_dataset`` — turns the captured trajectories (task,
   failure class, error message, outcome) into the natural-language feedback the
   reflection LM evolves the prompt from.

Reusing the Evaluator means GEPA optimizes against *exactly* the same metric the
Hermes loop already uses to gate patches — so an evolved prompt that scores well
here is scored the same way by the loop's downstream safety gate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from harness.improvement.gepa.reflection import CoroRunner
from harness.improvement.patch_generator import Patch

logger = logging.getLogger(__name__)

# Name of the single optimizable component (the agent system prompt).
COMPONENT = "system_prompt"


class HarnessGepaAdapter:
    """Implements GEPA's ``GEPAAdapter`` protocol over the Hermes Evaluator.

    Args:
        evaluator:  Object exposing
            ``async score(patch, test_cases, agent_type) -> EvalResult`` where
            ``EvalResult`` has ``.score`` (float) and ``.successes`` (int).
        agent_type: The agent type whose prompt is being optimized.
        run_coro:   Blocking coroutine runner (see ``make_coro_runner``) used to
            drive the async Evaluator from GEPA's synchronous worker thread.
    """

    # GEPA's reflective-mutation proposer probes this optional hook. Setting it to
    # None tells GEPA to use its built-in instruction proposer, which serializes
    # our reflective dataset and calls the reflection LM to draft the new prompt.
    propose_new_texts = None

    def __init__(
        self,
        evaluator: Any,
        agent_type: str,
        run_coro: CoroRunner,
    ) -> None:
        self._evaluator = evaluator
        self._agent_type = agent_type
        self._run_coro = run_coro

    # ------------------------------------------------------------------
    # GEPAAdapter.evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Score ``candidate`` on ``batch`` (one ErrorRecord per item).

        Never raises for per-example failures — a failed replay scores 0.0 so the
        optimizer can still learn from it (per GEPA's adapter contract).
        """
        from gepa.core.adapter import EvaluationBatch  # local import: optional dep

        prompt_text = candidate.get(COMPONENT, "")
        # A whole-prompt replacement, applied through the same Patch path the
        # Evaluator/PromptManager already understand (op="set").
        patch = Patch(
            agent_type=self._agent_type,
            target="prompt",
            op="set",
            path=COMPONENT,
            value=prompt_text,
            proposed_by="hermes-gepa",
        )

        async def _score_all() -> list[Any]:
            return await asyncio.gather(
                *(
                    self._evaluator.score(
                        patch=patch, test_cases=[record], agent_type=self._agent_type
                    )
                    for record in batch
                ),
                return_exceptions=True,
            )

        results = self._run_coro(_score_all())

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for record, result in zip(batch, results, strict=False):
            if isinstance(result, BaseException) or result is None:
                if isinstance(result, BaseException):
                    logger.debug("GEPA eval: record scored as failure: %s", result)
                score = 0.0
                success = False
            else:
                score = float(getattr(result, "score", 0.0) or 0.0)
                success = int(getattr(result, "successes", 0) or 0) > 0

            scores.append(score)
            outputs.append({"task": getattr(record, "task", ""), "success": success})

            if trajectories is not None:
                trajectories.append(
                    {
                        "task": getattr(record, "task", ""),
                        "failure_class": getattr(record, "failure_class", "UNKNOWN"),
                        "error_message": getattr(record, "error_message", ""),
                        "success": success,
                        "score": score,
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
        """Build the per-component feedback dataset GEPA reflects on."""
        trajectories = getattr(eval_batch, "trajectories", None) or []

        records: list[dict[str, Any]] = []
        for traj in trajectories:
            records.append(
                {
                    "Inputs": {"task": traj.get("task", "")},
                    "Generated Outputs": (
                        "Task succeeded."
                        if traj.get("success")
                        else "Task failed after the agent ran with this prompt."
                    ),
                    "Feedback": _build_feedback(traj),
                }
            )

        # Guard: GEPA requires a non-empty dataset per component being updated.
        if not records:
            records.append(
                {
                    "Inputs": {"task": ""},
                    "Generated Outputs": "No trajectories were captured.",
                    "Feedback": "No execution feedback available for this minibatch.",
                }
            )

        return dict.fromkeys(components_to_update, records)


def _build_feedback(traj: dict[str, Any]) -> str:
    """Compose natural-language feedback for one trajectory."""
    if traj.get("success"):
        return (
            "This task now succeeds with the current prompt. Preserve the guidance "
            "that makes it work; do not regress it."
        )

    failure_class = traj.get("failure_class", "UNKNOWN")
    error_message = str(traj.get("error_message", "")).strip()[:400]
    parts = [
        f"This task still FAILS. Failure class: {failure_class}.",
    ]
    if error_message:
        parts.append(f"Error: {error_message}")
    parts.append(
        "Revise the system prompt to prevent this specific failure — add or clarify "
        "instructions addressing its root cause without bloating the prompt."
    )
    return " ".join(parts)
