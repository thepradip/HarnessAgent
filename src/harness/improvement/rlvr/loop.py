"""
RLVRLoop — advantage-weighted prompt improvement using verified rewards.

Each completed episode:
  1. Load step rewards from StepRewardBuffer
  2. Compute per-step advantages (AdvantageEstimator)
  3. Reinforce high-advantage steps → add as positive few-shot examples
  4. Patch low-advantage steps → weighted Hermes patch generation
  5. Update rolling baseline
  6. Publish feedback scores to FeedbackChannel (mid-run, if run still active)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from harness.improvement.rlvr.advantage import AdvantageEstimator, StepAdvantage
from harness.improvement.rlvr.buffer import StepRewardBuffer
from harness.improvement.rlvr.verifiers import VerificationResult, get_verifier

logger = logging.getLogger(__name__)

_MIN_STEPS_FOR_UPDATE = 3     # skip episodes with fewer steps
_POS_THRESHOLD = 0.5          # advantage ≥ this → reinforce
_NEG_THRESHOLD = -0.5         # advantage ≤ this → patch
_MAX_FEWSHOT_PER_CYCLE = 3    # positive few-shots added per cycle
_MAX_ERRORS_FOR_PATCH = 5     # negative steps fed to PatchGenerator
_MAX_FEWSHOT_PER_AGENT = 50   # hard cap on stored few-shots per agent_type
_PROCESSED_RUN_MEMORY = 4096  # how many recent run_ids to remember for dedup


def _fewshot_key(task: str, action: str) -> str:
    """Stable hash of a (task, action) pair for few-shot dedup."""
    h = hashlib.sha1(f"{task}\x00{action}".encode("utf-8", "replace"))
    return h.hexdigest()


@dataclass
class RLVRCycleResult:
    run_id: str
    agent_type: str
    n_steps: int
    mean_reward: float
    baseline_before: float
    baseline_after: float
    n_positive: int
    n_negative: int
    patch_applied: bool
    fewshots_added: int
    cycle_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"RLVR cycle run={self.run_id[:8]} agent={self.agent_type} "
            f"steps={self.n_steps} mean_r={self.mean_reward:.3f} "
            f"baseline={self.baseline_before:.3f}→{self.baseline_after:.3f} "
            f"pos={self.n_positive} neg={self.n_negative} "
            f"patch={'yes' if self.patch_applied else 'no'} "
            f"fewshots={self.fewshots_added}"
        )


class RLVRLoop:
    """
    Advantage-weighted prompt improvement loop.

    Designed to run as a background task after each agent episode completes.
    Plugs into the existing Hermes infrastructure (PatchGenerator, PromptManager).
    """

    def __init__(
        self,
        reward_buffer: StepRewardBuffer,
        estimator: AdvantageEstimator,
        patch_generator: Any,        # harness.improvement.patch_generator.PatchGenerator
        prompt_store: Any,           # PromptManager / PromptStore
        feedback_channel: Any,       # FeedbackChannel (optional, for mid-run scores)
        min_steps: int = _MIN_STEPS_FOR_UPDATE,
        pos_threshold: float = _POS_THRESHOLD,
        neg_threshold: float = _NEG_THRESHOLD,
    ) -> None:
        self._buffer = reward_buffer
        self._estimator = estimator
        self._patch_gen = patch_generator
        self._prompt_store = prompt_store
        self._feedback = feedback_channel
        self._min_steps = min_steps
        self._pos_threshold = pos_threshold
        self._neg_threshold = neg_threshold
        # Idempotency + dedup state (in-memory; the loop is a long-lived object).
        self._processed_runs: set[str] = set()
        self._processed_order: deque[str] = deque(maxlen=_PROCESSED_RUN_MEMORY)
        # Per-agent set of (task, action) hashes already stored as few-shots,
        # and a count so we can cap how many we keep.
        self._fewshot_hashes: dict[str, set[str]] = {}

    async def process_episode(
        self,
        run_id: str,
        agent_type: str,
    ) -> RLVRCycleResult | None:
        """
        Process a completed episode. Returns None if episode has too few steps.
        Call this after receiving a 'completed' or 'failed' StepEvent for the run.

        Idempotent: a run_id is processed at most once. Re-delivery of the same
        'completed' event will not append duplicate few-shots or re-patch.
        """
        if run_id in self._processed_runs:
            logger.debug("RLVR: run=%s already processed — skipping", run_id[:8])
            return None

        episode = await self._buffer.get_episode(run_id)
        if len(episode) < self._min_steps:
            logger.debug(
                "RLVR: skipping run=%s (only %d steps, min=%d)",
                run_id[:8], len(episode), self._min_steps,
            )
            return None

        # 1. Advantage estimation
        baseline = await self._buffer.get_baseline(agent_type)
        advantages = self._estimator.compute(episode, baseline)
        positive, negative = self._estimator.split(
            advantages, self._pos_threshold, self._neg_threshold
        )

        mean_reward = sum(s.step.reward for s in advantages) / len(advantages)

        # 2. Reinforce positive steps (add as few-shot examples)
        fewshots_added = await self._reinforce(agent_type, positive)

        # 3. Patch negative steps (weighted error → PatchGenerator)
        patch_applied = await self._patch(agent_type, negative)

        # 4. Update baseline
        await self._buffer.update_baseline(agent_type, mean_reward)
        new_baseline = await self._buffer.get_baseline(agent_type)

        result = RLVRCycleResult(
            run_id=run_id,
            agent_type=agent_type,
            n_steps=len(episode),
            mean_reward=round(mean_reward, 4),
            baseline_before=round(baseline, 4),
            baseline_after=round(new_baseline, 4),
            n_positive=len(positive),
            n_negative=len(negative),
            patch_applied=patch_applied,
            fewshots_added=fewshots_added,
        )

        # 5. Mark processed and drop the episode buffer so a re-delivered
        #    'completed' event cannot trigger a second cycle for this run.
        self._mark_processed(run_id)
        await self._buffer.delete_episode(run_id)

        logger.info(result.summary())
        return result

    def _mark_processed(self, run_id: str) -> None:
        if len(self._processed_order) == self._processed_order.maxlen:
            self._processed_runs.discard(self._processed_order[0])
        self._processed_order.append(run_id)
        self._processed_runs.add(run_id)

    # ------------------------------------------------------------------
    # Reinforce: store high-advantage (task, action) pairs as few-shots
    # ------------------------------------------------------------------

    async def _reinforce(
        self,
        agent_type: str,
        positive: list[StepAdvantage],
        max_examples: int = _MAX_FEWSHOT_PER_CYCLE,
    ) -> int:
        if not positive or self._prompt_store is None:
            return 0

        seen = self._fewshot_hashes.setdefault(agent_type, set())
        if len(seen) >= _MAX_FEWSHOT_PER_AGENT:
            logger.debug(
                "RLVR: few-shot cap (%d) reached for %s — not adding more",
                _MAX_FEWSHOT_PER_AGENT, agent_type,
            )
            return 0

        # Sort by advantage (highest first) and take top N
        top = sorted(positive, key=lambda a: -a.advantage)[:max_examples]
        added = 0
        for adv in top:
            # Dedup on (task, action): the same winning pattern recorded across
            # episodes must not be appended repeatedly.
            fp = _fewshot_key(adv.step.task, adv.step.action)
            if fp in seen:
                continue
            if len(seen) >= _MAX_FEWSHOT_PER_AGENT:
                break
            try:
                example = (
                    f"# Task: {adv.step.task}\n"
                    f"# Action: {adv.step.action}\n"
                    f"# Result: {adv.step.result_preview[:200]}\n"
                    f"# Reward: {adv.step.reward:.2f} (advantage={adv.advantage:.2f})"
                )
                if hasattr(self._prompt_store, "add_fewshot"):
                    await _safe(self._prompt_store.add_fewshot(agent_type, example, label="positive"))
                elif hasattr(self._prompt_store, "apply_patch"):
                    from harness.improvement.patch_generator import Patch
                    patch = Patch(
                        agent_type=agent_type,
                        op="append",
                        value=f"\n\n## Example (reward={adv.step.reward:.2f})\n{example}",
                        rationale=f"High-advantage step (A={adv.advantage:.2f}) — reinforce this pattern.",
                        proposed_by="rlvr",
                    )
                    await _safe(self._prompt_store.apply_patch(patch))
                seen.add(fp)
                added += 1
            except Exception as exc:
                logger.debug("reinforce fewshot failed: %s", exc)

        logger.debug("RLVR: added %d few-shot examples for %s", added, agent_type)
        return added

    # ------------------------------------------------------------------
    # Patch: generate advantage-weighted prompt patch from negative steps
    # ------------------------------------------------------------------

    async def _patch(
        self,
        agent_type: str,
        negative: list[StepAdvantage],
        max_errors: int = _MAX_ERRORS_FOR_PATCH,
    ) -> bool:
        if not negative or self._patch_gen is None:
            return False

        # Sort by most negative advantage first (worst steps)
        worst = sorted(negative, key=lambda a: a.advantage)[:max_errors]

        # Convert to ErrorRecord with weight encoded in error_message
        from harness.improvement.error_collector import ErrorRecord
        error_records = [
            ErrorRecord(
                agent_type=agent_type,
                task=adv.step.task,
                failure_class="LOW_REWARD",
                error_message=(
                    f"reward={adv.step.reward:.2f} advantage={adv.advantage:.2f} "
                    f"verdict={adv.step.verdict} "
                    f"action={adv.step.action[:200]} "
                    f"reasoning={adv.step.reasoning[:200]}"
                ),
                context_snapshot={
                    "reward": adv.step.reward,
                    "advantage": adv.advantage,
                    "prompt_hash": adv.step.prompt_hash,
                    "source": adv.step.source,
                },
            )
            for adv in worst
        ]

        try:
            patch = await self._patch_gen.generate(
                errors=error_records,
                agent_type=agent_type,
            )
            if patch is None:
                return False

            # Apply patch via prompt store
            if self._prompt_store is not None and hasattr(self._prompt_store, "apply_patch"):
                await _safe(self._prompt_store.apply_patch(patch))
                logger.info(
                    "RLVR: applied patch %s for %s (based on %d low-advantage steps)",
                    patch.patch_id[:8], agent_type, len(worst),
                )
                return True
        except Exception as exc:
            logger.warning("RLVR patch generation failed: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Publish mid-run score feedback (if run is still active)
    # ------------------------------------------------------------------

    async def publish_step_feedback(
        self,
        run_id: str,
        verification: VerificationResult,
    ) -> None:
        """Push verification result back into the running agent via FeedbackChannel."""
        if self._feedback is None:
            return
        from harness.feedback.channel import FeedbackEvent
        try:
            ev = FeedbackEvent(
                run_id=run_id,
                type="score",
                content=verification.feedback_for_agent,
                score=verification.overall_reward,
                source="rlvr_verifier",
                priority=3 if verification.overall_reward < 0.4 else 2,
            )
            await self._feedback.publish(run_id, ev)
        except Exception as exc:
            logger.debug("publish_step_feedback failed: %s", exc)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _safe(coro: Any) -> None:
    try:
        if asyncio.iscoroutine(coro):
            await coro
    except Exception as exc:
        logger.debug("_safe coroutine error: %s", exc)


# ---------------------------------------------------------------------------
# Convenience: run RLVR on a completed episode with a verifier
# ---------------------------------------------------------------------------

async def run_episode_with_verification(
    run_id: str,
    agent_type: str,
    steps: list[dict],          # list of {task, action, result, gold}
    reward_buffer: StepRewardBuffer,
    rlvr_loop: RLVRLoop,
    verifier: Any,              # SQLVerifier | CodeVerifier | ReasoningVerifier
    prompt_hash: str = "",
) -> RLVRCycleResult | None:
    """
    Compute rewards for all steps via verifier, record them, then run one RLVR cycle.

    This is the high-level entry point used by the worker / background task.
    """
    from harness.improvement.rlvr.buffer import StepReward

    for i, step_data in enumerate(steps):
        task   = step_data.get("task", "")
        action = step_data.get("action", "")
        result = step_data.get("result")
        gold   = step_data.get("gold")

        try:
            vr: VerificationResult = await verifier.verify(
                task=task, action=action, result=result, gold=gold,
            )
            # Publish step-level feedback back to agent (if still running)
            await rlvr_loop.publish_step_feedback(run_id, vr)
        except Exception as exc:
            logger.warning("Verifier failed at step %d: %s", i, exc)
            from harness.improvement.rlvr.verifiers import VerificationResult as VR
            from harness.improvement.rlvr.verifiers import VerificationStep as VS
            vr = VR(overall_reward=0.5, verdict="partial", steps=[],
                    feedback_for_agent=str(exc), source="error")

        sr = StepReward(
            run_id=run_id,
            step=i,
            agent_type=agent_type,
            task=task,
            action=action,
            result_preview=str(result)[:500] if result else "",
            reward=vr.overall_reward,
            verdict=vr.verdict,
            confidence=vr._confidence(),
            source=vr.source,
            prompt_hash=prompt_hash,
            reasoning=vr.feedback_for_agent[:300],
        )
        await reward_buffer.record(sr)

    return await rlvr_loop.process_episode(run_id, agent_type)
