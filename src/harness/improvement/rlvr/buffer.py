"""StepRewardBuffer — Redis-backed per-episode reward store."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_EPISODE_PFX = "harness:rlvr:episode:"     # LIST   per run_id
_BASELINE_PFX = "harness:rlvr:baseline:"   # HASH   per agent_type
_EPISODE_TTL = 86_400 * 7                   # 7 days
_BASELINE_WINDOW = 50                       # episodes for rolling mean


@dataclass
class StepReward:
    run_id: str
    step: int
    agent_type: str
    task: str
    action: str                # SQL, code snippet, tool call, etc.
    result_preview: str        # first 500 chars of tool result
    reward: float              # 0.0 – 1.0
    verdict: str               # correct | partial | incorrect
    confidence: float
    source: str                # execution | llm_verifier | ensemble
    prompt_hash: str           # sha256 of the prompt used this step
    reasoning: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "agent_type": self.agent_type,
            "task": self.task,
            "action": self.action,
            "result_preview": self.result_preview,
            "reward": self.reward,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "source": self.source,
            "prompt_hash": self.prompt_hash,
            "reasoning": self.reasoning,
            "ts": self.ts.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepReward":
        ts = d.get("ts")
        return cls(
            run_id=d["run_id"],
            step=int(d["step"]),
            agent_type=d.get("agent_type", "base"),
            task=d.get("task", ""),
            action=d.get("action", ""),
            result_preview=d.get("result_preview", ""),
            reward=float(d["reward"]),
            verdict=d.get("verdict", "incorrect"),
            confidence=float(d.get("confidence", 0.5)),
            source=d.get("source", "unknown"),
            prompt_hash=d.get("prompt_hash", ""),
            reasoning=d.get("reasoning", ""),
            ts=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
        )


class StepRewardBuffer:
    """
    Redis-backed buffer for per-episode step rewards.

    Episode (one agent run):
        record(step_reward)  →  RPUSH harness:rlvr:episode:{run_id}

    End of episode:
        get_episode(run_id)  →  list[StepReward]

    Baseline (per agent type):
        update_baseline(agent_type, episode_mean)
        get_baseline(agent_type) → float
    """

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    # ------------------------------------------------------------------
    # Episode recording
    # ------------------------------------------------------------------

    async def record(self, step: StepReward) -> None:
        key = f"{_EPISODE_PFX}{step.run_id}"
        try:
            await self._redis.rpush(key, json.dumps(step.to_dict()))
            await self._redis.expire(key, _EPISODE_TTL)
        except Exception as exc:
            logger.debug("StepRewardBuffer.record failed: %s", exc)

    async def get_episode(self, run_id: str) -> list[StepReward]:
        key = f"{_EPISODE_PFX}{run_id}"
        try:
            raw_items = await self._redis.lrange(key, 0, -1)
        except Exception as exc:
            logger.debug("StepRewardBuffer.get_episode failed: %s", exc)
            return []
        steps: list[StepReward] = []
        for raw in raw_items:
            try:
                steps.append(StepReward.from_dict(json.loads(raw)))
            except Exception:
                pass
        return steps

    async def episode_mean(self, run_id: str) -> float:
        steps = await self.get_episode(run_id)
        if not steps:
            return 0.0
        return sum(s.reward for s in steps) / len(steps)

    async def delete_episode(self, run_id: str) -> None:
        try:
            await self._redis.delete(f"{_EPISODE_PFX}{run_id}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Baseline (rolling mean over last N episodes per agent_type)
    # ------------------------------------------------------------------

    async def update_baseline(self, agent_type: str, episode_mean: float) -> None:
        """Push episode_mean into a rolling list; recompute baseline."""
        key = f"{_BASELINE_PFX}{agent_type}"
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.rpush(f"{key}:history", json.dumps({"mean": episode_mean,
                                                      "ts": time.time()}))
            pipe.ltrim(f"{key}:history", -_BASELINE_WINDOW, -1)
            pipe.expire(f"{key}:history", _EPISODE_TTL)
            await pipe.execute()

            # Recompute rolling mean
            raw_items = await self._redis.lrange(f"{key}:history", 0, -1)
            means = [json.loads(r)["mean"] for r in raw_items]
            baseline = sum(means) / len(means) if means else 0.5
            await self._redis.hset(key, "baseline", str(baseline))
            await self._redis.expire(key, _EPISODE_TTL)
        except Exception as exc:
            logger.debug("update_baseline failed: %s", exc)

    async def get_baseline(self, agent_type: str) -> float:
        key = f"{_BASELINE_PFX}{agent_type}"
        try:
            val = await self._redis.hget(key, "baseline")
            if val:
                return float(val)
        except Exception:
            pass
        return 0.5  # neutral default

    async def get_history(self, agent_type: str, n: int = 20) -> list[float]:
        key = f"{_BASELINE_PFX}{agent_type}:history"
        try:
            raw_items = await self._redis.lrange(key, -n, -1)
            return [json.loads(r)["mean"] for r in raw_items]
        except Exception:
            return []
