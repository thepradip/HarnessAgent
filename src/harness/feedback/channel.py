"""
Real-time feedback channel for running agents.

A FeedbackChannel is a per-run Redis Stream where external systems (humans,
evaluators, monitoring) can publish FeedbackEvents. The agent reads and
applies pending feedback before each LLM call.

Feedback types
──────────────
correction  Inject a correction into the agent's context as a system message.
            The agent will see it on the next step and adjust its behaviour.
hint        Soft guidance — injected as a low-priority system message.
score       Real-time quality score + optional hint. Logged to metrics;
            if score < threshold also injects the hint as a correction.
stop        Signal the agent to stop cleanly after the current step.
redirect    Replace the agent's remaining task description.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

FeedbackType = Literal["correction", "hint", "score", "stop", "redirect"]

_STREAM_PFX = "harness:feedback:"     # Redis Stream per run_id
_STREAM_TTL = 86_400                  # 24 h
_SCORE_INJECT_THRESHOLD = 0.40        # inject hint when score < this


@dataclass
class FeedbackEvent:
    """A structured feedback event published into a running agent's channel."""

    feedback_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = ""
    type: FeedbackType = "hint"
    content: str = ""
    score: float | None = None        # for type="score"
    source: str = "human"             # "human" | "evaluator" | "auto"
    priority: int = 2                 # 1=low 2=medium 3=high
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied: bool = False
    applied_at: datetime | None = None

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "run_id": self.run_id,
            "type": self.type,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "applied": self.applied,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeedbackEvent":
        def _dt(v: Any) -> datetime | None:
            if not v:
                return None
            try:
                return datetime.fromisoformat(v)
            except (ValueError, TypeError):
                return None

        return cls(
            feedback_id=d.get("feedback_id", uuid.uuid4().hex),
            run_id=d.get("run_id", ""),
            type=d.get("type", "hint"),
            content=d.get("content", ""),
            score=float(d["score"]) if d.get("score") is not None else None,
            source=d.get("source", "human"),
            priority=int(d.get("priority", 2)),
            created_at=_dt(d.get("created_at")) or datetime.now(timezone.utc),
            applied=bool(d.get("applied", False)),
            applied_at=_dt(d.get("applied_at")),
        )

    def to_context_message(self) -> str:
        """Format this event as a system message to inject into agent context."""
        tag = {
            "correction": "[FEEDBACK:CORRECTION]",
            "hint":       "[FEEDBACK:HINT]",
            "score":      "[FEEDBACK:SCORE]",
            "stop":       "[FEEDBACK:STOP]",
            "redirect":   "[FEEDBACK:REDIRECT]",
        }.get(self.type, "[FEEDBACK]")
        score_str = f" (score={self.score:.2f})" if self.score is not None else ""
        return f"{tag}{score_str} {self.content}".strip()


class FeedbackChannel:
    """
    Per-run Redis Stream for real-time feedback to a running agent.

    Publisher side (external systems):
        await channel.publish(run_id, FeedbackEvent(...))

    Consumer side (agent, called each step):
        events = await channel.poll(run_id, last_id="0")
        for ev in events:
            # apply ev
        await channel.mark_applied(run_id, [ev.feedback_id for ev in events], new_last_id)
    """

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    def _key(self, run_id: str) -> str:
        return f"{_STREAM_PFX}{run_id}"

    # ------------------------------------------------------------------
    # Publish (external → agent)
    # ------------------------------------------------------------------

    async def publish(self, run_id: str, event: FeedbackEvent) -> str:
        """
        Append a FeedbackEvent to the run's Redis Stream.

        Returns the Redis stream entry ID.
        """
        event.run_id = run_id
        key = self._key(run_id)
        fields = {k: json.dumps(v, default=str) for k, v in event.to_dict().items()}
        try:
            entry_id = await self._redis.xadd(key, fields)
            await self._redis.expire(key, _STREAM_TTL)
            logger.debug(
                "FeedbackChannel: published %s/%s to run=%s",
                event.type, event.feedback_id[:8], run_id[:8],
            )
            return entry_id if isinstance(entry_id, str) else entry_id.decode()
        except Exception as exc:
            logger.warning("FeedbackChannel.publish failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Poll (agent, non-blocking)
    # ------------------------------------------------------------------

    async def poll(
        self,
        run_id: str,
        last_id: str = "0",
        count: int = 10,
    ) -> tuple[list[FeedbackEvent], str]:
        """
        Read up to *count* unread feedback events from the stream.

        Returns (events, new_last_id). Pass new_last_id back on the next
        poll call so you don't re-read the same entries.

        Non-blocking (block=0) — always returns immediately.
        """
        key = self._key(run_id)
        try:
            # block=None means non-blocking (returns immediately if no data).
            # block=0 would mean "block indefinitely" — never use that here.
            entries = await self._redis.xread({key: last_id}, count=count)
        except Exception as exc:
            logger.debug("FeedbackChannel.poll failed: %s", exc)
            return [], last_id

        if not entries:
            return [], last_id

        events: list[FeedbackEvent] = []
        new_last_id = last_id

        for _stream, messages in entries:
            for entry_id, raw_fields in messages:
                new_last_id = entry_id if isinstance(entry_id, str) else entry_id.decode()
                try:
                    d: dict[str, Any] = {}
                    for k, v in raw_fields.items():
                        key_str = k if isinstance(k, str) else k.decode()
                        val_str = v if isinstance(v, str) else v.decode()
                        try:
                            d[key_str] = json.loads(val_str)
                        except (json.JSONDecodeError, TypeError):
                            d[key_str] = val_str
                    events.append(FeedbackEvent.from_dict(d))
                except Exception as exc:
                    logger.debug("FeedbackChannel: bad entry %s: %s", new_last_id, exc)

        return events, new_last_id

    # ------------------------------------------------------------------
    # Mark applied
    # ------------------------------------------------------------------

    async def mark_applied(self, run_id: str, feedback_ids: list[str]) -> None:
        """Update applied=True on the stored feedback events (best-effort)."""
        # XRANGE to find entries and update them isn't atomic in Redis Streams;
        # instead we write a companion applied-set key for O(1) lookup.
        if not feedback_ids:
            return
        applied_key = f"{self._key(run_id)}:applied"
        try:
            await self._redis.sadd(applied_key, *feedback_ids)
            await self._redis.expire(applied_key, _STREAM_TTL)
        except Exception as exc:
            logger.debug("FeedbackChannel.mark_applied failed: %s", exc)

    async def is_applied(self, run_id: str, feedback_id: str) -> bool:
        applied_key = f"{self._key(run_id)}:applied"
        try:
            return bool(await self._redis.sismember(applied_key, feedback_id))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Read history (for API / dashboard)
    # ------------------------------------------------------------------

    async def history(
        self,
        run_id: str,
        count: int = 50,
    ) -> list[FeedbackEvent]:
        """Return all feedback events for a run (newest-last)."""
        key = self._key(run_id)
        try:
            entries = await self._redis.xrange(key, count=count)
        except Exception as exc:
            logger.debug("FeedbackChannel.history failed: %s", exc)
            return []

        events: list[FeedbackEvent] = []
        applied_key = f"{key}:applied"
        try:
            applied_ids: set[str] = set(await self._redis.smembers(applied_key))
        except Exception:
            applied_ids = set()

        for _entry_id, raw_fields in entries:
            try:
                d: dict[str, Any] = {}
                for k, v in raw_fields.items():
                    key_str = k if isinstance(k, str) else k.decode()
                    val_str = v if isinstance(v, str) else v.decode()
                    try:
                        d[key_str] = json.loads(val_str)
                    except (json.JSONDecodeError, TypeError):
                        d[key_str] = val_str
                ev = FeedbackEvent.from_dict(d)
                ev.applied = ev.feedback_id in applied_ids
                events.append(ev)
            except Exception:
                pass
        return events

    async def clear(self, run_id: str) -> None:
        """Delete the feedback stream for a run."""
        try:
            await self._redis.delete(self._key(run_id))
            await self._redis.delete(f"{self._key(run_id)}:applied")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Apply helper used by BaseAgent
# ---------------------------------------------------------------------------

def should_inject(event: FeedbackEvent) -> bool:
    """Return True when the event should be added to agent context."""
    if event.type in ("correction", "hint", "redirect"):
        return True
    if event.type == "score" and event.score is not None:
        return event.score < _SCORE_INJECT_THRESHOLD
    return False
