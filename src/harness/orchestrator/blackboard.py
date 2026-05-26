"""AgentBlackboard — shared typed artifact store for multi-agent plans.

The paper (arXiv:2605.18747 §4.3) identifies the lack of formal, persistent
shared state as the root cause of multi-agent brittleness:

    "The majority of the literature resides in the implicit/file-only category,
     lacking any formal model of the shared harness substrate."

AgentBlackboard provides a lightweight blackboard that agents in a plan can
read and write during execution. Instead of receiving only a text summary of
predecessor outputs, a downstream agent can query the exact artifact it needs:

    # Scheduler writes after each subtask:
    await bb.write("sql_agent", "sql", "SELECT * FROM orders WHERE ...")
    await bb.write("sql_agent", "test_result", "42 rows, schema ok")

    # Downstream code_agent reads before running:
    entries = await bb.read(artifact_types=["sql", "test_result"])
    context  = await bb.format_for_context(subtask_ids=["sql_agent"])

Storage: Redis hashes with TTL (default 24 h). Each artifact is stored at:
    harness:blackboard:{plan_id}:{artifact_type}:{subtask_id}
A plan-level ZSET index (scored by write timestamp) allows efficient listing.

All reads/writes are best-effort — failures are logged but never raise so the
Scheduler continues even if Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_BLACKBOARD_PREFIX = "harness:blackboard"
_DEFAULT_TTL = 86_400  # 24 hours


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BlackboardEntry:
    """A single artifact written to the blackboard."""

    plan_id: str
    subtask_id: str
    artifact_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    written_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "subtask_id": self.subtask_id,
            "artifact_type": self.artifact_type,
            "content": self.content,
            "metadata": self.metadata,
            "written_at": self.written_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlackboardEntry":
        d = dict(d)
        written_at = d.get("written_at")
        if isinstance(written_at, str):
            d["written_at"] = datetime.fromisoformat(written_at)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# AgentBlackboard
# ---------------------------------------------------------------------------

class AgentBlackboard:
    """Per-plan Redis-backed blackboard.

    Agents in a multi-agent plan write typed artifacts (code, SQL, test
    results, analysis output, file lists) and downstream agents read them
    during context assembly — replacing implicit text-only handoffs with an
    explicit, queryable shared substrate.

    Usage in Scheduler:

        bb = AgentBlackboard(redis, plan_id=plan.plan_id)
        # After subtask completes:
        await bb.write("sql_subtask", "sql", result.output)
        # Before dependent subtask runs:
        context = await bb.format_for_context(subtask_ids=["sql_subtask"])
    """

    # Known artifact types — open-ended, but these are the common ones
    TYPE_OUTPUT = "output"
    TYPE_CODE = "code"
    TYPE_SQL = "sql"
    TYPE_TEST_RESULT = "test_result"
    TYPE_FILE_LIST = "file_list"
    TYPE_ANALYSIS = "analysis"
    TYPE_PLAN = "plan"
    TYPE_ERROR = "error"

    def __init__(
        self,
        redis: Any,
        plan_id: str,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._redis = redis
        self._plan_id = plan_id
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def write(
        self,
        subtask_id: str,
        artifact_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist an artifact to the blackboard.

        Overwrites any existing entry for ``(subtask_id, artifact_type)``.
        Never raises — failures are logged and swallowed.
        """
        if not content:
            return
        entry = BlackboardEntry(
            plan_id=self._plan_id,
            subtask_id=subtask_id,
            artifact_type=artifact_type,
            content=content,
            metadata=metadata or {},
        )
        artifact_key = self._artifact_key(subtask_id, artifact_type)
        index_key = self._index_key()
        score = entry.written_at.timestamp()

        try:
            await self._redis.setex(
                artifact_key,
                self._ttl,
                json.dumps(entry.to_dict(), ensure_ascii=False, default=str),
            )
            await self._redis.zadd(index_key, {artifact_key: score})
            await self._redis.expire(index_key, self._ttl)
            logger.debug(
                "Blackboard write: plan=%s subtask=%s type=%s (%d chars)",
                self._plan_id[:8], subtask_id, artifact_type, len(content),
            )
        except Exception as exc:
            logger.warning("Blackboard.write failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def read(
        self,
        subtask_ids: list[str] | None = None,
        artifact_types: list[str] | None = None,
    ) -> list[BlackboardEntry]:
        """Return blackboard entries matching the given filters.

        Args:
            subtask_ids:    If given, only return entries for these subtasks.
            artifact_types: If given, only return entries of these types.

        Returns:
            Entries sorted by write time (oldest first).
        """
        try:
            # List all artifact keys from the plan index
            index_key = self._index_key()
            raw_keys: list[bytes | str] = await self._redis.zrange(
                index_key, 0, -1
            )
        except Exception as exc:
            logger.warning("Blackboard.read index scan failed: %s", exc)
            return []

        entries: list[BlackboardEntry] = []
        for raw_key in raw_keys:
            key_str = raw_key if isinstance(raw_key, str) else raw_key.decode()

            # Parse subtask_id and artifact_type from the key
            # Key format: harness:blackboard:{plan_id}:{artifact_type}:{subtask_id}
            parts = key_str.split(":")
            if len(parts) < 5:
                continue
            a_type = parts[3]
            s_id = ":".join(parts[4:])  # subtask_id may contain colons

            if subtask_ids is not None and s_id not in subtask_ids:
                continue
            if artifact_types is not None and a_type not in artifact_types:
                continue

            try:
                raw = await self._redis.get(key_str)
                if raw is None:
                    continue
                data = json.loads(raw if isinstance(raw, str) else raw.decode())
                entries.append(BlackboardEntry.from_dict(data))
            except Exception as exc:
                logger.debug("Blackboard: corrupt entry at %s: %s", key_str, exc)

        return sorted(entries, key=lambda e: e.written_at)

    async def get(
        self,
        subtask_id: str,
        artifact_type: str,
    ) -> BlackboardEntry | None:
        """Return a single entry by (subtask_id, artifact_type), or None."""
        try:
            raw = await self._redis.get(self._artifact_key(subtask_id, artifact_type))
            if raw is None:
                return None
            data = json.loads(raw if isinstance(raw, str) else raw.decode())
            return BlackboardEntry.from_dict(data)
        except Exception as exc:
            logger.debug("Blackboard.get failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    async def format_for_context(
        self,
        subtask_ids: list[str] | None = None,
        artifact_types: list[str] | None = None,
        max_chars: int = 4_000,
    ) -> str:
        """Return a formatted context block for agent injection.

        Downstream agents call this before their LLM turn to get a
        structured view of what predecessor agents produced.
        """
        entries = await self.read(subtask_ids=subtask_ids, artifact_types=artifact_types)
        if not entries:
            return ""

        parts = ["[Shared plan artifacts]"]
        total = len(parts[0])

        for entry in entries:
            block = (
                f"\n--- {entry.subtask_id} / {entry.artifact_type} "
                f"({entry.written_at.strftime('%H:%M:%S')}) ---\n"
                f"{entry.content}"
            )
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 80:
                    parts.append(block[:remaining] + "\n... [truncated]")
                break
            parts.append(block)
            total += len(block)

        return "\n".join(parts) if len(parts) > 1 else ""

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def list_entries(self) -> list[tuple[str, str, datetime]]:
        """Return (subtask_id, artifact_type, written_at) for all entries."""
        entries = await self.read()
        return [(e.subtask_id, e.artifact_type, e.written_at) for e in entries]

    async def delete(self, subtask_id: str, artifact_type: str) -> None:
        """Remove a single entry. Never raises."""
        key = self._artifact_key(subtask_id, artifact_type)
        try:
            await self._redis.delete(key)
            await self._redis.zrem(self._index_key(), key)
        except Exception as exc:
            logger.debug("Blackboard.delete failed: %s", exc)

    async def clear(self) -> None:
        """Remove all entries for this plan. Never raises."""
        try:
            raw_keys = await self._redis.zrange(self._index_key(), 0, -1)
            if raw_keys:
                await self._redis.delete(*raw_keys)
            await self._redis.delete(self._index_key())
        except Exception as exc:
            logger.debug("Blackboard.clear failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal key builders
    # ------------------------------------------------------------------

    def _artifact_key(self, subtask_id: str, artifact_type: str) -> str:
        return f"{_BLACKBOARD_PREFIX}:{self._plan_id}:{artifact_type}:{subtask_id}"

    def _index_key(self) -> str:
        return f"{_BLACKBOARD_PREFIX}:{self._plan_id}:index"
