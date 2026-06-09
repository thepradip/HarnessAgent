"""ErrorCollector — persists agent failure records for Hermes analysis."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ERROR_KEY_PREFIX = "harness:error"
_ERROR_INDEX_KEY = "harness:error_index"  # sorted set: score=timestamp, member=record_id


def _error_key(record_id: str) -> str:
    return f"{_ERROR_KEY_PREFIX}:{record_id}"


@dataclass
class ErrorRecord:
    """A single recorded agent failure.

    Attributes:
        record_id:        Unique identifier.
        agent_type:       Which agent failed.
        task:             The task that was being executed.
        failure_class:    FailureClass value string.
        error_message:    Human-readable error description.
        stack_trace:      Full Python traceback (optional).
        context_snapshot: Snapshot of relevant AgentContext fields.
        created_at:       UTC timestamp of when the error occurred.
    """

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_type: str = ""
    task: str = ""
    failure_class: str = "UNKNOWN"
    error_message: str = ""
    stack_trace: str = ""
    context_snapshot: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "failure_class": self.failure_class,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "context_snapshot": self.context_snapshot,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ErrorRecord":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)
        return cls(
            record_id=data.get("record_id", uuid.uuid4().hex),
            agent_type=data.get("agent_type", ""),
            task=data.get("task", ""),
            failure_class=data.get("failure_class", "UNKNOWN"),
            error_message=data.get("error_message", ""),
            stack_trace=data.get("stack_trace", ""),
            context_snapshot=dict(data.get("context_snapshot", {})),
            created_at=created_at,
        )

    @classmethod
    def from_json(cls, raw: str) -> "ErrorRecord":
        return cls.from_dict(json.loads(raw))


class ErrorCollector:
    """Records and retrieves agent failure events in Redis.

    Args:
        redis: Async Redis client.
        max_records: Maximum number of records to keep per agent_type index.
    """

    def __init__(
        self,
        redis: Any,
        max_records: int = 10_000,
        record_ttl_seconds: int = 86400 * 90,
    ) -> None:
        self._redis = redis
        self._max_records = max_records
        # TTL bounds the lifetime of any record key that slips past the index
        # trim (e.g. under concurrent writes). 0/None disables the TTL.
        self._record_ttl = record_ttl_seconds

    async def record(
        self,
        agent_type: str,
        task: str,
        failure_class: str,
        error_message: str,
        stack_trace: str = "",
        context_snapshot: Optional[dict] = None,
    ) -> ErrorRecord:
        """Persist a new ErrorRecord.

        Args:
            agent_type:      Which agent failed.
            task:            The task being executed.
            failure_class:   FailureClass string.
            error_message:   Error description.
            stack_trace:     Optional full traceback.
            context_snapshot: Optional dict snapshot of AgentContext.

        Returns:
            The created ErrorRecord.
        """
        rec = ErrorRecord(
            agent_type=agent_type,
            task=task,
            failure_class=failure_class,
            error_message=error_message,
            stack_trace=stack_trace,
            context_snapshot=context_snapshot or {},
        )

        timestamp = rec.created_at.timestamp()
        index_key = f"{_ERROR_INDEX_KEY}:{agent_type}"

        # Find the record ids that this insert will push out of the capped index
        # so we can delete their record keys too — trimming only the index would
        # orphan those keys forever (set() carried no TTL). The post-insert trim
        # removes ranks [0, -(max+1)]; before the insert (one fewer element) the
        # same victims are ranks [0, -max].
        try:
            stale_ids = await self._redis.zrange(index_key, 0, -self._max_records)
        except Exception:
            stale_ids = []

        async with self._redis.pipeline(transaction=True) as pipe:
            if self._record_ttl and self._record_ttl > 0:
                pipe.set(_error_key(rec.record_id), rec.to_json(), ex=self._record_ttl)
            else:
                pipe.set(_error_key(rec.record_id), rec.to_json())
            pipe.zadd(index_key, {rec.record_id: timestamp})
            # Trim the index to max_records and drop the evicted record keys.
            pipe.zremrangebyrank(index_key, 0, -(self._max_records + 1))
            for rid_raw in stale_ids or []:
                rid = rid_raw if isinstance(rid_raw, str) else rid_raw.decode()
                if rid != rec.record_id:
                    pipe.delete(_error_key(rid))
            await pipe.execute()

        logger.debug(
            "Recorded error for agent_type=%s failure_class=%s record_id=%s",
            agent_type,
            failure_class,
            rec.record_id[:8],
        )
        return rec

    async def get_recent(
        self,
        agent_type: str,
        limit: int = 50,
    ) -> list[ErrorRecord]:
        """Return the most recent error records for *agent_type*.

        Args:
            agent_type: Agent type to filter by.
            limit:      Maximum number of records to return.

        Returns:
            List of ErrorRecord objects, most recent first.
        """
        index_key = f"{_ERROR_INDEX_KEY}:{agent_type}"
        record_ids = await self._redis.zrevrange(index_key, 0, limit - 1)

        records: list[ErrorRecord] = []
        for rid_raw in record_ids:
            rid = rid_raw if isinstance(rid_raw, str) else rid_raw.decode()
            raw = await self._redis.get(_error_key(rid))
            if raw:
                try:
                    records.append(
                        ErrorRecord.from_json(raw if isinstance(raw, str) else raw.decode())
                    )
                except Exception as exc:
                    logger.warning("Failed to parse ErrorRecord %s: %s", rid, exc)

        return records

    async def count(self, agent_type: str) -> int:
        """Return total number of error records for *agent_type*."""
        index_key = f"{_ERROR_INDEX_KEY}:{agent_type}"
        count = await self._redis.zcard(index_key)
        return int(count)

    async def failure_heatmap(self) -> dict[str, dict[str, int]]:
        """Build a heatmap of failure_class counts per agent_type.

        Returns:
            Dict mapping agent_type -> {failure_class -> count}.
        """
        heatmap: dict[str, dict[str, int]] = {}

        # Discover all agent_type index keys
        pattern = f"{_ERROR_INDEX_KEY}:*"
        async for key in self._redis.scan_iter(match=pattern, count=100):
            key_str = key if isinstance(key, str) else key.decode()
            agent_type = key_str.removeprefix(f"{_ERROR_INDEX_KEY}:")
            records = await self.get_recent(agent_type, limit=1000)
            counts: dict[str, int] = {}
            for rec in records:
                counts[rec.failure_class] = counts.get(rec.failure_class, 0) + 1
            heatmap[agent_type] = counts

        return heatmap

    async def clear(self, agent_type: str) -> int:
        """Delete all error records for *agent_type*.

        Returns:
            Number of records deleted.
        """
        index_key = f"{_ERROR_INDEX_KEY}:{agent_type}"
        record_ids = await self._redis.zrange(index_key, 0, -1)

        deleted = 0
        async with self._redis.pipeline(transaction=True) as pipe:
            for rid_raw in record_ids:
                rid = rid_raw if isinstance(rid_raw, str) else rid_raw.decode()
                pipe.delete(_error_key(rid))
                deleted += 1
            pipe.delete(index_key)
            await pipe.execute()

        return deleted
