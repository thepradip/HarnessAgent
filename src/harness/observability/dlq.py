"""Dead-letter queue backed by Redis Lists."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from harness.core.context import AgentContext
    from harness.observability.failures import StepFailure


@dataclass
class DLQEntry:
    """An entry in the dead-letter queue representing a failed agent run."""

    run_id: str
    tenant_id: str
    agent_type: str
    task: str
    failure: dict[str, Any]  # serialised StepFailure
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    last_retry: datetime | None = None
    status: Literal["pending", "replaying", "resolved", "abandoned"] = "pending"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "failure": self.failure,
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
            "last_retry": self.last_retry.isoformat() if self.last_retry else None,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DLQEntry":
        d = dict(d)
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        if isinstance(d.get("last_retry"), str) and d["last_retry"]:
            d["last_retry"] = datetime.fromisoformat(d["last_retry"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DeadLetterQueue:
    """
    Redis List-backed dead-letter queue for failed agent tasks.

    Queue key pattern: ``harness:dlq:{tenant_id}``
    A secondary sorted set ``harness:dlq_index`` maps entry_id → expire_timestamp
    for global lookup.
    """

    _KEY_PREFIX = "harness:dlq:"
    _INDEX_KEY = "harness:dlq_index"

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(
        self,
        ctx: "AgentContext",
        failure: "StepFailure",
    ) -> DLQEntry:
        """
        Serialise the context + failure and LPUSH to the tenant queue.
        Updates the dlq_depth Prometheus gauge.
        """
        entry = DLQEntry(
            run_id=ctx.run_id,
            tenant_id=ctx.tenant_id,
            agent_type=ctx.agent_type,
            task=ctx.task,
            failure=failure.to_dict(),
        )

        try:
            serialized = json.dumps(entry.to_dict(), default=str)
            queue_key = f"{self._KEY_PREFIX}{ctx.tenant_id}"
            await self._redis.lpush(queue_key, serialized)

            # Update depth gauge
            await self._update_gauge(ctx.tenant_id)

        except Exception as exc:
            logger.error("DLQ push failed: %s", exc)

        return entry

    async def pop(
        self,
        tenant_id: str | None = None,
        k: int = 10,
    ) -> list[DLQEntry]:
        """
        Read the top ``k`` entries from the DLQ without removing them.

        If ``tenant_id`` is None, reads from all tenants (not recommended in production).
        """
        entries: list[DLQEntry] = []

        try:
            if tenant_id:
                queue_key = f"{self._KEY_PREFIX}{tenant_id}"
                raw_items = await self._redis.lrange(queue_key, 0, k - 1)
            else:
                # Scan all dlq keys
                raw_items = []
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor, match=f"{self._KEY_PREFIX}*", count=100
                    )
                    for key in keys:
                        batch = await self._redis.lrange(key, 0, k - 1)
                        raw_items.extend(batch)
                        if len(raw_items) >= k:
                            break
                    if cursor == 0:
                        break

            for raw in raw_items[:k]:
                try:
                    data = json.loads(raw)
                    entries.append(DLQEntry.from_dict(data))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning("DLQ deserialisation failed: %s", exc)

        except Exception as exc:
            logger.error("DLQ pop failed: %s", exc)

        return entries

    async def acknowledge(self, entry_id: str) -> None:
        """
        Mark an entry as resolved and remove it from the queue.

        Scans all tenant queues for the entry_id (suitable for small queues;
        for large deployments use a lookup index).
        """
        try:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=f"{self._KEY_PREFIX}*", count=100
                )
                for key in keys:
                    items = await self._redis.lrange(key, 0, -1)
                    for raw in items:
                        try:
                            data = json.loads(raw)
                            if data.get("id") == entry_id:
                                # Remove the element from the queue.
                                await self._redis.lrem(key, 1, raw)
                                # Sync the depth gauge — the queue just shrank,
                                # otherwise dlq_depth stays stale after ack.
                                tenant_id = data.get("tenant_id")
                                if not tenant_id:
                                    key_str = key.decode() if isinstance(key, bytes) else key
                                    tenant_id = key_str[len(self._KEY_PREFIX):]
                                if tenant_id:
                                    await self._update_gauge(tenant_id)
                                logger.info("DLQ entry %s acknowledged", entry_id)
                                return
                        except (json.JSONDecodeError, TypeError):
                            pass
                if cursor == 0:
                    break
        except Exception as exc:
            logger.error("DLQ acknowledge failed: %s", exc)

    async def get_depth(self, tenant_id: str | None = None) -> int:
        """Return total number of pending DLQ entries for the tenant (or all)."""
        try:
            if tenant_id:
                return await self._redis.llen(f"{self._KEY_PREFIX}{tenant_id}")

            total = 0
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=f"{self._KEY_PREFIX}*", count=100
                )
                for key in keys:
                    total += await self._redis.llen(key)
                if cursor == 0:
                    break
            return total
        except Exception as exc:
            logger.warning("DLQ get_depth failed: %s", exc)
            return 0

    async def requeue_for_retry(
        self,
        entry: DLQEntry,
        patch: str | None = None,
    ) -> None:
        """
        Re-enqueue an entry at lower priority (RPUSH) for retry.

        Increments retry_count and optionally stores the patch.
        """
        entry.retry_count += 1
        entry.last_retry = datetime.now(timezone.utc)
        entry.status = "replaying"
        if patch:
            entry.failure["hermes_patch_proposed"] = patch

        try:
            serialized = json.dumps(entry.to_dict(), default=str)
            queue_key = f"{self._KEY_PREFIX}{entry.tenant_id}"
            await self._redis.rpush(queue_key, serialized)
            await self._update_gauge(entry.tenant_id)
        except Exception as exc:
            logger.error("DLQ requeue_for_retry failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _update_gauge(self, tenant_id: str) -> None:
        """Sync dlq_depth Prometheus gauge with actual queue length."""
        try:
            from harness.observability.metrics import dlq_depth

            depth = await self.get_depth(tenant_id)
            dlq_depth.labels(tenant_id=tenant_id).set(float(depth))
        except Exception:
            pass
