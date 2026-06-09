"""Redis Stream event bus for real-time step streaming.

There is ONE event transport in the harness: a Redis *stream* keyed
``harness:events:{run_id}``.  The agent event sink
(``harness.workers.agent_worker._RedisStreamEventSink``) and the SSE endpoints
(``/runs/{id}/stream`` and ``/runs/{id}/steps``) all use this stream via
``XADD`` / ``XREAD``.

``EventBus`` is a thin wrapper over that same stream so framework adapters
(crewai/autogen/langgraph/openclaw) and ``api.deps.get_event_bus`` share the
single source of truth instead of a parallel pub/sub channel.  Historically
this class used Redis pub/sub, which nothing on the read side consumed; it now
serialises the identical field layout the stream sink writes so events flow to
the SSE endpoints.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from harness.core.context import StepEvent

logger = logging.getLogger(__name__)

_STREAM_PREFIX = "harness:events:"
_STREAM_MAXLEN = 1000


def _stream_key(run_id: str) -> str:
    return f"{_STREAM_PREFIX}{run_id}"


class EventBus:
    """
    Redis Stream event bus for streaming StepEvents to SSE endpoints.

    Publishers call ``publish(run_id, event)``; subscribers iterate over
    ``subscribe(run_id)`` which yields StepEvent instances in real time.

    Events are XADDed to ``harness:events:{run_id}`` using the same field
    layout as ``_RedisStreamEventSink`` so the ``/runs/{id}/stream`` and
    ``/runs/{id}/steps`` SSE endpoints read them via ``XREAD``.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: Any = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, run_id: str, event: StepEvent) -> None:
        """
        Serialise and XADD a StepEvent to the run's Redis stream.

        Uses the same field layout as
        ``harness.workers.agent_worker._RedisStreamEventSink`` so the SSE
        endpoints parse events consistently regardless of which producer
        wrote them.
        """
        try:
            client = await self._get_client()
            await client.xadd(
                _stream_key(run_id),
                {
                    "run_id": event.run_id,
                    "step": str(event.step),
                    "event_type": event.event_type,
                    "payload": json.dumps(event.payload, default=str),
                    "timestamp": event.timestamp.isoformat(),
                },
                maxlen=_STREAM_MAXLEN,
                approximate=True,
            )
        except Exception as exc:
            logger.warning("EventBus publish failed for run %s: %s", run_id, exc)

    async def subscribe(self, run_id: str) -> AsyncIterator[StepEvent]:
        """
        Async generator that yields StepEvents published for ``run_id``.

        Reads the Redis stream from the beginning with a blocking ``XREAD``
        loop and reconstructs StepEvent instances.  The generator runs until
        it is closed (GeneratorExit) or a terminal event arrives.
        """
        client = await self._get_client()
        last_id = "0"

        try:
            while True:
                try:
                    entries = await client.xread(
                        {_stream_key(run_id): last_id}, count=50, block=1000
                    )
                except Exception as exc:
                    logger.debug("EventBus: stream read error: %s", exc)
                    continue

                if not entries:
                    continue

                for _stream_name, messages in entries:
                    for msg_id, fields in messages:
                        last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                        try:
                            step_event = self._parse_fields(fields, run_id)
                        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                            logger.debug("EventBus: malformed entry skipped: %s", exc)
                            continue
                        yield step_event
                        if step_event.event_type in (
                            "run_end",
                            "completed",
                            "failed",
                            "cancelled",
                            "budget_exceeded",
                        ):
                            return

        except GeneratorExit:
            logger.debug("EventBus: subscriber for run %s closed", run_id)

    @staticmethod
    def _parse_fields(fields: dict, run_id: str) -> StepEvent:
        """Reconstruct a StepEvent from a decoded stream entry's fields."""
        norm: dict[str, Any] = {}
        for k, v in fields.items():
            key = k if isinstance(k, str) else k.decode()
            norm[key] = v if isinstance(v, str) else v.decode()

        ts_raw = norm.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)

        payload_raw = norm.get("payload", "{}")
        try:
            payload = json.loads(payload_raw)
        except (json.JSONDecodeError, TypeError):
            payload = {}

        return StepEvent(
            run_id=norm.get("run_id", run_id),
            step=int(norm.get("step", 0)),
            event_type=norm.get("event_type", "unknown"),
            payload=payload,
            timestamp=ts,
        )

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
