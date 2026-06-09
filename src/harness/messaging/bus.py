"""AgentMessageBus: Redis Streams-backed inter-agent messaging."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from harness.core.errors import InterAgentTimeout
from harness.messaging.schema import AgentMessage

logger = logging.getLogger(__name__)

_STREAM_PREFIX = "harness:stream:"
_BROADCAST_STREAM = "harness:stream:broadcast"
_MSG_INDEX_KEY = "harness:msg_index"
_BLOCK_MS = 100  # XREAD block timeout in milliseconds


class AgentMessageBus:
    """
    Redis Streams-backed message bus for inter-agent communication.

    Each agent has a dedicated stream: ``harness:stream:{agent_id}``
    A shared broadcast stream: ``harness:stream:broadcast``

    Message routing:
    - ``send(msg)``          → XADD to recipient or broadcast stream
    - ``subscribe(agent_id)``→ XREAD blocking on agent + broadcast streams
    - ``request(...)``       → send + await correlated reply
    - ``fan_out(...)``       → send to N agents, gather replies
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: Any = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                max_connections=30,
            )
        return self._client

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def send(self, msg: AgentMessage) -> str:
        """
        Publish a message to the appropriate Redis stream.

        Automatically injects the current W3C traceparent from the active
        OpenTelemetry span (if available) so the receiving agent can
        continue the distributed trace without a chain break.

        Returns the stream entry ID assigned by Redis.
        Also registers the message in the TTL index.
        """
        # Inject W3C traceparent for distributed trace propagation
        if msg.traceparent is None:
            try:
                from opentelemetry.propagate import inject as _otel_inject  # type: ignore
                carrier: dict[str, str] = {}
                _otel_inject(carrier)
                if "traceparent" in carrier:
                    msg.traceparent = carrier["traceparent"]
            except Exception:
                pass  # OTel not installed or no active span — skip silently

        client = await self._get_client()
        stream_key = (
            _BROADCAST_STREAM
            if msg.is_broadcast()
            else f"{_STREAM_PREFIX}{msg.recipient_id}"
        )
        serialized = json.dumps(msg.to_dict(), default=str)

        try:
            entry_id: str = await client.xadd(
                stream_key,
                {"data": serialized},
                maxlen=10_000,
                approximate=True,
            )

            # Register in TTL index (score = expire timestamp)
            expire_ts = time.time() + msg.ttl_seconds
            await client.zadd(_MSG_INDEX_KEY, {msg.id: expire_ts})

            return entry_id
        except Exception as exc:
            logger.error("AgentMessageBus.send failed: %s", exc)
            raise

    async def snapshot_stream_ids(self, agent_id: str) -> dict[str, str]:
        """
        Capture the current last entry ID of ``agent_id``'s direct stream and
        the broadcast stream.

        Pass the result as ``last_ids`` to :meth:`subscribe` to receive every
        message added *after* this point — including ones XADDed before the
        subscriber's first XREAD, which ``last_id="$"`` would silently skip.
        Nonexistent (or unreadable) streams map to ``"0"``.
        """
        client = await self._get_client()
        ids: dict[str, str] = {}
        for stream_key in (f"{_STREAM_PREFIX}{agent_id}", _BROADCAST_STREAM):
            try:
                entries = await client.xrevrange(stream_key, count=1)
                ids[stream_key] = entries[0][0] if entries else "0"
            except Exception as exc:
                logger.debug(
                    "snapshot_stream_ids: falling back to '0' for %s: %s",
                    stream_key,
                    exc,
                )
                ids[stream_key] = "0"
        return ids

    async def subscribe(
        self,
        agent_id: str,
        message_types: list[str] | None = None,
        last_id: str = "$",
        last_ids: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentMessage]:
        """
        Async generator yielding messages for ``agent_id``.

        Reads from both:
        - ``harness:stream:{agent_id}`` (direct messages)
        - ``harness:stream:broadcast`` (broadcast messages)

        Filters by ``message_types`` if specified.
        Expired messages are silently skipped.
        Uses XREAD with BLOCK=100ms to avoid busy-waiting.

        ``last_ids`` (per-stream start IDs, e.g. from
        :meth:`snapshot_stream_ids`) overrides ``last_id`` for the streams it
        names — use it to avoid the ``$`` race where entries added before the
        first XREAD are never seen.
        """
        client = await self._get_client()
        agent_stream = f"{_STREAM_PREFIX}{agent_id}"

        # Maintain per-stream cursor IDs
        cursors: dict[str, str] = {
            agent_stream: last_id,
            _BROADCAST_STREAM: last_id,
        }
        if last_ids:
            cursors.update({k: v for k, v in last_ids.items() if k in cursors})

        try:
            while True:
                streams_to_read = list(cursors.keys())
                ids_to_read = list(cursors.values())

                try:
                    results = await client.xread(
                        streams={k: v for k, v in zip(streams_to_read, ids_to_read)},
                        block=_BLOCK_MS,
                        count=50,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("XREAD failed: %s — retrying", exc)
                    await asyncio.sleep(0.5)
                    continue

                if not results:
                    continue

                for stream_name, entries in results:
                    # Update cursor
                    if entries:
                        cursors[stream_name] = entries[-1][0]

                    for entry_id, data in entries:
                        raw = data.get("data")
                        if not raw:
                            continue
                        try:
                            msg_dict = json.loads(raw)
                            msg = AgentMessage.from_dict(msg_dict)
                        except (json.JSONDecodeError, KeyError, TypeError) as exc:
                            logger.debug("Message deserialisation failed: %s", exc)
                            continue

                        # Skip expired messages
                        if msg.is_expired():
                            continue

                        # Filter by message type
                        if message_types and msg.message_type not in message_types:
                            continue

                        yield msg

        except GeneratorExit:
            pass

    async def request(
        self,
        sender_id: str,
        recipient_id: str,
        payload: dict[str, Any],
        timeout: float = 30.0,
        message_types: list[str] | None = None,
    ) -> AgentMessage:
        """
        Send a query and await a correlated reply.

        ``message_types`` filters which reply types are accepted
        (default: ``["result", "error"]``).

        Raises InterAgentTimeout if no reply arrives within ``timeout`` seconds.
        """
        import uuid as _uuid

        correlation_id = _uuid.uuid4().hex
        query_msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type="query",
            payload=payload,
            correlation_id=correlation_id,
        )

        # Snapshot the reply-stream position BEFORE sending so a reply
        # XADDed between send() and the subscriber's first XREAD isn't lost.
        start_ids = await self.snapshot_stream_ids(sender_id)
        await self.send(query_msg)

        async def _await_reply() -> AgentMessage | None:
            async for reply in self.subscribe(
                sender_id,
                message_types=message_types or ["result", "error"],
                last_ids=start_ids,
            ):
                if reply.correlation_id == correlation_id:
                    return reply
            return None  # subscriber was cancelled before a reply matched

        try:
            reply = await asyncio.wait_for(_await_reply(), timeout=timeout)
        except asyncio.TimeoutError:
            reply = None

        if reply is not None:
            return reply

        raise InterAgentTimeout(
            f"No reply from {recipient_id} within {timeout}s",
            context={"sender": sender_id, "correlation_id": correlation_id},
        )

    async def fan_out(
        self,
        sender_id: str,
        recipient_ids: list[str],
        payload: dict[str, Any],
        timeout: float = 60.0,
    ) -> list[AgentMessage]:
        """
        Send the same payload to multiple recipients and collect replies.

        Returns partial results — if some recipients time out their replies
        are simply omitted from the list.
        """
        import uuid as _uuid

        correlation_id = _uuid.uuid4().hex
        pending: set[str] = set(recipient_ids)
        replies: list[AgentMessage] = []

        # Snapshot the reply-stream position BEFORE sending (see request()).
        start_ids = await self.snapshot_stream_ids(sender_id)

        # Send to all recipients in parallel
        send_tasks = [
            self.send(
                AgentMessage(
                    sender_id=sender_id,
                    recipient_id=rid,
                    message_type="task",
                    payload=payload,
                    correlation_id=correlation_id,
                )
            )
            for rid in recipient_ids
        ]
        await asyncio.gather(*send_tasks, return_exceptions=True)

        # Collect replies until all recipients answered or timeout elapses
        async def _collect() -> None:
            async for reply in self.subscribe(
                sender_id, message_types=["result", "error"], last_ids=start_ids
            ):
                if (
                    reply.correlation_id == correlation_id
                    and reply.sender_id in pending
                ):
                    replies.append(reply)
                    pending.discard(reply.sender_id)
                    if not pending:
                        return

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # partial results are acceptable

        return replies

    async def broadcast(
        self,
        sender_id: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Publish a broadcast message to all subscribers."""
        msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,
            message_type=message_type,  # type: ignore[arg-type]
            payload=payload,
        )
        return await self.send(msg)

    async def cleanup_expired(self) -> int:
        """
        Remove expired message IDs from the TTL index.

        Returns the number of entries removed.
        """
        client = await self._get_client()
        try:
            now = time.time()
            removed: int = await client.zremrangebyscore(
                _MSG_INDEX_KEY, "-inf", now
            )
            if removed:
                logger.debug("DLQ cleanup: removed %d expired message index entries", removed)
            return removed
        except Exception as exc:
            logger.warning("cleanup_expired failed: %s", exc)
            return 0

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
