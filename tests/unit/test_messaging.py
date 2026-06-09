"""Unit tests for the inter-agent message bus (AgentMessageBus, Redis Streams)."""

from __future__ import annotations

import json
import time

import pytest
import pytest_asyncio

from harness.core.errors import InterAgentTimeout
from harness.messaging.bus import (
    _BROADCAST_STREAM,
    _MSG_INDEX_KEY,
    _STREAM_PREFIX,
    AgentMessageBus,
)
from harness.messaging.patterns import MessagePatterns
from harness.messaging.schema import AgentMessage


@pytest_asyncio.fixture
async def bus(redis_client):
    """AgentMessageBus wired to the fakeredis client (bypasses real connection)."""
    b = AgentMessageBus(redis_url="redis://localhost:6379/0")
    b._client = redis_client  # inject fakeredis; _get_client() returns it as-is
    return b


def _msg(**kw):
    base = {"sender_id": "planner", "message_type": "task",
            "payload": {"content": "List tables"}}
    base.update(kw)
    return AgentMessage(**base)


@pytest.mark.asyncio
async def test_send_routes_to_recipient_stream(bus, redis_client):
    entry_id = await bus.send(_msg(recipient_id="agent-sql"))
    assert entry_id  # Redis returns a non-empty stream entry id
    assert await redis_client.xlen(f"{_STREAM_PREFIX}agent-sql") == 1
    # Nothing leaked onto the broadcast stream
    assert await redis_client.xlen(_BROADCAST_STREAM) == 0


@pytest.mark.asyncio
async def test_send_broadcast_routes_to_broadcast_stream(bus, redis_client):
    # recipient_id=None -> broadcast
    await bus.send(_msg(recipient_id=None, message_type="status"))
    assert await redis_client.xlen(_BROADCAST_STREAM) == 1


@pytest.mark.asyncio
async def test_send_registers_message_in_ttl_index(bus, redis_client):
    msg = _msg(recipient_id="agent-sql", ttl_seconds=300.0)
    await bus.send(msg)
    score = await redis_client.zscore(_MSG_INDEX_KEY, msg.id)
    assert score is not None and score > 0  # expire timestamp recorded


@pytest.mark.asyncio
async def test_sent_payload_roundtrips_through_the_stream(bus, redis_client):
    sent = _msg(recipient_id="agent-sql", payload={"content": "describe orders"})
    await bus.send(sent)
    entries = await redis_client.xrange(f"{_STREAM_PREFIX}agent-sql")
    assert len(entries) == 1
    _, fields = entries[0]
    restored = AgentMessage.from_dict(json.loads(fields["data"]))
    assert restored.sender_id == "planner"
    assert restored.payload == {"content": "describe orders"}
    assert restored.id == sent.id


@pytest.mark.asyncio
async def test_multiple_sends_accumulate_on_stream(bus, redis_client):
    for i in range(3):
        await bus.send(_msg(recipient_id="agent-sql", payload={"n": i}))
    assert await redis_client.xlen(f"{_STREAM_PREFIX}agent-sql") == 3


@pytest.mark.asyncio
async def test_is_broadcast_reflects_recipient(bus):
    assert _msg(recipient_id=None).is_broadcast() is True
    assert _msg(recipient_id="agent-sql").is_broadcast() is False


# ---------------------------------------------------------------------------
# Request/reply regressions: lost-reply race + unreachable timeout
# ---------------------------------------------------------------------------


def _instant_responder(bus, reply_type: str = "result", responders: list[str] | None = None):
    """Wrap bus.send so a correlated reply is XADDed to the sender's stream
    IMMEDIATELY after the outgoing message — i.e. before request()/fan_out()
    has started its subscriber. Regression for the `last_id="$"` race."""
    original_send = bus.send

    async def send_and_reply(msg):
        entry_id = await original_send(msg)
        if msg.message_type in ("query", "task") and (
            responders is None or msg.recipient_id in responders
        ):
            await original_send(
                AgentMessage(
                    sender_id=msg.recipient_id,
                    recipient_id=msg.sender_id,
                    message_type=reply_type,
                    payload={"echo": msg.payload},
                    correlation_id=msg.correlation_id,
                )
            )
        return entry_id

    bus.send = send_and_reply


@pytest.mark.asyncio
async def test_request_reply_arriving_before_subscribe_is_not_lost(bus):
    _instant_responder(bus)
    reply = await bus.request(
        "planner", "agent-sql", {"content": "ping"}, timeout=5.0
    )
    assert reply.message_type == "result"
    assert reply.sender_id == "agent-sql"
    assert reply.payload == {"echo": {"content": "ping"}}


@pytest.mark.asyncio
async def test_request_with_zero_replies_raises_timeout(bus):
    start = time.monotonic()
    with pytest.raises(InterAgentTimeout):
        await bus.request("planner", "agent-sql", {"content": "ping"}, timeout=0.3)
    # Must return promptly instead of hanging forever
    assert time.monotonic() - start < 3.0


@pytest.mark.asyncio
async def test_request_custom_message_types_accepts_status_reply(bus):
    _instant_responder(bus, reply_type="status")
    reply = await bus.request(
        "monitor",
        "agent-sql",
        {"type": "ping"},
        timeout=5.0,
        message_types=["result", "error", "status", "heartbeat"],
    )
    assert reply.message_type == "status"


@pytest.mark.asyncio
async def test_fan_out_replies_arriving_before_subscribe_are_not_lost(bus):
    _instant_responder(bus)
    replies = await bus.fan_out(
        "planner", ["agent-a", "agent-b"], {"content": "go"}, timeout=5.0
    )
    assert sorted(r.sender_id for r in replies) == ["agent-a", "agent-b"]


@pytest.mark.asyncio
async def test_fan_out_with_zero_replies_returns_empty_after_timeout(bus):
    start = time.monotonic()
    replies = await bus.fan_out(
        "planner", ["agent-a", "agent-b"], {"content": "go"}, timeout=0.3
    )
    assert replies == []
    assert time.monotonic() - start < 3.0


@pytest.mark.asyncio
async def test_fan_out_returns_partial_results_on_timeout(bus):
    _instant_responder(bus, responders=["agent-a"])  # agent-b never replies
    replies = await bus.fan_out(
        "planner", ["agent-a", "agent-b"], {"content": "go"}, timeout=0.5
    )
    assert [r.sender_id for r in replies] == ["agent-a"]


@pytest.mark.asyncio
async def test_scatter_gather_replies_arriving_before_subscribe_are_not_lost(bus):
    _instant_responder(bus)
    patterns = MessagePatterns(bus)
    replies = await patterns.scatter_gather(
        sender_id="planner",
        recipient_ids=["agent-a", "agent-b"],
        payload={"content": "go"},
        timeout=5.0,
        min_responses=2,
    )
    assert sorted(r.sender_id for r in replies) == ["agent-a", "agent-b"]


@pytest.mark.asyncio
async def test_scatter_gather_with_zero_replies_returns_after_timeout(bus):
    patterns = MessagePatterns(bus)
    start = time.monotonic()
    replies = await patterns.scatter_gather(
        sender_id="planner",
        recipient_ids=["agent-a"],
        payload={"content": "go"},
        timeout=0.3,
    )
    assert replies == []
    assert time.monotonic() - start < 3.0


@pytest.mark.asyncio
async def test_snapshot_stream_ids_returns_zero_for_missing_streams(bus):
    ids = await bus.snapshot_stream_ids("nobody")
    assert ids == {f"{_STREAM_PREFIX}nobody": "0", _BROADCAST_STREAM: "0"}


@pytest.mark.asyncio
async def test_snapshot_stream_ids_captures_last_entry(bus, redis_client):
    entry_id = await bus.send(_msg(recipient_id="agent-sql"))
    ids = await bus.snapshot_stream_ids("agent-sql")
    assert ids[f"{_STREAM_PREFIX}agent-sql"] == entry_id
