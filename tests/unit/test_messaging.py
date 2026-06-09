"""Unit tests for the inter-agent message bus (AgentMessageBus, Redis Streams)."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from harness.messaging.bus import (
    _BROADCAST_STREAM,
    _MSG_INDEX_KEY,
    _STREAM_PREFIX,
    AgentMessageBus,
)
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
