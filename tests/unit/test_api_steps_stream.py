"""Regression tests for GET /runs/{run_id}/steps SSE delivery.

The /steps endpoint historically subscribed to a Redis *pub/sub* channel that
nothing published to, so it could never deliver events.  It now reads the same
Redis *stream* (``harness:events:{run_id}``) that the agent event sink and
EventBus XADD to.  These tests assert that:

- an event XADDed to the run stream is delivered by GET /runs/{run_id}/steps
- a ``token_delta`` event is forwarded (not filtered out)
- a terminal ``run_end`` event closes the stream with the [DONE] sentinel
- _RedisStreamEventSink and EventBus write entries the endpoint reads
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import pytest_asyncio

pytest.importorskip("httpx", reason="httpx required for API tests")
from httpx import ASGITransport, AsyncClient  # noqa: E402

TENANT_A = "tenant-a"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def api_app(redis_client):
    """A FastAPI app wired to fakeredis (lifespan not run, state set manually)."""
    from harness.api.main import create_app

    app = create_app()
    app.state.redis = redis_client
    yield app
    app.dependency_overrides.clear()


@asynccontextmanager
async def _client_as(app, tenant_id: str):
    """Yield an AsyncClient authenticated as *tenant_id* via dependency override."""
    from harness.api import deps

    app.dependency_overrides[deps.get_current_tenant] = lambda: tenant_id
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _create_run(redis_client, tenant_id: str) -> str:
    from harness.orchestrator.runner import AgentRunner

    runner = AgentRunner(redis=redis_client, agent_factory=lambda agent_type: None)
    record = await runner.create_run(
        tenant_id=tenant_id, agent_type="sql", task="list tables"
    )
    return record.run_id


def _xadd_fields(run_id: str, step: int, event_type: str, payload: dict) -> dict:
    """The exact field layout the sink / EventBus write to the stream."""
    return {
        "run_id": run_id,
        "step": str(step),
        "event_type": event_type,
        "payload": json.dumps(payload, default=str),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _parse_sse(text: str) -> list[dict | str]:
    """Parse an SSE body into a list of event payloads (dicts) / sentinels."""
    events: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(data))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_steps_delivers_stream_event(api_app, redis_client):
    """An event XADDed to the run stream is delivered by GET /runs/{id}/steps."""
    run_id = await _create_run(redis_client, TENANT_A)
    stream_key = f"harness:events:{run_id}"

    await redis_client.xadd(
        stream_key, _xadd_fields(run_id, 1, "step_start", {"detail": "begin"})
    )
    # Terminal event so the blocking read loop returns promptly.
    await redis_client.xadd(
        stream_key, _xadd_fields(run_id, 1, "run_end", {"status": "completed"})
    )

    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get(f"/runs/{run_id}/steps?timeout=5")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)

    types = [e.get("event_type") for e in events if isinstance(e, dict)]
    assert "step_start" in types
    assert "run_end" in types
    assert "[DONE]" in events


@pytest.mark.asyncio
async def test_steps_forwards_token_delta(api_app, redis_client):
    """token_delta events from the token-streaming feature are forwarded."""
    run_id = await _create_run(redis_client, TENANT_A)
    stream_key = f"harness:events:{run_id}"

    await redis_client.xadd(
        stream_key, _xadd_fields(run_id, 1, "token_delta", {"delta": "Hello"})
    )
    await redis_client.xadd(
        stream_key, _xadd_fields(run_id, 1, "token_delta", {"delta": " world"})
    )
    await redis_client.xadd(
        stream_key, _xadd_fields(run_id, 1, "run_end", {"status": "completed"})
    )

    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get(f"/runs/{run_id}/steps?timeout=5")

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    deltas = [
        e["payload"]["delta"]
        for e in events
        if isinstance(e, dict) and e.get("event_type") == "token_delta"
    ]
    assert deltas == ["Hello", " world"]


@pytest.mark.asyncio
async def test_event_bus_publish_is_read_by_steps(api_app, redis_client):
    """EventBus.publish XADDs entries the /steps endpoint reads (one event path)."""
    from harness.core.context import StepEvent
    from harness.observability.event_bus import EventBus

    run_id = await _create_run(redis_client, TENANT_A)

    bus = EventBus(redis_url="redis://unused")
    bus._client = redis_client  # reuse the fakeredis client directly

    await bus.publish(
        run_id,
        StepEvent(
            run_id=run_id,
            step=2,
            event_type="tool_call",
            payload={"tool": "sql"},
            timestamp=datetime.now(timezone.utc),
        ),
    )
    await bus.publish(
        run_id,
        StepEvent(
            run_id=run_id,
            step=2,
            event_type="run_end",
            payload={"status": "completed"},
            timestamp=datetime.now(timezone.utc),
        ),
    )

    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get(f"/runs/{run_id}/steps?timeout=5")

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e.get("event_type") for e in events if isinstance(e, dict)]
    assert "tool_call" in types
    assert "run_end" in types


@pytest.mark.asyncio
async def test_redis_stream_sink_is_read_by_steps(api_app, redis_client):
    """_RedisStreamEventSink.publish writes entries the /steps endpoint reads."""
    from harness.core.context import StepEvent
    from harness.workers.agent_worker import _RedisStreamEventSink

    run_id = await _create_run(redis_client, TENANT_A)
    sink = _RedisStreamEventSink(redis_client)

    await sink.publish(
        StepEvent(
            run_id=run_id,
            step=1,
            event_type="llm_call",
            payload={"model": "test"},
            timestamp=datetime.now(timezone.utc),
        )
    )
    await sink.publish(
        StepEvent(
            run_id=run_id,
            step=1,
            event_type="run_end",
            payload={"status": "completed"},
            timestamp=datetime.now(timezone.utc),
        )
    )

    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get(f"/runs/{run_id}/steps?timeout=5")

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e.get("event_type") for e in events if isinstance(e, dict)]
    assert "llm_call" in types
    assert "run_end" in types
