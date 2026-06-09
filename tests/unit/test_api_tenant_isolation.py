"""Regression tests for cross-tenant authorization on the API routes.

Covers the security fixes for:
- /runs/{run_id}/feedback (POST/GET/DELETE) — run ownership enforcement
- /hitl/{request_id}/approve|reject       — HITL request tenant check
- /runs/spans/{span_id}                   — span tenant check
- /runs/{run_id}/steps                    — fail-closed on corrupt run records
- jwt_secret_key default rejected in prod (config validator + deps guard)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

pytest.importorskip("httpx", reason="httpx required for API tests")
from httpx import ASGITransport, AsyncClient  # noqa: E402

from harness.core.config import DEFAULT_JWT_SECRET, Settings  # noqa: E402

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


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
    """Persist a RunRecord owned by *tenant_id* and return its run_id."""
    from harness.orchestrator.runner import AgentRunner

    runner = AgentRunner(redis=redis_client, agent_factory=lambda agent_type: None)
    record = await runner.create_run(
        tenant_id=tenant_id, agent_type="sql", task="list tables"
    )
    return record.run_id


# ---------------------------------------------------------------------------
# Feedback endpoints — run ownership
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_feedback_cross_tenant_403(api_app, redis_client):
    run_id = await _create_run(redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_B) as client:
        resp = await client.post(
            f"/runs/{run_id}/feedback", json={"type": "stop", "content": "halt"}
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_post_feedback_unknown_run_404(api_app):
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.post(
            "/runs/no-such-run/feedback", json={"type": "hint", "content": "x"}
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_post_feedback_owner_succeeds(api_app, redis_client):
    run_id = await _create_run(redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.post(
            f"/runs/{run_id}/feedback", json={"type": "hint", "content": "go left"}
        )
    assert resp.status_code == 201
    assert resp.json()["run_id"] == run_id


@pytest.mark.asyncio
async def test_list_feedback_cross_tenant_403(api_app, redis_client):
    run_id = await _create_run(redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_B) as client:
        resp = await client.get(f"/runs/{run_id}/feedback")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_feedback_owner_succeeds(api_app, redis_client):
    run_id = await _create_run(redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get(f"/runs/{run_id}/feedback")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_clear_feedback_cross_tenant_403(api_app, redis_client):
    run_id = await _create_run(redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_B) as client:
        resp = await client.delete(f"/runs/{run_id}/feedback")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_clear_feedback_unknown_run_404(api_app):
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.delete("/runs/missing-run/feedback")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# HITL approve / reject — tenant ownership
# ---------------------------------------------------------------------------

async def _create_hitl_request(api_app, redis_client, tenant_id: str):
    from harness.orchestrator.hitl import HITLManager

    hitl = HITLManager(redis=redis_client)
    api_app.state.hitl_manager = hitl
    req = await hitl.request_approval(
        run_id="run-1",
        tenant_id=tenant_id,
        tool_name="execute_sql",
        tool_args={"query": "DROP TABLE users"},
        reason="dangerous",
    )
    return hitl, req


@pytest.mark.asyncio
async def test_approve_hitl_cross_tenant_404(api_app, redis_client):
    hitl, req = await _create_hitl_request(api_app, redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_B) as client:
        resp = await client.post(
            f"/hitl/{req.request_id}/approve", json={"resolved_by": "mallory"}
        )
    assert resp.status_code == 404
    # The request must remain pending — no cross-tenant resolution.
    stored = await hitl.get(req.request_id)
    assert stored.status == "pending"


@pytest.mark.asyncio
async def test_reject_hitl_cross_tenant_404(api_app, redis_client):
    hitl, req = await _create_hitl_request(api_app, redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_B) as client:
        resp = await client.post(
            f"/hitl/{req.request_id}/reject", json={"resolved_by": "mallory"}
        )
    assert resp.status_code == 404
    stored = await hitl.get(req.request_id)
    assert stored.status == "pending"


@pytest.mark.asyncio
async def test_approve_hitl_owner_succeeds(api_app, redis_client):
    hitl, req = await _create_hitl_request(api_app, redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.post(
            f"/hitl/{req.request_id}/approve", json={"resolved_by": "alice"}
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"


@pytest.mark.asyncio
async def test_approve_hitl_unknown_request_404(api_app, redis_client):
    from harness.orchestrator.hitl import HITLManager

    api_app.state.hitl_manager = HITLManager(redis=redis_client)
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.post(
            "/hitl/deadbeef/approve", json={"resolved_by": "alice"}
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/spans/{span_id} — tenant ownership
# ---------------------------------------------------------------------------

async def _record_span(redis_client, tmp_path: Path, tenant_id: str | None) -> str:
    """Record one span (with or without a tenant) and return its span_id."""
    from harness.core.context import AgentContext
    from harness.observability.trace_recorder import TraceRecorder
    from harness.observability.trace_schema import SpanKind

    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    ctx = None
    if tenant_id is not None:
        ctx = AgentContext(
            run_id="span-run", tenant_id=tenant_id,
            agent_type="sql", task="t",
            memory=None, workspace_path=tmp_path,
        )
    span_id = await recorder.start_span("span-run", SpanKind.LLM, "llm:call", ctx)
    await recorder.end_span("span-run", span_id, output_preview="done")
    return span_id


def _patch_recorder(redis_client, tmp_path):
    """Patch TraceRecorder.create in the traces route to use fakeredis."""
    from harness.observability.trace_recorder import TraceRecorder

    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    return patch(
        "harness.observability.trace_recorder.TraceRecorder.create",
        return_value=recorder,
    )


@pytest.mark.asyncio
async def test_get_span_cross_tenant_404(api_app, redis_client, tmp_path):
    span_id = await _record_span(redis_client, tmp_path, TENANT_A)
    with _patch_recorder(redis_client, tmp_path):
        async with _client_as(api_app, TENANT_B) as client:
            resp = await client.get(f"/runs/spans/{span_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_span_owner_succeeds(api_app, redis_client, tmp_path):
    span_id = await _record_span(redis_client, tmp_path, TENANT_A)
    with _patch_recorder(redis_client, tmp_path):
        async with _client_as(api_app, TENANT_A) as client:
            resp = await client.get(f"/runs/spans/{span_id}")
    assert resp.status_code == 200
    assert resp.json()["span_id"] == span_id


@pytest.mark.asyncio
async def test_get_span_missing_tenant_is_denied(api_app, redis_client, tmp_path):
    """Spans recorded without a tenant (tenant_id='') are not world-readable."""
    span_id = await _record_span(redis_client, tmp_path, tenant_id=None)
    with _patch_recorder(redis_client, tmp_path):
        async with _client_as(api_app, TENANT_A) as client:
            resp = await client.get(f"/runs/spans/{span_id}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/steps — fail closed on corrupt run records
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_steps_corrupt_run_record_404(api_app, redis_client):
    """A non-JSON run record must 404, not skip the tenant check (fail-open)."""
    await redis_client.set("harness:run:corrupt-run", "{not valid json")
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get("/runs/corrupt-run/steps")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_steps_cross_tenant_403(api_app, redis_client):
    run_id = await _create_run(redis_client, TENANT_A)
    async with _client_as(api_app, TENANT_B) as client:
        resp = await client.get(f"/runs/{run_id}/steps")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_steps_unknown_run_404(api_app):
    async with _client_as(api_app, TENANT_A) as client:
        resp = await client.get("/runs/missing/steps")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# jwt_secret_key default — config validator
# ---------------------------------------------------------------------------

def test_settings_prod_with_default_jwt_secret_refuses_to_start():
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        Settings(environment="prod", jwt_secret_key=DEFAULT_JWT_SECRET)


def test_settings_prod_with_real_jwt_secret_ok():
    s = Settings(environment="prod", jwt_secret_key="a-strong-random-secret")
    assert s.jwt_secret_key == "a-strong-random-secret"


def test_settings_dev_with_default_jwt_secret_ok():
    s = Settings(environment="dev", jwt_secret_key=DEFAULT_JWT_SECRET)
    assert s.environment == "dev"


def test_settings_staging_with_default_jwt_secret_ok():
    s = Settings(environment="staging", jwt_secret_key=DEFAULT_JWT_SECRET)
    assert s.environment == "staging"


# ---------------------------------------------------------------------------
# _decode_jwt — defense in depth + exception narrowing
# ---------------------------------------------------------------------------

def _mock_config(environment: str, secret: str) -> MagicMock:
    cfg = MagicMock()
    cfg.environment = environment
    cfg.jwt_secret_key = secret
    return cfg


def test_decode_jwt_rejects_default_secret_in_prod():
    pytest.importorskip("jose", reason="python-jose required")
    from fastapi import HTTPException

    from harness.api.deps import _decode_jwt

    with patch(
        "harness.api.deps.get_config",
        return_value=_mock_config("prod", DEFAULT_JWT_SECRET),
    ):
        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt("any-token")
    assert exc_info.value.status_code == 401


def test_decode_jwt_valid_token_with_real_secret():
    jose = pytest.importorskip("jose", reason="python-jose required")
    from harness.api.deps import _decode_jwt

    secret = "unit-test-secret"
    token = jose.jwt.encode({"tenant_id": TENANT_A}, secret, algorithm="HS256")
    with patch(
        "harness.api.deps.get_config",
        return_value=_mock_config("prod", secret),
    ):
        payload = _decode_jwt(token)
    assert payload["tenant_id"] == TENANT_A


def test_decode_jwt_invalid_token_is_401():
    pytest.importorskip("jose", reason="python-jose required")
    from fastapi import HTTPException

    from harness.api.deps import _decode_jwt

    with patch(
        "harness.api.deps.get_config",
        return_value=_mock_config("dev", "some-secret"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt("not.a.jwt")
    assert exc_info.value.status_code == 401


def test_decode_jwt_missing_jose_surfaces_import_error():
    """A missing JWT library is a server misconfiguration, not a 401."""
    from harness.api.deps import _decode_jwt

    with patch.dict("sys.modules", {"jose": None}):
        with pytest.raises(ImportError):
            _decode_jwt("any-token")
