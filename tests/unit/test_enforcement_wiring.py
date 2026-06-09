"""
Tests for the three wired enforcement paths:

1. CostTracker.check_budget is called in the agent loop and stops a run when
   the tenant is over its monthly cap (failure_class BUDGET_COST).
2. ctx.metadata["cancel_check"] lets an operator cancel a running agent
   (failure_class CANCELLED), and the runner wires it from cancel_run.
3. RateLimitMiddleware enforces per-tenant limits from app.state, exempts
   infra paths, and fails open when no limiter is configured.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from harness.agents.base import BaseAgent
from harness.core.context import LLMResponse
from harness.core.errors import FailureClass, RateLimitError


def _llm_resp(content="done"):
    return LLMResponse(
        content=content, tool_calls=[], input_tokens=5, output_tokens=5,
        model="m", provider="p",
    )


def _make_agent(*, router=None, cost_tracker=None):
    if router is None:
        router = AsyncMock()
        router.complete = AsyncMock(return_value=_llm_resp())
    return BaseAgent(
        llm_router=router,
        memory_manager=None,
        tool_registry=AsyncMock(),
        safety_pipeline=None,
        step_tracer=None,
        mlflow_tracer=None,
        failure_tracker=None,
        audit_logger=None,
        event_bus=None,
        cost_tracker=cost_tracker,
        checkpoint_manager=None,
        trace_recorder=None,
    )


# ---------------------------------------------------------------------------
# 1. Cost budget enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_over_budget_tenant_stops_before_llm_call(agent_context):
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_llm_resp())

    cost_tracker = MagicMock()
    cost_tracker._enforce_budget = True
    cost_tracker.check_budget = AsyncMock(
        side_effect=RateLimitError("over budget", retry_after=0.0)
    )
    cost_tracker.record = AsyncMock(return_value=MagicMock(cost_usd=0.0))

    agent = _make_agent(router=router, cost_tracker=cost_tracker)
    result = await agent.run(agent_context())

    assert not result.success
    assert result.failure_class == FailureClass.BUDGET_COST.value
    # The cap is checked before the LLM call, so no spend occurs.
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_budget_infra_error_fails_open(agent_context):
    """A Redis/infra error in check_budget must not kill a healthy run."""
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_llm_resp())

    cost_tracker = MagicMock()
    cost_tracker._enforce_budget = True
    cost_tracker.check_budget = AsyncMock(side_effect=ConnectionError("redis down"))
    cost_tracker.record = AsyncMock(return_value=MagicMock(cost_usd=0.0))

    agent = _make_agent(router=router, cost_tracker=cost_tracker)
    result = await agent.run(agent_context())

    assert result.success
    router.complete.assert_called()


@pytest.mark.asyncio
async def test_enforcement_disabled_does_not_check(agent_context):
    cost_tracker = MagicMock()
    cost_tracker._enforce_budget = False
    cost_tracker.check_budget = AsyncMock(
        side_effect=RateLimitError("over budget", retry_after=0.0)
    )
    cost_tracker.record = AsyncMock(return_value=MagicMock(cost_usd=0.0))

    agent = _make_agent(cost_tracker=cost_tracker)
    result = await agent.run(agent_context())

    assert result.success
    cost_tracker.check_budget.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_check_stops_run_before_llm(agent_context):
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_llm_resp())

    async def _cancelled() -> bool:
        return True

    ctx = agent_context()
    ctx.metadata["cancel_check"] = _cancelled

    agent = _make_agent(router=router)
    result = await agent.run(ctx)

    assert not result.success
    assert result.failure_class == FailureClass.CANCELLED.value
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_check_infra_error_does_not_abort(agent_context):
    async def _boom() -> bool:
        raise ConnectionError("redis down")

    ctx = agent_context()
    ctx.metadata["cancel_check"] = _boom

    agent = _make_agent()
    result = await agent.run(ctx)

    assert result.success


@pytest.mark.asyncio
async def test_runner_wires_cancel_check_from_status(monkeypatch):
    """execute_run injects a cancel_check that reflects the persisted status."""
    from harness.orchestrator import runner as runner_mod

    captured = {}

    from harness.core.context import AgentResult

    async def _fake_run_agent(agent, record, workspace, cancel_check=None):
        captured["cancel_check"] = cancel_check
        return AgentResult(run_id="r1", output="ok", steps=1, tokens=1, success=True)

    monkeypatch.setattr(runner_mod, "_run_agent", _fake_run_agent)

    from harness.orchestrator.runner import AgentRunner, RunRecord

    redis = MagicMock()
    state = {"status": "running"}

    async def _get(key):
        rec = RunRecord(run_id="r1", tenant_id="t1", agent_type="sql", task="x",
                        status=state["status"])
        return rec.to_json()

    redis.get = AsyncMock(side_effect=_get)
    redis.set = AsyncMock()

    runner = AgentRunner(redis=redis, agent_factory=lambda t: MagicMock(),
                         workspace_base=__import__("pathlib").Path("/tmp/haas-test"))
    await runner.execute_run("r1")

    check = captured["cancel_check"]
    assert check is not None
    # The wired check reflects the live persisted status.
    state["status"] = "running"
    assert await check() is False
    state["status"] = "cancelled"
    assert await check() is True


# ---------------------------------------------------------------------------
# 3. Rate limit middleware
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def rl_app(redis_client):
    from fastapi import FastAPI
    from harness.core.rate_limiter import RateLimiter, RateLimitMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    app.state.rate_limiter = RateLimiter(redis_client, default_rpm=3, window_seconds=60)

    @app.get("/runs/x")
    async def _runs():
        return {"ok": True}

    @app.get("/health")
    async def _health():
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_middleware_blocks_over_limit(rl_app):
    import httpx

    transport = httpx.ASGITransport(app=rl_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        headers = {"X-Tenant-ID": "tenant-rl"}
        codes = [(await client.get("/runs/x", headers=headers)).status_code
                 for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


@pytest.mark.asyncio
async def test_middleware_exempts_health(rl_app):
    import httpx

    transport = httpx.ASGITransport(app=rl_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        headers = {"X-Tenant-ID": "tenant-health"}
        for _ in range(10):
            resp = await client.get("/health", headers=headers)
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_middleware_fails_open_without_limiter():
    import httpx
    from fastapi import FastAPI
    from harness.core.rate_limiter import RateLimitMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    app.state.rate_limiter = None

    @app.get("/runs/x")
    async def _runs():
        return {"ok": True}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        for _ in range(10):
            resp = await client.get("/runs/x", headers={"X-Tenant-ID": "t"})
            assert resp.status_code == 200
