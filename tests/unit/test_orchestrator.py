"""Tests for orchestrator: ApprovalRequest, HITLManager, SubTask, TaskPlan, Scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from harness.orchestrator.hitl import ApprovalRequest, HITLManager
from harness.orchestrator.planner import SubTask, TaskPlan


# ===========================================================================
# ApprovalRequest
# ===========================================================================

def test_approval_request_defaults():
    req = ApprovalRequest(
        run_id="r1", tenant_id="t1", tool_name="delete_file", tool_args={}
    )
    assert req.status == "pending"
    assert req.resolved_at is None
    assert req.resolved_by is None
    assert len(req.request_id) == 32


def test_approval_request_is_expired_false():
    req = ApprovalRequest(
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    assert req.is_expired is False


def test_approval_request_is_expired_true():
    req = ApprovalRequest(
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    assert req.is_expired is True


def test_approval_request_is_resolved_pending():
    req = ApprovalRequest(status="pending")
    assert req.is_resolved is False


def test_approval_request_is_resolved_approved():
    req = ApprovalRequest(status="approved")
    assert req.is_resolved is True


def test_approval_request_is_resolved_rejected():
    req = ApprovalRequest(status="rejected")
    assert req.is_resolved is True


def test_approval_request_is_resolved_expired():
    req = ApprovalRequest(status="expired")
    assert req.is_resolved is False


def test_approval_request_round_trip():
    req = ApprovalRequest(
        run_id="r1", tenant_id="t1", tool_name="drop_table",
        tool_args={"table": "users"}, reason="DDL blocked",
        status="pending",
    )
    d = req.to_dict()
    req2 = ApprovalRequest.from_dict(d)
    assert req2.run_id == "r1"
    assert req2.tenant_id == "t1"
    assert req2.tool_name == "drop_table"
    assert req2.tool_args == {"table": "users"}
    assert req2.reason == "DDL blocked"
    assert req2.status == "pending"


def test_approval_request_to_json_parseable():
    import json
    req = ApprovalRequest(run_id="r1", tool_name="execute_sql")
    parsed = json.loads(req.to_json())
    assert parsed["run_id"] == "r1"
    assert parsed["tool_name"] == "execute_sql"


# ===========================================================================
# HITLManager
# ===========================================================================

def _fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def hitl():
    return HITLManager(_fake_redis(), ttl_seconds=3600)


@pytest.mark.asyncio
async def test_hitl_request_approval_creates_pending(hitl):
    req = await hitl.request_approval(
        run_id="r1", tenant_id="t1",
        tool_name="drop_table", tool_args={"table": "users"},
        reason="DDL blocked by policy",
    )
    assert req.status == "pending"
    assert req.tool_name == "drop_table"
    assert req.run_id == "r1"


@pytest.mark.asyncio
async def test_hitl_get_returns_request(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})
    fetched = await hitl.get(req.request_id)
    assert fetched is not None
    assert fetched.request_id == req.request_id


@pytest.mark.asyncio
async def test_hitl_get_nonexistent_returns_none(hitl):
    result = await hitl.get("does_not_exist")
    assert result is None


@pytest.mark.asyncio
async def test_hitl_approve(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})
    approved = await hitl.approve(req.request_id, resolved_by="admin")
    assert approved.status == "approved"
    assert approved.resolved_by == "admin"
    assert approved.resolved_at is not None


@pytest.mark.asyncio
async def test_hitl_reject(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})
    rejected = await hitl.reject(req.request_id, resolved_by="admin")
    assert rejected.status == "rejected"
    assert rejected.resolved_by == "admin"


@pytest.mark.asyncio
async def test_hitl_approve_nonexistent_raises(hitl):
    with pytest.raises(KeyError):
        await hitl.approve("nonexistent_id")


@pytest.mark.asyncio
async def test_hitl_reject_nonexistent_raises(hitl):
    with pytest.raises(KeyError):
        await hitl.reject("nonexistent_id")


@pytest.mark.asyncio
async def test_hitl_approve_already_resolved_raises(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})
    await hitl.approve(req.request_id)
    with pytest.raises(ValueError):
        await hitl.approve(req.request_id)


@pytest.mark.asyncio
async def test_hitl_reject_already_resolved_raises(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})
    await hitl.reject(req.request_id)
    with pytest.raises(ValueError):
        await hitl.reject(req.request_id)


@pytest.mark.asyncio
async def test_hitl_list_pending_returns_pending(hitl):
    req1 = await hitl.request_approval("r1", "t1", "tool_a", {})
    req2 = await hitl.request_approval("r2", "t1", "tool_b", {})
    await hitl.approve(req1.request_id)
    pending = await hitl.list_pending()
    pending_ids = {r.request_id for r in pending}
    assert req2.request_id in pending_ids
    assert req1.request_id not in pending_ids


@pytest.mark.asyncio
async def test_hitl_list_pending_empty(hitl):
    result = await hitl.list_pending()
    assert result == []


@pytest.mark.asyncio
async def test_hitl_list_pending_tenant_filter(hitl):
    await hitl.request_approval("r1", "acme", "tool", {})
    await hitl.request_approval("r2", "beta", "tool", {})
    acme_pending = await hitl.list_pending(tenant_id="acme")
    assert all(r.tenant_id == "acme" for r in acme_pending)


@pytest.mark.asyncio
async def test_hitl_await_decision_approved(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})

    async def _approve_after_delay():
        await asyncio.sleep(0.05)
        await hitl.approve(req.request_id)

    asyncio.create_task(_approve_after_delay())
    result = await hitl.await_decision(req.request_id, timeout=5.0, poll_interval=0.02)
    assert result == "approved"


@pytest.mark.asyncio
async def test_hitl_await_decision_rejected(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})

    async def _reject_after_delay():
        await asyncio.sleep(0.05)
        await hitl.reject(req.request_id)

    asyncio.create_task(_reject_after_delay())
    result = await hitl.await_decision(req.request_id, timeout=5.0, poll_interval=0.02)
    assert result == "rejected"


@pytest.mark.asyncio
async def test_hitl_await_decision_timeout(hitl):
    req = await hitl.request_approval("r1", "t1", "tool", {})
    result = await hitl.await_decision(req.request_id, timeout=0.1, poll_interval=0.03)
    assert result == "expired"


@pytest.mark.asyncio
async def test_hitl_await_decision_nonexistent_returns_expired(hitl):
    result = await hitl.await_decision("no_such_id", timeout=0.1, poll_interval=0.05)
    assert result == "expired"


# ===========================================================================
# SubTask
# ===========================================================================

def test_subtask_defaults():
    st = SubTask(id="t1", agent_type="sql", task="count users")
    assert st.depends_on == []
    assert st.metadata == {}


def test_subtask_with_deps():
    st = SubTask(id="t2", agent_type="code", task="plot data", depends_on=["t1"])
    assert st.depends_on == ["t1"]


# ===========================================================================
# TaskPlan
# ===========================================================================

def _make_plan():
    subtasks = [
        SubTask(id="t1", agent_type="sql",  task="query data",    depends_on=[]),
        SubTask(id="t2", agent_type="code", task="plot results",  depends_on=["t1"]),
        SubTask(id="t3", agent_type="base", task="write summary", depends_on=["t2"]),
    ]
    return TaskPlan(plan_id="plan1", original_task="analyse data", subtasks=subtasks)


def test_task_plan_get_ready_tasks_empty_completed():
    plan = _make_plan()
    ready = plan.get_ready_tasks(set())
    assert len(ready) == 1
    assert ready[0].id == "t1"


def test_task_plan_get_ready_tasks_after_t1():
    plan = _make_plan()
    ready = plan.get_ready_tasks({"t1"})
    assert len(ready) == 1
    assert ready[0].id == "t2"


def test_task_plan_get_ready_tasks_after_t1_t2():
    plan = _make_plan()
    ready = plan.get_ready_tasks({"t1", "t2"})
    assert len(ready) == 1
    assert ready[0].id == "t3"


def test_task_plan_get_ready_tasks_all_completed():
    plan = _make_plan()
    ready = plan.get_ready_tasks({"t1", "t2", "t3"})
    assert ready == []


def test_task_plan_topological_order_linear():
    plan = _make_plan()
    order = plan.topological_order()
    ids = [s.id for s in order]
    assert ids.index("t1") < ids.index("t2")
    assert ids.index("t2") < ids.index("t3")


def test_task_plan_topological_order_parallel():
    subtasks = [
        SubTask(id="t1", agent_type="sql",  task="query A", depends_on=[]),
        SubTask(id="t2", agent_type="sql",  task="query B", depends_on=[]),
        SubTask(id="t3", agent_type="code", task="merge",   depends_on=["t1", "t2"]),
    ]
    plan = TaskPlan(plan_id="p1", original_task="merge", subtasks=subtasks)
    order = plan.topological_order()
    ids = [s.id for s in order]
    assert ids.index("t1") < ids.index("t3")
    assert ids.index("t2") < ids.index("t3")
    assert len(order) == 3


def test_task_plan_topological_order_single_task():
    plan = TaskPlan(
        plan_id="p1", original_task="simple",
        subtasks=[SubTask(id="t1", agent_type="base", task="do it")],
    )
    order = plan.topological_order()
    assert len(order) == 1
    assert order[0].id == "t1"


def test_task_plan_ready_tasks_excludes_already_completed():
    plan = _make_plan()
    # If t1 is already done but t2 isn't, t2 should be ready but t1 should not
    ready = plan.get_ready_tasks({"t1"})
    ready_ids = {s.id for s in ready}
    assert "t1" not in ready_ids
    assert "t2" in ready_ids


def test_task_plan_multiple_parallel_tasks_all_ready_at_start():
    subtasks = [
        SubTask(id="a", agent_type="sql",  task="A", depends_on=[]),
        SubTask(id="b", agent_type="code", task="B", depends_on=[]),
        SubTask(id="c", agent_type="base", task="C", depends_on=["a", "b"]),
    ]
    plan = TaskPlan(plan_id="p", original_task="all", subtasks=subtasks)
    ready = plan.get_ready_tasks(set())
    ready_ids = {s.id for s in ready}
    assert "a" in ready_ids
    assert "b" in ready_ids
    assert "c" not in ready_ids


# ===========================================================================
# Scheduler (lightweight — mock agent runner)
# ===========================================================================

def _make_mock_runner():
    from harness.core.context import AgentResult
    runner = MagicMock()
    runner.create_run = AsyncMock(return_value=MagicMock(run_id="mock_run"))
    runner.execute_run = AsyncMock(return_value=MagicMock(
        result={"run_id": "mock_run", "output": "done", "success": True,
                "steps": 2, "tokens": 100, "cost_usd": 0.001,
                "elapsed_seconds": 0.5, "tool_calls": 1, "tool_errors": 0,
                "guardrail_hits": 0, "handoff_count": 0,
                "cache_hits": 0, "cache_read_tokens": 0}
    ))
    return runner


@pytest.mark.asyncio
async def test_scheduler_executes_single_task():
    from harness.orchestrator.scheduler import Scheduler

    plan = TaskPlan(
        plan_id="p1", original_task="count users",
        subtasks=[SubTask(id="t1", agent_type="sql", task="SELECT COUNT(*) FROM users")],
    )
    scheduler = Scheduler(agent_runner=_make_mock_runner())
    results = await scheduler.execute_plan(plan, tenant_id="t1", timeout=30.0)
    assert "t1" in results


@pytest.mark.asyncio
async def test_scheduler_executes_parallel_tasks():
    from harness.orchestrator.scheduler import Scheduler

    plan = TaskPlan(
        plan_id="p1", original_task="parallel",
        subtasks=[
            SubTask(id="a", agent_type="sql",  task="query A", depends_on=[]),
            SubTask(id="b", agent_type="code", task="code B",  depends_on=[]),
        ],
    )
    scheduler = Scheduler(agent_runner=_make_mock_runner())
    results = await scheduler.execute_plan(plan, tenant_id="t1", timeout=30.0)
    assert "a" in results
    assert "b" in results


@pytest.mark.asyncio
async def test_scheduler_respects_dependency_order():
    from harness.orchestrator.scheduler import Scheduler

    execution_order = []

    async def _mock_execute(run_id):
        execution_order.append(run_id)
        return MagicMock(result={
            "run_id": run_id, "output": f"done_{run_id}", "success": True,
            "steps": 1, "tokens": 50, "cost_usd": 0.0, "elapsed_seconds": 0.1,
            "tool_calls": 0, "tool_errors": 0, "guardrail_hits": 0,
            "handoff_count": 0, "cache_hits": 0, "cache_read_tokens": 0
        })

    runner = MagicMock()
    runner.create_run = AsyncMock(side_effect=lambda **kw: MagicMock(run_id=kw.get("task", "run")))
    runner.execute_run = AsyncMock(side_effect=_mock_execute)

    plan = TaskPlan(
        plan_id="p1", original_task="chain",
        subtasks=[
            SubTask(id="t1", agent_type="sql",  task="first",  depends_on=[]),
            SubTask(id="t2", agent_type="code", task="second", depends_on=["t1"]),
        ],
    )
    scheduler = Scheduler(agent_runner=runner)
    results = await scheduler.execute_plan(plan, tenant_id="t1", timeout=30.0)
    assert "t1" in results
    assert "t2" in results


@pytest.mark.asyncio
async def test_scheduler_blackboard_metadata_is_json_serialisable():
    # Regression: the scheduler used to stuff the AgentBlackboard OBJECT into
    # run metadata, which blew up RunRecord.to_json() (plain json.dumps) and
    # failed every subtask whenever the scheduler had a redis client.
    import json

    from harness.orchestrator.scheduler import Scheduler

    redis = fakeredis.FakeRedis(decode_responses=True)
    runner = _make_mock_runner()
    plan = TaskPlan(
        plan_id="p_bb", original_task="bb",
        subtasks=[SubTask(id="t1", agent_type="sql", task="query")],
    )
    scheduler = Scheduler(agent_runner=runner, redis=redis)
    results = await scheduler.execute_plan(plan, tenant_id="t1", timeout=30.0)
    assert "t1" in results

    metadata = runner.create_run.call_args.kwargs["metadata"]
    json.dumps(metadata)  # must not raise
    assert "blackboard" not in metadata
    assert metadata["blackboard_plan_id"] == "p_bb"


# ===========================================================================
# AgentRunner — concurrent cancel during execute_run
# ===========================================================================

@pytest.mark.asyncio
async def test_execute_run_preserves_concurrent_cancel(tmp_path):
    # Regression: execute_run used to unconditionally persist its local record
    # at the end, clobbering a concurrent cancel_run()'s "cancelled" status.
    from harness.core.context import AgentResult
    from harness.orchestrator.runner import AgentRunner

    redis = fakeredis.FakeRedis(decode_responses=True)
    holder: dict = {}

    class _Agent:
        async def run(self, ctx):
            # Simulate an operator cancelling the run while the agent works
            await holder["runner"].cancel_run(ctx.run_id)
            return AgentResult(
                run_id=ctx.run_id, output="done", steps=1, tokens=10, success=True
            )

    runner = AgentRunner(
        redis=redis,
        agent_factory=lambda agent_type: _Agent(),
        workspace_base=str(tmp_path),
    )
    holder["runner"] = runner

    record = await runner.create_run(tenant_id="t1", agent_type="base", task="x")
    result = await runner.execute_run(record.run_id)

    assert result.status == "cancelled"
    persisted = await runner.get_run(record.run_id)
    assert persisted is not None
    assert persisted.status == "cancelled"
