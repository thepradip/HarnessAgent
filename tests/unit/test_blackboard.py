"""Tests for AgentBlackboard and Scheduler blackboard integration."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import fakeredis.aioredis as fakeredis

from harness.orchestrator.blackboard import AgentBlackboard, BlackboardEntry


def _redis():
    return fakeredis.FakeRedis(decode_responses=True)


def _bb(plan_id="plan-1"):
    return AgentBlackboard(_redis(), plan_id=plan_id)


# ===========================================================================
# BlackboardEntry
# ===========================================================================

def test_entry_round_trip():
    e = BlackboardEntry(
        plan_id="p1", subtask_id="sql_agent", artifact_type="sql",
        content="SELECT * FROM users", metadata={"rows": 42},
    )
    e2 = BlackboardEntry.from_dict(e.to_dict())
    assert e2.subtask_id == "sql_agent"
    assert e2.artifact_type == "sql"
    assert e2.content == "SELECT * FROM users"
    assert e2.metadata == {"rows": 42}


def test_entry_datetime_preserved():
    ts = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
    e = BlackboardEntry("p", "s", "code", "x = 1", written_at=ts)
    e2 = BlackboardEntry.from_dict(e.to_dict())
    assert e2.written_at == ts


# ===========================================================================
# write / get
# ===========================================================================

@pytest.mark.asyncio
async def test_write_and_get():
    bb = _bb()
    await bb.write("sql_agent", "sql", "SELECT 1")
    entry = await bb.get("sql_agent", "sql")
    assert entry is not None
    assert entry.content == "SELECT 1"
    assert entry.artifact_type == "sql"
    assert entry.subtask_id == "sql_agent"


@pytest.mark.asyncio
async def test_write_empty_content_skipped():
    bb = _bb()
    await bb.write("sql_agent", "sql", "")
    entry = await bb.get("sql_agent", "sql")
    assert entry is None


@pytest.mark.asyncio
async def test_write_overwrites_existing():
    bb = _bb()
    await bb.write("sql_agent", "sql", "SELECT 1")
    await bb.write("sql_agent", "sql", "SELECT 2")
    entry = await bb.get("sql_agent", "sql")
    assert entry.content == "SELECT 2"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    bb = _bb()
    assert await bb.get("nonexistent", "sql") is None


@pytest.mark.asyncio
async def test_write_with_metadata():
    bb = _bb()
    await bb.write("code_agent", "code", "def foo(): pass", metadata={"language": "python"})
    entry = await bb.get("code_agent", "code")
    assert entry.metadata["language"] == "python"


# ===========================================================================
# read — filtering
# ===========================================================================

@pytest.mark.asyncio
async def test_read_all():
    bb = _bb()
    await bb.write("a", "sql", "SELECT 1")
    await bb.write("b", "code", "x = 1")
    entries = await bb.read()
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_read_filter_by_subtask():
    bb = _bb()
    await bb.write("sql_agent", "sql", "SELECT 1")
    await bb.write("code_agent", "code", "x = 1")
    entries = await bb.read(subtask_ids=["sql_agent"])
    assert len(entries) == 1
    assert entries[0].subtask_id == "sql_agent"


@pytest.mark.asyncio
async def test_read_filter_by_type():
    bb = _bb()
    await bb.write("a", "sql", "SELECT 1")
    await bb.write("b", "sql", "SELECT 2")
    await bb.write("c", "code", "x = 1")
    entries = await bb.read(artifact_types=["sql"])
    assert len(entries) == 2
    assert all(e.artifact_type == "sql" for e in entries)


@pytest.mark.asyncio
async def test_read_filter_by_both():
    bb = _bb()
    await bb.write("sql_agent", "sql", "SELECT 1")
    await bb.write("sql_agent", "output", "42 rows")
    await bb.write("code_agent", "sql", "SELECT 2")
    entries = await bb.read(subtask_ids=["sql_agent"], artifact_types=["sql"])
    assert len(entries) == 1
    assert entries[0].subtask_id == "sql_agent"
    assert entries[0].artifact_type == "sql"


@pytest.mark.asyncio
async def test_read_empty_store():
    bb = _bb()
    assert await bb.read() == []


@pytest.mark.asyncio
async def test_read_sorted_by_time():
    bb = _bb()
    await bb.write("first", "output", "first output")
    await bb.write("second", "output", "second output")
    entries = await bb.read()
    times = [e.written_at for e in entries]
    assert times == sorted(times)


# ===========================================================================
# format_for_context
# ===========================================================================

@pytest.mark.asyncio
async def test_format_for_context_includes_content():
    bb = _bb()
    await bb.write("sql_agent", "sql", "SELECT COUNT(*) FROM users")
    result = await bb.format_for_context(subtask_ids=["sql_agent"])
    assert "sql_agent" in result
    assert "SELECT COUNT(*) FROM users" in result
    assert "[Shared plan artifacts]" in result


@pytest.mark.asyncio
async def test_format_for_context_empty_returns_empty():
    bb = _bb()
    result = await bb.format_for_context()
    assert result == ""


@pytest.mark.asyncio
async def test_format_for_context_respects_max_chars():
    bb = _bb()
    await bb.write("agent", "output", "x" * 5000)
    result = await bb.format_for_context(max_chars=500)
    assert len(result) <= 600  # header + some content


@pytest.mark.asyncio
async def test_format_for_context_filters_by_subtask():
    bb = _bb()
    await bb.write("a", "output", "output from A")
    await bb.write("b", "output", "output from B")
    result = await bb.format_for_context(subtask_ids=["a"])
    assert "output from A" in result
    assert "output from B" not in result


# ===========================================================================
# delete / clear
# ===========================================================================

@pytest.mark.asyncio
async def test_delete_single_entry():
    bb = _bb()
    await bb.write("a", "sql", "SELECT 1")
    await bb.write("a", "output", "done")
    await bb.delete("a", "sql")
    assert await bb.get("a", "sql") is None
    assert await bb.get("a", "output") is not None


@pytest.mark.asyncio
async def test_delete_nonexistent_no_error():
    bb = _bb()
    await bb.delete("ghost", "sql")  # must not raise


@pytest.mark.asyncio
async def test_clear_removes_all():
    bb = _bb()
    await bb.write("a", "sql", "SELECT 1")
    await bb.write("b", "code", "x = 1")
    await bb.clear()
    assert await bb.read() == []


# ===========================================================================
# plan isolation
# ===========================================================================

@pytest.mark.asyncio
async def test_separate_plans_isolated():
    r = _redis()
    bb1 = AgentBlackboard(r, plan_id="plan-A")
    bb2 = AgentBlackboard(r, plan_id="plan-B")
    await bb1.write("agent", "output", "plan A result")
    await bb2.write("agent", "output", "plan B result")
    entries1 = await bb1.read()
    entries2 = await bb2.read()
    assert all(e.content == "plan A result" for e in entries1)
    assert all(e.content == "plan B result" for e in entries2)


# ===========================================================================
# Redis failure resilience
# ===========================================================================

@pytest.mark.asyncio
async def test_write_survives_redis_error():
    bad_redis = MagicMock()
    bad_redis.setex = AsyncMock(side_effect=RuntimeError("connection refused"))
    bad_redis.zadd = AsyncMock(side_effect=RuntimeError("connection refused"))
    bb = AgentBlackboard(bad_redis, plan_id="p1")
    await bb.write("a", "sql", "SELECT 1")  # must not raise


@pytest.mark.asyncio
async def test_read_survives_redis_error():
    bad_redis = MagicMock()
    bad_redis.zrange = AsyncMock(side_effect=RuntimeError("connection refused"))
    bb = AgentBlackboard(bad_redis, plan_id="p1")
    entries = await bb.read()  # must not raise
    assert entries == []


# ===========================================================================
# Scheduler integration — blackboard wired in
# ===========================================================================

@pytest.mark.asyncio
async def test_scheduler_creates_blackboard_when_redis_provided():
    from harness.orchestrator.scheduler import Scheduler
    from harness.orchestrator.planner import TaskPlan, SubTask

    r = _redis()
    mock_runner = MagicMock()
    mock_runner.create_run = AsyncMock(return_value=MagicMock(run_id="run-1"))
    mock_runner.execute_run = AsyncMock(return_value=MagicMock(
        result={"output": "done", "success": True, "steps": 1,
                "tokens": 10, "cost_usd": 0.001, "elapsed_seconds": 1.0,
                "tool_calls": 0, "tool_errors": 0, "guardrail_hits": 0,
                "cache_hits": 0, "cache_read_tokens": 0},
        status="completed",
    ))

    scheduler = Scheduler(agent_runner=mock_runner, redis=r)
    plan = TaskPlan(
        plan_id="test-plan",
        original_task="query users",
        subtasks=[SubTask(id="t1", agent_type="sql", task="query users")],
    )

    results = await scheduler.execute_plan(plan, tenant_id="acme")
    assert "t1" in results

    # Blackboard should have written the result
    bb = AgentBlackboard(r, plan_id="test-plan")
    entry = await bb.get("t1", "output")
    assert entry is not None
    assert entry.content == "done"


@pytest.mark.asyncio
async def test_scheduler_without_redis_no_blackboard():
    from harness.orchestrator.scheduler import Scheduler
    from harness.orchestrator.planner import TaskPlan, SubTask

    mock_runner = MagicMock()
    mock_runner.create_run = AsyncMock(return_value=MagicMock(run_id="run-1"))
    mock_runner.execute_run = AsyncMock(return_value=MagicMock(
        result={"output": "done", "success": True, "steps": 1,
                "tokens": 10, "cost_usd": 0.0, "elapsed_seconds": 1.0,
                "tool_calls": 0, "tool_errors": 0, "guardrail_hits": 0,
                "cache_hits": 0, "cache_read_tokens": 0},
        status="completed",
    ))

    # No redis — must work exactly as before, no blackboard
    scheduler = Scheduler(agent_runner=mock_runner)
    plan = TaskPlan(
        plan_id="test-plan-2",
        original_task="write code",
        subtasks=[SubTask(id="t1", agent_type="code", task="write code")],
    )
    results = await scheduler.execute_plan(plan, tenant_id="acme")
    assert results["t1"].success is True


@pytest.mark.asyncio
async def test_scheduler_injects_blackboard_context_into_dependent_task():
    """Downstream agent task should include blackboard context from predecessors."""
    from harness.orchestrator.scheduler import Scheduler
    from harness.orchestrator.planner import TaskPlan, SubTask

    r = _redis()
    captured_tasks: list[str] = []

    mock_runner = MagicMock()

    async def create_run(tenant_id, agent_type, task, metadata):
        captured_tasks.append(task)
        return MagicMock(run_id=f"run-{len(captured_tasks)}")

    mock_runner.create_run = AsyncMock(side_effect=create_run)
    mock_runner.execute_run = AsyncMock(return_value=MagicMock(
        result={"output": "SELECT * FROM users", "success": True,
                "steps": 1, "tokens": 10, "cost_usd": 0.0,
                "elapsed_seconds": 1.0, "tool_calls": 0, "tool_errors": 0,
                "guardrail_hits": 0, "cache_hits": 0, "cache_read_tokens": 0},
        status="completed",
    ))

    scheduler = Scheduler(agent_runner=mock_runner, redis=r)
    plan = TaskPlan(
        plan_id="chained-plan",
        original_task="query then analyse",
        subtasks=[
            SubTask(id="sql_task", agent_type="sql", task="query users", depends_on=[]),
            SubTask(id="code_task", agent_type="code", task="analyse results", depends_on=["sql_task"]),
        ],
    )
    await scheduler.execute_plan(plan, tenant_id="acme")

    # The second task (code_task) should include blackboard context from sql_task
    code_task_input = next((t for t in captured_tasks if "analyse" in t), "")
    assert "Shared plan artifacts" in code_task_input or "sql_task" in code_task_input
