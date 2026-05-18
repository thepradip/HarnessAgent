"""Tests for filesystem layer: CheckpointManager, WorkspaceManager."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from harness.filesystem.checkpoint import CheckpointData, CheckpointManager
from harness.filesystem.workspace import WorkspaceManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ctx(
    run_id="run1",
    tenant_id="t1",
    agent_type="sql",
    task="count users",
    step_count=5,
    token_count=1200,
    metadata=None,
    failed=False,
    failure_class=None,
):
    ctx = MagicMock()
    ctx.run_id = run_id
    ctx.tenant_id = tenant_id
    ctx.agent_type = agent_type
    ctx.task = task
    ctx.step_count = step_count
    ctx.token_count = token_count
    ctx.started_at = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    ctx.metadata = metadata or {}
    ctx.failed = failed
    ctx.failure_class = failure_class
    return ctx


# ===========================================================================
# CheckpointData
# ===========================================================================

def test_checkpoint_data_to_dict():
    now = datetime.now(timezone.utc)
    d = CheckpointData(
        run_id="r1", tenant_id="t1", agent_type="sql",
        task="query", step_count=3, token_count=500,
        started_at=now, metadata={"key": "val"},
        failed=False, failure_class=None, history_snapshot=[],
    )
    result = d.to_dict()
    assert result["run_id"] == "r1"
    assert result["step_count"] == 3
    assert result["token_count"] == 500
    assert isinstance(result["started_at"], str)
    assert result["metadata"] == {"key": "val"}
    assert result["failed"] is False
    assert result["failure_class"] is None


def test_checkpoint_data_round_trip():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    original = CheckpointData(
        run_id="r42", tenant_id="acme", agent_type="code",
        task="debug code", step_count=7, token_count=2000,
        started_at=now, metadata={"x": 1},
        failed=True, failure_class="BUDGET_EXCEEDED",
        history_snapshot=[{"role": "user", "content": "hi"}],
    )
    restored = CheckpointData.from_dict(original.to_dict())
    assert restored.run_id == "r42"
    assert restored.tenant_id == "acme"
    assert restored.agent_type == "code"
    assert restored.step_count == 7
    assert restored.token_count == 2000
    assert restored.started_at == now
    assert restored.metadata == {"x": 1}
    assert restored.failed is True
    assert restored.failure_class == "BUDGET_EXCEEDED"
    assert restored.history_snapshot == [{"role": "user", "content": "hi"}]


def test_checkpoint_data_from_dict_parses_datetime_strings():
    d = {
        "run_id": "r1", "tenant_id": "t1", "agent_type": "base",
        "task": "task", "step_count": 0, "token_count": 0,
        "started_at": "2026-05-17T10:00:00+00:00",
        "metadata": {}, "failed": False, "failure_class": None,
        "history_snapshot": [],
    }
    cp = CheckpointData.from_dict(d)
    assert isinstance(cp.started_at, datetime)


# ===========================================================================
# CheckpointManager
# ===========================================================================

@pytest.fixture
def tmp_workspace(tmp_path):
    return CheckpointManager(tmp_path)


@pytest.mark.asyncio
async def test_checkpoint_save_creates_file(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx()
    path = await mgr.save(ctx, [])
    assert path.exists()
    assert path.name == "checkpoint.json"


@pytest.mark.asyncio
async def test_checkpoint_save_contains_correct_data(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx(step_count=10, token_count=3000)
    await mgr.save(ctx, [])
    data = json.loads((tmp_path / "t1" / "run1" / "checkpoint.json").read_text())
    assert data["step_count"] == 10
    assert data["token_count"] == 3000
    assert data["run_id"] == "run1"
    assert data["tenant_id"] == "t1"


@pytest.mark.asyncio
async def test_checkpoint_save_no_tmp_left_behind(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx()
    await mgr.save(ctx, [])
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == []


@pytest.mark.asyncio
async def test_checkpoint_save_history_dicts(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx()
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    await mgr.save(ctx, history)
    data = json.loads((tmp_path / "t1" / "run1" / "checkpoint.json").read_text())
    assert len(data["history_snapshot"]) == 2
    assert data["history_snapshot"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_checkpoint_load_missing_returns_none(tmp_path):
    mgr = CheckpointManager(tmp_path)
    result = await mgr.load("nonexistent", "t1")
    assert result is None


@pytest.mark.asyncio
async def test_checkpoint_load_returns_correct_data(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx(step_count=5, token_count=500)
    await mgr.save(ctx, [{"role": "user", "content": "hi"}])
    cp = await mgr.load("run1", "t1")
    assert cp is not None
    assert cp.step_count == 5
    assert cp.token_count == 500
    assert cp.run_id == "run1"
    assert len(cp.history_snapshot) == 1


@pytest.mark.asyncio
async def test_checkpoint_load_corrupt_returns_none(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx()
    checkpoint_dir = tmp_path / "t1" / "run1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint.json").write_text("{invalid json!!}")
    result = await mgr.load("run1", "t1")
    assert result is None


@pytest.mark.asyncio
async def test_checkpoint_exists_true(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx()
    await mgr.save(ctx, [])
    assert await mgr.exists("run1", "t1") is True


@pytest.mark.asyncio
async def test_checkpoint_exists_false(tmp_path):
    mgr = CheckpointManager(tmp_path)
    assert await mgr.exists("no_run", "t1") is False


@pytest.mark.asyncio
async def test_checkpoint_delete(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx()
    await mgr.save(ctx, [])
    assert await mgr.exists("run1", "t1") is True
    await mgr.delete("run1", "t1")
    assert await mgr.exists("run1", "t1") is False


@pytest.mark.asyncio
async def test_checkpoint_delete_nonexistent_no_error(tmp_path):
    mgr = CheckpointManager(tmp_path)
    await mgr.delete("no_run", "t1")  # must not raise


@pytest.mark.asyncio
async def test_checkpoint_multiple_runs_isolated(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx1 = _mock_ctx(run_id="run_a", step_count=3)
    ctx2 = _mock_ctx(run_id="run_b", step_count=7)
    await mgr.save(ctx1, [])
    await mgr.save(ctx2, [])
    cp1 = await mgr.load("run_a", "t1")
    cp2 = await mgr.load("run_b", "t1")
    assert cp1.step_count == 3
    assert cp2.step_count == 7


@pytest.mark.asyncio
async def test_checkpoint_overwrites_previous(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx(step_count=1)
    await mgr.save(ctx, [])
    ctx.step_count = 99
    await mgr.save(ctx, [])
    cp = await mgr.load("run1", "t1")
    assert cp.step_count == 99


@pytest.mark.asyncio
async def test_checkpoint_metadata_preserved(tmp_path):
    mgr = CheckpointManager(tmp_path)
    ctx = _mock_ctx(metadata={"db_path": "/tmp/test.db", "gold_sql": "SELECT 1"})
    await mgr.save(ctx, [])
    cp = await mgr.load("run1", "t1")
    assert cp.metadata["db_path"] == "/tmp/test.db"
    assert cp.metadata["gold_sql"] == "SELECT 1"


# ===========================================================================
# WorkspaceManager
# ===========================================================================

@pytest.fixture
def workspace_mgr(tmp_path):
    return WorkspaceManager(tmp_path)


@pytest.mark.asyncio
async def test_workspace_create_returns_path(workspace_mgr):
    path = await workspace_mgr.create("run1", "t1")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.is_dir()


@pytest.mark.asyncio
async def test_workspace_create_idempotent(workspace_mgr):
    p1 = await workspace_mgr.create("run1", "t1")
    p2 = await workspace_mgr.create("run1", "t1")
    assert p1 == p2
    assert p1.exists()


@pytest.mark.asyncio
async def test_workspace_create_different_runs_isolated(workspace_mgr):
    p1 = await workspace_mgr.create("run_a", "t1")
    p2 = await workspace_mgr.create("run_b", "t1")
    assert p1 != p2


@pytest.mark.asyncio
async def test_workspace_create_different_tenants_isolated(workspace_mgr):
    p1 = await workspace_mgr.create("run1", "tenant_a")
    p2 = await workspace_mgr.create("run1", "tenant_b")
    assert p1 != p2
    assert "tenant_a" in str(p1)
    assert "tenant_b" in str(p2)


@pytest.mark.asyncio
async def test_workspace_resolve_safe_path(workspace_mgr):
    await workspace_mgr.create("run1", "t1")
    resolved = workspace_mgr.resolve("run1", "t1", "output.txt")
    assert "output.txt" in str(resolved)


@pytest.mark.asyncio
async def test_workspace_resolve_blocks_path_traversal(workspace_mgr):
    await workspace_mgr.create("run1", "t1")
    with pytest.raises(PermissionError):
        workspace_mgr.resolve("run1", "t1", "../../etc/passwd")


@pytest.mark.asyncio
async def test_workspace_resolve_blocks_absolute_escape(workspace_mgr, tmp_path):
    await workspace_mgr.create("run1", "t1")
    with pytest.raises(PermissionError):
        workspace_mgr.resolve("run1", "t1", "/etc/passwd")


@pytest.mark.asyncio
async def test_workspace_resolve_nested_path_ok(workspace_mgr):
    await workspace_mgr.create("run1", "t1")
    resolved = workspace_mgr.resolve("run1", "t1", "subdir/output.txt")
    assert "subdir" in str(resolved)
    assert "output.txt" in str(resolved)


@pytest.mark.asyncio
async def test_workspace_files_persist(workspace_mgr):
    ws = await workspace_mgr.create("run1", "t1")
    (ws / "test.txt").write_text("hello")
    assert (ws / "test.txt").read_text() == "hello"
