"""Tests for filesystem layer: CheckpointManager, WorkspaceManager."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


# ===========================================================================
# DockerSandbox — runtime flag
# ===========================================================================

def test_docker_sandbox_default_runtime_is_runc():
    from harness.filesystem.sandbox import DockerSandbox
    sb = DockerSandbox()
    assert sb._runtime == "runc"


def test_docker_sandbox_custom_runtime_stored():
    from harness.filesystem.sandbox import DockerSandbox
    sb = DockerSandbox(runtime="runsc")
    assert sb._runtime == "runsc"


@pytest.mark.asyncio
async def test_docker_sandbox_runc_does_not_add_runtime_flag(tmp_path):
    """Default runc runtime must NOT add --runtime to the docker command."""
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import DockerSandbox

    (tmp_path / "run_test.py").write_text("print('hello')")
    sb = DockerSandbox(runtime="runc", timeout=5.0)

    captured_cmd: list[str] = []

    async def fake_run_command(cmd, workspace_path, env=None):
        captured_cmd.extend(cmd)
        from harness.filesystem.sandbox import SandboxResult
        return SandboxResult(stdout="hello\n", stderr="", exit_code=0,
                             timed_out=False, execution_time_ms=10.0)

    with patch.object(sb, "run_command", side_effect=fake_run_command):
        await sb.run_code("print('hello')", tmp_path)

    assert "--runtime=runc" not in captured_cmd


@pytest.mark.asyncio
async def test_docker_sandbox_gvisor_adds_runtime_flag(tmp_path):
    """Non-default runtime must inject --runtime=<name> into the docker command."""
    from unittest.mock import patch
    from harness.filesystem.sandbox import DockerSandbox

    sb = DockerSandbox(runtime="runsc", timeout=5.0)
    captured_cmd: list[str] = []

    async def fake_run_command(cmd, workspace_path, env=None):
        captured_cmd.extend(cmd)
        from harness.filesystem.sandbox import SandboxResult
        return SandboxResult(stdout="", stderr="", exit_code=0,
                             timed_out=False, execution_time_ms=10.0)

    with patch.object(sb, "run_command", side_effect=fake_run_command):
        await sb.run_code("x = 1", tmp_path)

    assert "--runtime=runsc" in captured_cmd


# ===========================================================================
# SessionDockerSandbox — unit tests (no real Docker)
# ===========================================================================

@pytest.mark.asyncio
async def test_session_sandbox_run_code_raises_when_not_started(tmp_path):
    from harness.filesystem.sandbox import SessionDockerSandbox, SandboxError
    sb = SessionDockerSandbox(tmp_path)
    with pytest.raises(SandboxError, match="not running"):
        await sb.run_code("print('hi')")


@pytest.mark.asyncio
async def test_session_sandbox_is_alive_false_before_start(tmp_path):
    from harness.filesystem.sandbox import SessionDockerSandbox
    sb = SessionDockerSandbox(tmp_path)
    assert sb.is_alive is False


@pytest.mark.asyncio
async def test_session_sandbox_is_alive_true_after_start(tmp_path):
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox

    async def fake_communicate():
        return (b"fake-container-id-123\n", b"")

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = fake_communicate

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        sb = SessionDockerSandbox(tmp_path)
        await sb._start_container()
        assert sb.is_alive is True
        assert sb._container_id == "fake-container-id-123"


@pytest.mark.asyncio
async def test_session_sandbox_stop_clears_container_id(tmp_path):
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox

    sb = SessionDockerSandbox(tmp_path)
    sb._container_id = "abc123"

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        await sb._stop_container()

    assert sb._container_id is None
    assert sb.is_alive is False


@pytest.mark.asyncio
async def test_session_sandbox_stop_noop_when_not_started(tmp_path):
    from harness.filesystem.sandbox import SessionDockerSandbox
    sb = SessionDockerSandbox(tmp_path)
    await sb._stop_container()  # must not raise


@pytest.mark.asyncio
async def test_session_sandbox_run_code_returns_result(tmp_path):
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox

    sb = SessionDockerSandbox(tmp_path, timeout=10.0)
    sb._container_id = "live-container"

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"42\n", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await sb.run_code("print(42)")

    assert result.stdout == "42\n"
    assert result.exit_code == 0
    assert result.success is True


@pytest.mark.asyncio
async def test_session_sandbox_run_code_timeout(tmp_path):
    import asyncio as _asyncio
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox

    sb = SessionDockerSandbox(tmp_path, timeout=1.0)
    sb._container_id = "live-container"

    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()

    async def slow_communicate():
        await _asyncio.sleep(99)
        return b"", b""

    mock_proc.communicate = slow_communicate

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await sb.run_code("import time; time.sleep(99)", timeout=0.05)

    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()


@pytest.mark.asyncio
async def test_session_sandbox_detects_container_death(tmp_path):
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox, SandboxError

    sb = SessionDockerSandbox(tmp_path)
    sb._container_id = "dead-container"

    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"Error response from daemon: No such container: dead-container")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(SandboxError, match="Session container died"):
            await sb.run_code("x = 1")

    assert sb.is_alive is False


@pytest.mark.asyncio
async def test_session_sandbox_start_failure_raises_sandbox_error(tmp_path):
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox, SandboxError

    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"docker: image not found"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        sb = SessionDockerSandbox(tmp_path)
        with pytest.raises(SandboxError, match="Failed to start session container"):
            await sb._start_container()


@pytest.mark.asyncio
async def test_session_sandbox_gvisor_runtime_in_start_cmd(tmp_path):
    """gVisor runtime flag must appear in the docker run command."""
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox

    captured_cmds: list[list[str]] = []

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"container-xyz\n", b""))

    async def capture(*args, **kwargs):
        captured_cmds.append(list(args))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=capture):
        sb = SessionDockerSandbox(tmp_path, runtime="runsc")
        await sb._start_container()

    flat = [a for cmd in captured_cmds for a in cmd]
    assert "--runtime=runsc" in flat


@pytest.mark.asyncio
async def test_session_sandbox_runc_not_in_start_cmd(tmp_path):
    """Default runc must NOT add --runtime flag."""
    from unittest.mock import AsyncMock, patch
    from harness.filesystem.sandbox import SessionDockerSandbox

    captured_cmds: list[list[str]] = []

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"container-xyz\n", b""))

    async def capture(*args, **kwargs):
        captured_cmds.append(list(args))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=capture):
        sb = SessionDockerSandbox(tmp_path, runtime="runc")
        await sb._start_container()

    flat = [a for cmd in captured_cmds for a in cmd]
    assert "--runtime=runc" not in flat


# ===========================================================================
# RunCodeTool — session priority
# ===========================================================================

@pytest.mark.asyncio
async def test_run_code_tool_uses_session_when_alive(tmp_path):
    """When a live session is in ctx.metadata, RunCodeTool should use it."""
    from unittest.mock import AsyncMock, MagicMock
    from harness.tools.code_tools import RunCodeTool
    from harness.filesystem.sandbox import SandboxResult

    mock_session = MagicMock()
    mock_session.is_alive = True
    mock_session.run_code = AsyncMock(return_value=SandboxResult(
        stdout="session_result\n", stderr="", exit_code=0,
        timed_out=False, execution_time_ms=5.0,
    ))

    tool = RunCodeTool()
    ctx = MagicMock()
    ctx.metadata = {"docker_session": mock_session}
    ctx.run_id = "r1"
    ctx.step_count = 1
    ctx.workspace_path = tmp_path

    result = await tool.execute(ctx, {"code": "print('hi')"})

    mock_session.run_code.assert_called_once()
    assert result.data["stdout"] == "session_result\n"


@pytest.mark.asyncio
async def test_run_code_tool_skips_dead_session(tmp_path):
    """A session with is_alive=False must be skipped silently."""
    from unittest.mock import AsyncMock, MagicMock
    from harness.tools.code_tools import RunCodeTool

    dead_session = MagicMock()
    dead_session.is_alive = False

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"subprocess_out\n", b""))

    tool = RunCodeTool()
    ctx = MagicMock()
    ctx.metadata = {"docker_session": dead_session}
    ctx.run_id = "r1"
    ctx.step_count = 1
    ctx.workspace_path = tmp_path

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.execute(ctx, {"code": "print('hi')"})

    dead_session.run_code.assert_not_called()
    assert result.data is not None


# ---------------------------------------------------------------------------
# Sandbox security limits — cpu_time, disk_quota, pids_limit, exec timeout wrap
# ---------------------------------------------------------------------------

def test_docker_sandbox_includes_cpu_time_limit():
    """DockerSandbox docker run command must include --ulimit=cpu=<N>:<N>."""
    from harness.filesystem.sandbox import DockerSandbox
    sb = DockerSandbox(cpu_time_seconds=45)
    # Inspect the command built inside run_code by checking __init__ attributes
    assert sb._cpu_time == 45
    assert sb._fsize_blocks == (512 * 1024 * 1024) // 512  # default 512 MiB


def test_docker_sandbox_custom_disk_quota():
    """DockerSandbox disk_quota_mb converts correctly to 512-byte blocks."""
    from harness.filesystem.sandbox import DockerSandbox
    sb = DockerSandbox(disk_quota_mb=256)
    expected_blocks = (256 * 1024 * 1024) // 512  # 524288 blocks
    assert sb._fsize_blocks == expected_blocks


def test_docker_sandbox_pids_limit_stored():
    """DockerSandbox stores pids_limit for injection into docker run."""
    from harness.filesystem.sandbox import DockerSandbox
    sb = DockerSandbox(pids_limit=32)
    assert sb._pids_limit == 32


def test_session_sandbox_timeout_wrap_uses_timeout_command(tmp_path):
    """SessionDockerSandbox exec must wrap python with `timeout -k <grace> <N>`."""
    from harness.filesystem.sandbox import SessionDockerSandbox
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    sb = SessionDockerSandbox(workspace_path=tmp_path, timeout=20.0)
    sb._container_id = "abc123"

    captured_cmd: list = []

    async def _fake_exec(*args, **kwargs):
        captured_cmd.extend(args)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"ok", b""))
        return proc

    async def _run():
        with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
            await sb.run_code("print('hi')")

    asyncio.get_event_loop().run_until_complete(_run())

    cmd_str = " ".join(str(c) for c in captured_cmd)
    assert "timeout" in cmd_str, "exec must use timeout command inside container"
    assert "-k" in cmd_str, "timeout must pass SIGKILL grace flag"


def test_session_sandbox_stores_limits():
    """SessionDockerSandbox stores all three new limit params."""
    import tempfile
    from pathlib import Path
    from harness.filesystem.sandbox import SessionDockerSandbox

    with tempfile.TemporaryDirectory() as tmp:
        sb = SessionDockerSandbox(
            workspace_path=Path(tmp),
            cpu_time_seconds=30,
            disk_quota_mb=128,
            pids_limit=16,
        )
    assert sb._cpu_time == 30
    assert sb._fsize_blocks == (128 * 1024 * 1024) // 512
    assert sb._pids_limit == 16
