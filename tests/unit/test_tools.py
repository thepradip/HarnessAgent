"""Unit tests for ToolRegistry and tool implementations."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.core.context import AgentContext, ToolCall, ToolResult
from harness.core.errors import FailureClass, SafetyViolation, ToolError
from harness.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path: Path) -> AgentContext:
    return AgentContext(
        run_id=uuid.uuid4().hex,
        tenant_id="test",
        agent_type="test",
        task="test",
        memory=MagicMock(),
        workspace_path=tmp_path / "ws",
    )


def _make_tool(name: str, response: dict = None, error: str = None):
    """Create a minimal ToolExecutor mock."""
    tool = MagicMock()
    tool.name = name
    tool.description = f"{name} tool"
    tool.input_schema = {"type": "object", "properties": {}, "required": []}
    tool.output_schema = None   # explicit None so registry skips output validation
    tool.timeout_seconds = 5.0

    async def _execute(ctx, args):
        if error:
            return ToolResult(data=None, error=error)
        return ToolResult(data=response or {"status": "ok"})

    tool.execute = _execute
    return tool


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_executes_registered_tool(tmp_path):
    """Registry should execute a registered tool and return its result."""
    registry = ToolRegistry()
    tool = _make_tool("my_tool", response={"value": 42})
    registry.register(tool)

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-1", name="my_tool", args={})
    result = await registry.execute(ctx, call)

    assert result.data == {"value": 42}
    assert result.is_error is False


@pytest.mark.asyncio
async def test_registry_raises_on_unknown_tool(tmp_path):
    """Executing an unknown tool should raise ToolError with TOOL_NOT_FOUND."""
    registry = ToolRegistry()
    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-1", name="nonexistent_tool", args={})

    with pytest.raises(ToolError) as exc_info:
        await registry.execute(ctx, call)

    assert exc_info.value.failure_class == FailureClass.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_registry_validates_args_against_schema(tmp_path):
    """Registry should raise ToolError(TOOL_SCHEMA_ERROR) for invalid args."""
    registry = ToolRegistry()

    tool = _make_tool("strict_tool")
    tool.input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    registry.register(tool)

    ctx = _make_ctx(tmp_path)
    # Missing required 'name' field
    call = ToolCall(id="call-1", name="strict_tool", args={"wrong_field": 123})

    with pytest.raises(ToolError) as exc_info:
        await registry.execute(ctx, call)

    assert exc_info.value.failure_class == FailureClass.TOOL_SCHEMA_ERROR


@pytest.mark.asyncio
async def test_registry_enforces_safety_block(tmp_path):
    """Registry should raise SafetyViolation when the safety pipeline blocks a tool call."""
    safety_pipeline = MagicMock()

    class _BlockResult:
        blocked = True
        reason = "Dangerous operation"

    async def _check_step(payload):
        return _BlockResult()

    safety_pipeline.check_step = _check_step

    registry = ToolRegistry(safety_pipeline=safety_pipeline)
    registry.register(_make_tool("dangerous_tool"))

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-1", name="dangerous_tool", args={})

    with pytest.raises(SafetyViolation):
        await registry.execute(ctx, call)


@pytest.mark.asyncio
async def test_mcp_tool_wrapper_calls_mcp_session(tmp_path):
    """MCPToolWrapper should call the MCP session's execute_tool method."""
    try:
        from harness.tools.mcp_client import MCPToolWrapper  # type: ignore
    except ImportError:
        pytest.skip("MCPToolWrapper not implemented")

    # MCPToolWrapper uses session.call_tool(name, arguments=args)
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "mcp_response"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.isError = False

    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_response)

    wrapper = MCPToolWrapper(
        name="mcp_list_tables",
        description="List tables via MCP",
        input_schema={"type": "object", "properties": {}},
        session=mock_session,
    )

    ctx = _make_ctx(tmp_path)
    result = await wrapper.execute(ctx, {"database": "test_db"})

    mock_session.call_tool.assert_called_once()
    assert not result.is_error


@pytest.mark.asyncio
async def test_sql_tool_rejects_write_in_read_only_mode(tmp_path):
    """ExecuteQueryTool in read-only mode rejects DML before touching the DB."""
    from harness.tools.sql_tools import ExecuteQueryTool, SQLConnectionConfig

    config = SQLConnectionConfig(
        connection_string="sqlite+aiosqlite:///:memory:", read_only=True
    )
    pool = MagicMock()  # must never be queried — rejection happens first
    pool.execute_query = AsyncMock()
    tool = ExecuteQueryTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"sql": "DELETE FROM users"})

    assert result.is_error is True
    assert "select" in result.error.lower() or "read-only" in result.error.lower()
    pool.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_sql_tool_adds_limit_if_missing(tmp_path):
    """ExecuteQueryTool auto-appends LIMIT to a SELECT that has none."""
    from harness.core.context import AgentContext
    from harness.tools.sql_tools import ExecuteQueryTool, SQLConnectionConfig

    config = SQLConnectionConfig(
        connection_string="sqlite+aiosqlite:///:memory:", read_only=True, max_rows=100
    )
    pool = MagicMock()
    pool.execute_query = AsyncMock(return_value=[{"id": 1, "name": "Alice"}])
    tool = ExecuteQueryTool(pool=pool, config=config)

    # memory=None so the success path skips the optional GraphRAG recording branch
    ctx = AgentContext(
        run_id=uuid.uuid4().hex, tenant_id="test", agent_type="test",
        task="test", memory=None, workspace_path=tmp_path / "ws",
    )
    result = await tool.execute(ctx, {"sql": "SELECT * FROM users"})

    assert not result.is_error
    executed_sql = pool.execute_query.call_args.args[0]
    assert "LIMIT" in executed_sql.upper()


@pytest.mark.asyncio
async def test_workspace_tool_rejects_path_traversal(tmp_path):
    """ReadFileTool rejects a path that escapes the workspace boundary."""
    from harness.tools.file_tools import ReadFileTool

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    tool = ReadFileTool()  # workspace_manager=None -> resolves against ctx.workspace_path

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"path": "../../etc/passwd"})

    assert result.is_error is True
    err = result.error.lower()
    assert "escape" in err or "boundary" in err or "outside" in err


# ---------------------------------------------------------------------------
# Tool result size cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_registry_caps_large_tool_result(tmp_path):
    """Tool outputs exceeding 8 k chars should be truncated before returning."""
    from harness.tools.registry import _TOOL_RESULT_MAX_CHARS

    large_data = "x" * (_TOOL_RESULT_MAX_CHARS + 5_000)
    registry = ToolRegistry()
    tool = _make_tool("big_tool", response=large_data)
    registry.register(tool)

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-big", name="big_tool", args={})
    result = await registry.execute(ctx, call)

    assert not result.is_error
    assert result.metadata.get("truncated") is True
    assert result.metadata.get("original_chars") == len(f'"{large_data}"')  # json.dumps adds quotes
    assert len(result.to_text()) < _TOOL_RESULT_MAX_CHARS + 200  # truncated + suffix


@pytest.mark.asyncio
async def test_registry_does_not_cap_small_tool_result(tmp_path):
    """Tool outputs under the 8 k cap should pass through unchanged."""

    small_data = {"result": "short answer"}
    registry = ToolRegistry()
    tool = _make_tool("small_tool", response=small_data)
    registry.register(tool)

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-small", name="small_tool", args={})
    result = await registry.execute(ctx, call)

    assert not result.is_error
    assert result.metadata.get("truncated") is None
    assert result.data == small_data


@pytest.mark.asyncio
async def test_registry_does_not_cap_error_result(tmp_path):
    """Error results should never be truncated."""
    registry = ToolRegistry()
    tool = _make_tool("err_tool", error="something went wrong")
    registry.register(tool)

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-err", name="err_tool", args={})
    result = await registry.execute(ctx, call)

    assert result.is_error
    assert result.metadata.get("truncated") is None


# ---------------------------------------------------------------------------
# Security regressions — SQL injection via schema argument
# ---------------------------------------------------------------------------

_PG_CONN = "postgresql+asyncpg://user:pass@localhost/db"


def _mock_pool():
    pool = MagicMock()
    pool.execute_query = AsyncMock(return_value=[])
    return pool


@pytest.mark.asyncio
async def test_list_tables_rejects_malicious_schema(tmp_path):
    """ListTablesTool must reject a schema argument containing SQL injection."""
    from harness.tools.sql_tools import ListTablesTool, SQLConnectionConfig

    config = SQLConnectionConfig(connection_string=_PG_CONN)
    pool = _mock_pool()
    tool = ListTablesTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(
        ctx, {"schema": "public' UNION SELECT password FROM users --"}
    )

    assert result.is_error is True
    assert "schema" in result.error.lower()
    pool.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_describe_table_rejects_malicious_schema(tmp_path):
    """DescribeTableTool must reject a schema argument containing SQL injection."""
    from harness.tools.sql_tools import DescribeTableTool, SQLConnectionConfig

    config = SQLConnectionConfig(connection_string=_PG_CONN)
    pool = _mock_pool()
    tool = DescribeTableTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(
        ctx, {"table_name": "users", "schema": "public'; DROP TABLE users; --"}
    )

    assert result.is_error is True
    assert "schema" in result.error.lower()
    pool.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_sample_rows_rejects_malicious_schema(tmp_path):
    """SampleRowsTool must reject a schema argument containing SQL injection."""
    from harness.tools.sql_tools import SampleRowsTool, SQLConnectionConfig

    config = SQLConnectionConfig(connection_string=_PG_CONN)
    pool = _mock_pool()
    tool = SampleRowsTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(
        ctx, {"table_name": "users", "schema": "public.users; DROP TABLE users"}
    )

    assert result.is_error is True
    assert "schema" in result.error.lower()
    pool.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_sql_tools_accept_valid_schema(tmp_path):
    """A normal identifier schema still works."""
    from harness.tools.sql_tools import ListTablesTool, SQLConnectionConfig

    config = SQLConnectionConfig(connection_string=_PG_CONN)
    pool = _mock_pool()
    tool = ListTablesTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"schema": "analytics_v2"})

    assert not result.is_error
    pool.execute_query.assert_called_once()


# ---------------------------------------------------------------------------
# Security regressions — read-only bypass via CTE / buried write keywords
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sql_tool_rejects_cte_write_bypass(tmp_path):
    """WITH ... DELETE must be rejected even though it doesn't start with DELETE."""
    from harness.tools.sql_tools import ExecuteQueryTool, SQLConnectionConfig

    config = SQLConnectionConfig(
        connection_string="sqlite+aiosqlite:///:memory:", read_only=True
    )
    pool = _mock_pool()
    tool = ExecuteQueryTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(
        ctx, {"sql": "WITH t AS (SELECT 1) DELETE FROM users"}
    )

    assert result.is_error is True
    assert "delete" in result.error.lower()
    pool.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_sql_tool_rejects_buried_pragma(tmp_path):
    """PRAGMA buried after a comment must be rejected in read-only mode."""
    from harness.tools.sql_tools import ExecuteQueryTool, SQLConnectionConfig

    config = SQLConnectionConfig(
        connection_string="sqlite+aiosqlite:///:memory:", read_only=True
    )
    pool = _mock_pool()
    tool = ExecuteQueryTool(pool=pool, config=config)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"sql": "SELECT 1; PRAGMA writable_schema=1"})

    assert result.is_error is True
    pool.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_sql_tool_allows_write_keyword_inside_string_literal(tmp_path):
    """A legitimate SELECT containing 'DELETE' inside a string literal passes."""
    import uuid as _uuid
    from harness.tools.sql_tools import ExecuteQueryTool, SQLConnectionConfig

    config = SQLConnectionConfig(
        connection_string="sqlite+aiosqlite:///:memory:", read_only=True
    )
    pool = _mock_pool()
    tool = ExecuteQueryTool(pool=pool, config=config)

    ctx = AgentContext(
        run_id=_uuid.uuid4().hex, tenant_id="test", agent_type="test",
        task="test", memory=None, workspace_path=tmp_path / "ws",
    )
    result = await tool.execute(
        ctx,
        {"sql": "SELECT * FROM logs WHERE message = 'please DELETE this -- now' LIMIT 5"},
    )

    assert not result.is_error
    pool.execute_query.assert_called_once()


def test_find_write_keyword_strips_comments():
    """Write keywords hidden in comments don't count, real ones do."""
    from harness.tools.sql_tools import _find_write_keyword

    assert _find_write_keyword("SELECT 1 -- DROP TABLE users") is None
    assert _find_write_keyword("SELECT 1 /* DELETE */ FROM t") is None
    assert _find_write_keyword("WITH t AS (SELECT 1) DELETE FROM users") == "DELETE"
    assert _find_write_keyword('SELECT "drop" FROM t') is None


# ---------------------------------------------------------------------------
# Security regressions — RunCodeTool unsandboxed fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_code_refuses_unsandboxed_fallback_by_default(tmp_path, monkeypatch):
    """With no sandbox available and no opt-in, run_python must refuse to execute."""
    from harness.tools.code_tools import RunCodeTool

    monkeypatch.delenv("HARNESS_ALLOW_UNSANDBOXED", raising=False)
    tool = RunCodeTool()  # no session, no docker, no restricted executor

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"code": "print('pwned host')"})

    assert result.is_error is True
    assert "HARNESS_ALLOW_UNSANDBOXED" in result.error
    # nothing written to the (nonexistent) workspace → no execution side effects
    assert not (tmp_path / "ws").exists()


@pytest.mark.asyncio
async def test_run_code_env_var_enables_unsandboxed_fallback(tmp_path, monkeypatch):
    """HARNESS_ALLOW_UNSANDBOXED=1 restores the subprocess fallback."""
    from harness.tools.code_tools import RunCodeTool

    monkeypatch.setenv("HARNESS_ALLOW_UNSANDBOXED", "1")
    tool = RunCodeTool()

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"code": "print('hello')"})

    assert not result.is_error
    assert result.data["exit_code"] == 0
    assert "hello" in result.data["stdout"]


# ---------------------------------------------------------------------------
# Security regressions — ApplyPatchTool path traversal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_patch_rejects_sibling_prefix_escape(tmp_path):
    """A sibling dir sharing the workspace path prefix (ws → ws_evil) is rejected."""
    from harness.tools.code_tools import ApplyPatchTool

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    evil = tmp_path / "ws_evil"
    evil.mkdir(parents=True, exist_ok=True)

    tool = ApplyPatchTool()
    ctx = _make_ctx(tmp_path)
    result = await tool.execute(
        ctx, {"path": "../ws_evil/target.txt", "patch": "--- a\n+++ b\n"}
    )

    assert result.is_error is True
    err = result.error.lower()
    assert "escape" in err or "boundary" in err
    assert not (evil / "target.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_rejects_dotdot_escape(tmp_path):
    """Plain ../ traversal out of the workspace is rejected."""
    from harness.tools.code_tools import ApplyPatchTool

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    tool = ApplyPatchTool()
    ctx = _make_ctx(tmp_path)
    result = await tool.execute(
        ctx, {"path": "../../etc/passwd", "patch": "--- a\n+++ b\n"}
    )

    assert result.is_error is True


# ---------------------------------------------------------------------------
# Security regressions — registry fails closed on safety pipeline errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_registry_fails_closed_on_safety_pipeline_error(tmp_path, monkeypatch):
    """An exception from check_step must block the tool call by default."""
    monkeypatch.delenv("HARNESS_SAFETY_FAIL_OPEN", raising=False)

    safety_pipeline = MagicMock()

    async def _check_step(payload):
        raise RuntimeError("guard model unavailable")

    safety_pipeline.check_step = _check_step

    registry = ToolRegistry(safety_pipeline=safety_pipeline)
    executed = []

    tool = _make_tool("guarded_tool")

    async def _execute(ctx, args):
        executed.append(True)
        return ToolResult(data={"status": "ok"})

    tool.execute = _execute
    registry.register(tool)

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-1", name="guarded_tool", args={})

    with pytest.raises(SafetyViolation):
        await registry.execute(ctx, call)
    assert executed == []


@pytest.mark.asyncio
async def test_registry_safety_fail_open_flag_restores_legacy(tmp_path):
    """safety_fail_open=True restores the old log-and-continue behaviour."""
    safety_pipeline = MagicMock()

    async def _check_step(payload):
        raise RuntimeError("guard model unavailable")

    safety_pipeline.check_step = _check_step

    registry = ToolRegistry(safety_pipeline=safety_pipeline, safety_fail_open=True)
    registry.register(_make_tool("guarded_tool", response={"ok": 1}))

    ctx = _make_ctx(tmp_path)
    call = ToolCall(id="call-1", name="guarded_tool", args={})
    result = await registry.execute(ctx, call)

    assert not result.is_error
    assert result.data == {"ok": 1}


# ---------------------------------------------------------------------------
# Security regressions — MCP subprocess env allowlist + transport cleanup
# ---------------------------------------------------------------------------

def test_mcp_subprocess_env_allowlist(monkeypatch):
    """Only allowlisted host env vars reach MCP server subprocesses."""
    from harness.tools.mcp_client import _build_subprocess_env

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = _build_subprocess_env({"SERVER_TOKEN": "abc"})

    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["LC_ALL"] == "en_US.UTF-8"
    assert env["SERVER_TOKEN"] == "abc"


def test_mcp_subprocess_env_inherit_opt_in(monkeypatch):
    """inherit_env=True passes the full host environment (explicit escape hatch)."""
    from harness.tools.mcp_client import _build_subprocess_env

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret")

    env = _build_subprocess_env({}, inherit_env=True)
    assert env["AWS_SECRET_ACCESS_KEY"] == "super-secret"


@pytest.mark.asyncio
async def test_mcp_adapter_disconnect_closes_transports(monkeypatch):
    """connect() must hold transport CMs so disconnect() runs their __aexit__."""
    import contextlib as _contextlib

    from harness.tools.mcp_client import MCPServerConfig, MCPToolAdapter

    closed = {"transport": False, "session": False}

    @_contextlib.asynccontextmanager
    async def fake_stdio_client(server_params):
        try:
            yield (MagicMock(name="read"), MagicMock(name="write"))
        finally:
            closed["transport"] = True

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            closed["session"] = True

        async def initialize(self):
            return None

        async def list_tools(self):
            result = MagicMock()
            result.tools = []
            return result

    import mcp
    import mcp.client.stdio

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", FakeSession)

    adapter = MCPToolAdapter(
        MCPServerConfig(name="test", transport="stdio", command=["echo", "hi"])
    )
    tools = await adapter.connect()
    assert tools == []
    assert closed == {"transport": False, "session": False}

    await adapter.disconnect()
    assert closed == {"transport": True, "session": True}


@pytest.mark.asyncio
async def test_mcp_adapter_connect_failure_unwinds_transports(monkeypatch):
    """If initialize() fails, the transport CM must still be exited."""
    import contextlib as _contextlib

    from harness.core.errors import HarnessError
    from harness.tools.mcp_client import MCPServerConfig, MCPToolAdapter

    closed = {"transport": False}

    @_contextlib.asynccontextmanager
    async def fake_stdio_client(server_params):
        try:
            yield (MagicMock(name="read"), MagicMock(name="write"))
        finally:
            closed["transport"] = True

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def initialize(self):
            raise RuntimeError("handshake failed")

    import mcp
    import mcp.client.stdio

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", FakeSession)

    adapter = MCPToolAdapter(
        MCPServerConfig(name="test", transport="stdio", command=["echo", "hi"])
    )
    with pytest.raises(HarnessError):
        await adapter.connect()

    assert closed["transport"] is True
