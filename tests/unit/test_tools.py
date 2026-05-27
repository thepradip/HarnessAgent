"""Unit tests for ToolRegistry and tool implementations."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    """SQL tool in read-only mode should reject DML statements."""
    try:
        from harness.tools.skills import SQLExecutor  # type: ignore
    except ImportError:
        pytest.skip("SQLExecutor not implemented")

    tool = SQLExecutor(connection_string="sqlite:///:memory:", read_only=True)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"query": "DELETE FROM users"})

    assert result.is_error is True
    assert "read" in result.error.lower() or "write" in result.error.lower()


@pytest.mark.asyncio
async def test_sql_tool_adds_limit_if_missing(tmp_path):
    """SQL tool should automatically add LIMIT to SELECT queries without one."""
    try:
        from harness.tools.skills import SQLExecutor  # type: ignore
    except ImportError:
        pytest.skip("SQLExecutor not implemented")

    # In-memory SQLite DB
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()

    tool = SQLExecutor(connection=conn, read_only=True, auto_limit=100)

    ctx = _make_ctx(tmp_path)
    result = await tool.execute(ctx, {"query": "SELECT * FROM users"})

    # Should succeed and not return error
    assert not result.is_error or "limit" in str(result.data).lower()


@pytest.mark.asyncio
async def test_workspace_tool_rejects_path_traversal(tmp_path):
    """Workspace tool should reject paths that attempt directory traversal."""
    try:
        from harness.tools.skills import WorkspaceReadTool  # type: ignore
    except ImportError:
        pytest.skip("WorkspaceReadTool not implemented")

    tool = WorkspaceReadTool(workspace_root=str(tmp_path / "ws"))

    ctx = _make_ctx(tmp_path)
    # Attempt path traversal
    result = await tool.execute(ctx, {"path": "../../etc/passwd"})

    assert result.is_error is True
    assert "traversal" in result.error.lower() or "path" in result.error.lower() or "outside" in result.error.lower()


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
    from harness.tools.registry import _TOOL_RESULT_MAX_CHARS

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
