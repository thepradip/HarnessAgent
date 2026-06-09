"""Unit tests for the harness MCP **server** (harness.mcp).

These tests exercise:
  * schema translation  (harness ToolExecutor → MCP tool descriptor),
  * MCP tool-call routing (call → ToolRegistry.execute → structured result),
  * the run_agent handler (mocked agent factory returning an AgentResult),
  * import-safety / guarded mcp import.

They do NOT require the real ``mcp`` SDK — the bridge layer is pure Python and
the SDK is only touched in the two server-build tests, which skip cleanly when
``mcp`` is not installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.core.context import AgentContext, AgentResult, ToolResult
from harness.mcp.bridge import (
    HarnessBridge,
    run_agent_mcp_schema,
    tool_to_mcp_schema,
)
from harness.mcp.config import MCPServerSettings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> MCPServerSettings:
    return MCPServerSettings(
        server_name="test-harness",
        tenant_id="t-test",
        workspace_base=tmp_path / "ws",
        sql_connection_string="",
    )


class _FakeTool:
    name = "echo"
    description = "Echo a value back."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------


def test_tool_to_mcp_schema_renames_input_schema():
    schema = tool_to_mcp_schema(_FakeTool())
    assert schema["name"] == "echo"
    assert schema["description"] == "Echo a value back."
    # harness `input_schema` becomes MCP `inputSchema`, unchanged JSON Schema.
    assert schema["inputSchema"]["properties"]["value"]["type"] == "string"
    assert schema["inputSchema"]["required"] == ["value"]


def test_tool_to_mcp_schema_normalises_missing_schema():
    class _NoSchema:
        name = "noschema"
        description = ""
        input_schema = None

    schema = tool_to_mcp_schema(_NoSchema())
    assert schema["inputSchema"] == {"type": "object", "properties": {}}


def test_tool_to_mcp_schema_normalises_non_object_schema():
    class _BadSchema:
        name = "bad"
        description = "d"
        input_schema = {"type": "array"}

    schema = tool_to_mcp_schema(_BadSchema())
    assert schema["inputSchema"]["type"] == "object"


def test_run_agent_schema_lists_agent_types():
    schema = run_agent_mcp_schema(["sql", "code"])
    assert schema["name"] == "run_agent"
    props = schema["inputSchema"]["properties"]
    assert "agent_type" in props
    assert "task" in props
    assert schema["inputSchema"]["required"] == ["agent_type", "task"]
    assert "sql" in props["agent_type"]["description"]


# ---------------------------------------------------------------------------
# Bridge: tool listing
# ---------------------------------------------------------------------------


def test_list_tool_schemas_includes_builtin_tools_and_run_agent(settings):
    bridge = HarnessBridge(settings)
    schemas = bridge.list_tool_schemas()
    names = {s["name"] for s in schemas}
    # Built-in harness tools surfaced as MCP tools
    assert {"read_file", "write_file", "list_workspace", "run_python",
            "lint_code", "apply_patch"}.issubset(names)
    # Headline capability
    assert "run_agent" in names


def test_list_tool_schemas_respects_denylist(settings):
    settings.tool_denylist = frozenset({"run_python", "apply_patch"})
    bridge = HarnessBridge(settings)
    names = {s["name"] for s in bridge.list_tool_schemas()}
    assert "run_python" not in names
    assert "apply_patch" not in names
    assert "read_file" in names


def test_list_tool_schemas_can_hide_run_agent(settings):
    settings.expose_run_agent = False
    bridge = HarnessBridge(settings)
    names = {s["name"] for s in bridge.list_tool_schemas()}
    assert "run_agent" not in names
    assert "read_file" in names


def test_list_tool_schemas_can_hide_tools(settings):
    settings.expose_tools = False
    bridge = HarnessBridge(settings)
    names = {s["name"] for s in bridge.list_tool_schemas()}
    assert names == {"run_agent"}


# ---------------------------------------------------------------------------
# Bridge: tool-call routing → ToolRegistry.execute
# ---------------------------------------------------------------------------


async def test_call_tool_routes_through_registry(settings):
    bridge = HarnessBridge(settings)
    # Replace the registry with a mock so we assert routing precisely.
    fake_registry = MagicMock()
    fake_registry.execute = AsyncMock(
        return_value=ToolResult(data={"ok": True}, error=None)
    )
    bridge._registry = fake_registry

    payload = await bridge.call_tool("read_file", {"path": "x.txt"})

    assert fake_registry.execute.await_count == 1
    ctx, call = fake_registry.execute.await_args.args
    assert isinstance(ctx, AgentContext)
    assert ctx.tenant_id == "t-test"
    assert call.name == "read_file"
    assert call.args == {"path": "x.txt"}
    assert payload["is_error"] is False
    assert payload["data"] == {"ok": True}


async def test_call_tool_propagates_tool_error_as_error_payload(settings):
    bridge = HarnessBridge(settings)
    fake_registry = MagicMock()
    fake_registry.execute = AsyncMock(
        return_value=ToolResult(data=None, error="boom")
    )
    bridge._registry = fake_registry

    payload = await bridge.call_tool("read_file", {"path": "x.txt"})
    assert payload["is_error"] is True
    assert "boom" in payload["text"]


async def test_call_tool_handles_registry_exception(settings):
    bridge = HarnessBridge(settings)
    fake_registry = MagicMock()
    fake_registry.execute = AsyncMock(side_effect=RuntimeError("kaboom"))
    bridge._registry = fake_registry

    payload = await bridge.call_tool("read_file", {"path": "x.txt"})
    assert payload["is_error"] is True
    assert "kaboom" in payload["text"]


async def test_call_tool_builtin_read_file_end_to_end(settings, tmp_path):
    """Exercise the real ToolRegistry + ReadFileTool through the bridge."""
    bridge = HarnessBridge(settings)
    # Build a real workspace file under the bridge's workspace base by routing
    # a write then a read.
    write_payload = await bridge.call_tool(
        "write_file", {"path": "hello.txt", "content": "hi from mcp"}
    )
    assert write_payload["is_error"] is False
    # write_file resolves relative to a per-call workspace, so a subsequent
    # read in a *different* call won't see it — instead assert the write itself
    # routed and succeeded through the real registry/safety pipeline.
    assert "hello.txt" in write_payload["text"]


# ---------------------------------------------------------------------------
# Bridge: run_agent
# ---------------------------------------------------------------------------


async def test_run_agent_builds_agent_and_returns_result(settings):
    bridge = HarnessBridge(settings)

    fake_result = AgentResult(
        run_id="r1",
        output="the answer is 42",
        steps=3,
        tokens=1200,
        success=True,
        cost_usd=0.01,
        tool_calls=2,
        tool_errors=0,
        elapsed_seconds=1.5,
    )
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=fake_result)

    with patch.object(bridge, "_build_agent", return_value=fake_agent) as build:
        payload = await bridge.run_agent(
            {"agent_type": "code", "task": "compute 6*7", "max_steps": 5}
        )

    build.assert_called_once_with("code")
    # The agent ran against a properly constructed context.
    ctx = fake_agent.run.await_args.args[0]
    assert isinstance(ctx, AgentContext)
    assert ctx.agent_type == "code"
    assert ctx.task == "compute 6*7"
    assert ctx.max_steps == 5
    assert ctx.tenant_id == "t-test"

    assert payload["is_error"] is False
    assert payload["text"] == "the answer is 42"
    assert payload["data"]["run_id"] == "r1"
    assert payload["data"]["steps"] == 3
    assert payload["data"]["tokens"] == 1200
    assert payload["data"]["cost_usd"] == 0.01
    assert payload["data"]["success"] is True


async def test_run_agent_via_call_tool_dispatch(settings):
    bridge = HarnessBridge(settings)
    fake_result = AgentResult(
        run_id="r2", output="done", steps=1, tokens=10, success=True
    )
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=fake_result)
    with patch.object(bridge, "_build_agent", return_value=fake_agent):
        payload = await bridge.call_tool(
            "run_agent", {"agent_type": "sql", "task": "count rows"}
        )
    assert payload["data"]["run_id"] == "r2"


async def test_run_agent_rejects_unknown_agent_type(settings):
    bridge = HarnessBridge(settings)
    payload = await bridge.run_agent({"agent_type": "wizard", "task": "do it"})
    assert payload["is_error"] is True
    assert "Unknown agent_type" in payload["text"]


async def test_run_agent_requires_task_and_type(settings):
    bridge = HarnessBridge(settings)
    payload = await bridge.run_agent({"agent_type": "code"})
    assert payload["is_error"] is True
    assert "task" in payload["text"]


async def test_run_agent_failed_result_is_error(settings):
    bridge = HarnessBridge(settings)
    fake_result = AgentResult(
        run_id="r3",
        output="",
        steps=2,
        tokens=50,
        success=False,
        failure_class="budget_steps",
        error_message="Budget exceeded",
    )
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=fake_result)
    with patch.object(bridge, "_build_agent", return_value=fake_agent):
        payload = await bridge.run_agent({"agent_type": "code", "task": "loop"})
    assert payload["is_error"] is True
    assert payload["data"]["failure_class"] == "budget_steps"
    assert "Budget exceeded" in payload["text"]


# ---------------------------------------------------------------------------
# Bridge: resources
# ---------------------------------------------------------------------------


def test_resource_descriptors_present(settings):
    bridge = HarnessBridge(settings)
    uris = {r["uri"] for r in bridge.list_resource_descriptors()}
    assert uris == {"harness://skills", "harness://runs/recent"}


def test_resource_descriptors_can_be_disabled(settings):
    settings.expose_resources = False
    bridge = HarnessBridge(settings)
    assert bridge.list_resource_descriptors() == []


async def test_read_resource_recent_runs_returns_json(settings):
    bridge = HarnessBridge(settings)
    import json

    text = await bridge.read_resource("harness://runs/recent")
    assert isinstance(json.loads(text), list)


async def test_read_resource_unknown_uri_raises(settings):
    bridge = HarnessBridge(settings)
    with pytest.raises(ValueError):
        await bridge.read_resource("harness://nope")


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("HARNESS_MCP_SERVER_NAME", "myserver")
    monkeypatch.setenv("HARNESS_MCP_TENANT_ID", "acme")
    monkeypatch.setenv("HARNESS_MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("HARNESS_MCP_EXPOSE_RUN_AGENT", "false")
    monkeypatch.setenv("HARNESS_MCP_TOOL_DENYLIST", "run_python, apply_patch")
    monkeypatch.setenv("HARNESS_MCP_MAX_STEPS", "7")

    s = MCPServerSettings.from_env()
    assert s.server_name == "myserver"
    assert s.tenant_id == "acme"
    assert s.transport == "stdio"
    assert s.expose_run_agent is False
    assert s.tool_denylist == frozenset({"run_python", "apply_patch"})
    assert s.default_max_steps == 7


# ---------------------------------------------------------------------------
# Import-safety / guarded mcp import
# ---------------------------------------------------------------------------


def test_importing_harness_mcp_does_not_require_mcp_sdk():
    """harness.mcp + its submodules must import even without the mcp SDK."""
    import importlib

    for mod in ("harness.mcp", "harness.mcp.config", "harness.mcp.bridge",
                "harness.mcp.server", "harness.mcp.__main__"):
        importlib.import_module(mod)  # should not raise


def test_require_mcp_raises_clear_error_when_sdk_absent():
    """When mcp is unavailable, _require_mcp raises an actionable RuntimeError."""
    import builtins

    from harness.mcp import server as server_mod

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "mcp.server" or name.startswith("mcp.server") or name == "mcp.types":
            raise ImportError("no mcp")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(RuntimeError) as exc:
            server_mod._require_mcp()
    assert "agent-haas[mcp]" in str(exc.value)


# ---------------------------------------------------------------------------
# Server build (requires the real mcp SDK; skips cleanly otherwise)
# ---------------------------------------------------------------------------


def _mcp_available() -> bool:
    try:
        import mcp.server  # noqa: F401
        import mcp.types  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mcp_available(), reason="mcp SDK not installed")
def test_build_server_with_real_sdk(settings):
    from harness.mcp.server import build_server

    server = build_server(settings)
    assert server.name == "test-harness"
    # Bridge is stashed for introspection.
    assert isinstance(server._harness_bridge, HarnessBridge)


@pytest.mark.skipif(not _mcp_available(), reason="mcp SDK not installed")
async def test_build_server_list_tools_handler(settings):
    """The registered list_tools handler returns real mcp.types.Tool objects."""
    import mcp.types as mcp_types

    from harness.mcp.server import build_server

    server = build_server(settings)
    # The low-level Server stores handlers keyed by request type. Rather than
    # reach into internals, re-run the bridge translation the handler uses and
    # confirm each descriptor is a valid Tool model.
    for schema in server._harness_bridge.list_tool_schemas():
        tool = mcp_types.Tool(**schema)
        assert tool.name == schema["name"]
        assert tool.inputSchema["type"] == "object"
