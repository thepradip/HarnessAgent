"""MCP **server** for the HarnessAgent harness.

Wires :class:`harness.mcp.bridge.HarnessBridge` (pure, SDK-free logic) to the
official ``mcp`` Python SDK's low-level :class:`mcp.server.Server`.

The ``mcp`` import is guarded exactly like :mod:`harness.tools.mcp_client`: it
is imported lazily inside the functions that actually need it, so importing
``harness`` (or even ``harness.mcp``) never requires the optional ``mcp`` SDK.
A clear, actionable error is raised only when the server is started without it.
"""

from __future__ import annotations

import logging
from typing import Any

from harness.mcp.bridge import HarnessBridge
from harness.mcp.config import MCPServerSettings

logger = logging.getLogger(__name__)

_MCP_MISSING_MSG = (
    "The 'mcp' package is required to run the harness MCP server. "
    "Install it with:  pip install 'agent-haas[mcp]'  (or  pip install mcp)."
)


def _require_mcp() -> tuple[Any, Any, Any]:
    """Import the ``mcp`` SDK pieces we need, raising a clear error if absent.

    Returns ``(Server, stdio_server, mcp_types)``.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        import mcp.types as mcp_types
    except ImportError as exc:  # pragma: no cover - exercised only without mcp
        raise RuntimeError(_MCP_MISSING_MSG) from exc
    return Server, stdio_server, mcp_types


def _to_content_blocks(payload: dict[str, Any], mcp_types: Any) -> list[Any]:
    """Turn a bridge payload into a list of MCP content blocks.

    The structured ``data`` is attached on the TextContent's metadata where the
    SDK supports it; the human-readable ``text`` is always present.
    """
    text = payload.get("text") or ""
    return [mcp_types.TextContent(type="text", text=text)]


def build_server(settings: MCPServerSettings | None = None) -> Any:
    """Build and return a configured ``mcp.server.Server`` for the harness.

    Registers handlers for ``list_tools`` / ``call_tool`` and (when enabled)
    ``list_resources`` / ``read_resource``. Requires the ``mcp`` SDK.
    """
    settings = settings or MCPServerSettings.from_env()
    Server, _stdio_server, mcp_types = _require_mcp()

    bridge = HarnessBridge(settings)
    server = Server(settings.server_name)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [mcp_types.Tool(**schema) for schema in bridge.list_tool_schemas()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        payload = await bridge.call_tool(name, arguments or {})
        blocks = _to_content_blocks(payload, mcp_types)
        if payload.get("is_error"):
            # Surface the error to the host. Newer SDKs accept the isError flag
            # via the returned content; raising keeps behaviour explicit and
            # portable across SDK versions.
            raise _ToolCallError(payload.get("text") or "tool call failed", blocks)
        return blocks

    if settings.expose_resources:

        @server.list_resources()
        async def _list_resources() -> list[Any]:
            return [
                mcp_types.Resource(**desc)
                for desc in bridge.list_resource_descriptors()
            ]

        @server.read_resource()
        async def _read_resource(uri: Any) -> str:
            return await bridge.read_resource(str(uri))

    # Stash the bridge so callers/tests can introspect without rebuilding.
    server._harness_bridge = bridge  # type: ignore[attr-defined]
    return server


class _ToolCallError(Exception):
    """Internal: carries content blocks for an errored MCP tool call."""

    def __init__(self, message: str, content: list[Any]) -> None:
        super().__init__(message)
        self.content = content


def run_stdio(settings: MCPServerSettings | None = None) -> None:
    """Run the harness MCP server over stdio (blocking until the host closes)."""
    import anyio

    settings = settings or MCPServerSettings.from_env()
    Server, stdio_server, _mcp_types = _require_mcp()
    server = build_server(settings)

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    logger.info("Starting harness MCP server '%s' over stdio", settings.server_name)
    anyio.run(_serve)


def run_http(settings: MCPServerSettings | None = None) -> None:
    """Streamable-HTTP transport — deferred.

    The low-level streamable-HTTP transport requires an ASGI host (Starlette /
    uvicorn) and session management that is out of scope for this first cut.
    See the module docstring / README follow-up note.
    """
    raise NotImplementedError(
        "HTTP transport for the harness MCP server is not implemented yet; "
        "use the default stdio transport. Track this as a follow-up."
    )


def run(settings: MCPServerSettings | None = None) -> None:
    """Run the server using the transport selected in settings."""
    settings = settings or MCPServerSettings.from_env()
    if settings.transport == "http":
        run_http(settings)
    else:
        run_stdio(settings)
