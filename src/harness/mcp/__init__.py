"""harness.mcp — expose the HarnessAgent harness as an MCP **server**.

This package lets MCP hosts (Claude Code, IDEs, other agents) consume the
harness: the built-in harness tools are surfaced as MCP tools and a whole
agent run can be delegated through the ``run_agent`` MCP tool.

Importing this package is always safe — it pulls in no optional dependency.
The ``mcp`` SDK is imported lazily inside :mod:`harness.mcp.server` and only
when the server is actually started, mirroring the guarded-import discipline
of :mod:`harness.tools.mcp_client`.
"""

from __future__ import annotations

from harness.mcp.config import MCPServerSettings

__all__ = ["MCPServerSettings", "build_server", "run_stdio", "main"]


def build_server(settings: "MCPServerSettings | None" = None):  # noqa: F821
    """Lazily build and return the harness MCP ``Server`` instance.

    Imported lazily so ``import harness.mcp`` never requires the ``mcp`` SDK.
    """
    from harness.mcp.server import build_server as _build

    return _build(settings)


def run_stdio(settings: "MCPServerSettings | None" = None) -> None:  # noqa: F821
    """Run the harness MCP server over stdio (blocking)."""
    from harness.mcp.server import run_stdio as _run

    _run(settings)


def main() -> None:
    """Console-script / ``python -m harness.mcp`` entry point."""
    from harness.mcp.__main__ import main as _main

    _main()
