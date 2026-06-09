"""Entry point for ``python -m harness.mcp`` and the ``harness-mcp`` script.

Launches the harness MCP server. Transport defaults to stdio and is
configurable via ``HARNESS_MCP_TRANSPORT`` (``stdio`` | ``http``).
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse args / env and start the harness MCP server."""
    parser = argparse.ArgumentParser(
        prog="harness-mcp",
        description="Run the HarnessAgent harness as an MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="Transport to serve on (default: stdio, or $HARNESS_MCP_TRANSPORT).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="MCP server name advertised to hosts (default: 'harness').",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    # Logs must go to stderr so they never corrupt the stdio JSON-RPC stream.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from harness.mcp.config import MCPServerSettings
    from harness.mcp.server import run

    settings = MCPServerSettings.from_env()
    if args.transport:
        settings.transport = args.transport
    if args.name:
        settings.server_name = args.name

    run(settings)


if __name__ == "__main__":
    main()
