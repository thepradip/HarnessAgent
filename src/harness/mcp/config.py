"""Configuration for the harness MCP server.

Kept dependency-free (plain dataclass + ``os.environ``) so the module can be
imported without pulling in pydantic-settings or the ``mcp`` SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class MCPServerSettings:
    """Settings controlling how the harness MCP server exposes the harness.

    All fields are overridable via ``HARNESS_MCP_*`` environment variables so
    the server can be configured from an MCP host's launch config without a
    file.
    """

    server_name: str = "harness"
    transport: str = "stdio"  # "stdio" (default) | "http" (deferred)

    # Tenant id stamped onto every AgentContext built by the server.
    tenant_id: str = "mcp"

    # Base directory for per-call workspaces. Defaults to a temp dir created at
    # startup when left as None.
    workspace_base: Path | None = None

    # SQL connection string — when set, the SQL tools are exposed as MCP tools.
    sql_connection_string: str = ""

    # Whether to expose the headline ``run_agent`` MCP tool.
    expose_run_agent: bool = True

    # Whether to expose the individual harness tools (read_file, run_python, …).
    expose_tools: bool = True

    # Whether to expose recent runs + skills as MCP resources (cheap, best-effort).
    expose_resources: bool = True

    # Per-tool execution defaults applied to the constructed AgentContext.
    default_max_steps: int = 30
    default_max_tokens: int = 100_000

    # HTTP transport options (only used when transport == "http").
    http_host: str = "127.0.0.1"
    http_port: int = 8765

    # Tools that should never be exposed even when expose_tools is True.
    tool_denylist: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_env(cls) -> "MCPServerSettings":
        """Build settings from ``HARNESS_MCP_*`` environment variables.

        Falls back to the core harness config for the SQL connection string and
        workspace base when those are not explicitly set for the MCP server.
        """
        sql = os.environ.get("HARNESS_MCP_SQL_CONNECTION_STRING", "")
        workspace_raw = os.environ.get("HARNESS_MCP_WORKSPACE_BASE")

        # Fall back to the harness core config (best-effort — never fatal).
        if not sql or workspace_raw is None:
            try:
                from harness.core.config import get_config

                cfg = get_config()
                if not sql:
                    sql = getattr(cfg, "sql_connection_string", "") or ""
            except Exception:
                pass

        deny = os.environ.get("HARNESS_MCP_TOOL_DENYLIST", "")
        denylist = frozenset(t.strip() for t in deny.split(",") if t.strip())

        return cls(
            server_name=os.environ.get("HARNESS_MCP_SERVER_NAME", "harness"),
            transport=os.environ.get("HARNESS_MCP_TRANSPORT", "stdio").strip().lower(),
            tenant_id=os.environ.get("HARNESS_MCP_TENANT_ID", "mcp"),
            workspace_base=Path(workspace_raw) if workspace_raw else None,
            sql_connection_string=sql,
            expose_run_agent=_env_bool("HARNESS_MCP_EXPOSE_RUN_AGENT", True),
            expose_tools=_env_bool("HARNESS_MCP_EXPOSE_TOOLS", True),
            expose_resources=_env_bool("HARNESS_MCP_EXPOSE_RESOURCES", True),
            default_max_steps=int(os.environ.get("HARNESS_MCP_MAX_STEPS", "30")),
            default_max_tokens=int(os.environ.get("HARNESS_MCP_MAX_TOKENS", "100000")),
            http_host=os.environ.get("HARNESS_MCP_HTTP_HOST", "127.0.0.1"),
            http_port=int(os.environ.get("HARNESS_MCP_HTTP_PORT", "8765")),
            tool_denylist=denylist,
        )
