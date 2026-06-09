"""MCPToolAdapter: connect to any MCP server and expose its tools as ToolExecutors."""

from __future__ import annotations

import contextlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from harness.core.context import AgentContext, ToolResult
from harness.core.errors import FailureClass, HarnessError, ToolError

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Host env vars safe to pass to MCP server subprocesses. Everything else
# (API keys, cloud credentials, tokens) is withheld unless the server config
# explicitly sets it in `env` or opts in via `inherit_env: true`.
_SAFE_ENV_VARS = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "TZ", "TMPDIR"}
)


def _build_subprocess_env(
    config_env: dict[str, str], *, inherit_env: bool = False
) -> dict[str, str]:
    """Build the environment for an MCP server subprocess.

    By default only a minimal allowlist (PATH, HOME, USER, LANG, LC_*, TMPDIR,
    …) is inherited from the host, merged with the server's configured env.
    Set ``inherit_env=True`` in the server config to pass the full host
    environment (explicit opt-in escape hatch).
    """
    if inherit_env:
        base = dict(os.environ)
    else:
        base = {
            k: v
            for k, v in os.environ.items()
            if k in _SAFE_ENV_VARS or k.startswith("LC_")
        }
    return {**base, **config_env}


def _interpolate_env(value: str) -> str:
    """Replace ${ENV_VAR} placeholders with environment variable values."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return _ENV_VAR_RE.sub(replacer, value)


def _interpolate_dict(d: Any) -> Any:
    """Recursively interpolate environment variables in a nested dict/list/str."""
    if isinstance(d, str):
        return _interpolate_env(d)
    if isinstance(d, dict):
        return {k: _interpolate_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_interpolate_dict(item) for item in d]
    return d


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    name: str
    transport: Literal["stdio", "sse"]
    command: list[str] | None = None  # for stdio transport
    url: str | None = None            # for sse transport
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    inherit_env: bool = False  # opt-in: pass full host env to the subprocess


class MCPToolWrapper:
    """Wraps a single MCP tool as a ToolExecutor.

    Implements the ToolExecutor protocol so it can be registered in ToolRegistry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        session: Any,  # mcp.ClientSession
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.timeout_seconds = timeout_seconds
        self._session = session

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Call the MCP tool and return a ToolResult."""
        try:
            import asyncio
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._session.call_tool(self.name, arguments=args)
            # MCP returns a list of content blocks
            content_parts: list[str] = []
            is_error = False
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    content_parts.append(block.text)
                elif block_type == "error":
                    is_error = True
                    content_parts.append(getattr(block, "text", str(block)))
                else:
                    content_parts.append(str(block))

            text_result = "\n".join(content_parts)
            if is_error or getattr(response, "isError", False):
                return ToolResult(data=None, error=text_result)
            return ToolResult(data=text_result)

        except asyncio.TimeoutError:
            msg = f"MCP tool '{self.name}' timed out after {self.timeout_seconds}s"
            logger.warning(msg)
            raise ToolError(
                msg,
                tool_name=self.name,
                failure_class=FailureClass.TOOL_TIMEOUT,
                context={"run_id": ctx.run_id},
            )
        except ToolError:
            raise
        except Exception as exc:
            msg = f"MCP tool '{self.name}' execution failed: {exc}"
            logger.exception("MCP tool '%s' raised: %s", self.name, exc)
            raise ToolError(
                msg,
                tool_name=self.name,
                failure_class=FailureClass.MCP_TOOL_ERROR,
                context={"run_id": ctx.run_id, "original_error": str(exc)},
            ) from exc


class MCPToolAdapter:
    """Connects to an MCP server and wraps its tools as ToolExecutors.

    Supports both stdio and SSE transports via the mcp package.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._session: Any = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._tools: list[MCPToolWrapper] = []

    async def connect(self) -> list[MCPToolWrapper]:
        """Establish connection to the MCP server and discover available tools.

        Returns a list of MCPToolWrapper instances, one per tool.
        Raises HarnessError(MCP_CONNECT_ERROR) on connection failure.

        Transport/session context managers are held on an AsyncExitStack so
        their __aexit__ runs in disconnect() and the server subprocess is
        reaped. Note: anyio cancel-scope rules require connect() and
        disconnect() to be called from the same task.
        """
        exit_stack = contextlib.AsyncExitStack()
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            if self._config.transport == "stdio":
                if not self._config.command:
                    raise HarnessError(
                        f"MCP server '{self._config.name}' configured as stdio "
                        "but no command was provided.",
                        failure_class=FailureClass.MCP_CONNECT_ERROR,
                    )
                env = _build_subprocess_env(
                    self._config.env, inherit_env=self._config.inherit_env
                )
                server_params = StdioServerParameters(
                    command=self._config.command[0],
                    args=self._config.command[1:],
                    env=env,
                )
                read_stream, write_stream = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self._session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

            elif self._config.transport == "sse":
                if not self._config.url:
                    raise HarnessError(
                        f"MCP server '{self._config.name}' configured as sse "
                        "but no url was provided.",
                        failure_class=FailureClass.MCP_CONNECT_ERROR,
                    )
                try:
                    from mcp.client.sse import sse_client
                except ImportError:
                    from mcp.client.http import http_client as sse_client  # type: ignore[no-redef]

                read_stream, write_stream = await exit_stack.enter_async_context(
                    sse_client(self._config.url)
                )
                self._session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
            else:
                raise HarnessError(
                    f"Unsupported transport: {self._config.transport}",
                    failure_class=FailureClass.MCP_CONNECT_ERROR,
                )

            # Initialize the MCP session
            await self._session.initialize()

            # List tools
            tools_result = await self._session.list_tools()
            self._tools = []
            for mcp_tool in tools_result.tools:
                wrapper = MCPToolWrapper(
                    name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    input_schema=mcp_tool.inputSchema or {"type": "object", "properties": {}},
                    session=self._session,
                    timeout_seconds=self._config.timeout,
                )
                self._tools.append(wrapper)

            self._exit_stack = exit_stack
            logger.info(
                "Connected to MCP server '%s', discovered %d tools",
                self._config.name,
                len(self._tools),
            )
            return self._tools

        except HarnessError:
            await self._safe_close(exit_stack)
            raise
        except Exception as exc:
            await self._safe_close(exit_stack)
            raise HarnessError(
                f"Failed to connect to MCP server '{self._config.name}': {exc}",
                failure_class=FailureClass.MCP_CONNECT_ERROR,
                context={"server_name": self._config.name, "transport": self._config.transport},
            ) from exc

    async def _safe_close(self, exit_stack: contextlib.AsyncExitStack) -> None:
        """Unwind an exit stack, swallowing (but logging) cleanup errors."""
        try:
            await exit_stack.aclose()
        except Exception as exc:
            logger.warning(
                "Error closing MCP transport for '%s': %s", self._config.name, exc
            )

    async def disconnect(self) -> None:
        """Close the MCP session and underlying transport.

        Must run in the same task that called connect() (anyio cancel scopes).
        """
        self._session = None
        exit_stack, self._exit_stack = self._exit_stack, None
        if exit_stack is not None:
            await self._safe_close(exit_stack)

    async def list_resources(self) -> list[Any]:
        """List MCP resources exposed by the server."""
        if self._session is None:
            return []
        try:
            result = await self._session.list_resources()
            return list(result.resources)
        except Exception as exc:
            logger.warning("list_resources failed for '%s': %s", self._config.name, exc)
            return []

    @property
    def tools(self) -> list[MCPToolWrapper]:
        """Return the list of wrapped tools (after connect())."""
        return self._tools


def load_mcp_servers_from_config(
    config_path: str = "configs/mcp_servers.yaml",
) -> list[MCPServerConfig]:
    """Parse a YAML file and return MCPServerConfig objects.

    Supports ${ENV_VAR} interpolation in all string values.
    """
    import pathlib

    path = pathlib.Path(config_path)
    if not path.exists():
        logger.warning("MCP config file not found: %s", config_path)
        return []

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not raw or "servers" not in raw:
        return []

    configs: list[MCPServerConfig] = []
    for entry in raw["servers"]:
        entry = _interpolate_dict(entry)
        transport = entry.get("transport", "stdio")
        cfg = MCPServerConfig(
            name=entry["name"],
            transport=transport,
            command=entry.get("command"),
            url=entry.get("url"),
            env=entry.get("env", {}),
            timeout=float(entry.get("timeout", 30.0)),
            inherit_env=bool(entry.get("inherit_env", False)),
        )
        configs.append(cfg)
        logger.debug("Loaded MCP server config: %s (%s)", cfg.name, cfg.transport)

    return configs
