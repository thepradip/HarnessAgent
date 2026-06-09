"""Pure bridge logic between the harness and MCP — no ``mcp`` SDK import.

Everything here is unit-testable without the ``mcp`` package installed:

* :func:`tool_to_mcp_schema` — harness ToolExecutor → MCP tool descriptor.
* :class:`HarnessBridge` — builds a ToolRegistry, constructs an AgentContext
  per call, routes MCP tool calls through ``ToolRegistry.execute``, and runs a
  whole agent via the harness ``_build_agent`` factory for ``run_agent``.

The bridge returns plain Python structures (dicts / strings). The thin
:mod:`harness.mcp.server` layer wraps those in ``mcp.types`` content blocks.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from harness.core.context import AgentContext, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------

_RUN_AGENT_TOOL_NAME = "run_agent"


def tool_to_mcp_schema(tool: Any) -> dict[str, Any]:
    """Translate a harness ToolExecutor into an MCP tool descriptor dict.

    The MCP ``Tool`` model wants ``name`` / ``description`` / ``inputSchema``.
    Harness tools expose ``name`` / ``description`` / ``input_schema`` (a JSON
    Schema), so this is largely a rename, but we normalise a missing/empty
    schema to a valid empty object schema (MCP requires an object schema).
    """
    schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
    # MCP requires the top-level schema to be an object type.
    if not isinstance(schema, dict) or schema.get("type") != "object":
        schema = {"type": "object", "properties": {}}
    return {
        "name": tool.name,
        "description": getattr(tool, "description", "") or "",
        "inputSchema": schema,
    }


def run_agent_mcp_schema(supported_agent_types: list[str]) -> dict[str, Any]:
    """MCP tool descriptor for the headline ``run_agent`` capability."""
    return {
        "name": _RUN_AGENT_TOOL_NAME,
        "description": (
            "Delegate a complete task to a HarnessAgent agent. The agent runs "
            "its full tool-using loop (LLM + tools + safety + verification) and "
            "returns the final result. Use this to hand off an entire task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "description": (
                        "Which harness agent to run. "
                        f"Supported: {', '.join(supported_agent_types)}."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": "Natural-language task for the agent to complete.",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Optional cap on agent loop iterations.",
                    "minimum": 1,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional token budget for the run.",
                    "minimum": 1,
                },
            },
            "required": ["agent_type", "task"],
        },
    }


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class HarnessBridge:
    """Owns the harness wiring used to serve MCP requests.

    A single bridge is created per server process. It lazily builds a
    ToolRegistry (with the same safety pipeline the harness uses) and constructs
    a fresh AgentContext for every MCP tool call so each call is workspace- and
    run-scoped.
    """

    #: Agent types that ``run_agent`` will accept.
    SUPPORTED_AGENT_TYPES = ("sql", "code")

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._registry: Any = None
        self._workspace_base: Path | None = None

    # -- workspace ------------------------------------------------------

    @property
    def workspace_base(self) -> Path:
        """Base directory under which per-call workspaces are created."""
        if self._workspace_base is None:
            base = getattr(self._settings, "workspace_base", None)
            if base is not None:
                self._workspace_base = Path(base)
            else:
                self._workspace_base = Path(
                    tempfile.mkdtemp(prefix="harness-mcp-")
                )
            self._workspace_base.mkdir(parents=True, exist_ok=True)
        return self._workspace_base

    def _new_workspace(self, run_id: str) -> Path:
        ws = self.workspace_base / run_id
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def _make_context(
        self,
        *,
        agent_type: str,
        task: str,
        max_steps: int | None = None,
        max_tokens: int | None = None,
    ) -> AgentContext:
        run_id = uuid.uuid4().hex
        return AgentContext.create(
            tenant_id=getattr(self._settings, "tenant_id", "mcp"),
            agent_type=agent_type,
            task=task,
            memory=None,
            workspace_path=self._new_workspace(run_id),
            max_steps=max_steps or getattr(self._settings, "default_max_steps", 30),
            max_tokens=max_tokens or getattr(self._settings, "default_max_tokens", 100_000),
        )

    # -- tool registry --------------------------------------------------

    def _build_registry(self) -> Any:
        """Build a ToolRegistry with the built-in harness tools + safety.

        Reuses the harness safety pipeline so MCP tool calls go through the same
        validation / safety / audit pipeline as in-process agent runs.
        """
        from harness.safety.pipeline_factory import build_pipeline, get_default_config
        from harness.tools.code_tools import ApplyPatchTool, LintCodeTool, RunCodeTool
        from harness.tools.file_tools import (
            ListWorkspaceTool,
            ReadFileTool,
            WriteFileTool,
        )
        from harness.tools.registry import ToolRegistry

        # "code" is the most permissive default policy (allows code exec + writes).
        safety = build_pipeline("code", get_default_config("code"))
        registry = ToolRegistry(safety_pipeline=safety)

        for tool in (
            ReadFileTool(),
            WriteFileTool(),
            ListWorkspaceTool(),
            RunCodeTool(),
            LintCodeTool(),
            ApplyPatchTool(),
        ):
            registry.register(tool)

        sql = getattr(self._settings, "sql_connection_string", "") or ""
        if sql:
            try:
                from harness.tools.sql_tools import (
                    SQLConnectionConfig,
                    build_sql_tools,
                )

                for tool in build_sql_tools(SQLConnectionConfig(connection_string=sql)):
                    registry.register(tool)
            except Exception as exc:  # pragma: no cover - depends on [sql] extra
                logger.warning("Failed to register SQL tools for MCP server: %s", exc)

        return registry

    @property
    def registry(self) -> Any:
        if self._registry is None:
            self._registry = self._build_registry()
        return self._registry

    # -- public surface -------------------------------------------------

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Return MCP tool descriptors for everything this server exposes."""
        schemas: list[dict[str, Any]] = []
        deny = getattr(self._settings, "tool_denylist", frozenset())

        if getattr(self._settings, "expose_tools", True):
            for tool in self.registry.list_tools():
                if tool.name in deny:
                    continue
                schemas.append(tool_to_mcp_schema(tool))

        if getattr(self._settings, "expose_run_agent", True):
            schemas.append(run_agent_mcp_schema(list(self.SUPPORTED_AGENT_TYPES)))

        return schemas

    def is_run_agent(self, name: str) -> bool:
        return name == _RUN_AGENT_TOOL_NAME

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Route an MCP tool call.

        Returns a structured dict ``{"text": str, "is_error": bool, "data": Any}``
        that the server layer turns into MCP content blocks.
        """
        if self.is_run_agent(name):
            return await self.run_agent(arguments or {})
        return await self._call_harness_tool(name, arguments or {})

    async def _call_harness_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        ctx = self._make_context(agent_type="tool", task=f"mcp tool call: {name}")
        call = ToolCall(id=uuid.uuid4().hex, name=name, args=arguments)
        try:
            result = await self.registry.execute(ctx, call)
        except Exception as exc:
            logger.warning("MCP tool '%s' failed: %s", name, exc)
            return {"text": f"Tool '{name}' failed: {exc}", "is_error": True, "data": None}

        return {
            "text": result.to_text(),
            "is_error": result.is_error,
            "data": result.data,
        }

    async def run_agent(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Build and run a harness agent for the given task.

        Returns a structured dict with the AgentResult fields plus a ``text``
        summary, and ``is_error`` reflecting agent success.
        """
        agent_type = arguments.get("agent_type")
        task = arguments.get("task")
        if not agent_type or not task:
            return {
                "text": "run_agent requires 'agent_type' and 'task'.",
                "is_error": True,
                "data": None,
            }
        if agent_type not in self.SUPPORTED_AGENT_TYPES:
            return {
                "text": (
                    f"Unknown agent_type {agent_type!r}. "
                    f"Supported: {', '.join(self.SUPPORTED_AGENT_TYPES)}."
                ),
                "is_error": True,
                "data": None,
            }

        agent = self._build_agent(agent_type)
        ctx = self._make_context(
            agent_type=agent_type,
            task=task,
            max_steps=arguments.get("max_steps"),
            max_tokens=arguments.get("max_tokens"),
        )
        result = await agent.run(ctx)
        return self._agent_result_payload(result)

    def _build_agent(self, agent_type: str) -> Any:
        """Build an agent via the harness factory.

        Reuses ``harness.workers.agent_worker._build_agent`` so the MCP server
        gets exactly the same agent wiring as the RQ worker / API server.
        """
        from harness.core.config import get_config
        from harness.workers.agent_worker import _build_agent

        return _build_agent(agent_type, get_config(), redis_client=None)

    @staticmethod
    def _agent_result_payload(result: Any) -> dict[str, Any]:
        data = {
            "run_id": result.run_id,
            "output": result.output,
            "steps": result.steps,
            "tokens": result.tokens,
            "cost_usd": result.cost_usd,
            "success": result.success,
            "failure_class": result.failure_class,
            "error_message": result.error_message,
            "tool_calls": result.tool_calls,
            "tool_errors": result.tool_errors,
            "elapsed_seconds": result.elapsed_seconds,
        }
        return {
            "text": result.output or (result.error_message or ""),
            "is_error": not result.success,
            "data": data,
        }

    # -- resources ------------------------------------------------------

    def list_resource_descriptors(self) -> list[dict[str, Any]]:
        """Cheap, static MCP resource descriptors (no I/O).

        Two resources are advertised:
          * ``harness://skills`` — the captured skill library (best-effort).
          * ``harness://runs/recent`` — recent run log summaries (best-effort).
        """
        if not getattr(self._settings, "expose_resources", True):
            return []
        return [
            {
                "uri": "harness://skills",
                "name": "Harness skills",
                "description": "Reusable skills captured by the harness skill store.",
                "mimeType": "application/json",
            },
            {
                "uri": "harness://runs/recent",
                "name": "Recent harness runs",
                "description": "Summaries of recent local agent runs.",
                "mimeType": "application/json",
            },
        ]

    async def read_resource(self, uri: str) -> str:
        """Return a JSON string for a known resource URI (best-effort)."""
        import json

        if uri == "harness://runs/recent":
            return json.dumps(self._recent_runs(), default=str)
        if uri == "harness://skills":
            return json.dumps(await self._skills(), default=str)
        raise ValueError(f"Unknown resource: {uri}")

    @staticmethod
    def _recent_runs(limit: int = 20) -> list[dict[str, Any]]:
        """Read the tail of local per-run JSONL logs (best-effort, never raises)."""
        import json

        runs_dir = Path("logs") / "runs"
        out: list[dict[str, Any]] = []
        if not runs_dir.exists():
            return out
        try:
            files = sorted(
                runs_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:limit]
            for f in files:
                summary: dict[str, Any] = {"run_id": f.stem}
                try:
                    lines = f.read_text(encoding="utf-8").splitlines()
                    for line in reversed(lines):
                        entry = json.loads(line)
                        if entry.get("event") == "run_finished":
                            summary.update(
                                {
                                    "success": entry.get("success"),
                                    "agent_type": entry.get("agent_type"),
                                    "steps": entry.get("total_steps"),
                                    "cost_usd": entry.get("total_cost_usd"),
                                }
                            )
                            break
                except Exception:
                    pass
                out.append(summary)
        except Exception as exc:  # pragma: no cover - filesystem edge
            logger.debug("recent_runs read failed: %s", exc)
        return out

    async def _skills(self) -> list[dict[str, Any]]:  # pragma: no cover - optional store
        """Best-effort skill listing — empty when no skill store is configured."""
        return []
