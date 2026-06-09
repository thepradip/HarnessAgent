"""Base adapter contracts for framework integration with HarnessAgent."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from harness.core.context import AgentContext, StepEvent

logger = logging.getLogger(__name__)

# Mirrors tools/registry.py: a safety-pipeline ERROR (not a clean "blocked"
# decision) fails CLOSED by default — the call is treated as a violation. Set
# HARNESS_SAFETY_FAIL_OPEN=1 to restore the old fail-open behaviour.
_SAFETY_FAIL_OPEN_ENV = "HARNESS_SAFETY_FAIL_OPEN"


def _safety_fails_open() -> bool:
    """True when a safety-pipeline error should let execution proceed."""
    return os.environ.get(_SAFETY_FAIL_OPEN_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@dataclass
class FrameworkResult:
    """Normalised result produced by any framework adapter after a run completes.

    Attributes:
        framework: Canonical framework name — "langgraph" | "autogen" | "crewai".
        output:    Final text output extracted from the framework's state.
        steps:     Total number of observable steps (nodes / messages / tasks).
        metadata:  Framework-specific raw data (final state, message list, etc.).
    """

    framework: str
    output: str
    steps: int
    metadata: dict = field(default_factory=dict)


class FrameworkAdapter(ABC):
    """Wraps an external framework's execution inside the harness lifecycle.

    Concrete subclasses implement ``run()`` and ``get_result()``.

    Production harness features (safety, cost tracking, audit) are injected via
    ``attach_harness()`` and activated by calling ``run_with_harness()`` instead
    of ``run()`` directly::

        adapter = LangGraphAdapter(graph=my_graph)
        adapter.attach_harness(safety_pipeline=pipeline, cost_tracker=tracker)

        async for event in adapter.run_with_harness(ctx, input):
            ...
        result = await adapter.get_result()
    """

    framework_name: str = "unknown"

    def __init__(self) -> None:
        self._safety_pipeline: Any | None = None
        self._cost_tracker: Any | None = None
        self._audit_logger: Any | None = None
        self._mcp_clients: list[tuple[Any, list[str] | None]] = []
        # None → defer to HARNESS_SAFETY_FAIL_OPEN env var at call time.
        self._safety_fail_open: bool | None = None

    def _safety_fails_open(self) -> bool:
        """True when a safety-pipeline error should let execution proceed."""
        if getattr(self, "_safety_fail_open", None) is not None:
            return bool(self._safety_fail_open)
        return _safety_fails_open()

    def attach_harness(
        self,
        safety_pipeline: Any | None = None,
        cost_tracker: Any | None = None,
        audit_logger: Any | None = None,
        safety_fail_open: bool | None = None,
    ) -> "FrameworkAdapter":
        """Inject production components into this adapter.

        Call this after construction, before running.  Returns ``self`` for
        method chaining::

            adapter = CrewAIAdapter(crew).attach_harness(safety_pipeline=pipe)

        Args:
            safety_pipeline: Guardrail pipeline with check_input / check_output.
            cost_tracker:    CostTracker for per-run cost recording.
            audit_logger:    AuditLogger for compliance records.

        Returns:
            This adapter instance (fluent interface).
        """
        self._safety_pipeline = safety_pipeline
        self._cost_tracker = cost_tracker
        self._audit_logger = audit_logger
        if safety_fail_open is not None:
            self._safety_fail_open = safety_fail_open
        return self

    def attach_mcp(
        self,
        mcp_client: Any,
        tool_names: list[str] | None = None,
    ) -> "FrameworkAdapter":
        """Inject tools from an MCP server into this adapter.

        MCP tools are injected into the underlying framework before execution.
        Each concrete adapter (CrewAI, AutoGen, OpenClaw) handles the
        framework-specific injection in its ``run()`` method.

        Args:
            mcp_client:  A connected ``MCPClient`` instance.
            tool_names:  Optional allowlist of tool names to expose.
                         ``None`` means all tools from the server are exposed.

        Returns:
            This adapter instance (fluent interface).
        """
        if not hasattr(self, "_mcp_clients"):
            self._mcp_clients = []
        self._mcp_clients.append((mcp_client, tool_names))
        return self

    async def _resolve_mcp_tools(self) -> list[dict]:
        """Fetch and normalize tool schemas from all attached MCP clients.

        Returns a list of harness-format tool dicts ready for injection.
        """
        mcp_clients: list = getattr(self, "_mcp_clients", [])
        tools: list[dict] = []
        for client, name_filter in mcp_clients:
            try:
                raw_tools = await client.list_tools()
                for t in raw_tools:
                    name = getattr(t, "name", None) or t.get("name", "")
                    if name_filter and name not in name_filter:
                        continue
                    tools.append({
                        "name": name,
                        "description": getattr(t, "description", "") or t.get("description", ""),
                        "input_schema": getattr(t, "inputSchema", {}) or t.get("inputSchema", {}),
                        "_mcp_client": client,
                    })
            except Exception as exc:
                logger.warning("MCP tool listing failed: %s", exc)
        return tools

    async def run_with_harness(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002
    ) -> AsyncIterator["StepEvent"]:
        """Run the framework with full harness production features.

        Wraps ``run()`` with:
        - Input safety check (blocks prompt injection before the framework starts)
        - Output safety check on each yielded StepEvent
        - Cost recording at run completion

        Raises SafetyViolation if any check fails.  Falls back gracefully if
        safety pipeline or cost tracker are not attached.
        """
        from harness.core.errors import SafetyViolation, FailureClass

        safety_pipeline = getattr(self, "_safety_pipeline", None)
        cost_tracker = getattr(self, "_cost_tracker", None)

        # --- Input safety check ---
        if safety_pipeline is not None:
            task = input.get("task") or input.get("message") or str(input)
            try:
                guard = await safety_pipeline.check_input({"content": task})
                if getattr(guard, "blocked", False):
                    raise SafetyViolation(
                        f"Input blocked by {self.framework_name} adapter: "
                        f"{getattr(guard, 'reason', '')}",
                        guard_source="input_guard",
                        failure_class=FailureClass.SAFETY_INPUT,
                    )
            except SafetyViolation:
                raise
            except Exception as exc:
                # The pipeline ERRORED (couldn't reach a decision). Fail closed
                # by default — a broken guard must not silently pass input.
                if self._safety_fails_open():
                    logger.warning(
                        "Adapter input safety check errored; failing OPEN "
                        "(HARNESS_SAFETY_FAIL_OPEN set): %s", exc,
                    )
                else:
                    logger.error("Adapter input safety check errored; failing CLOSED: %s", exc)
                    raise SafetyViolation(
                        f"Input safety check failed in {self.framework_name} adapter "
                        f"(failing closed): {exc}. Set HARNESS_SAFETY_FAIL_OPEN=1 to "
                        "restore fail-open behaviour.",
                        guard_source="input_guard",
                        failure_class=FailureClass.SAFETY_INPUT,
                    ) from exc

        # --- Run framework and check each step ---
        async for event in self.run(ctx, input):
            if safety_pipeline is not None:
                content = ""
                if event.payload:
                    content = str(event.payload.get("content", event.payload.get("output", "")))
                if content:
                    try:
                        guard = await safety_pipeline.check_output({"content": content})
                        if getattr(guard, "blocked", False):
                            raise SafetyViolation(
                                f"Output blocked by {self.framework_name} adapter: "
                                f"{getattr(guard, 'reason', '')}",
                                guard_source="output_guard",
                                failure_class=FailureClass.SAFETY_OUTPUT,
                            )
                    except SafetyViolation:
                        raise
                    except Exception as exc:
                        # Pipeline ERRORED — fail closed by default so a broken
                        # guard cannot silently leak unchecked output.
                        if self._safety_fails_open():
                            logger.warning(
                                "Adapter output safety check errored; failing OPEN "
                                "(HARNESS_SAFETY_FAIL_OPEN set): %s", exc,
                            )
                        else:
                            logger.error(
                                "Adapter output safety check errored; failing CLOSED: %s", exc
                            )
                            raise SafetyViolation(
                                f"Output safety check failed in {self.framework_name} "
                                f"adapter (failing closed): {exc}. Set "
                                "HARNESS_SAFETY_FAIL_OPEN=1 to restore fail-open.",
                                guard_source="output_guard",
                                failure_class=FailureClass.SAFETY_OUTPUT,
                            ) from exc
            yield event

        # --- Cost recording ---
        if cost_tracker is not None:
            try:
                await cost_tracker.record(
                    run_id=ctx.run_id,
                    tenant_id=ctx.tenant_id,
                    model=self.framework_name,
                    input_tokens=ctx.token_count,
                    output_tokens=0,
                )
            except Exception as exc:
                logger.debug("Adapter cost tracking failed: %s", exc)

    @abstractmethod
    async def run(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002
    ) -> AsyncIterator["StepEvent"]:
        """Execute the framework and yield a StepEvent per observable step.

        Adapters MUST call ``ctx.tick()`` and check ``ctx.is_budget_ok()``
        to respect budget limits.  Use ``run_with_harness()`` for production.
        """
        raise NotImplementedError  # pragma: no cover
        yield  # type: ignore[misc]

    @abstractmethod
    async def get_result(self) -> FrameworkResult:
        """Return the final FrameworkResult after ``run()`` has completed."""
        raise NotImplementedError  # pragma: no cover
