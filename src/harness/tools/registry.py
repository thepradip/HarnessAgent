"""ToolRegistry: central tool management with full validation pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import jsonschema

from harness.core.context import AgentContext, ToolCall, ToolResult
from harness.core.errors import FailureClass, SafetyViolation, ToolError
from harness.core.protocols import ToolExecutor

logger = logging.getLogger(__name__)

_TOOL_CALLS_METRIC = "tool_calls_total"
_TOOL_RESULT_MAX_CHARS = 8_000  # max chars before truncating tool output entering agent history


class ToolRegistry:
    """Central registry for all tools available to agents.

    Handles registration, lookup, schema validation, safety checks,
    execution with timeout, audit logging, and metrics.
    """

    def __init__(
        self,
        safety_pipeline: Any | None = None,
        audit_logger: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        self._tools: dict[str, ToolExecutor] = {}
        self._safety_pipeline = safety_pipeline
        self._audit_logger = audit_logger
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolExecutor) -> None:
        """Register a tool executor, keyed by tool.name."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool registration: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> ToolExecutor | None:
        """Return the registered ToolExecutor for name, or None."""
        return self._tools.get(name)

    def list_tools(self, agent_type: str | None = None) -> list[ToolExecutor]:
        """Return all registered tools. agent_type filter is reserved for future use."""
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Format converters
    # ------------------------------------------------------------------

    def to_anthropic_format(self) -> list[dict[str, Any]]:
        """Convert all registered tools to Anthropic API tool format."""
        result: list[dict[str, Any]] = []
        for tool in self._tools.values():
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return result

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all registered tools to OpenAI API function-calling format."""
        result: list[dict[str, Any]] = []
        for tool in self._tools.values():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, ctx: AgentContext, call: ToolCall) -> ToolResult:
        """Execute a tool call with full validation, safety, timeout, and logging.

        Steps:
        1. Lookup tool — raise ToolError(TOOL_NOT_FOUND) if missing.
        2. JSON Schema validate args — raise ToolError(TOOL_SCHEMA_ERROR) if fails.
        3. Safety check via pipeline.check_step — raise SafetyViolation if blocked.
        4. Execute with asyncio.timeout(tool.timeout_seconds).
        5. Audit log the call.
        6. Increment tool_calls_total counter.
        7. On exception: wrap in ToolError(TOOL_EXEC_ERROR), record failure.
        """
        # 1. Lookup
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(
                f"Tool '{call.name}' not found in registry.",
                tool_name=call.name,
                failure_class=FailureClass.TOOL_NOT_FOUND,
                context={"run_id": ctx.run_id, "available": list(self._tools.keys())},
            )

        # 2. Schema validation
        if hasattr(tool, "input_schema") and tool.input_schema:
            try:
                jsonschema.validate(instance=call.args, schema=tool.input_schema)
            except jsonschema.ValidationError as exc:
                raise ToolError(
                    f"Tool '{call.name}' arg schema validation failed: {exc.message}",
                    tool_name=call.name,
                    failure_class=FailureClass.TOOL_SCHEMA_ERROR,
                    context={
                        "run_id": ctx.run_id,
                        "args": call.args,
                        "validation_error": exc.message,
                    },
                ) from exc

        # 3. Safety check
        if self._safety_pipeline is not None:
            try:
                step_payload = {"tool_name": call.name, "args": call.args}
                guard_result = await _maybe_await(
                    self._safety_pipeline.check_step(step_payload)
                )
                if guard_result is not None and hasattr(guard_result, "blocked") and guard_result.blocked:
                    raise SafetyViolation(
                        f"Tool call '{call.name}' blocked by safety pipeline: "
                        f"{getattr(guard_result, 'reason', 'policy violation')}",
                        guard_source="tool_registry",
                        failure_class=FailureClass.SAFETY_STEP,
                        context={"run_id": ctx.run_id, "tool_name": call.name},
                    )
            except SafetyViolation:
                raise
            except Exception as exc:
                logger.warning(
                    "Safety pipeline check_step raised unexpected error: %s", exc
                )

        # 4. Execute with timeout
        timeout_seconds = getattr(tool, "timeout_seconds", 30.0)
        result: ToolResult
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await tool.execute(ctx, call.args)
        except asyncio.TimeoutError:
            error_msg = (
                f"Tool '{call.name}' timed out after {timeout_seconds:.1f}s"
            )
            logger.warning(error_msg)
            result = ToolResult(data=None, error=error_msg)
            # Audit log failure
            await self._audit(ctx, call, result)
            self._increment_metric(call.name, success=False)
            raise ToolError(
                error_msg,
                tool_name=call.name,
                failure_class=FailureClass.TOOL_TIMEOUT,
                context={"run_id": ctx.run_id, "timeout_seconds": timeout_seconds},
            )
        except ToolError:
            raise
        except SafetyViolation:
            raise
        except Exception as exc:
            error_msg = f"Tool '{call.name}' execution failed: {exc}"
            logger.exception("Tool '%s' raised unhandled exception", call.name)
            result = ToolResult(data=None, error=error_msg)
            await self._audit(ctx, call, result)
            self._increment_metric(call.name, success=False)
            raise ToolError(
                error_msg,
                tool_name=call.name,
                failure_class=FailureClass.TOOL_EXEC_ERROR,
                context={"run_id": ctx.run_id, "original_error": str(exc)},
            ) from exc

        # 5a. Cap tool result size — prevent large outputs from bloating agent history
        result = _cap_tool_result(result)

        # 5. Output schema validation (if tool declares output_schema)
        if hasattr(tool, "output_schema") and tool.output_schema and not result.is_error:
            try:
                jsonschema.validate(instance=result.data, schema=tool.output_schema)
            except jsonschema.ValidationError as exc:
                logger.warning(
                    "Tool '%s' output failed schema validation: %s",
                    call.name,
                    exc.message,
                )
                result = ToolResult(
                    data=result.data,
                    error=f"Output schema error: {exc.message}",
                    metadata=result.metadata,
                )

        # 6. Audit log
        await self._audit(ctx, call, result)

        # 7. Metrics
        self._increment_metric(call.name, success=not result.is_error)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _audit(
        self, ctx: AgentContext, call: ToolCall, result: ToolResult
    ) -> None:
        """Fire-and-forget audit log for a tool execution."""
        if self._audit_logger is None:
            return
        try:
            await _maybe_await(
                self._audit_logger.log(
                    event_type="tool_executed",
                    run_id=ctx.run_id,
                    tenant_id=ctx.tenant_id,
                    payload={
                        "tool_name": call.name,
                        "tool_id": call.id,
                        "args": call.args,
                        "is_error": result.is_error,
                        "error": result.error,
                    },
                )
            )
        except Exception as exc:
            logger.warning("Audit logger raised: %s", exc)

    def _increment_metric(self, tool_name: str, *, success: bool) -> None:
        """Increment the tool_calls_total Prometheus counter."""
        if self._metrics is None:
            return
        try:
            labels = {"tool_name": tool_name, "success": str(success).lower()}
            counter = getattr(self._metrics, _TOOL_CALLS_METRIC, None)
            if counter is not None:
                counter.labels(**labels).inc()
        except Exception as exc:
            logger.debug("Metrics increment failed: %s", exc)


def _cap_tool_result(result: ToolResult) -> ToolResult:
    """Truncate tool result data if its text representation exceeds _TOOL_RESULT_MAX_CHARS.

    Large tool outputs (e.g. full-table SQL dumps, verbose file reads) would otherwise
    consume most of the agent's context window.  We cap at 8 k chars and record the
    original length in ``metadata`` so callers can detect the truncation.
    """
    if result.is_error:
        return result
    try:
        text = result.to_text()
    except Exception:
        return result
    if len(text) <= _TOOL_RESULT_MAX_CHARS:
        return result
    truncated = text[:_TOOL_RESULT_MAX_CHARS] + (
        f"\n…[truncated — original output was {len(text):,} chars; "
        f"showing first {_TOOL_RESULT_MAX_CHARS:,}]"
    )
    logger.debug(
        "Tool result capped: original=%d chars, limit=%d", len(text), _TOOL_RESULT_MAX_CHARS
    )
    return ToolResult(
        data=truncated,
        error=None,
        metadata={**result.metadata, "truncated": True, "original_chars": len(text)},
    )


async def _maybe_await(obj: Any) -> Any:
    """Await obj if it is a coroutine, otherwise return it directly."""
    if asyncio.iscoroutine(obj):
        return await obj
    return obj
