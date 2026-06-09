"""LangGraph adapter — runs a StateGraph inside the HarnessAgent lifecycle."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from harness.adapters.base import FrameworkAdapter, FrameworkResult
from harness.core.errors import BudgetExceeded

if TYPE_CHECKING:
    from harness.core.context import AgentContext, StepEvent
    from harness.observability.event_bus import EventBus
    from harness.observability.tracer import StepTracer

logger = logging.getLogger(__name__)

# Keys tried in order when extracting a text output from the final graph state.
_OUTPUT_KEYS = ("messages", "output", "result", "answer", "response")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LangGraphAdapter(FrameworkAdapter):
    """Wraps a LangGraph ``StateGraph`` or compiled graph inside the harness.

    The adapter streams the graph via ``astream()`` and emits one
    :class:`~harness.core.context.StepEvent` (``event_type="tool_call"``) per
    node execution.  Budget enforcement via
    :pymeth:`~harness.core.context.AgentContext.is_budget_ok` happens before
    each node's event is yielded.

    Args:
        graph:        A LangGraph ``StateGraph`` *or* an already-compiled graph
                      (anything with an ``astream`` method).
        event_bus:    Optional :class:`~harness.observability.event_bus.EventBus`
                      that receives every :class:`StepEvent` as it is emitted.
        step_tracer:  Optional :class:`~harness.observability.tracer.StepTracer`
                      used to open OTel spans around each node.
    """

    framework_name = "langgraph"

    def __init__(
        self,
        graph: Any,
        event_bus: "EventBus | None" = None,
        step_tracer: "StepTracer | None" = None,
    ) -> None:
        # Initialise base adapter state (_safety_pipeline, _cost_tracker,
        # _mcp_clients, _safety_fail_open). Without this, run_with_harness and
        # attach_* only work by accident via getattr/hasattr fallbacks.
        super().__init__()
        self._graph = graph
        self._event_bus = event_bus
        self._step_tracer = step_tracer
        self._final_state: Any = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # FrameworkAdapter implementation
    # ------------------------------------------------------------------

    async def run(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002
    ) -> AsyncIterator["StepEvent"]:
        """Stream the graph and yield one StepEvent per node execution.

        Raises:
            ImportError: If ``langgraph`` is not installed.
            BudgetExceeded: Propagated from ``ctx.tick()`` when a budget limit
                is crossed (step count, token count, or elapsed time).
        """
        try:
            from langgraph.graph import StateGraph  # noqa: F401 — import check only
        except ImportError as exc:
            raise ImportError(
                "LangGraph is not installed.  Install it with:  pip install langgraph"
            ) from exc

        # Compile lazily if a raw StateGraph was supplied.
        compiled = (
            self._graph
            if hasattr(self._graph, "astream")
            else self._graph.compile()
        )

        step = 0
        last_event: Any = {}

        async for node_events in compiled.astream(input):
            # node_events is a dict mapping node_name → node_output_state.

            # --- Budget guard (checked *before* yielding) ---
            if not ctx.is_budget_ok():
                from harness.core.context import StepEvent

                budget_event = StepEvent(
                    run_id=ctx.run_id,
                    step=step,
                    event_type="budget_exceeded",
                    payload={
                        "framework": self.framework_name,
                        "step_count": ctx.step_count,
                        "max_steps": ctx.max_steps,
                    },
                    timestamp=_utcnow(),
                )
                await self._publish(budget_event)
                yield budget_event
                return

            # Advance the harness step counter; raises BudgetExceeded if over.
            try:
                ctx.tick()
            except BudgetExceeded as exc:
                from harness.core.context import StepEvent

                budget_event = StepEvent(
                    run_id=ctx.run_id,
                    step=ctx.step_count,
                    event_type="budget_exceeded",
                    payload={
                        "framework": self.framework_name,
                        "error": str(exc),
                    },
                    timestamp=_utcnow(),
                )
                await self._publish(budget_event)
                yield budget_event
                return

            step += 1

            # Emit one StepEvent per active node in this chunk.
            for node_name, node_output in node_events.items():
                from harness.core.context import StepEvent

                output_keys: list[str] = (
                    list(node_output.keys())
                    if isinstance(node_output, dict)
                    else []
                )

                # Optional: log token usage if the node output exposes it.
                token_usage = self._extract_token_usage(node_output)
                if token_usage:
                    logger.debug(
                        "LangGraph node '%s' used %d tokens",
                        node_name,
                        token_usage,
                    )

                event = StepEvent(
                    run_id=ctx.run_id,
                    step=step,
                    event_type="tool_call",
                    payload={
                        "framework": self.framework_name,
                        "node": node_name,
                        "output_keys": output_keys,
                        "token_usage": token_usage,
                    },
                    timestamp=_utcnow(),
                )

                await self._publish(event)
                yield event

            last_event = node_events  # keep the most recent chunk as final state

        self._final_state = last_event
        self._step_count = step

    async def get_result(self) -> FrameworkResult:
        """Return the :class:`FrameworkResult` built from the graph's final state.

        Raises:
            RuntimeError: If ``run()`` has not been called yet.
        """
        if self._final_state is None:
            raise RuntimeError(
                "LangGraphAdapter.get_result() called before run() completed."
            )

        state = self._final_state
        # The final event from astream is {node_name: output}.  Merge all node
        # outputs to build a single state dict so we can probe common keys.
        merged: dict[str, Any] = {}
        if isinstance(state, dict):
            for node_output in state.values():
                if isinstance(node_output, dict):
                    merged.update(node_output)
            # Also include top-level keys (some graphs expose them directly).
            merged.update(
                {k: v for k, v in state.items() if not isinstance(v, dict)}
            )

        output = self._extract_output(merged) or self._extract_output(
            state if isinstance(state, dict) else {}
        )

        return FrameworkResult(
            framework=self.framework_name,
            output=output,
            steps=self._step_count,
            metadata=merged if merged else (state if isinstance(state, dict) else {}),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output(state: dict[str, Any]) -> str:
        """Probe common output keys and return the best string representation."""
        for key in _OUTPUT_KEYS:
            val = state.get(key)
            if val is None:
                continue
            if isinstance(val, list) and val:
                return str(val[-1])
            if val:
                return str(val)
        return ""

    @staticmethod
    def _extract_token_usage(node_output: Any) -> int:
        """Try to read token usage from common LangChain / LangGraph structures."""
        if not isinstance(node_output, dict):
            return 0

        # Check for LangChain AIMessage objects in "messages" list.
        messages = node_output.get("messages", [])
        if isinstance(messages, list):
            for msg in reversed(messages):
                usage = None
                if hasattr(msg, "usage_metadata"):
                    usage = msg.usage_metadata
                elif isinstance(msg, dict):
                    usage = msg.get("usage_metadata") or msg.get("usage")

                if isinstance(usage, dict):
                    return usage.get("total_tokens", 0) or (
                        usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0)
                    )

        # Direct usage keys on the state dict.
        for key in ("total_tokens", "token_usage", "usage"):
            val = node_output.get(key)
            if isinstance(val, int):
                return val
            if isinstance(val, dict):
                return val.get("total_tokens", 0)

        return 0

    async def _publish(self, event: "StepEvent") -> None:
        """Fire-and-forget publish to the optional EventBus."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(event.run_id, event)
        except Exception as exc:  # pragma: no cover
            logger.debug("LangGraphAdapter: event_bus.publish failed: %s", exc)
