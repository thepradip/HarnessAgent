"""CrewAI adapter — runs a Crew inside the HarnessAgent lifecycle."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from harness.adapters.base import FrameworkAdapter, FrameworkResult

if TYPE_CHECKING:
    from harness.core.context import AgentContext, StepEvent
    from harness.observability.event_bus import EventBus

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CrewAIAdapter(FrameworkAdapter):
    """Wraps a CrewAI ``Crew`` object inside the harness lifecycle.

    CrewAI is synchronous; the adapter runs ``crew.kickoff()`` inside
    :func:`asyncio.get_running_loop().run_in_executor` so the event loop
    stays unblocked.

    Individual agent steps are captured via ``crew.step_callback``:

    * If a ``step_callback`` already exists on the crew, the adapter wraps it
      (calling through to the original) so existing tooling is not disrupted.
    * After ``kickoff()`` completes, one :class:`~harness.core.context.StepEvent`
      (``event_type="tool_call"``) is yielded per captured step output, allowing
      full harness observability.

    Args:
        crew:    A CrewAI ``Crew`` instance ready to run.
        verbose: If ``True``, log step outputs at DEBUG level (default False).
        event_bus: Optional event bus for real-time publishing.
    """

    framework_name = "crewai"

    def __init__(
        self,
        crew: Any,
        verbose: bool = False,
        event_bus: "EventBus | None" = None,
    ) -> None:
        super().__init__()
        self._crew = crew
        self._verbose = verbose
        self._event_bus = event_bus
        self._task_outputs: list[str] = []
        self._step_count: int = 0
        self._final_output: str = ""
        self._kickoff_done: bool = False

    # ------------------------------------------------------------------
    # FrameworkAdapter implementation
    # ------------------------------------------------------------------

    async def run(
        self,
        ctx: "AgentContext",
        input: dict,  # noqa: A002
    ) -> AsyncIterator["StepEvent"]:
        """Kick off the crew and yield one StepEvent per captured agent step.

        The ``input`` dict is passed directly to ``crew.kickoff(inputs=input)``.

        Raises:
            ImportError: If ``crewai`` is not installed.
        """
        try:
            import crewai  # noqa: F401 — import check only
        except ImportError as exc:
            raise ImportError(
                "CrewAI is not installed.  Install it with:  pip install crewai"
            ) from exc

        # Reset per-run state so the adapter is reusable.
        self._task_outputs = []
        self._step_count = 0
        self._final_output = ""
        self._kickoff_done = False

        # --- Inject MCP tools into crew agents before kickoff ---
        await self._inject_mcp_tools()

        # --- Wrap step_callback to capture individual agent steps ---
        original_callback = getattr(self._crew, "step_callback", None)

        def _step_cb(output: Any) -> None:
            """Intercept each agent step; forward to the original callback if set.

            Enforce the harness budget LIVE: ``ctx.tick()`` advances the step
            counter and raises :class:`BudgetExceeded` once a limit is crossed.
            Raising here propagates out of ``crew.kickoff()`` (run in the
            executor) and aborts the crew mid-run instead of letting it spend
            to completion and only checking the budget during replay.
            """
            self._step_count += 1
            output_str = str(output)
            self._task_outputs.append(output_str)

            # tick() raises BudgetExceeded when a budget limit is crossed,
            # which aborts kickoff(). Best-effort: CrewAI may swallow the
            # exception, so the replay loop below still guards as a backstop.
            ctx.tick()
            if self._verbose:
                logger.debug(
                    "CrewAI step %d: %s", self._step_count, output_str[:120]
                )
            if original_callback is not None:
                try:
                    original_callback(output)
                except Exception as cb_exc:
                    logger.debug("CrewAI original step_callback raised: %s", cb_exc)

        self._crew.step_callback = _step_cb

        from harness.core.errors import BudgetExceeded

        budget_aborted = False
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._crew.kickoff(inputs=input),
            )
            self._final_output = str(result)
            self._kickoff_done = True
        except BudgetExceeded as exc:
            # ctx.tick() in the step callback aborted the crew mid-run. Mark
            # the run done so get_result() works, and emit a budget event below.
            logger.warning("CrewAI run aborted by budget limit: %s", exc)
            budget_aborted = True
            self._final_output = self._task_outputs[-1] if self._task_outputs else ""
            self._kickoff_done = True
        finally:
            # Restore the crew's original callback regardless of outcome.
            self._crew.step_callback = original_callback

        if budget_aborted:
            from harness.core.context import StepEvent

            budget_event = StepEvent(
                run_id=ctx.run_id,
                step=ctx.step_count,
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

        # --- Yield one StepEvent per captured step output ---
        for i, task_out in enumerate(self._task_outputs):
            from harness.core.context import StepEvent

            if not ctx.is_budget_ok():
                budget_event = StepEvent(
                    run_id=ctx.run_id,
                    step=i,
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

            event = StepEvent(
                run_id=ctx.run_id,
                step=i,
                event_type="tool_call",
                payload={
                    "framework": self.framework_name,
                    "task_output": task_out[:300],
                },
                timestamp=_utcnow(),
            )
            await self._publish(event)
            yield event

    async def get_result(self) -> FrameworkResult:
        """Return the :class:`FrameworkResult` after the crew has finished.

        Raises:
            RuntimeError: If ``run()`` has not been called yet.
        """
        if not self._kickoff_done:
            raise RuntimeError(
                "CrewAIAdapter.get_result() called before run() completed."
            )

        return FrameworkResult(
            framework=self.framework_name,
            output=self._final_output,
            steps=self._step_count,
            metadata={"task_outputs": list(self._task_outputs)},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _inject_mcp_tools(self) -> None:
        """Add MCP tools to every agent in the crew before kickoff."""
        mcp_tools = await self._resolve_mcp_tools()
        if not mcp_tools or not hasattr(self._crew, "agents"):
            return

        try:
            from crewai.tools import BaseTool  # type: ignore
            import pydantic

            for tool_schema in mcp_tools:
                mcp_client = tool_schema.pop("_mcp_client", None)
                tool_name = tool_schema["name"]
                tool_desc = tool_schema.get("description", "")

                def _make_tool(name: str, client: Any, schema: dict) -> Any:
                    class _MCPTool(BaseTool):
                        name: str = name
                        description: str = schema.get("description", "")

                        def _run(self, **kwargs: Any) -> str:
                            import asyncio
                            # CrewAI calls _run from a worker thread with no
                            # running loop, so get_event_loop().run_until_complete
                            # raises RuntimeError. Fall back to asyncio.run, which
                            # creates and manages its own loop (mirrors AutoGen).
                            try:
                                loop = asyncio.get_event_loop()
                                result = loop.run_until_complete(
                                    client.call_tool(name, kwargs)
                                )
                            except RuntimeError:
                                result = asyncio.run(client.call_tool(name, kwargs))
                            return str(result)

                    return _MCPTool()

                crewai_tool = _make_tool(tool_name, mcp_client, tool_schema)
                for agent in self._crew.agents:
                    if not hasattr(agent, "tools") or agent.tools is None:
                        agent.tools = []
                    agent.tools.append(crewai_tool)

            logger.info(
                "CrewAIAdapter: injected %d MCP tool(s) into %d agent(s)",
                len(mcp_tools),
                len(getattr(self._crew, "agents", [])),
            )
        except ImportError:
            logger.debug("crewai not installed — MCP injection skipped")
        except Exception as exc:
            logger.warning("CrewAIAdapter MCP injection failed: %s", exc)

    async def _publish(self, event: "StepEvent") -> None:
        """Fire-and-forget publish to the optional EventBus."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(event.run_id, event)
        except Exception as exc:  # pragma: no cover
            logger.debug("CrewAIAdapter: event_bus.publish failed: %s", exc)
