"""BaseAgent: the full agent lifecycle with all production features."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.core.context import (
    AgentContext,
    AgentResult,
    LLMResponse,
    StepEvent,
    ToolCall,
    ToolResult,
)
from harness.core.errors import (
    BudgetExceeded,
    FailureClass,
    HarnessError,
    HITLRejected,
    SafetyViolation,
    ToolError,
)
from harness.core.prompt_overrides import gepa_override
from harness.observability.failures import StepFailure
from harness.observability.trace_schema import SpanKind, SpanStatus

logger = logging.getLogger(__name__)

# How often (in steps) to save a checkpoint
_CHECKPOINT_INTERVAL = 10

# Max times the verifier may inject feedback before accepting the output anyway
_MAX_VERIFICATION_ATTEMPTS = 3

# Tool names whose results are tracked in ctx.metadata["last_code_result"]
_CODE_EXEC_RESULT_TOOLS: frozenset[str] = frozenset({
    "run_python", "run_code", "execute_code", "exec_python",
})

# Truncate large tool outputs before they enter the LLM context window
_TOOL_RESULT_MAX_CHARS = 8_000

# Summarize history when it grows beyond this length
_MAX_HISTORY_MESSAGES = 40
# Keep this many recent messages verbatim after summarization
_RECENT_MESSAGES_KEEP = 20

# Directory for per-run logs and conversation history (gitignored)
_LOG_DIR = Path("logs")

# Tool names whose execution requires allow_code_execution=True in tenant policy
_CODE_EXEC_TOOLS: frozenset[str] = frozenset({
    "run_python", "run_code", "execute_code", "exec_python",
})

# Tool names whose execution requires allow_file_write=True in tenant policy
_FILE_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file", "apply_patch", "create_file", "write_code",
})


def _run_log_path(run_id: str) -> Path:
    """Return path for the per-run JSONL log file."""
    p = _LOG_DIR / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{run_id}.jsonl"


def _history_path(workspace_path: Path, run_id: str) -> Path:
    """Return path for the full conversation history saved alongside checkpoint."""
    return workspace_path / "conversation.jsonl"


def _cap_tool_result(result: ToolResult) -> ToolResult:
    """Truncate oversized tool output before it enters the LLM context window."""
    if isinstance(result.data, str) and len(result.data) > _TOOL_RESULT_MAX_CHARS:
        overflow = len(result.data) - _TOOL_RESULT_MAX_CHARS
        return ToolResult(
            data=result.data[:_TOOL_RESULT_MAX_CHARS]
            + f"\n...[{overflow} chars truncated]",
            error=result.error,
            metadata={**result.metadata, "truncated": True},
        )
    return result


def _append_run_log(run_id: str, entry: dict) -> None:
    """Append one JSON line to the per-run log. Never raises."""
    try:
        line = json.dumps({**entry, "run_id": run_id,
                           "ts": datetime.now(UTC).isoformat()},
                          ensure_ascii=False)
        with _run_log_path(run_id).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.debug("run log write failed: %s", exc)


def _save_conversation(workspace_path: Path, run_id: str,
                       history: list[dict]) -> None:
    """Write full conversation history to workspace. Never raises."""
    try:
        hist_path = _history_path(workspace_path, run_id)
        with hist_path.open("w", encoding="utf-8") as fh:
            for turn in history:
                fh.write(json.dumps({**turn, "run_id": run_id},
                                    ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("conversation save failed: %s", exc)


class BaseAgent:
    """Production-grade base agent with full lifecycle management.

    Subclasses override:
    - agent_type (class attribute)
    - build_system_prompt(ctx)
    - Optionally override run() to add pre/post processing
    """

    agent_type: str = "base"

    def __init__(
        self,
        llm_router: Any,
        memory_manager: Any,
        tool_registry: Any,
        safety_pipeline: Any,
        step_tracer: Any,
        mlflow_tracer: Any,
        failure_tracker: Any,
        audit_logger: Any,
        event_bus: Any,
        cost_tracker: Any,
        checkpoint_manager: Any,
        message_bus: Any | None = None,
        online_monitor: Any | None = None,
        prompt_manager: Any | None = None,
        trace_recorder: Any | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._memory = memory_manager
        self._tool_registry = tool_registry
        self._safety_pipeline = safety_pipeline
        self._step_tracer = step_tracer
        self._mlflow_tracer = mlflow_tracer
        self._failure_tracker = failure_tracker
        self._audit_logger = audit_logger
        self._event_bus = event_bus
        self._cost_tracker = cost_tracker
        self._checkpoint_manager = checkpoint_manager
        self._message_bus = message_bus
        self._online_monitor = online_monitor
        self._prompt_manager = prompt_manager
        self._trace_recorder = trace_recorder
        self._feedback_last_id: dict[str, str] = {}  # run_id → last stream entry id

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Execute the full agent lifecycle and return an AgentResult."""
        run_start = time.monotonic()
        output: str = ""
        total_cost_usd: float = 0.0
        tool_calls_total = 0
        tool_errors_total = 0
        guardrail_hits = 0
        cache_hits = 0
        cache_read_tokens = 0
        metrics = _get_metrics()

        # 1. Emit started event and begin MLflow run
        await self._emit_event(StepEvent.started(ctx))
        if metrics is not None:
            try:
                metrics.active_runs.labels(agent_type=self.agent_type).inc()
            except Exception:
                pass

        # Open root RUN span (no-op when trace_recorder is None)
        run_span_id = await self._start_trace_span(
            ctx, SpanKind.RUN, f"run:{self.agent_type}",
            input_preview=ctx.task[:500],
        )

        history: list[dict[str, Any]] = []

        async with self._mlflow_context(ctx):
            try:
                # Log run start locally
                _append_run_log(ctx.run_id, {
                    "event": "run_started",
                    "agent_type": ctx.agent_type,
                    "tenant_id": ctx.tenant_id,
                    "task": ctx.task,
                    "max_steps": ctx.max_steps,
                    "max_tokens": ctx.max_tokens,
                })

                # 2. Resume from checkpoint — restores ctx counters and history
                history = await self._maybe_resume_checkpoint(ctx)

                # 3. Start a persistent sandbox session if configured
                await self._start_docker_session(ctx)

                while ctx.is_budget_ok():
                    # 3a. Check real-time feedback channel
                    await self._apply_feedback(ctx, history)

                    # 3b. Fit history to context window
                    history = await self._fit_history(ctx, history)

                    # 3b. Build retrieval context from memory and skill library
                    retrieval_context = await self._smart_retrieve(ctx)
                    skill_context = await self._retrieve_skills(ctx)

                    # 3c. Build messages
                    messages = self.build_messages(ctx, history, retrieval_context, skill_context)
                    system_prompt = gepa_override(
                        ctx, "system_prompt", self.build_system_prompt(ctx)
                    )

                    # 3d. LLM call with OTel span
                    async with self._llm_span(ctx) as llm_span_id:
                        response = await self._call_llm(ctx, messages, system_prompt)

                        # 3e-f. Record token usage
                        total_tokens = response.input_tokens + response.output_tokens
                        ctx.tick(tokens=total_tokens)
                        if response.cached:
                            cache_hits += 1
                            cache_read_tokens += response.input_tokens

                        # 3g. Cost tracking
                        call_cost_usd = 0.0
                        if self._cost_tracker is not None:
                            try:
                                run_cost = await self._cost_tracker.record(
                                    run_id=ctx.run_id,
                                    tenant_id=ctx.tenant_id,
                                    model=response.model,
                                    input_tokens=response.input_tokens,
                                    output_tokens=response.output_tokens,
                                )
                                call_cost_usd = float(run_cost.cost_usd)
                                total_cost_usd += call_cost_usd
                            except Exception as exc:
                                logger.warning("Cost tracking failed: %s", exc)

                        # 3h. MLflow LLM span + annotate trace span with token counts
                        await self._log_llm_span(ctx, response)
                        if self._trace_recorder is not None and llm_span_id is not None:
                            self._trace_recorder.set_llm_usage(
                                llm_span_id,
                                input_tokens=response.input_tokens,
                                output_tokens=response.output_tokens,
                                cost_usd=call_cost_usd,
                                cached=response.cached,
                            )

                    # 3i. Emit llm_called event
                    await self._emit_event(StepEvent.llm_called(ctx, response))

                    # Log LLM call locally
                    _append_run_log(ctx.run_id, {
                        "event": "llm_call",
                        "step": ctx.step_count,
                        "model": response.model,
                        "provider": response.provider,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cached": response.cached,
                        "has_tool_calls": bool(response.tool_calls),
                        "response_preview": response.content[:300] if response.content else "",
                    })

                    # 3j. Safety check on output — wrapped in GUARDRAIL span
                    if self._safety_pipeline is not None:
                        _gs_id = await self._start_trace_span(
                            ctx, SpanKind.GUARDRAIL, "guardrail:output",
                            input_preview=response.content[:300],
                        )
                        _guard_blocked = False
                        try:
                            guard = await _safe_call(
                                self._safety_pipeline.check_output,
                                {"content": response.content},
                            )
                            if guard is not None and getattr(guard, "blocked", False):
                                _guard_blocked = True
                                raise SafetyViolation(
                                    f"Output blocked by safety pipeline: {getattr(guard, 'reason', '')}",
                                    guard_source="output_guard",
                                    failure_class=FailureClass.SAFETY_OUTPUT,
                                )
                            await self._end_trace_span(
                                ctx, _gs_id, SpanStatus.OK,
                                output_preview="passed",
                            )
                        except SafetyViolation:
                            await self._end_trace_span(
                                ctx, _gs_id, SpanStatus.ERROR,
                                error="blocked",
                            )
                            raise
                        except Exception as exc:
                            await self._end_trace_span(ctx, _gs_id, SpanStatus.OK)
                            logger.debug("Safety output check raised: %s", exc)

                    # Redact any leaked secrets/PII before the content enters history,
                    # checkpoints, or memory — do this regardless of pipeline type.
                    safe_content = response.content
                    if self._safety_pipeline is not None and hasattr(self._safety_pipeline, "redact"):
                        try:
                            safe_content = self._safety_pipeline.redact(response.content)
                        except Exception:
                            pass

                    # Add assistant message to history
                    history.append({"role": "assistant", "content": safe_content})

                    # 3k. Push assistant response to short-term memory
                    if ctx.memory is not None:
                        try:
                            await ctx.memory.push_message(
                                run_id=ctx.run_id,
                                role="assistant",
                                content=response.content,
                                tokens=response.output_tokens,
                            )
                        except Exception as exc:
                            logger.debug("Failed to push assistant message to memory: %s", exc)

                    # 3l. If no tool calls — verify before accepting as done
                    if not response.tool_calls:
                        output = self.extract_final_answer(history)
                        vr = await self._verify_output(ctx, output, history)
                        if vr.passed:
                            break
                        # Verification failed — inject feedback and continue
                        attempts = ctx.metadata.get("_verification_attempts", 0) + 1
                        ctx.metadata["_verification_attempts"] = attempts
                        if attempts >= _MAX_VERIFICATION_ATTEMPTS:
                            logger.warning(
                                "Verification failed %d times for run %s — accepting output",
                                attempts, ctx.run_id,
                            )
                            break
                        history.append({
                            "role": "user",
                            "content": (
                                f"[Verification failed — attempt {attempts}/{_MAX_VERIFICATION_ATTEMPTS}]\n"
                                f"{vr.feedback}\n"
                                "Please fix the issue and try again."
                            ),
                        })
                        # Loop continues without break
                    tool_calls_total += len(response.tool_calls)

                    # 3m. Execute tool calls — HITL sequential, execution parallel
                    tool_results_for_history: list[dict[str, Any]] = []

                    # Policy and HITL checks are sequential — both are blocking gates
                    for call in response.tool_calls:
                        await self._check_policy(ctx, call)
                        await self._check_hitl(ctx, call)

                    # Execute all approved tool calls in parallel — each in a TOOL span
                    async def _execute_one(call: Any) -> ToolResult:
                        _ts_id = await self._start_trace_span(
                            ctx, SpanKind.TOOL, f"tool:{call.name}",
                            input_preview=str(call.args)[:300],
                        )
                        try:
                            result = await self._tool_registry.execute(ctx, call)
                            await self._end_trace_span(
                                ctx, _ts_id, SpanStatus.OK,
                                output_preview=str(result.data)[:300] if result.data else "",
                                error=result.error,
                            )
                            return result
                        except ToolError as exc:
                            logger.warning("Tool '%s' failed: %s", call.name, exc)
                            await self._end_trace_span(
                                ctx, _ts_id, SpanStatus.ERROR, error=str(exc)
                            )
                            return ToolResult(
                                data=None,
                                error=str(exc),
                                metadata={"failure_class": exc.failure_class.value},
                            )
                        except SafetyViolation as exc:
                            logger.warning("Tool '%s' blocked: %s", call.name, exc)
                            await self._end_trace_span(
                                ctx, _ts_id, SpanStatus.ERROR, error=f"blocked: {exc}"
                            )
                            return ToolResult(
                                data=None,
                                error=f"Blocked by safety policy: {exc}",
                                metadata={"guardrail_hit": True},
                            )

                    tool_results: list[ToolResult] = [
                        _cap_tool_result(r)
                        for r in await asyncio.gather(
                            *[_execute_one(c) for c in response.tool_calls]
                        )
                    ]

                    # Process results sequentially (history order must be preserved)
                    for call, result in zip(response.tool_calls, tool_results, strict=True):
                        # Push tool result to memory
                        if result.is_error:
                            tool_errors_total += 1
                        if result.metadata.get("guardrail_hit") or (
                            result.error
                            and "blocked by safety policy" in result.error.lower()
                        ):
                            guardrail_hits += 1

                        if ctx.memory is not None:
                            try:
                                await ctx.memory.push_message(
                                    run_id=ctx.run_id,
                                    role="tool",
                                    content=result.to_text(),
                                    tokens=0,
                                )
                            except Exception as mem_exc:
                                logger.debug("Failed to push tool result to memory: %s", mem_exc)

                        # Emit tool_called step event
                        await self._emit_event(StepEvent.tool_called(ctx, call, result))

                        # Log tool call locally
                        _append_run_log(ctx.run_id, {
                            "event": "tool_call",
                            "step": ctx.step_count,
                            "tool": call.name,
                            "args": call.args,
                            "success": not result.is_error,
                            "error": result.error,
                            "result_preview": str(result.data)[:300] if result.data else "",
                        })

                        # Track last code-execution result for the PEV verifier
                        if call.name in _CODE_EXEC_RESULT_TOOLS and isinstance(result.data, dict):
                            ctx.metadata["last_code_result"] = result.data

                        # Audit log
                        await self._audit(ctx, call, result)

                        # RLVR: record per-step reward (non-blocking, best-effort)
                        await self._record_rlvr_reward(ctx, call, result)

                        tool_results_for_history.append(
                            {
                                "role": "tool",
                                "tool_use_id": call.id,
                                "content": result.to_text(),
                            }
                        )

                    # Append tool results to history
                    history.extend(tool_results_for_history)

                    # 3n. Checkpoint every N steps
                    if ctx.step_count % _CHECKPOINT_INTERVAL == 0:
                        await self._save_checkpoint(ctx, history)

                else:
                    # Budget exceeded — loop condition failed
                    ctx.failed = True
                    if ctx.step_count >= ctx.max_steps:
                        ctx.failure_class = FailureClass.BUDGET_STEPS.value
                    elif ctx.token_count >= ctx.max_tokens:
                        ctx.failure_class = FailureClass.BUDGET_TOKENS.value
                    else:
                        ctx.failure_class = FailureClass.BUDGET_TIME.value
                    await self._emit_event(
                        StepEvent(
                            run_id=ctx.run_id,
                            step=ctx.step_count,
                            event_type="budget_exceeded",
                            payload={"failure_class": ctx.failure_class},
                            timestamp=_utcnow(),
                        )
                    )

            except BudgetExceeded as exc:
                ctx.failed = True
                ctx.failure_class = exc.failure_class.value
                output = f"Budget exceeded: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except HITLRejected as exc:
                ctx.failed = True
                ctx.failure_class = FailureClass.INTER_AGENT_REJECT.value
                guardrail_hits += 1
                output = f"HITL rejected: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except SafetyViolation as exc:
                ctx.failed = True
                ctx.failure_class = exc.failure_class.value
                guardrail_hits += 1
                output = f"Safety violation: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except HarnessError as exc:
                ctx.failed = True
                ctx.failure_class = exc.failure_class.value
                output = f"Harness error: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except asyncio.CancelledError:
                ctx.failed = True
                ctx.failure_class = FailureClass.UNKNOWN.value
                output = "Run was cancelled."
                await self._emit_event(StepEvent.failed(ctx, "cancelled"))
                raise

            except Exception as exc:
                ctx.failed = True
                ctx.failure_class = self._classify_exception(exc).value
                output = f"Unexpected error: {exc}"
                logger.exception("Unhandled exception in agent run %s", ctx.run_id)
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            finally:
                # Stop the persistent sandbox session (always, even on exception)
                await self._stop_docker_session(ctx)

                # Always checkpoint so runs can be resumed or inspected after any exit
                await self._save_checkpoint(ctx, history)

                # Save full conversation history to disk (gitignored, local only)
                if history:
                    _save_conversation(ctx.workspace_path, ctx.run_id, history)

                # Feed online learning monitor so Hermes can detect regressions
                if self._online_monitor is not None:
                    try:
                        version_id = ""
                        version_number = 0
                        if self._prompt_manager is not None:
                            pv = await _safe_call(
                                self._prompt_manager.get_version, self.agent_type
                            )
                            if pv is not None:
                                version_id = pv.version_id
                                version_number = pv.version_number
                        if version_id:
                            await self._online_monitor.record_run(
                                agent_type=self.agent_type,
                                version_id=version_id,
                                version_number=version_number,
                                success=not ctx.failed,
                                cost_usd=total_cost_usd,
                                steps=ctx.step_count,
                            )
                    except Exception as exc:
                        logger.debug("online_monitor.record_run failed: %s", exc)

                # Auto-capture reusable skills from high-scoring successful runs
                await self._maybe_capture_skill(ctx, output, total_cost_usd)

                # Log run completion locally
                elapsed = time.monotonic() - run_start
                _append_run_log(ctx.run_id, {
                    "event": "run_finished",
                    "agent_type": ctx.agent_type,
                    "success": not ctx.failed,
                    "failure_class": ctx.failure_class,
                    "total_steps": ctx.step_count,
                    "total_tokens": ctx.token_count,
                    "total_cost_usd": total_cost_usd,
                    "elapsed_seconds": round(elapsed, 2),
                    "output_preview": output[:300] if output else "",
                })

                # Close root RUN span
                await self._end_trace_span(
                    ctx, run_span_id,
                    status=SpanStatus.ERROR if ctx.failed else SpanStatus.OK,
                    output_preview=output[:500] if output else "",
                    error=ctx.failure_class if ctx.failed else None,
                )

                # 8. Decrement active_runs gauge
                if metrics is not None:
                    try:
                        metrics.active_runs.labels(agent_type=self.agent_type).dec()
                        metrics.agent_runs_total.labels(
                            agent_type=self.agent_type,
                            success=str(not ctx.failed).lower(),
                        ).inc()
                    except Exception:
                        pass

        # Emit completed event if success
        if not ctx.failed:
            await self._emit_event(StepEvent.completed(ctx, output))

        elapsed = time.monotonic() - run_start

        return AgentResult(
            run_id=ctx.run_id,
            output=output,
            steps=ctx.step_count,
            tokens=ctx.token_count,
            success=not ctx.failed,
            failure_class=ctx.failure_class,
            error_message=output if ctx.failed else None,
            elapsed_seconds=elapsed,
            cost_usd=total_cost_usd,
            tool_calls=tool_calls_total,
            tool_errors=tool_errors_total,
            guardrail_hits=guardrail_hits,
            handoff_count=int(ctx.metadata.get("handoff_count", 0) or 0),
            cache_hits=cache_hits,
            cache_read_tokens=cache_read_tokens,
        )

    # ------------------------------------------------------------------
    # Overridable methods
    # ------------------------------------------------------------------

    def build_system_prompt(self, ctx: AgentContext) -> str:
        """Return the system prompt for this agent. Override in subclasses."""
        return (
            f"You are a helpful AI agent of type '{self.agent_type}'. "
            f"Task: {ctx.task}\n"
            "Use the available tools to complete the task. "
            "When you have a final answer, respond without requesting any more tools."
        )

    def build_messages(
        self,
        ctx: AgentContext,
        history: list[dict[str, Any]],
        retrieval_context: str,
        skill_context: str = "",
    ) -> list[dict[str, Any]]:
        """Assemble the messages list for the LLM call."""
        messages: list[dict[str, Any]] = []

        # If we have retrieval context, prepend it as a user message
        if (retrieval_context or skill_context) and not history:
            parts: list[str] = []
            if retrieval_context:
                parts.append(f"Relevant context from memory:\n{retrieval_context}")
            if skill_context:
                parts.append(skill_context)
            parts.append(f"Task: {ctx.task}")
            messages.append({"role": "user", "content": "\n\n".join(parts)})
        elif not history:
            messages.append({"role": "user", "content": ctx.task})

        messages.extend(history)

        # Ensure first message is from user
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": ctx.task})

        return messages

    def extract_final_answer(self, history: list[dict[str, Any]]) -> str:
        """Extract the final answer text from agent history."""
        # Walk history in reverse to find the last assistant message
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Anthropic block content
                    text_parts = [
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                        if not (isinstance(b, dict) and b.get("type") == "tool_use")
                    ]
                    text = " ".join(text_parts).strip()
                    if text:
                        return text
                elif isinstance(content, str) and content.strip():
                    return content.strip()
        return "Task completed."

    def _classify_exception(self, e: Exception) -> FailureClass:
        """Map an exception to a FailureClass."""
        if isinstance(e, BudgetExceeded):
            return e.failure_class
        if isinstance(e, ToolError):
            return e.failure_class
        if isinstance(e, SafetyViolation):
            return e.failure_class
        if isinstance(e, HITLRejected):
            return FailureClass.INTER_AGENT_REJECT
        if isinstance(e, HarnessError):
            return e.failure_class
        if isinstance(e, TimeoutError):
            return FailureClass.BUDGET_TIME
        return FailureClass.UNKNOWN

    # ------------------------------------------------------------------
    # Real-time feedback
    # ------------------------------------------------------------------

    async def _apply_feedback(
        self,
        ctx: AgentContext,
        history: list[dict[str, Any]],
    ) -> None:
        """
        Poll the feedback channel and inject any pending events into history.

        Called at the start of every loop iteration before the LLM call.
        The feedback channel is optional — this is a no-op when not configured.
        """
        feedback_channel = ctx.metadata.get("feedback_channel")
        if feedback_channel is None:
            return

        last_id = self._feedback_last_id.get(ctx.run_id, "0")

        try:
            events, new_last_id = await feedback_channel.poll(ctx.run_id, last_id)
        except Exception as exc:
            logger.debug("feedback poll failed for run %s: %s", ctx.run_id, exc)
            return

        if not events:
            return

        self._feedback_last_id[ctx.run_id] = new_last_id
        applied_ids: list[str] = []

        for event in events:
            logger.info(
                "Feedback received run=%s type=%s source=%s priority=%d",
                ctx.run_id[:8], event.type, event.source, event.priority,
            )

            # stop — raise clean termination
            if event.type == "stop":
                await feedback_channel.mark_applied(ctx.run_id, [event.feedback_id])
                await self._emit_event(_feedback_step_event(ctx, event))
                from harness.core.errors import HarnessError
                raise HarnessError(
                    f"Agent stopped by feedback: {event.content or '(no reason)'}",
                    failure_class=FailureClass.UNKNOWN,
                )

            # redirect — update remaining task in context metadata
            if event.type == "redirect" and event.content:
                ctx.metadata["redirected_task"] = event.content
                history.append({
                    "role": "system",
                    "content": event.to_context_message(),
                })
                applied_ids.append(event.feedback_id)
                await self._emit_event(_feedback_step_event(ctx, event))
                continue

            # correction / hint / score-with-low-score → inject into history
            from harness.feedback.channel import should_inject
            if should_inject(event):
                history.append({
                    "role": "system",
                    "content": event.to_context_message(),
                })
                applied_ids.append(event.feedback_id)
                await self._emit_event(_feedback_step_event(ctx, event))

            # score → record metric regardless
            if event.type == "score" and event.score is not None:
                try:
                    from harness.observability.metrics import get_prometheus_metrics
                    m = get_prometheus_metrics()
                    if m and hasattr(m, "feedback_scores"):
                        m.feedback_scores.labels(
                            agent_type=self.agent_type,
                            run_id=ctx.run_id,
                        ).observe(event.score)
                except Exception:
                    pass
                applied_ids.append(event.feedback_id)

        if applied_ids:
            try:
                await feedback_channel.mark_applied(ctx.run_id, applied_ids)
            except Exception as exc:
                logger.debug("mark_applied failed: %s", exc)

    async def _record_rlvr_reward(
        self,
        ctx: AgentContext,
        call: ToolCall,
        result: ToolResult,
    ) -> None:
        """
        Compute and record a per-step reward for RLVR.
        Uses the verifier and reward buffer from ctx.metadata when configured.
        Non-blocking — failures are silently logged.
        """
        reward_buffer = ctx.metadata.get("rlvr_reward_buffer")
        verifier = ctx.metadata.get("rlvr_verifier")
        rlvr_loop = ctx.metadata.get("rlvr_loop")
        if reward_buffer is None:
            return

        try:
            from harness.improvement.rlvr.buffer import StepReward
            action = call.args.get("sql") or call.args.get("code") or str(call.args)
            result_text = result.to_text() if not result.is_error else ""
            gold = ctx.metadata.get("gold_action") or ctx.metadata.get("gold_sql")

            reward = 0.5  # default neutral
            verdict = "partial"
            source = "default"
            reasoning = ""
            vr = None

            if verifier is not None and action:
                try:
                    vr = await verifier.verify(
                        task=ctx.task,
                        action=action,
                        result=result_text,
                        gold=gold,
                    )
                    reward = vr.overall_reward
                    verdict = vr.verdict
                    source = vr.source
                    reasoning = vr.feedback_for_agent
                    # Publish step feedback back to agent
                    if rlvr_loop is not None:
                        await rlvr_loop.publish_step_feedback(ctx.run_id, vr)
                except Exception as exc:
                    logger.debug("RLVR verifier failed: %s", exc)
            elif result.is_error:
                reward, verdict, source = 0.0, "incorrect", "error_signal"
                reasoning = result.error or ""
            else:
                reward, verdict, source = 0.8, "correct", "success_signal"

            # Prompt hash (stable key for this prompt version)
            import hashlib
            prompt_hash = hashlib.sha256(
                ctx.metadata.get("prompt_version", "unknown").encode()
            ).hexdigest()[:16]

            sr = StepReward(
                run_id=ctx.run_id,
                step=ctx.step_count,
                agent_type=self.agent_type,
                task=ctx.task,
                action=action[:500],
                result_preview=result_text[:500],
                reward=reward,
                verdict=verdict,
                confidence=vr._confidence() if vr else 0.5,
                source=source,
                prompt_hash=prompt_hash,
                reasoning=reasoning[:300],
            )
            await reward_buffer.record(sr)

        except Exception as exc:
            logger.debug("_record_rlvr_reward failed silently: %s", exc)

    # ------------------------------------------------------------------
    # HITL
    # ------------------------------------------------------------------

    async def _check_hitl(self, ctx: AgentContext, call: ToolCall) -> None:
        """Check if human-in-the-loop approval is required for this tool call."""
        # Policy-based check: look for hitl manager in metadata
        hitl_manager = ctx.metadata.get("hitl_manager")
        policy = ctx.metadata.get("policy")

        if hitl_manager is None or policy is None:
            return

        if not policy.requires_hitl(call.name):
            return

        logger.info(
            "HITL required for tool '%s' in run '%s'", call.name, ctx.run_id
        )

        # Create approval request
        request = await hitl_manager.request_approval(
            run_id=ctx.run_id,
            tenant_id=ctx.tenant_id,
            tool_name=call.name,
            tool_args=call.args,
        )

        # Await the decision
        decision = await hitl_manager.await_decision(
            request_id=request.request_id,
            timeout=3600.0,
        )

        if decision == "rejected":
            raise HITLRejected(
                f"Human reviewer rejected tool call '{call.name}'",
                request_id=request.request_id,
            )
        if decision == "expired":
            raise HITLRejected(
                f"HITL approval for '{call.name}' expired",
                request_id=request.request_id,
            )

    async def _verify_output(
        self,
        ctx: AgentContext,
        output: str,
        history: list[dict[str, Any]],
    ) -> Any:
        """Run the configured verifier against the agent's current output.

        Returns a VerificationResult. If no verifier is configured (or the
        verifier raises), returns a passing result so the agent always exits
        cleanly when there is no objective oracle.
        """
        from harness.verification.verifier import VerificationResult
        verifier = ctx.metadata.get("verifier")
        if verifier is None:
            return VerificationResult.skipped()
        try:
            result = await verifier.verify(ctx, output, history)
            if not result.passed:
                logger.info(
                    "Verification failed for run %s: %s (score=%.2f)",
                    ctx.run_id[:8], result.verdict, result.score,
                )
            return result
        except Exception as exc:
            logger.warning("Verifier raised unexpectedly: %s — treating as skipped", exc)
            return VerificationResult.skipped()

    async def _check_policy(self, ctx: AgentContext, call: ToolCall) -> None:
        """Enforce per-tenant policy restrictions before a tool call executes."""
        policy = ctx.metadata.get("policy")
        if policy is None:
            return

        if policy.is_tool_blocked(call.name):
            raise SafetyViolation(
                f"Tool '{call.name}' is blocked by tenant policy",
                guard_source="policy",
                failure_class=FailureClass.SAFETY_STEP,
            )

        tool_lower = call.name.lower()
        is_code_exec = (
            call.name in _CODE_EXEC_TOOLS
            or tool_lower.startswith(("run_", "exec_", "execute_"))
        )
        if is_code_exec and not policy.allow_code_execution:
            raise SafetyViolation(
                f"Tool '{call.name}' requires code execution, which is disabled by tenant policy",
                guard_source="policy",
                failure_class=FailureClass.SAFETY_STEP,
            )

        tool_lower = call.name.lower()
        is_file_write = (
            call.name in _FILE_WRITE_TOOLS
            or tool_lower.startswith("write_")
            or tool_lower.endswith("_patch")
        )
        if is_file_write and not policy.allow_file_write:
            raise SafetyViolation(
                f"Tool '{call.name}' requires file write access, which is disabled by tenant policy",
                guard_source="policy",
                failure_class=FailureClass.SAFETY_STEP,
            )

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        ctx: AgentContext,
        messages: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse:
        """Call the LLM router with tools from the registry."""
        tools = (
            self._tool_registry.to_anthropic_format()
            if self._tool_registry is not None
            else []
        )

        remaining_tokens = ctx.max_tokens - ctx.token_count
        max_tokens = min(4096, max(256, remaining_tokens // 2))

        # Use streaming when requested and no tools are needed (streaming + tools
        # requires special handling that varies by provider)
        if ctx.metadata.get("stream_tokens") and not tools:
            return await self._call_llm_streaming(ctx, messages, system, max_tokens)

        return await self._llm_router.complete(
            messages=messages,
            system=system,
            tools=tools if tools else None,
            max_tokens=max_tokens,
            tenant_id=ctx.tenant_id,
        )

    async def _call_llm_streaming(
        self,
        ctx: AgentContext,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> LLMResponse:
        """Stream tokens from the LLM, publishing each delta as a StepEvent.

        Accumulates the full text and returns a synthetic LLMResponse so the
        rest of the agent loop (history, cost tracking, safety) works unchanged.
        """
        chunks: list[str] = []
        try:
            async for delta in self._llm_router.stream(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
            ):
                chunks.append(delta)
                # Publish token delta for SSE clients
                await self._emit_event(
                    StepEvent(
                        run_id=ctx.run_id,
                        step=ctx.step_count,
                        event_type="token_delta",
                        payload={"delta": delta},
                        timestamp=_utcnow(),
                    )
                )
        except Exception as exc:
            logger.warning("Streaming LLM call failed, falling back to complete(): %s", exc)
            return await self._llm_router.complete(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                tenant_id=ctx.tenant_id,
            )

        full_text = "".join(chunks)
        # Estimate tokens (streaming responses don't always return exact counts)
        estimated_tokens = max(1, len(full_text) // 4)
        return LLMResponse(
            content=full_text,
            tool_calls=[],
            input_tokens=estimated_tokens,
            output_tokens=estimated_tokens,
            model=getattr(self._llm_router, "model", "unknown"),
            provider=getattr(self._llm_router, "provider_name", "unknown"),
            cached=False,
        )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _smart_retrieve(self, ctx: AgentContext) -> str:
        """Retrieve relevant context from long-term memory for the current task."""
        if ctx.memory is None:
            return ""
        try:
            if hasattr(ctx.memory, "smart_retrieve"):
                result = await ctx.memory.smart_retrieve(
                    query=ctx.task,
                    run_id=ctx.run_id,
                    k=5,
                )
                if result:
                    if isinstance(result, str):
                        return result
                    if hasattr(result, "formatted"):
                        return result.formatted
                    return str(result)
        except Exception as exc:
            logger.debug("smart_retrieve failed: %s", exc)
        return ""

    async def _retrieve_skills(self, ctx: AgentContext) -> str:
        """Retrieve relevant skill artifacts from the skill library for this task.

        Returns a formatted context block, or empty string when no skill store
        is configured or no relevant skills are found.
        """
        skill_store = ctx.metadata.get("skill_store")
        if skill_store is None:
            return ""
        try:
            from harness.tools.skill_store import format_skills_for_context
            skills = await skill_store.retrieve_relevant(
                query=ctx.task,
                tenant_id=ctx.tenant_id,
                k=3,
            )
            if skills:
                # Record usage for each retrieved skill (non-blocking)
                for skill in skills:
                    try:
                        await skill_store.record_use(skill.skill_id, ctx.tenant_id)
                    except Exception:
                        pass
            return format_skills_for_context(skills)
        except Exception as exc:
            logger.debug("_retrieve_skills failed: %s", exc)
            return ""

    async def _maybe_capture_skill(
        self,
        ctx: AgentContext,
        output: str,
        cost_usd: float,
    ) -> None:
        """Auto-capture a skill from a successful high-scoring run.

        Triggered when all of these are true:
        - Run succeeded (not ctx.failed)
        - ``ctx.metadata["skill_capture"]`` is a configured SkillCapture instance
        - ``ctx.metadata["skill_score"]`` (float 0–1) meets the capture threshold

        Callers set ``ctx.metadata["skill_score"]`` from their RLVR verifier,
        eval scorer, or a static value. If not set, defaults to 0.0 (skipped).
        """
        if ctx.failed or not output:
            return
        capture = ctx.metadata.get("skill_capture")
        if capture is None:
            return
        score = float(ctx.metadata.get("skill_score", 0.0))
        try:
            from harness.tools.skill_store import SkillType
            title = ctx.metadata.get("skill_title") or f"{self.agent_type}: {ctx.task[:60]}"
            description = ctx.metadata.get("skill_description") or ctx.task
            skill_type_str = ctx.metadata.get("skill_type", "output")
            try:
                skill_type = SkillType(skill_type_str)
            except ValueError:
                skill_type = SkillType.CODE  # safe default

            saved = await capture.capture(
                title=title,
                description=description,
                content=output,
                skill_type=skill_type,
                tenant_id=ctx.tenant_id,
                score=score,
                run_id=ctx.run_id,
                metadata={"cost_usd": cost_usd, "steps": ctx.step_count},
            )
            if saved:
                logger.info(
                    "Skill captured: run=%s score=%.2f title=%r",
                    ctx.run_id[:8], score, title,
                )
        except Exception as exc:
            logger.debug("_maybe_capture_skill failed (non-fatal): %s", exc)

    async def _start_docker_session(self, ctx: AgentContext) -> None:
        """Start a persistent sandbox container for this run if configured.

        Enabled when ``ctx.metadata["sandbox_session"]`` is truthy or the
        global ``sandbox_session_reuse`` config flag is set.  The running
        session is stored at ``ctx.metadata["docker_session"]`` so
        ``RunCodeTool`` can find it without any additional wiring.
        """
        enabled = ctx.metadata.get("sandbox_session", False)
        if not enabled:
            try:
                from harness.core.config import get_config
                enabled = get_config().sandbox_session_reuse
            except Exception:
                pass
        if not enabled:
            return

        try:
            from harness.core.config import get_config
            cfg = get_config()
            provider = ctx.metadata.get("sandbox_provider") or getattr(cfg, "sandbox_provider", "docker")

            if provider == "e2b":
                # E2B cloud sandbox — drop-in for SessionDockerSandbox.
                from harness.filesystem.e2b_sandbox import E2BSandbox
                if not await E2BSandbox.is_available():
                    logger.debug("E2B unavailable (missing SDK or E2B_API_KEY); skipping session sandbox")
                    return
                session: Any = E2BSandbox(
                    api_key=cfg.e2b_api_key or None,
                    template=cfg.e2b_template or None,
                    workspace_path=ctx.workspace_path,
                )
            else:
                from harness.filesystem.sandbox import SessionDockerSandbox, memory_for_workload
                if not await SessionDockerSandbox.is_available():
                    logger.debug("Docker not available; skipping session sandbox")
                    return
                session = SessionDockerSandbox(
                    workspace_path=ctx.workspace_path,
                    memory_limit=memory_for_workload(cfg.sandbox_workload),
                    runtime=cfg.sandbox_runtime,
                )

            await session.__aenter__()
            # Stored under "docker_session" for back-compat — RunCodeTool reads this key.
            ctx.metadata["docker_session"] = session
            logger.info("%s session started for run %s", provider, ctx.run_id[:8])
        except Exception as exc:
            logger.warning("Failed to start sandbox session: %s", exc)

    async def _stop_docker_session(self, ctx: AgentContext) -> None:
        """Stop the persistent sandbox container, if one is running."""
        session = ctx.metadata.pop("docker_session", None)
        if session is None:
            return
        try:
            await session.__aexit__(None, None, None)
            logger.info("Docker session stopped for run %s", ctx.run_id[:8])
        except Exception as exc:
            logger.warning("Failed to stop Docker session: %s", exc)

    async def _fit_history(
        self,
        ctx: AgentContext,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Trim history to fit within the context window budget.

        When history exceeds _MAX_HISTORY_MESSAGES, the older portion is
        summarized via LLM call and replaced with a single summary message.
        Keeps the last _RECENT_MESSAGES_KEEP messages verbatim.
        Falls back to a naive slice if summarization fails.
        """
        if len(history) <= _MAX_HISTORY_MESSAGES:
            return history

        recent = history[-_RECENT_MESSAGES_KEEP:]
        older = history[:-_RECENT_MESSAGES_KEEP]

        # Try memory manager's fit_history first (most efficient path)
        if ctx.memory is not None and hasattr(ctx.memory, "fit_history"):
            try:
                fit_result = await ctx.memory.fit_history(
                    run_id=ctx.run_id,
                    history=older,
                    keep_last=0,
                )
                summary_text = getattr(fit_result, "summary", None)
                if summary_text:
                    logger.info(
                        "Run %s: summarized %d messages via memory manager",
                        ctx.run_id,
                        len(older),
                    )
                    return [
                        {"role": "user", "content": f"[Conversation summary]: {summary_text}"},
                        *recent,
                    ]
            except Exception as exc:
                logger.debug("memory.fit_history failed: %s", exc)

        # Try LLM summarization directly
        try:
            turns_text = "\n".join(
                f"{m['role'].upper()}: {m.get('content', '')[:500]}"
                for m in older
                if isinstance(m.get("content"), str)
            )
            if turns_text:
                # Context-compression prompt — optimizable as the "context_summary"
                # component (GEPA injects an override via ctx.metadata).
                summary_instruction = gepa_override(
                    ctx,
                    "context_summary",
                    "Summarize the following conversation history concisely "
                    "in 3-5 sentences, preserving key decisions, findings, "
                    "and tool call outcomes:",
                )
                summary_response = await self._llm_router.complete(
                    messages=[
                        {
                            "role": "user",
                            "content": summary_instruction + "\n\n" + turns_text,
                        }
                    ],
                    system="You are a helpful assistant that summarizes conversations concisely.",
                    max_tokens=512,
                    tenant_id=ctx.tenant_id,
                    tier="cheap",  # summarization is simple — never burn the premium model
                )
                summary_text = summary_response.content.strip()
                if summary_text:
                    logger.info(
                        "Run %s: summarized %d messages via LLM (%d tokens)",
                        ctx.run_id,
                        len(older),
                        summary_response.output_tokens,
                    )
                    return [
                        {"role": "user", "content": f"[Conversation summary]: {summary_text}"},
                        *recent,
                    ]
        except Exception as exc:
            logger.debug("LLM history summarization failed: %s", exc)

        # Fallback: naive slice
        logger.debug(
            "Run %s: falling back to naive history slice (%d → %d messages)",
            ctx.run_id,
            len(history),
            _MAX_HISTORY_MESSAGES,
        )
        return history[-_MAX_HISTORY_MESSAGES:]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    async def _maybe_resume_checkpoint(
        self, ctx: AgentContext
    ) -> list[dict[str, Any]]:
        """Load a checkpoint and return the restored conversation history.

        Also restores step_count and token_count onto ctx so budget accounting
        is correct after a resume. Returns an empty list when no checkpoint
        exists or when checkpoint manager is not configured.
        """
        if self._checkpoint_manager is None:
            return []
        try:
            checkpoint = await self._checkpoint_manager.load(ctx.run_id, ctx.tenant_id)
            if checkpoint is not None:
                ctx.step_count = checkpoint.step_count
                ctx.token_count = checkpoint.token_count
                logger.info(
                    "Resumed run %s from checkpoint at step %d",
                    ctx.run_id,
                    checkpoint.step_count,
                )
                return list(checkpoint.history_snapshot)
        except Exception as exc:
            logger.debug("Checkpoint load failed (will start fresh): %s", exc)
        return []

    async def _save_checkpoint(
        self, ctx: AgentContext, history: list[dict[str, Any]]
    ) -> None:
        """Persist ctx state and full conversation history to a local checkpoint."""
        if self._checkpoint_manager is None:
            return
        try:
            await self._checkpoint_manager.save(ctx, history)
            logger.debug(
                "Checkpoint saved: run=%s step=%d history=%d",
                ctx.run_id, ctx.step_count, len(history),
            )
        except Exception as exc:
            logger.debug("Checkpoint save failed: %s", exc)

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------

    async def _emit_event(self, event: StepEvent) -> None:
        """Publish a StepEvent to the event bus and message bus."""
        if self._event_bus is not None:
            try:
                await _safe_call(self._event_bus.publish, event)
            except Exception as exc:
                logger.debug("event_bus.publish failed: %s", exc)
        if self._message_bus is not None:
            try:
                await _safe_call(self._message_bus.publish, event)
            except Exception as exc:
                logger.debug("message_bus.publish failed: %s", exc)

    async def _record_failure(self, ctx: AgentContext, exc: Exception) -> None:
        """Record a failure in the failure tracker and potentially DLQ."""
        if self._failure_tracker is None:
            return
        try:
            failure = StepFailure.from_exception(
                exc,
                run_id=ctx.run_id,
                step_number=ctx.step_count,
                agent_type=self.agent_type,
                failure_class=self._classify_exception(exc),
            )
            await _safe_call(self._failure_tracker.record, failure)
        except Exception as track_exc:
            logger.debug("failure_tracker.record failed: %s", track_exc)

    async def _audit(
        self, ctx: AgentContext, call: ToolCall, result: ToolResult
    ) -> None:
        """Fire-and-forget audit log entry for a tool call."""
        if self._audit_logger is None:
            return
        try:
            await _safe_call(
                self._audit_logger.log,
                event_type="tool_call",
                run_id=ctx.run_id,
                tenant_id=ctx.tenant_id,
                payload={
                    "tool_name": call.name,
                    "tool_id": call.id,
                    "is_error": result.is_error,
                },
            )
        except Exception as exc:
            logger.debug("audit_logger.log failed: %s", exc)

    async def _log_llm_span(self, ctx: AgentContext, response: LLMResponse) -> None:
        """Record an MLflow LLM span for this response."""
        if self._mlflow_tracer is None:
            return
        try:
            await _safe_call(
                self._mlflow_tracer.log_llm_call,
                run_id=ctx.run_id,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except Exception as exc:
            logger.debug("mlflow_tracer.log_llm_call failed: %s", exc)

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _mlflow_context(self, ctx: AgentContext):  # type: ignore[return]
        """Start an MLflow agent run if tracer is available."""
        mlflow_run_id = None
        if self._mlflow_tracer is not None:
            try:
                async with self._mlflow_tracer.agent_run(ctx) as run:
                    yield getattr(run, "info", {}).get("run_id") if run else None
                    return
            except Exception as exc:
                logger.debug("MLflow agent_run context failed: %s", exc)
        yield mlflow_run_id

    @asynccontextmanager
    async def _llm_span(self, ctx: AgentContext):  # type: ignore[return]
        """Open an OTel span + TraceRecorder LLM span for the LLM call."""
        # TraceRecorder span (durable)
        llm_span_id = await self._start_trace_span(
            ctx, SpanKind.LLM, "llm:call",
        )
        try:
            # OTel span (live export)
            if self._step_tracer is not None:
                try:
                    with self._step_tracer.span("llm_call", ctx):
                        yield llm_span_id
                        return
                except Exception as exc:
                    logger.debug("StepTracer.span failed: %s", exc)
            yield llm_span_id
        finally:
            await self._end_trace_span(ctx, llm_span_id)

    # ------------------------------------------------------------------
    # TraceRecorder helpers — no-op when recorder is None
    # ------------------------------------------------------------------

    async def _start_trace_span(
        self,
        ctx: AgentContext,
        kind: SpanKind,
        name: str,
        input_preview: str = "",
    ) -> str | None:
        if self._trace_recorder is None:
            return None
        try:
            return await self._trace_recorder.start_span(
                run_id=ctx.run_id,
                kind=kind,
                name=name,
                ctx=ctx,
                input_preview=input_preview,
            )
        except Exception as exc:
            logger.debug("TraceRecorder.start_span failed: %s", exc)
            return None

    async def _end_trace_span(
        self,
        ctx: AgentContext,
        span_id: str | None,
        status: SpanStatus = SpanStatus.OK,
        output_preview: str = "",
        error: str | None = None,
    ) -> None:
        if self._trace_recorder is None or span_id is None:
            return
        try:
            await self._trace_recorder.end_span(
                run_id=ctx.run_id,
                span_id=span_id,
                status=status,
                output_preview=output_preview,
                error=error,
            )
        except Exception as exc:
            logger.debug("TraceRecorder.end_span failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs); await if coroutine. Never raises."""
    import asyncio
    try:
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception as exc:
        logger.debug("_safe_call(%s) raised: %s", getattr(fn, "__name__", fn), exc)
        return None


def _get_metrics() -> Any:
    """Return the HarnessMetrics instance without failing if unavailable."""
    try:
        from harness.observability.metrics import get_prometheus_metrics
        return get_prometheus_metrics()
    except Exception:
        return None


def _utcnow():
    return datetime.now(UTC)


def _feedback_step_event(ctx: AgentContext, event: Any) -> StepEvent:
    """Wrap a FeedbackEvent as a StepEvent for the event bus."""
    return StepEvent(
        run_id=ctx.run_id,
        step=ctx.step_count,
        event_type="feedback_applied",
        payload={
            "feedback_id": event.feedback_id,
            "type": event.type,
            "source": event.source,
            "priority": event.priority,
            "content_preview": (event.content or "")[:200],
            "score": event.score,
        },
        timestamp=_utcnow(),
    )
