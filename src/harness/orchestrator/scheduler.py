"""Scheduler: execute a TaskPlan with parallel execution of independent tasks."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from harness.core.context import AgentResult, StepEvent
from harness.orchestrator.planner import SubTask, TaskPlan

logger = logging.getLogger(__name__)


class Scheduler:
    """Execute a TaskPlan with maximal parallelism.

    Independent tasks (no dependency relationship) are executed concurrently
    with asyncio.gather(). As tasks complete their results are passed to
    dependents via metadata.

    Production features:
    - Back-pressure via max_concurrent semaphore
    - Per-subtask retry (max_retries attempts on transient failure)
    - Dependency-failure propagation: tasks whose deps failed are skipped
    """

    def __init__(
        self,
        agent_runner: Any,
        message_bus: Any | None = None,
        max_concurrent: int = 10,
        max_retries: int = 1,
        memory_manager: Any | None = None,
        child_context_budget: int = 8_000,
        redis: Any | None = None,
    ) -> None:
        """
        Args:
            agent_runner:          AgentRunner exposing create_run() / execute_run().
            message_bus:           Optional AgentMessageBus for plan events.
            max_concurrent:        Max parallel subtasks (back-pressure semaphore).
            max_retries:           Retry attempts per subtask on transient failure.
            memory_manager:        Optional MemoryManager.  When provided the scheduler
                                   slices parent context into each child's hot window
                                   before execution and injects the child's result back
                                   into the parent after completion.
            child_context_budget:  Token budget given to each child agent from the
                                   parent's context (default 8 000 tokens).
            redis:                 Optional async Redis client.  When provided, the
                                   Scheduler creates an AgentBlackboard per plan so
                                   agents can read typed predecessor artifacts instead
                                   of only receiving text summaries.
        """
        self._runner = agent_runner
        self._message_bus = message_bus
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._memory = memory_manager
        self._child_budget = child_context_budget
        self._redis = redis

    async def execute_plan(
        self,
        plan: TaskPlan,
        tenant_id: str,
        timeout: float = 300.0,
        parent_run_id: str | None = None,
    ) -> dict[str, AgentResult]:
        """Execute a TaskPlan and return results keyed by SubTask.id.

        Algorithm:
        1. Validate the DAG and compute topological order.
        2. Maintain a set of completed task ids and their results.
        3. In each iteration, find all tasks ready to run (deps satisfied),
           execute them in parallel with asyncio.gather, then add to completed.
        4. Repeat until all tasks are done or timeout.

        Args:
            plan:      The TaskPlan to execute.
            tenant_id: The tenant executing this plan.
            timeout:   Total wall-clock timeout for the entire plan (seconds).

        Returns:
            Dict mapping subtask_id to AgentResult.
        """
        errors = plan.validate()
        if errors:
            raise ValueError(f"Cannot execute invalid plan: {errors}")

        completed_ids: set[str] = set()
        failed_ids: set[str] = set()           # subtasks that failed after all retries
        results: dict[str, Any] = {}
        all_ids = {st.id for st in plan.subtasks}
        semaphore = asyncio.Semaphore(self._max_concurrent)

        # Create a per-plan blackboard when Redis is available
        blackboard = None
        if self._redis is not None:
            from harness.orchestrator.blackboard import AgentBlackboard
            blackboard = AgentBlackboard(self._redis, plan_id=plan.plan_id)
            logger.debug("Blackboard created for plan %s", plan.plan_id[:8])

        logger.info(
            "Starting plan %s execution: %d tasks, tenant=%s",
            plan.plan_id,
            len(plan.subtasks),
            tenant_id,
        )
        await self._publish("plan_started", plan.plan_id, {"total_tasks": len(plan.subtasks)})

        try:
            async with asyncio.timeout(timeout):
                while completed_ids != all_ids:
                    ready = plan.get_ready_tasks(completed_ids)

                    # Propagate dependency failures: skip tasks whose deps failed
                    skippable = [
                        st for st in ready
                        if any(dep in failed_ids for dep in st.depends_on)
                    ]
                    for st in skippable:
                        failed_deps = [d for d in st.depends_on if d in failed_ids]
                        logger.warning(
                            "Plan %s: skipping '%s' — failed deps: %s",
                            plan.plan_id,
                            st.id,
                            failed_deps,
                        )
                        from harness.core.context import AgentResult
                        results[st.id] = AgentResult(
                            run_id=f"skipped_{st.id}",
                            output="",
                            steps=0,
                            tokens=0,
                            success=False,
                            failure_class="UNKNOWN",
                            error_message=f"Skipped: dependency {failed_deps} failed",
                        )
                        completed_ids.add(st.id)
                        failed_ids.add(st.id)
                        await self._publish(
                            "subtask_skipped",
                            plan.plan_id,
                            {"subtask_id": st.id, "failed_deps": failed_deps},
                        )
                    ready = [st for st in ready if st not in skippable]

                    if not ready:
                        if completed_ids == all_ids:
                            break
                        remaining = [st.id for st in plan.subtasks if st.id not in completed_ids]
                        logger.error(
                            "Plan %s deadlocked: no ready tasks, remaining=%s",
                            plan.plan_id,
                            remaining,
                        )
                        raise RuntimeError(
                            f"Plan deadlock: no tasks ready to execute. "
                            f"Remaining: {remaining}"
                        )

                    logger.info(
                        "Plan %s: executing %d tasks in parallel: %s",
                        plan.plan_id,
                        len(ready),
                        [st.id for st in ready],
                    )

                    # Execute all ready tasks concurrently, respecting back-pressure
                    batch_results = await asyncio.gather(
                        *[
                            self._execute_subtask_with_retry(
                                st, tenant_id, results, semaphore, parent_run_id,
                                blackboard=blackboard,
                            )
                            for st in ready
                        ],
                        return_exceptions=True,
                    )

                    for st, result in zip(ready, batch_results, strict=True):
                        if isinstance(result, Exception):
                            logger.error(
                                "Plan %s sub-task '%s' failed after retries: %s",
                                plan.plan_id,
                                st.id,
                                result,
                            )
                            from harness.core.context import AgentResult
                            agent_result = AgentResult(
                                run_id=f"failed_{st.id}",
                                output="",
                                steps=0,
                                tokens=0,
                                success=False,
                                failure_class="UNKNOWN",
                                error_message=str(result),
                            )
                            results[st.id] = agent_result
                            failed_ids.add(st.id)
                            if blackboard is not None:
                                await blackboard.write(
                                    st.id, "error", str(result),
                                    metadata={"success": False},
                                )
                        else:
                            results[st.id] = result
                            if not getattr(result, "success", True):
                                failed_ids.add(st.id)
                            # Write structured artifacts to blackboard for downstream agents
                            if blackboard is not None:
                                await _write_result_to_blackboard(
                                    blackboard, st.id, result
                                )

                        completed_ids.add(st.id)
                        await self._publish(
                            "subtask_completed",
                            plan.plan_id,
                            {
                                "subtask_id": st.id,
                                "success": (
                                    results[st.id].success
                                    if hasattr(results[st.id], "success")
                                    else False
                                ),
                            },
                        )

        except TimeoutError:
            logger.error("Plan %s timed out after %.1f seconds", plan.plan_id, timeout)
            await self._publish(
                "plan_timeout", plan.plan_id, {"completed": list(completed_ids)}
            )
            raise

        succeeded = sum(1 for r in results.values() if getattr(r, "success", False))
        logger.info(
            "Plan %s completed: %d/%d tasks succeeded",
            plan.plan_id,
            succeeded,
            len(results),
        )
        await self._publish(
            "plan_completed",
            plan.plan_id,
            {"total": len(results), "succeeded": succeeded},
        )

        return results

    async def _execute_subtask_with_retry(
        self,
        subtask: SubTask,
        tenant_id: str,
        predecessor_results: dict[str, Any],
        semaphore: asyncio.Semaphore,
        parent_run_id: str | None = None,
        blackboard: Any | None = None,
    ) -> Any:
        """Execute a subtask with back-pressure and retry on failure."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            async with semaphore:
                try:
                    result = await self._execute_subtask(
                        subtask, tenant_id, predecessor_results, parent_run_id,
                        blackboard=blackboard,
                    )
                    if getattr(result, "success", True):
                        return result
                    # Agent ran but reported failure — retry if attempts remain
                    last_exc = RuntimeError(
                        getattr(result, "error_message", "subtask returned success=False")
                    )
                    if attempt < self._max_retries:
                        logger.warning(
                            "Plan subtask '%s' failed (attempt %d/%d), retrying...",
                            subtask.id,
                            attempt + 1,
                            self._max_retries + 1,
                        )
                        await asyncio.sleep(2 ** attempt)  # brief exponential back-off
                    else:
                        return result  # return the failed result after exhausting retries
                except Exception as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        logger.warning(
                            "Plan subtask '%s' raised (attempt %d/%d): %s — retrying...",
                            subtask.id,
                            attempt + 1,
                            self._max_retries + 1,
                            exc,
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

        raise last_exc  # type: ignore[misc]

    async def _execute_subtask(
        self,
        subtask: SubTask,
        tenant_id: str,
        predecessor_results: dict[str, Any],
        parent_run_id: str | None = None,
        blackboard: Any | None = None,
    ) -> Any:
        """Execute a single SubTask, enriching its context with predecessor outputs.

        When a ``memory_manager`` is configured and ``parent_run_id`` is
        provided, the parent's relevant context is sliced into the child's
        hot window before execution so the child agent does not start blind.
        After execution the child's result is injected back into the parent's
        context as a compressed tool message.

        Args:
            subtask:             The sub-task to run.
            tenant_id:           Owning tenant.
            predecessor_results: Results from previously completed tasks.
            parent_run_id:       Run ID of the orchestrating parent agent.

        Returns:
            AgentResult for this subtask.
        """
        # Build enriched task description with predecessor outputs
        task_with_context = _build_task_with_context(subtask, predecessor_results)

        # Append structured blackboard artifacts from direct predecessors
        if blackboard is not None and subtask.depends_on:
            try:
                bb_context = await blackboard.format_for_context(
                    subtask_ids=subtask.depends_on,
                )
                if bb_context:
                    task_with_context = f"{task_with_context}\n\n{bb_context}"
            except Exception as exc:
                logger.debug("Blackboard context read failed (non-fatal): %s", exc)

        # Build metadata for this run
        metadata: dict[str, Any] = dict(subtask.metadata)
        metadata["plan_subtask_id"] = subtask.id
        metadata["depends_on"] = subtask.depends_on
        metadata["handoff_count"] = len(subtask.depends_on)
        if blackboard is not None:
            metadata["blackboard"] = blackboard

        # Include predecessor result summaries
        for dep_id in subtask.depends_on:
            dep_result = predecessor_results.get(dep_id)
            if dep_result is not None:
                metadata[f"predecessor_{dep_id}_output"] = _summarise_result(dep_result)

        logger.info(
            "Executing sub-task '%s' (agent_type=%s)", subtask.id, subtask.agent_type
        )

        try:
            record = await self._runner.create_run(
                tenant_id=tenant_id,
                agent_type=subtask.agent_type,
                task=task_with_context,
                metadata=metadata,
            )
            child_run_id = record.run_id

            # ── Slice parent context into child's hot window ─────────────
            if self._memory is not None and parent_run_id:
                try:
                    await self._memory.slice_for_subagent(
                        parent_run_id=parent_run_id,
                        child_run_id=child_run_id,
                        task=subtask.task,
                        token_budget=self._child_budget,
                        skill_ns=subtask.agent_type,
                    )
                    logger.debug(
                        "Sliced parent context %s → child %s (budget=%d tokens)",
                        parent_run_id[:8], child_run_id[:8], self._child_budget,
                    )
                except Exception as exc:
                    logger.debug("Context slicing failed (continuing): %s", exc)

            updated_record = await self._runner.execute_run(child_run_id)

            # ── Inject child result back into parent context ──────────────
            if self._memory is not None and parent_run_id:
                try:
                    result_data = updated_record.result or {}
                    child_output = result_data.get("output", "")
                    if child_output:
                        summary = (
                            f"SubTask '{subtask.id}' ({subtask.agent_type}): "
                            + child_output[:800]
                        )
                        await self._memory.inject_subagent_result(
                            parent_run_id=parent_run_id,
                            child_run_id=child_run_id,
                            result_summary=summary,
                            skill_ns=subtask.agent_type,
                        )
                        logger.debug(
                            "Injected child %s result into parent %s",
                            child_run_id[:8], parent_run_id[:8],
                        )
                except Exception as exc:
                    logger.debug("Result injection failed (non-fatal): %s", exc)

            # Extract AgentResult from the record
            result_data = updated_record.result or {}
            from harness.core.context import AgentResult

            return AgentResult(
                run_id=result_data.get("run_id", record.run_id),
                output=result_data.get("output", ""),
                steps=result_data.get("steps", 0),
                tokens=result_data.get("tokens", 0),
                success=result_data.get("success", updated_record.status == "completed"),
                failure_class=result_data.get("failure_class"),
                error_message=result_data.get("error_message"),
                elapsed_seconds=result_data.get("elapsed_seconds", 0.0),
                cost_usd=result_data.get("cost_usd", 0.0),
                tool_calls=result_data.get("tool_calls", 0),
                tool_errors=result_data.get("tool_errors", 0),
                guardrail_hits=result_data.get("guardrail_hits", 0),
                handoff_count=len(subtask.depends_on),
                cache_hits=result_data.get("cache_hits", 0),
                cache_read_tokens=result_data.get("cache_read_tokens", 0),
            )

        except Exception as exc:
            logger.exception(
                "Sub-task '%s' execution raised: %s", subtask.id, exc
            )
            raise

    async def _publish(
        self, event_type: str, plan_id: str, payload: dict[str, Any]
    ) -> None:
        """Publish a plan-level event to the message bus."""
        if self._message_bus is None:
            return
        try:
            event = StepEvent(
                run_id=f"plan:{plan_id}",
                step=0,
                event_type=event_type,
                payload={"plan_id": plan_id, **payload},
                timestamp=datetime.now(UTC),
            )
            result = self._message_bus.publish(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Scheduler._publish failed: %s", exc)


def _build_task_with_context(
    subtask: SubTask, predecessor_results: dict[str, Any]
) -> str:
    """Enrich a sub-task description with predecessor outputs."""
    task = subtask.task
    if not subtask.depends_on:
        return task

    context_parts: list[str] = []
    for dep_id in subtask.depends_on:
        dep_result = predecessor_results.get(dep_id)
        if dep_result is None:
            continue
        output = _summarise_result(dep_result)
        if output:
            context_parts.append(f"[Output from {dep_id}]: {output}")

    if context_parts:
        context_str = "\n".join(context_parts)
        task = f"{task}\n\nContext from prerequisite tasks:\n{context_str}"

    return task


async def _write_result_to_blackboard(
    blackboard: Any,
    subtask_id: str,
    result: Any,
) -> None:
    """Write an AgentResult to the blackboard after a subtask completes.

    Stores the primary output as TYPE_OUTPUT plus any typed artifacts the
    agent embedded in its metadata (code, sql, test_result, file_list).
    Failures are logged and swallowed — the main execution must never stall.
    """
    output = getattr(result, "output", "") or ""
    success = getattr(result, "success", False)
    try:
        await blackboard.write(
            subtask_id,
            "output",
            output,
            metadata={
                "success": success,
                "steps": getattr(result, "steps", 0),
                "cost_usd": getattr(result, "cost_usd", 0.0),
            },
        )
        # Extract typed artifacts from result metadata when present
        meta = getattr(result, "metadata", {}) or {}
        for artifact_type in ("code", "sql", "test_result", "file_list", "analysis"):
            artifact = meta.get(f"blackboard_{artifact_type}")
            if artifact:
                await blackboard.write(subtask_id, artifact_type, str(artifact))
    except Exception as exc:
        logger.debug("_write_result_to_blackboard failed: %s", exc)


def _summarise_result(result: Any, max_chars: int = 2000) -> str:
    """Extract a short textual summary from an AgentResult."""
    output = getattr(result, "output", "")
    if not output and isinstance(result, dict):
        output = result.get("output", "")
    if len(output) > max_chars:
        output = output[:max_chars] + "... [truncated]"
    return output
