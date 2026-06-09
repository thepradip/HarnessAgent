"""Planner: decompose complex multi-step tasks into a DAG of sub-tasks."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from harness.core.prompt_overrides import gepa_override

logger = logging.getLogger(__name__)

_PLAN_PROMPT_TEMPLATE = """\
You are a task decomposition specialist. Break down the user's task into discrete sub-tasks,
each handled by a specialised agent.

Available agent types: {available_agents}

Task:
{task}

Return a JSON array of sub-task objects. Each object must have:
- "id": a short unique identifier (e.g. "t1", "t2")
- "agent_type": one of the available agent types
- "task": a clear, self-contained description for that agent
- "depends_on": list of sub-task ids that must complete first (empty list if none)
- "metadata": optional dict of additional context

Rules:
1. Tasks that can run in parallel should have no dependency relationship.
2. Pass relevant output from predecessor tasks to dependent tasks via the task description.
3. Keep each sub-task atomic and focused.
4. Maximum 10 sub-tasks.
5. Output ONLY valid JSON — no prose, no code fences.

Example output:
[
  {{"id": "t1", "agent_type": "sql", "task": "Query the orders table to get total sales by region", "depends_on": [], "metadata": {{}}}},
  {{"id": "t2", "agent_type": "code", "task": "Visualise the sales data from t1 as a bar chart", "depends_on": ["t1"], "metadata": {{}}}}
]
"""


@dataclass
class SubTask:
    """A single node in the task execution DAG."""

    id: str
    agent_type: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """A directed acyclic graph of SubTasks derived from a complex task."""

    plan_id: str
    original_task: str
    subtasks: list[SubTask]

    def get_ready_tasks(self, completed_ids: set[str]) -> list[SubTask]:
        """Return sub-tasks whose prerequisites are all in completed_ids."""
        ready = []
        for st in self.subtasks:
            if st.id in completed_ids:
                continue  # already done
            if all(dep in completed_ids for dep in st.depends_on):
                ready.append(st)
        return ready

    def topological_order(self) -> list[SubTask]:
        """Return sub-tasks in a valid topological execution order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {st.id: 0 for st in self.subtasks}
        adjacency: dict[str, list[str]] = {st.id: [] for st in self.subtasks}

        for st in self.subtasks:
            for dep in st.depends_on:
                adjacency.setdefault(dep, []).append(st.id)
                in_degree[st.id] = in_degree.get(st.id, 0) + 1

        # Re-compute correct in-degrees
        for st in self.subtasks:
            in_degree[st.id] = len(st.depends_on)

        queue = [st for st in self.subtasks if in_degree[st.id] == 0]
        ordered: list[SubTask] = []
        id_to_task = {st.id: st for st in self.subtasks}

        while queue:
            # Pick first available (stable order)
            current = queue.pop(0)
            ordered.append(current)
            for neighbour_id in adjacency.get(current.id, []):
                in_degree[neighbour_id] -= 1
                if in_degree[neighbour_id] == 0:
                    queue.append(id_to_task[neighbour_id])

        if len(ordered) != len(self.subtasks):
            raise ValueError(
                f"TaskPlan '{self.plan_id}' contains a cycle — "
                f"processed {len(ordered)}/{len(self.subtasks)} tasks"
            )
        return ordered

    def validate(self) -> list[str]:
        """Return a list of validation errors, or empty list if the plan is valid."""
        errors: list[str] = []
        ids = {st.id for st in self.subtasks}

        for st in self.subtasks:
            for dep in st.depends_on:
                if dep not in ids:
                    errors.append(
                        f"SubTask '{st.id}' depends on unknown id '{dep}'"
                    )

        # Check for cycles
        try:
            self.topological_order()
        except ValueError as exc:
            errors.append(str(exc))

        return errors


class Planner:
    """Uses an LLM to decompose a complex task into a TaskPlan.

    The plan is a DAG of SubTasks, each executed by a specialised agent.
    The Planner validates the DAG for cycles and unknown dependencies
    before returning.
    """

    def __init__(self, llm_provider: Any) -> None:
        """
        Args:
            llm_provider: Any LLMProvider-compatible object with a complete() method.
        """
        self._llm = llm_provider

    async def plan(
        self,
        task: str,
        available_agents: list[str],
        ctx: Any,  # AgentContext — for logging/tracing, not used in LLM call
    ) -> TaskPlan:
        """Decompose task into a validated DAG of SubTasks.

        Args:
            task:             The complex user task to decompose.
            available_agents: List of agent type names that can be assigned.
            ctx:              AgentContext for tracing and metadata.

        Returns:
            A validated TaskPlan.

        Raises:
            ValueError: If the LLM output cannot be parsed or the plan has cycles.
        """
        agents_str = ", ".join(f'"{a}"' for a in available_agents)
        # Multi-agent planning/coordination prompt — optimizable as the
        # "planner_prompt" component (GEPA injects an override via ctx.metadata).
        # Fall back to the default template if an evolved one breaks .format().
        template = gepa_override(ctx, "planner_prompt", _PLAN_PROMPT_TEMPLATE)
        try:
            prompt = template.format(available_agents=agents_str, task=task)
        except (KeyError, IndexError, ValueError):
            logger.warning("planner_prompt override failed to format; using default")
            prompt = _PLAN_PROMPT_TEMPLATE.format(
                available_agents=agents_str,
                task=task,
            )

        logger.info(
            "Planning task (run_id=%s, agents=%s): %.100s",
            getattr(ctx, "run_id", "n/a"),
            agents_str,
            task,
        )

        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            system="You are a task decomposition specialist. Output only valid JSON.",
        )

        plan_text = response.content.strip()
        subtasks = await self._parse_plan(plan_text, available_agents)

        plan_id = uuid.uuid4().hex
        plan = TaskPlan(
            plan_id=plan_id,
            original_task=task,
            subtasks=subtasks,
        )

        # Validate
        errors = plan.validate()
        if errors:
            raise ValueError(
                f"Generated plan has validation errors: {errors}. "
                f"Raw LLM output: {plan_text[:500]}"
            )

        logger.info(
            "Plan %s: %d sub-tasks for task: %.80s",
            plan_id,
            len(subtasks),
            task,
        )
        return plan

    async def _parse_plan(
        self,
        llm_output: str,
        available_agents: list[str],
    ) -> list[SubTask]:
        """Parse the LLM JSON output into a list of SubTask objects.

        Handles code fences and extracts JSON from the response.
        Falls back to a single-subtask plan on parse failure.
        """
        # Strip markdown code fences if present
        text = llm_output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last lines if they are fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            raw_list = json.loads(text)
        except json.JSONDecodeError as exc:
            # Try to find a JSON array anywhere in the text
            import re
            # Greedy match so we capture the FULL outer array. A non-greedy
            # match stops at the first ']' (e.g. inside a subtask's
            # "depends_on": []), yielding invalid JSON for real multi-task plans.
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    raw_list = json.loads(match.group(0))
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Cannot parse plan JSON: {exc}. Text: {text[:300]}"
                    ) from exc
            else:
                raise ValueError(
                    f"No JSON array found in plan output: {text[:300]}"
                ) from exc

        if not isinstance(raw_list, list):
            raise ValueError(
                f"Expected a JSON array, got {type(raw_list).__name__}"
            )

        subtasks: list[SubTask] = []
        seen_ids: set[str] = set()

        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict plan item %d: %s", i, item)
                continue

            # Normalise id
            task_id = str(item.get("id", f"t{i + 1}")).strip()
            if not task_id or task_id in seen_ids:
                task_id = f"t{i + 1}_{uuid.uuid4().hex[:4]}"
            seen_ids.add(task_id)

            # Normalise agent_type
            agent_type = str(item.get("agent_type", "base")).strip().lower()
            if agent_type not in available_agents:
                logger.warning(
                    "Plan task '%s' specifies unknown agent_type '%s'; "
                    "using first available: %s",
                    task_id,
                    agent_type,
                    available_agents[0] if available_agents else "base",
                )
                agent_type = available_agents[0] if available_agents else "base"

            task_desc = str(item.get("task", "")).strip()
            if not task_desc:
                logger.warning("Plan task '%s' has empty task description", task_id)
                task_desc = f"Sub-task {task_id}"

            depends_on = [str(d) for d in item.get("depends_on", [])]
            metadata = dict(item.get("metadata", {}))

            subtasks.append(
                SubTask(
                    id=task_id,
                    agent_type=agent_type,
                    task=task_desc,
                    depends_on=depends_on,
                    metadata=metadata,
                )
            )

        if not subtasks:
            raise ValueError("Plan produced no sub-tasks")

        return subtasks
