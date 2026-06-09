"""Regression tests for Planner JSON-recovery parsing (item 5)."""

from __future__ import annotations

import pytest

from harness.orchestrator.planner import Planner


@pytest.mark.asyncio
async def test_greedy_recovery_parses_full_array_with_empty_depends_on():
    """The recovery regex must capture the FULL array, not stop at the first ']'.

    Regression: a non-greedy r"\\[.*?\\]" stops at the first ']' — which is the
    empty "depends_on": [] of the first subtask — yielding invalid JSON and
    losing every later subtask.
    """
    planner = Planner(llm_provider=None)

    # Wrap the array in prose so the direct json.loads() fails and the
    # regex-recovery fallback path runs.
    llm_output = (
        "Here is the plan you asked for:\n"
        '[\n'
        '  {"id": "t1", "agent_type": "sql", "task": "load data", "depends_on": []},\n'
        '  {"id": "t2", "agent_type": "code", "task": "analyze", "depends_on": ["t1"]}\n'
        ']\n'
        "Let me know if you need changes."
    )

    subtasks = await planner._parse_plan(llm_output, available_agents=["sql", "code"])

    assert len(subtasks) == 2, "non-greedy regex truncated the plan to one subtask"
    assert subtasks[0].id == "t1"
    assert subtasks[1].id == "t2"
    assert subtasks[1].depends_on == ["t1"]
