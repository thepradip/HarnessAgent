"""
Tests for the provider-neutral history format and its per-provider translation.

The agent loop stores assistant turns with a ``tool_calls`` list and tool
results as ``{"role": "tool", "tool_use_id": ...}``. Each provider must
translate that into its own wire format:

- Anthropic: tool_use blocks on the assistant turn, tool_result blocks in a
  *user* turn (consecutive results merged into one user message).
- OpenAI: assistant ``tool_calls`` entries with JSON-string arguments,
  followed by role "tool" messages keyed by ``tool_call_id``.

Also covers the _llm_span fix (body exceptions must propagate unchanged when
a step tracer is configured) and the orphan-safe history split.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness.agents.base import BaseAgent, _orphan_safe_split
from harness.core.context import LLMResponse, ToolCall, ToolResult
from harness.core.errors import FailureClass, LLMError
from harness.llm.anthropic import AnthropicProvider
from harness.llm.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Neutral history used across provider tests
# ---------------------------------------------------------------------------

NEUTRAL_HISTORY = [
    {"role": "user", "content": "List the tables"},
    {
        "role": "assistant",
        "content": "I'll check the schema.",
        "tool_calls": [
            {"id": "call_1", "name": "list_tables", "args": {"schema": "public"}},
            {"id": "call_2", "name": "describe_table", "args": {"table_name": "users"}},
        ],
    },
    {"role": "tool", "tool_use_id": "call_1", "content": "users, orders"},
    {"role": "tool", "tool_use_id": "call_2", "content": "id, name"},
    {"role": "assistant", "content": "There are two tables."},
]


# ---------------------------------------------------------------------------
# Anthropic _map_messages
# ---------------------------------------------------------------------------

def test_anthropic_assistant_tool_calls_become_tool_use_blocks():
    p = AnthropicProvider(api_key="test-key")
    mapped = p._map_messages(NEUTRAL_HISTORY)

    assistant = mapped[1]
    assert assistant["role"] == "assistant"
    types = [b["type"] for b in assistant["content"]]
    assert types == ["text", "tool_use", "tool_use"]
    assert assistant["content"][1]["id"] == "call_1"
    assert assistant["content"][1]["name"] == "list_tables"
    assert assistant["content"][1]["input"] == {"schema": "public"}


def test_anthropic_tool_results_become_user_tool_result_blocks():
    p = AnthropicProvider(api_key="test-key")
    mapped = p._map_messages(NEUTRAL_HISTORY)

    # No "tool" role may survive — Anthropic only accepts user/assistant
    assert all(m["role"] in ("user", "assistant") for m in mapped)
    tool_turn = mapped[2]
    assert tool_turn["role"] == "user"
    assert [b["type"] for b in tool_turn["content"]] == ["tool_result", "tool_result"]
    assert tool_turn["content"][0]["tool_use_id"] == "call_1"
    assert tool_turn["content"][1]["tool_use_id"] == "call_2"


def test_anthropic_merges_consecutive_tool_results_into_one_user_turn():
    p = AnthropicProvider(api_key="test-key")
    mapped = p._map_messages(NEUTRAL_HISTORY)
    # user, assistant(tool_use), user(2 tool_results), assistant
    assert len(mapped) == 4


def test_anthropic_plain_messages_pass_through():
    p = AnthropicProvider(api_key="test-key")
    mapped = p._map_messages(
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert mapped == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ---------------------------------------------------------------------------
# OpenAI _prepare_messages
# ---------------------------------------------------------------------------

def test_openai_translates_assistant_tool_calls_and_tool_messages():
    p = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    prepared = p._prepare_messages(NEUTRAL_HISTORY, system="sys")

    assert prepared[0] == {"role": "system", "content": "sys"}
    assistant = prepared[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "list_tables"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "schema": "public"
    }

    tool_msg = prepared[3]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "users, orders",
    }


def test_openai_reasoning_model_maps_inline_system_to_developer():
    p = OpenAIProvider(api_key="sk-test", model="o1-mini")
    prepared = p._prepare_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "mid-run note"},
        ],
        system=None,
    )
    assert prepared[1]["role"] == "developer"


@pytest.mark.asyncio
async def test_openai_reasoning_model_receives_tools():
    """o-series / gpt-5 models support function calling — tools must be sent."""
    p = OpenAIProvider(api_key="sk-test", model="o1-mini")
    mock_client = MagicMock()
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = []
    resp.choices = [choice]
    resp.usage = None
    mock_client.chat.completions.create = AsyncMock(return_value=resp)
    p._client = mock_client

    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    await p.complete(
        [{"role": "user", "content": "hi"}], max_tokens=10, tools=tools
    )

    request = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" in request
    assert request["tool_choice"] == "auto"
    assert "temperature" not in request


# ---------------------------------------------------------------------------
# Agent loop: history carries tool_calls; splits never orphan tool results
# ---------------------------------------------------------------------------

def test_orphan_safe_split_backs_up_over_tool_results():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_use_id": "1", "content": "r1"},
        {"role": "tool", "tool_use_id": "1", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]
    # A split landing on either tool result must back up to the assistant turn
    assert _orphan_safe_split(history, 2) == 1
    assert _orphan_safe_split(history, 3) == 1
    # Splits on non-tool messages stay put
    assert _orphan_safe_split(history, 4) == 4
    assert _orphan_safe_split(history, 0) == 0


@pytest.mark.asyncio
async def test_agent_history_pairs_tool_calls_with_results(agent_context):
    """The messages sent on the second LLM call must contain the assistant
    tool_calls entry followed by the matching tool result."""
    captured: list[list[dict]] = []

    tool_call = ToolCall(id="tc_1", name="echo", args={"x": 1})

    async def _complete(messages, **kwargs):
        captured.append([dict(m) for m in messages])
        if len(captured) == 1:
            return LLMResponse(
                content="calling tool",
                tool_calls=[tool_call],
                input_tokens=5,
                output_tokens=5,
                model="m",
                provider="p",
            )
        return LLMResponse(
            content="final answer",
            input_tokens=5,
            output_tokens=5,
            model="m",
            provider="p",
        )

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=_complete)
    registry = AsyncMock()
    registry.execute = AsyncMock(return_value=ToolResult(data="echoed"))

    agent = BaseAgent(
        llm_router=router,
        memory_manager=None,
        tool_registry=registry,
        safety_pipeline=None,
        step_tracer=None,
        mlflow_tracer=None,
        failure_tracker=None,
        audit_logger=None,
        event_bus=None,
        cost_tracker=None,
        checkpoint_manager=None,
        trace_recorder=None,
    )
    ctx = agent_context()
    result = await agent.run(ctx)

    assert result.success
    assert len(captured) == 2
    second = captured[1]
    assistant_turns = [m for m in second if m.get("tool_calls")]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["tool_calls"] == [
        {"id": "tc_1", "name": "echo", "args": {"x": 1}}
    ]
    idx = second.index(assistant_turns[0])
    assert second[idx + 1]["role"] == "tool"
    assert second[idx + 1]["tool_use_id"] == "tc_1"
    # No system-role injections mid-history
    assert all(m.get("role") != "system" for m in second)


# ---------------------------------------------------------------------------
# _llm_span: body exceptions propagate unchanged with a step tracer attached
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_error_classified_correctly_with_step_tracer(agent_context):
    """With a step tracer configured, an LLMError from the call must keep its
    failure class instead of being replaced by a generator RuntimeError."""
    router = AsyncMock()
    router.complete = AsyncMock(
        side_effect=LLMError("429", failure_class=FailureClass.LLM_RATE_LIMIT)
    )
    step_tracer = MagicMock()
    step_tracer.span = MagicMock(return_value=contextlib.nullcontext())

    agent = BaseAgent(
        llm_router=router,
        memory_manager=None,
        tool_registry=AsyncMock(),
        safety_pipeline=None,
        step_tracer=step_tracer,
        mlflow_tracer=None,
        failure_tracker=None,
        audit_logger=None,
        event_bus=None,
        cost_tracker=None,
        checkpoint_manager=None,
        trace_recorder=None,
    )
    ctx = agent_context()
    result = await agent.run(ctx)

    assert not result.success
    assert result.failure_class == FailureClass.LLM_RATE_LIMIT.value
    step_tracer.span.assert_called()
