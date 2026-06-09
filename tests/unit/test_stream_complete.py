"""Regression tests for stream-with-tools (`stream_complete`).

Covers:
- Anthropic provider: text deltas via on_text, tool_use accumulation, EXACT
  usage from message_start + message_delta (never len//4).
- OpenAI provider: tool_call delta accumulation + exact usage from the
  include_usage trailing chunk.
- Router: forwards on_text, returns the LLMResponse, falls back only BEFORE
  the first token, never implements stream for providers lacking it.
- BaseAgent._call_llm: streams when stream_tokens is set (even WITH tools),
  emits token_delta events with the agreed payload, uses real usage, and falls
  back to complete() when stream_complete is unavailable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.core.context import LLMResponse, ToolCall
from harness.core.errors import FailureClass, LLMError
from harness.llm.anthropic import AnthropicProvider
from harness.llm.openai_provider import OpenAIProvider
from harness.llm.router import LLMRouter


# ---------------------------------------------------------------------------
# Fake SDK event/chunk builders + async-iterable stream contexts
# ---------------------------------------------------------------------------


def _ns(**kw):
    """Lightweight attribute bag (SimpleNamespace-like, but allows .get-free)."""
    obj = MagicMock()
    for k, v in kw.items():
        setattr(obj, k, v)
    return obj


class _FakeAnthropicStream:
    """Async context manager whose `async for` yields raw Anthropic events."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for e in self._events:
            yield e


class _FakeOpenAIStream:
    """Async-iterable returned by chat.completions.create(stream=True)."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


def _seed_anthropic_exc(p: AnthropicProvider) -> None:
    p._exc_rate_limit = type("RL", (Exception,), {})
    p._exc_timeout = type("TO", (Exception,), {})
    p._exc_connection = type("CE", (Exception,), {})
    p._exc_status = type("SE", (Exception,), {"status_code": 500})


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_stream_complete_text_and_exact_usage():
    p = AnthropicProvider(api_key="k", model="claude-sonnet-4-6")
    events = [
        _ns(type="message_start",
            message=_ns(model="claude-sonnet-4-6",
                        usage=_ns(input_tokens=123, cache_read_input_tokens=0))),
        _ns(type="content_block_delta",
            index=0, delta=_ns(type="text_delta", text="Hel")),
        _ns(type="content_block_delta",
            index=0, delta=_ns(type="text_delta", text="lo")),
        _ns(type="message_delta", usage=_ns(output_tokens=45)),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    p._client = client
    _seed_anthropic_exc(p)

    seen: list[str] = []
    resp = await p.stream_complete(
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
        on_text=lambda d: seen.append(d),
    )

    assert seen == ["Hel", "lo"]
    assert resp.content == "Hello"
    # EXACT provider usage — not len("Hello")//4
    assert resp.input_tokens == 123
    assert resp.output_tokens == 45
    assert resp.tool_calls == []
    assert resp.provider == "anthropic"


@pytest.mark.asyncio
async def test_anthropic_stream_complete_accumulates_tool_calls():
    p = AnthropicProvider(api_key="k", model="claude-sonnet-4-6")
    events = [
        _ns(type="message_start",
            message=_ns(model="claude-sonnet-4-6",
                        usage=_ns(input_tokens=10, cache_read_input_tokens=0))),
        _ns(type="content_block_start", index=0,
            content_block=_ns(type="tool_use", id="tu_1", name="search")),
        _ns(type="content_block_delta", index=0,
            delta=_ns(type="input_json_delta", partial_json='{"q": "wid')),
        _ns(type="content_block_delta", index=0,
            delta=_ns(type="input_json_delta", partial_json='gets"}')),
        _ns(type="message_delta", usage=_ns(output_tokens=7)),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    p._client = client
    _seed_anthropic_exc(p)

    resp = await p.stream_complete([{"role": "user", "content": "x"}], max_tokens=50)

    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "tu_1"
    assert tc.name == "search"
    assert tc.args == {"q": "widgets"}
    assert resp.input_tokens == 10 and resp.output_tokens == 7


@pytest.mark.asyncio
async def test_anthropic_stream_complete_supports_async_callback():
    p = AnthropicProvider(api_key="k", model="claude-sonnet-4-6")
    events = [
        _ns(type="message_start", message=_ns(model="m", usage=_ns(input_tokens=1, cache_read_input_tokens=0))),
        _ns(type="content_block_delta", index=0, delta=_ns(type="text_delta", text="yo")),
        _ns(type="message_delta", usage=_ns(output_tokens=2)),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    p._client = client
    _seed_anthropic_exc(p)

    seen: list[str] = []

    async def _cb(d):
        seen.append(d)

    resp = await p.stream_complete([{"role": "user", "content": "x"}], max_tokens=10, on_text=_cb)
    assert seen == ["yo"] and resp.content == "yo"


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _seed_openai_exc(p: OpenAIProvider) -> None:
    p._exc_rate_limit = type("RL", (Exception,), {})
    p._exc_timeout = type("TO", (Exception,), {})
    p._exc_connection = type("CE", (Exception,), {})
    p._exc_status = type("SE", (Exception,), {})


@pytest.mark.asyncio
async def test_openai_stream_complete_text_tools_and_exact_usage():
    p = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    chunks = [
        _ns(model="gpt-4o-mini",
            choices=[_ns(delta=_ns(content="Hi", tool_calls=None))],
            usage=None),
        _ns(model="gpt-4o-mini",
            choices=[_ns(delta=_ns(content=None, tool_calls=[
                _ns(index=0, id="call_1", function=_ns(name="lookup", arguments='{"id":'))
            ]))],
            usage=None),
        _ns(model="gpt-4o-mini",
            choices=[_ns(delta=_ns(content=None, tool_calls=[
                _ns(index=0, id=None, function=_ns(name=None, arguments='42}'))
            ]))],
            usage=None),
        # Trailing usage-only chunk (stream_options include_usage)
        _ns(model="gpt-4o-mini", choices=[],
            usage=_ns(prompt_tokens=88, completion_tokens=12, prompt_tokens_details=None)),
    ]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_FakeOpenAIStream(chunks))
    p._client = client
    _seed_openai_exc(p)

    seen: list[str] = []
    resp = await p.stream_complete(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
        tools=[{"name": "lookup", "description": "", "input_schema": {}}],
        on_text=lambda d: seen.append(d),
    )

    assert seen == ["Hi"]
    assert resp.content == "Hi"
    # EXACT usage from the include_usage chunk
    assert resp.input_tokens == 88
    assert resp.output_tokens == 12
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].name == "lookup"
    assert resp.tool_calls[0].args == {"id": 42}

    # include_usage was actually requested
    _, called_kwargs = client.chat.completions.create.call_args
    assert called_kwargs["stream"] is True
    assert called_kwargs["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _provider_with_stream_complete(name, model, *, healthy=True, resp=None,
                                   tokens=None, exc=None, raise_before_text=False):
    """Mock provider exposing stream_complete; emits tokens then returns resp."""
    p = MagicMock()
    p.provider_name = name
    p.model = model
    p.health_check = AsyncMock(return_value=healthy)

    async def _sc(messages, *, max_tokens, system=None, tools=None, on_text=None, **kw):
        if exc is not None and raise_before_text:
            raise exc
        for t in (tokens or []):
            if on_text is not None:
                r = on_text(t)
                if hasattr(r, "__await__"):
                    await r
        if exc is not None and not raise_before_text:
            raise exc
        return resp or LLMResponse(content="".join(tokens or []), tool_calls=[],
                                   input_tokens=5, output_tokens=6,
                                   model=model, provider=name)

    p.stream_complete = _sc
    return p


@pytest.mark.asyncio
async def test_router_stream_complete_forwards_on_text_and_returns_response():
    p = _provider_with_stream_complete("primary", "m1", tokens=["a", "b", "c"])
    router = LLMRouter()
    router.register(p, priority=0)

    seen: list[str] = []
    resp = await router.stream_complete(
        [{"role": "user", "content": "hi"}], max_tokens=10,
        on_text=lambda d: seen.append(d),
    )
    assert seen == ["a", "b", "c"]
    assert resp.content == "abc"
    assert resp.input_tokens == 5 and resp.output_tokens == 6


@pytest.mark.asyncio
async def test_router_stream_complete_falls_back_before_first_token():
    p1 = _provider_with_stream_complete(
        "primary", "m1", raise_before_text=True,
        exc=LLMError("429", failure_class=FailureClass.LLM_RATE_LIMIT),
    )
    p2 = _provider_with_stream_complete("fallback", "m2", tokens=["ok"])
    router = LLMRouter()
    router.register(p1, priority=0)
    router.register(p2, priority=1)

    seen: list[str] = []
    resp = await router.stream_complete(
        [{"role": "user", "content": "hi"}], max_tokens=10,
        on_text=lambda d: seen.append(d),
    )
    assert seen == ["ok"]
    assert resp.provider == "fallback"


@pytest.mark.asyncio
async def test_router_stream_complete_no_fallback_after_first_token():
    p1 = _provider_with_stream_complete(
        "primary", "m1", tokens=["par", "tial"],
        exc=LLMError("boom", failure_class=FailureClass.LLM_ERROR),
    )
    p2 = _provider_with_stream_complete("fallback", "m2", tokens=["SHOULD_NOT"])
    router = LLMRouter()
    router.register(p1, priority=0)
    router.register(p2, priority=1)

    seen: list[str] = []
    with pytest.raises(LLMError):
        await router.stream_complete(
            [{"role": "user", "content": "hi"}], max_tokens=10,
            on_text=lambda d: seen.append(d),
        )
    # Partial output streamed, but the fallback's tokens never leaked in.
    assert seen == ["par", "tial"]


@pytest.mark.asyncio
async def test_router_stream_complete_skips_providers_without_capability():
    # Provider WITHOUT stream_complete must be skipped, not crash.
    no_cap = MagicMock(spec=["provider_name", "model", "health_check", "complete", "stream"])
    no_cap.provider_name = "nocap"
    no_cap.model = "m0"
    no_cap.health_check = AsyncMock(return_value=True)
    cap = _provider_with_stream_complete("cap", "m1", tokens=["hi"])

    router = LLMRouter()
    router.register(no_cap, priority=0)
    router.register(cap, priority=1)

    resp = await router.stream_complete([{"role": "user", "content": "x"}], max_tokens=10)
    assert resp.provider == "cap"


@pytest.mark.asyncio
async def test_router_stream_complete_raises_when_none_support_it():
    no_cap = MagicMock(spec=["provider_name", "model", "health_check", "complete", "stream"])
    no_cap.provider_name = "nocap"
    no_cap.model = "m0"
    no_cap.health_check = AsyncMock(return_value=True)
    router = LLMRouter()
    router.register(no_cap, priority=0)

    with pytest.raises(LLMError):
        await router.stream_complete([{"role": "user", "content": "x"}], max_tokens=10)


# ---------------------------------------------------------------------------
# BaseAgent._call_llm gating + token_delta emission + fallback
# ---------------------------------------------------------------------------


def _make_streaming_agent(router, tools_format):
    from harness.agents.base import BaseAgent

    tool_registry = MagicMock()
    tool_registry.to_anthropic_format = MagicMock(return_value=tools_format)

    event_bus = MagicMock()
    event_bus.publish = AsyncMock()

    agent = BaseAgent(
        llm_router=router,
        memory_manager=MagicMock(),
        tool_registry=tool_registry,
        safety_pipeline=None,
        step_tracer=None,
        mlflow_tracer=None,
        failure_tracker=MagicMock(),
        audit_logger=None,
        event_bus=event_bus,
        cost_tracker=None,
        checkpoint_manager=None,
    )
    return agent, event_bus


@pytest.mark.asyncio
async def test_base_streams_with_tools_and_emits_token_delta(agent_context):
    """Key fix: streaming is enabled even WHEN tools are registered, and emits
    token_delta events with the agreed {"text", "step"} payload + REAL usage."""
    tools_format = [{"name": "echo", "description": "", "input_schema": {}}]

    captured_on_text = {}

    async def _stream_complete(*, messages, system, tools, max_tokens, tenant_id, on_text):
        # Tools ARE forwarded — proves the no-tools gate was dropped.
        assert tools == tools_format
        for piece in ["Wor", "king"]:
            await on_text(piece)
        return LLMResponse(
            content="Working", tool_calls=[ToolCall(id="c1", name="echo", args={})],
            input_tokens=200, output_tokens=17, model="m", provider="mock",
        )

    router = MagicMock()
    router.stream_complete = AsyncMock(side_effect=_stream_complete)
    router.complete = AsyncMock()  # must NOT be used

    agent, event_bus = _make_streaming_agent(router, tools_format)
    ctx = agent_context()
    ctx.metadata["stream_tokens"] = True

    resp = await agent._call_llm(ctx, [{"role": "user", "content": "go"}], "sys")

    # Exact usage, not len//4
    assert resp.input_tokens == 200 and resp.output_tokens == 17
    assert resp.tool_calls and resp.tool_calls[0].name == "echo"
    router.complete.assert_not_called()

    # token_delta events emitted with the agreed payload shape
    token_events = [
        c.args[0] for c in event_bus.publish.call_args_list
        if c.args and getattr(c.args[0], "event_type", None) == "token_delta"
    ]
    assert [e.payload["text"] for e in token_events] == ["Wor", "king"]
    assert all("step" in e.payload for e in token_events)


@pytest.mark.asyncio
async def test_base_falls_back_to_complete_when_stream_complete_missing(agent_context):
    """When the router has no stream_complete, _call_llm uses complete()."""
    router = MagicMock(spec=["complete"])
    router.complete = AsyncMock(return_value=LLMResponse(
        content="done", tool_calls=[], input_tokens=3, output_tokens=4,
        model="m", provider="mock",
    ))
    agent, _ = _make_streaming_agent(router, [])
    ctx = agent_context()
    ctx.metadata["stream_tokens"] = True

    resp = await agent._call_llm(ctx, [{"role": "user", "content": "go"}], "sys")
    assert resp.content == "done"
    router.complete.assert_called_once()


@pytest.mark.asyncio
async def test_base_falls_back_when_stream_complete_raises(agent_context):
    """stream_complete raising (e.g. before first token) → fall back to complete()."""
    router = MagicMock()
    router.stream_complete = AsyncMock(
        side_effect=LLMError("nope", failure_class=FailureClass.LLM_ERROR)
    )
    router.complete = AsyncMock(return_value=LLMResponse(
        content="recovered", tool_calls=[], input_tokens=1, output_tokens=2,
        model="m", provider="mock",
    ))
    agent, _ = _make_streaming_agent(router, [])
    ctx = agent_context()
    ctx.metadata["stream_tokens"] = True

    resp = await agent._call_llm(ctx, [{"role": "user", "content": "go"}], "sys")
    assert resp.content == "recovered"
    router.complete.assert_called_once()


@pytest.mark.asyncio
async def test_base_no_streaming_when_flag_off(agent_context):
    """Default behaviour preserved: without stream_tokens, complete() is used."""
    router = MagicMock()
    router.stream_complete = AsyncMock()
    router.complete = AsyncMock(return_value=LLMResponse(
        content="plain", tool_calls=[], input_tokens=1, output_tokens=1,
        model="m", provider="mock",
    ))
    agent, _ = _make_streaming_agent(router, [])
    ctx = agent_context()  # stream_tokens not set

    resp = await agent._call_llm(ctx, [{"role": "user", "content": "go"}], "sys")
    assert resp.content == "plain"
    router.stream_complete.assert_not_called()
    router.complete.assert_called_once()
