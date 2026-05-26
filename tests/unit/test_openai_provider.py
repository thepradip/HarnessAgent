"""Unit tests for OpenAIProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.core.context import LLMResponse, ToolCall
from harness.core.errors import FailureClass, LLMError
from harness.llm.openai_provider import OpenAIProvider


def _make_choice(content="Hello", tool_calls=None, finish_reason="stop"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    return choice


def _make_usage(input_tokens=10, output_tokens=20, cached=0):
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    details = MagicMock()
    details.cached_tokens = cached
    usage.prompt_tokens_details = details
    return usage


def _make_response(content="Hello", tool_calls=None, cached=0):
    resp = MagicMock()
    resp.choices = [_make_choice(content, tool_calls)]
    resp.usage = _make_usage(cached=cached)
    return resp


def _seed_mock_client(p: OpenAIProvider) -> MagicMock:
    """Pre-seed a mock AsyncOpenAI client so tests can patch it directly."""
    mock_client = MagicMock()
    p._client = mock_client
    # Seed exception classes (lazy-loaded in production, needed by error handlers)
    from openai import APIConnectionError, APIStatusError, APITimeoutError
    from openai import RateLimitError as OpenAIRateLimitError
    p._exc_rate_limit = OpenAIRateLimitError
    p._exc_timeout = APITimeoutError
    p._exc_connection = APIConnectionError
    p._exc_status = APIStatusError
    return mock_client


@pytest.fixture
def provider():
    p = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    _seed_mock_client(p)
    return p


@pytest.fixture
def o1_provider():
    p = OpenAIProvider(api_key="sk-test", model="o1-mini")
    _seed_mock_client(p)
    return p


@pytest.mark.asyncio
async def test_complete_returns_llm_response(provider):
    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(
        return_value=_make_response("Test response")
    )):
        result = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=100)

    assert isinstance(result, LLMResponse)
    assert result.content == "Test response"
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_complete_parses_tool_calls(provider):
    raw_tc = MagicMock()
    raw_tc.id = "call_abc"
    raw_tc.function.name = "execute_sql"
    raw_tc.function.arguments = '{"sql": "SELECT 1"}'

    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(
        return_value=_make_response("", tool_calls=[raw_tc])
    )):
        result = await provider.complete([{"role": "user", "content": "query"}], max_tokens=100)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "execute_sql"
    assert result.tool_calls[0].args == {"sql": "SELECT 1"}


@pytest.mark.asyncio
async def test_complete_detects_cache_hit(provider):
    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(
        return_value=_make_response("cached", cached=500)
    )):
        result = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=100)

    assert result.cached is True


@pytest.mark.asyncio
async def test_reasoning_model_uses_max_completion_tokens(o1_provider):
    captured = {}

    async def _mock_create(**kwargs):
        captured.update(kwargs)
        return _make_response("reasoning output")

    with patch.object(o1_provider._client.chat.completions, "create", new=_mock_create):
        await o1_provider.complete([{"role": "user", "content": "think"}], max_tokens=500)

    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] == 500


@pytest.mark.asyncio
async def test_reasoning_model_uses_developer_role_for_system(o1_provider):
    captured = {}

    async def _mock_create(**kwargs):
        captured.update(kwargs)
        return _make_response("ok")

    with patch.object(o1_provider._client.chat.completions, "create", new=_mock_create):
        await o1_provider.complete(
            [{"role": "user", "content": "go"}],
            max_tokens=100,
            system="You are a helpful assistant",
        )

    roles = [m["role"] for m in captured["messages"]]
    assert "developer" in roles
    assert "system" not in roles


@pytest.mark.asyncio
async def test_standard_model_uses_system_role(provider):
    captured = {}

    async def _mock_create(**kwargs):
        captured.update(kwargs)
        return _make_response("ok")

    with patch.object(provider._client.chat.completions, "create", new=_mock_create):
        await provider.complete(
            [{"role": "user", "content": "hi"}],
            max_tokens=100,
            system="You are an expert",
        )

    roles = [m["role"] for m in captured["messages"]]
    assert "system" in roles
    assert "developer" not in roles


@pytest.mark.asyncio
async def test_rate_limit_raises_llm_error(provider):
    from openai import RateLimitError

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_request = MagicMock()

    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(
        side_effect=RateLimitError("Rate limit exceeded", response=mock_response, body={})
    )):
        with pytest.raises(LLMError) as exc_info:
            await provider.complete([{"role": "user", "content": "hi"}], max_tokens=100)

    assert exc_info.value.failure_class == FailureClass.LLM_RATE_LIMIT


@pytest.mark.asyncio
async def test_health_check_returns_true_on_success(provider):
    with patch.object(provider._client.models, "list", new=AsyncMock(return_value=MagicMock())):
        assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_returns_false_on_error(provider):
    with patch.object(provider._client.models, "list", new=AsyncMock(side_effect=Exception("down"))):
        assert await provider.health_check() is False


def test_to_openai_tool_converts_anthropic_format(provider):
    anthropic_tool = {
        "name": "execute_sql",
        "description": "Run SQL",
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}}
    }
    result = provider._to_openai_tool(anthropic_tool)
    assert result["type"] == "function"
    assert result["function"]["name"] == "execute_sql"
    assert result["function"]["parameters"] == anthropic_tool["input_schema"]


def test_cost_tracker_knows_openai_models():
    from harness.core.cost_tracker import MODEL_COSTS
    for model in ["gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini", "o1", "o3-mini", "o4-mini"]:
        assert model in MODEL_COSTS, f"{model} missing from MODEL_COSTS"
        assert MODEL_COSTS[model]["input"] >= 0
        assert MODEL_COSTS[model]["output"] >= 0


def test_factory_builds_router_with_openai_only():
    from harness.llm.factory import build_router
    cfg = MagicMock()
    cfg.anthropic_api_key = ""
    cfg.openai_api_key = "sk-test"
    cfg.openai_models = "gpt-4o-mini,gpt-4o"
    cfg.openai_base_url = ""
    cfg.default_model = "gpt-4o-mini"
    cfg.vllm_base_url = ""
    cfg.sglang_base_url = ""
    cfg.llamacpp_base_url = ""
    cfg.hermes_base_url = ""

    router = build_router(cfg)
    assert len(router._config.providers) == 2
    models = {e.provider.model for e in router._config.providers}
    assert "gpt-4o-mini" in models
    assert "gpt-4o" in models


def test_factory_builds_router_with_both_providers():
    from harness.llm.factory import build_router
    cfg = MagicMock()
    cfg.anthropic_api_key = "sk-ant-test"
    cfg.openai_api_key = "sk-openai-test"
    cfg.openai_models = "gpt-4o-mini"
    cfg.openai_base_url = ""
    cfg.default_model = "claude-sonnet-4-6"
    cfg.vllm_base_url = ""
    cfg.sglang_base_url = ""
    cfg.llamacpp_base_url = ""
    cfg.hermes_base_url = ""

    router = build_router(cfg)
    assert len(router._config.providers) == 2
    provider_names = {e.provider.provider_name for e in router._config.providers}
    assert "anthropic" in provider_names
    assert "openai" in provider_names
