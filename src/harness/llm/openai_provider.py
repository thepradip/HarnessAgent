"""OpenAI API provider for HarnessAgent.

Supports: gpt-4o, gpt-4o-mini, o1, o1-mini, o3, o3-mini, o4-mini,
          gpt-4.5, gpt-5 (and any future model via provider_name/model config).

Key differences from OpenAICompatProvider (local.py):
- Targets api.openai.com, not a local endpoint
- Handles o1/o3/o4 series: no system prompt, max_completion_tokens instead of max_tokens
- Uses OpenAI native function calling (not ReAct text injection)
- Tracks prompt-cache savings via usage.prompt_tokens_details
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from harness.core.context import LLMResponse, ToolCall
from harness.core.errors import FailureClass, LLMError

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens.
# Includes o1/o3/o4 reasoning series and gpt-5 family (including Azure variants).
_REASONING_MODELS = frozenset({
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4-mini",
    "gpt-5", "gpt-5-mini", "gpt-5.2", "gpt-5.5",
})

# Prefix patterns for reasoning/completion-token models not listed above
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

# Models that support prompt caching (auto, no extra config needed)
_CACHED_MODELS = frozenset({
    "gpt-4o", "gpt-4o-mini", "gpt-4.5",
    "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18",
    "gpt-5", "gpt-5-mini", "gpt-5.2", "gpt-5.5",
    "o1", "o3", "o4-mini",
})


class OpenAIProvider:
    """Official OpenAI API provider.

    Works for any model served at api.openai.com.
    Automatically adjusts request format for reasoning models (o1/o3/o4 series).
    """

    provider_name: str = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 120.0,
        max_retries: int = 0,       # retry handled by LLMRouter
        organization: str | None = None,
        base_url: str | None = None, # override for Azure or proxy
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            **({"base_url": base_url} if base_url else {}),
        )
        # A model is a "reasoning" model if it is in the known set, matches a
        # known prefix, or the caller explicitly flags it with use_completion_tokens.
        self._is_reasoning = (
            model in _REASONING_MODELS
            or any(model.startswith(p) for p in _REASONING_PREFIXES)
        )

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request to the OpenAI API."""
        prepared = self._prepare_messages(messages, system)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": prepared,
        }

        # Reasoning models use max_completion_tokens, not max_tokens
        if self._is_reasoning:
            request["max_completion_tokens"] = max_tokens
            # temperature not supported on reasoning models
        else:
            request["max_tokens"] = max_tokens
            if temperature is not None:
                request["temperature"] = temperature

        if tools and not self._is_reasoning:
            request["tools"] = [self._to_openai_tool(t) for t in tools]
            request["tool_choice"] = "auto"

        try:
            resp = await self._client.chat.completions.create(**request)
        except OpenAIRateLimitError as exc:
            raise LLMError(str(exc), failure_class=FailureClass.LLM_RATE_LIMIT) from exc
        except APITimeoutError as exc:
            raise LLMError(str(exc), failure_class=FailureClass.LLM_TIMEOUT) from exc
        except APIConnectionError as exc:
            raise LLMError(str(exc), failure_class=FailureClass.LLM_ERROR) from exc
        except APIStatusError as exc:
            if exc.status_code == 429:
                raise LLMError(str(exc), failure_class=FailureClass.LLM_RATE_LIMIT) from exc
            if exc.status_code in (500, 502, 503):
                raise LLMError(str(exc), failure_class=FailureClass.LLM_ERROR) from exc
            raise LLMError(str(exc), failure_class=FailureClass.LLM_ERROR) from exc

        choice = resp.choices[0]
        content = choice.message.content or ""
        tool_calls = self._extract_tool_calls(choice)

        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Track cache savings if available (gpt-4o series auto-caches)
        cached = False
        if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            cached = cached_tokens > 0

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider=self.provider_name,
            cached=cached,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from the OpenAI API."""
        prepared = self._prepare_messages(messages, system)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": prepared,
            "stream": True,
        }
        if self._is_reasoning:
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens

        try:
            async with await self._client.chat.completions.create(**request) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content
        except (OpenAIRateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise LLMError(str(exc), failure_class=FailureClass.LLM_ERROR) from exc

    async def health_check(self) -> bool:
        """Check reachability by listing one model."""
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> list[dict[str, Any]]:
        """Build the messages list, handling reasoning-model constraints."""
        result: list[dict[str, Any]] = []

        if system and not self._is_reasoning:
            result.append({"role": "system", "content": system})
        elif system and self._is_reasoning:
            # o1/o3/o4 don't support system role — prepend as developer message
            result.append({"role": "developer", "content": system})

        result.extend(messages)
        return result

    @staticmethod
    def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert a harness tool definition to OpenAI function format."""
        # Harness tools use Anthropic format; convert to OpenAI format
        if "input_schema" in tool:
            return {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
        # Already in OpenAI format
        return tool

    @staticmethod
    def _extract_tool_calls(choice: Any) -> list[ToolCall]:
        """Parse OpenAI tool_calls from a completion choice."""
        raw_calls = getattr(choice.message, "tool_calls", None) or []
        result: list[ToolCall] = []
        for tc in raw_calls:
            import json
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {"_raw": tc.function.arguments}
            result.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        return result


class AzureOpenAIProvider(OpenAIProvider):
    """
    Azure OpenAI provider — uses AzureOpenAI client instead of OpenAI.

    Configured via:
        AZURE_OPENAI_API_KEY      — Azure API key
        AZURE_OPENAI_ENDPOINT     — https://your-resource.openai.azure.com/
        AZURE_OPENAI_API_VERSION  — e.g. 2025-01-01-preview
        AZURE_OPENAI_DEPLOYMENT   — deployment name in Azure portal (e.g. gpt-5.2)

    The deployment name IS the model name passed to the API.
    """

    provider_name: str = "azure_openai"

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        deployment: str = "gpt-5.5",
        api_version: str = "2025-04-01-preview",
        timeout: float = 120.0,
    ) -> None:
        from openai import AsyncAzureOpenAI
        from urllib.parse import urlparse, urlunparse

        self.model = deployment   # Azure uses deployment name as model identifier

        # Normalize endpoint: strip any /openai/* suffix the user included.
        # AzureOpenAI client builds its own path on top of the base URL.
        parsed = urlparse(azure_endpoint)
        base = parsed.path.split("/openai")[0].rstrip("/")
        clean_endpoint = urlunparse(parsed._replace(path=base + "/"))

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=clean_endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            timeout=timeout,
            max_retries=0,
        )
        self._is_reasoning = (
            deployment in _REASONING_MODELS
            or any(deployment.startswith(p) for p in _REASONING_PREFIXES)
        )
        logger.info(
            "AzureOpenAIProvider: deployment=%s endpoint=%s api_version=%s",
            deployment, azure_endpoint.split(".openai.azure.com")[0].split("//")[-1],
            api_version,
        )

    async def health_check(self) -> bool:
        """Azure health check — minimal completion with the correct token parameter."""
        try:
            await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "developer", "content": "You are a health check assistant."},
                    {"role": "user", "content": "Reply with the word OK."},
                ],
                max_completion_tokens=5,
            )
            return True
        except Exception:
            return False
