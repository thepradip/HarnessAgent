"""Anthropic Claude provider adapter for HarnessAgent."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

from harness.core.context import LLMResponse, ToolCall
from harness.core.errors import FailureClass, LLMError

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic Claude LLM provider with prompt caching and tool calling."""

    provider_name: str = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: Any = None          # loaded on first call
        # Exception classes cached after first load — used in error handlers
        self._exc_rate_limit: Any = None
        self._exc_timeout: Any = None
        self._exc_connection: Any = None
        self._exc_status: Any = None

    def _ensure_client(self) -> None:
        """Lazy-load the Anthropic SDK on first use."""
        if self._client is not None:
            return
        try:
            from anthropic import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AsyncAnthropic,
                RateLimitError as AnthropicRateLimitError,
            )
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicProvider.\n"
                "Install: pip install anthropic\n"
                "     or: pip install agent-haas[anthropic]"
            ) from exc
        self._client = AsyncAnthropic(
            api_key=self._api_key,
            max_retries=self._max_retries,
            timeout=self._timeout,
        )
        self._exc_rate_limit = AnthropicRateLimitError
        self._exc_timeout = APITimeoutError
        self._exc_connection = APIConnectionError
        self._exc_status = APIStatusError

    def _build_system_block(self, system: str) -> list[dict[str, Any]]:
        """Build a system block with ephemeral cache_control for prompt caching."""
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool dicts to Anthropic format, adding cache control."""
        anthropic_tools: list[dict[str, Any]] = []
        for i, tool in enumerate(tools):
            t: dict[str, Any] = {
                "name": tool.get("name") or tool.get("function", {}).get("name", ""),
                "description": tool.get("description")
                or tool.get("function", {}).get("description", ""),
                "input_schema": tool.get("input_schema")
                or tool.get("parameters")
                or tool.get("function", {}).get("parameters", {}),
            }
            # Apply cache_control to the last tool when there are more than 3
            if len(tools) > 3 and i == len(tools) - 1:
                t["cache_control"] = {"type": "ephemeral"}
            anthropic_tools.append(t)
        return anthropic_tools

    def _map_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert generic message dicts to Anthropic-compatible format.

        The harness history is provider-neutral: assistant entries may carry
        ``tool_calls`` ([{id, name, args}]) and tool results use role "tool"
        with a ``tool_use_id``. Anthropic expects tool_use blocks on the
        assistant turn and tool_result blocks inside a *user* turn, with
        consecutive results merged into one user message.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Skip system messages — handled separately
            if role == "system":
                continue
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_use_id", ""),
                    "content": content if isinstance(content, str) else str(content),
                }
                prev = result[-1] if result else None
                if (
                    prev is not None
                    and prev["role"] == "user"
                    and isinstance(prev["content"], list)
                    and prev["content"]
                    and isinstance(prev["content"][-1], dict)
                    and prev["content"][-1].get("type") == "tool_result"
                ):
                    prev["content"].append(block)
                else:
                    result.append({"role": "user", "content": [block]})
                continue
            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    blocks.extend(content)
                for call in msg["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "input": call.get("args", {}),
                    })
                result.append({"role": "assistant", "content": blocks})
                continue
            result.append({"role": role, "content": content})
        return result

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request and return a normalised LLMResponse."""
        kwargs.pop("model", None)  # caller must not override via kwargs
        anthropic_messages = self._map_messages(messages)

        # Extract system from messages if not passed explicitly
        if system is None:
            for msg in messages:
                if msg.get("role") == "system":
                    system = str(msg.get("content", ""))
                    break

        build_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            build_kwargs["system"] = self._build_system_block(system)
        if tools:
            build_kwargs["tools"] = self._build_tools(tools)

        self._ensure_client()
        RateLimitError = self._exc_rate_limit
        APITimeoutError = self._exc_timeout
        APIConnectionError = self._exc_connection
        APIStatusError = self._exc_status

        try:
            response = await self._client.messages.create(**build_kwargs, **kwargs)
        except RateLimitError as exc:
            raise LLMError(
                f"Anthropic rate limit: {exc}",
                failure_class=FailureClass.LLM_RATE_LIMIT,
                context={"model": self.model},
            ) from exc
        except APITimeoutError as exc:
            raise LLMError(
                f"Anthropic request timed out: {exc}",
                failure_class=FailureClass.LLM_TIMEOUT,
                context={"model": self.model},
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                f"Anthropic connection error: {exc}",
                failure_class=FailureClass.LLM_ERROR,
                context={"model": self.model},
            ) from exc
        except APIStatusError as exc:
            if exc.status_code == 400 and "too large" in str(exc).lower():
                raise LLMError(
                    f"Anthropic context limit exceeded: {exc}",
                    failure_class=FailureClass.LLM_CONTEXT_LIMIT,
                    context={"model": self.model, "status_code": exc.status_code},
                ) from exc
            raise LLMError(
                f"Anthropic API error ({exc.status_code}): {exc}",
                failure_class=FailureClass.LLM_ERROR,
                context={"model": self.model, "status_code": exc.status_code},
            ) from exc

        # Parse content
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=dict(block.input) if block.input else {},
                    )
                )

        # Detect prompt cache usage
        cached = False
        usage = response.usage
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            cached = True

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=response.model,
            provider=self.provider_name,
            cached=cached,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text deltas from the Anthropic API."""
        max_tokens = kwargs.pop("max_tokens", 4096)
        system: str | None = kwargs.pop("system", None)
        anthropic_messages = self._map_messages(messages)

        if system is None:
            for msg in messages:
                if msg.get("role") == "system":
                    system = str(msg.get("content", ""))
                    break

        build_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            build_kwargs["system"] = self._build_system_block(system)

        self._ensure_client()
        RateLimitError = self._exc_rate_limit
        APITimeoutError = self._exc_timeout
        APIStatusError = self._exc_status

        try:
            async with self._client.messages.stream(**build_kwargs, **kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except RateLimitError as exc:
            raise LLMError(
                f"Anthropic rate limit during stream: {exc}",
                failure_class=FailureClass.LLM_RATE_LIMIT,
            ) from exc
        except APITimeoutError as exc:
            raise LLMError(
                f"Anthropic stream timed out: {exc}",
                failure_class=FailureClass.LLM_TIMEOUT,
            ) from exc
        except APIStatusError as exc:
            raise LLMError(
                f"Anthropic API error during stream ({exc.status_code}): {exc}",
                failure_class=FailureClass.LLM_ERROR,
            ) from exc

    async def health_check(self) -> bool:
        """Return True if the Anthropic API is reachable."""
        self._ensure_client()
        RateLimitError = self._exc_rate_limit
        try:
            await self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except RateLimitError:
            return True  # rate-limited but the API is up
        except Exception as exc:
            logger.warning("Anthropic health check failed: %s", exc)
            return False
