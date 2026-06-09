"""OpenAI-compatible provider adapter for local/self-hosted LLMs."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from harness.core.context import LLMResponse, ToolCall
from harness.core.errors import FailureClass, LLMError

logger = logging.getLogger(__name__)

_TOOL_JSON_RE = re.compile(
    r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,[^{}]*"args"\s*:\s*\{[^{}]*\}[^{}]*\}',
    re.DOTALL,
)


@dataclass
class ModelCapabilities:
    """Describes the capabilities of a local model endpoint."""

    supports_tool_calling: bool
    supports_system_prompt: bool
    context_window: int
    supports_vision: bool = False


class OpenAICompatProvider:
    """LLM provider adapter for vLLM, SGLang, llama.cpp, Ollama and similar."""

    provider_name: str = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-required",
        capabilities: ModelCapabilities | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._capabilities = capabilities or ModelCapabilities(
            supports_tool_calling=True,
            supports_system_prompt=True,
            context_window=8192,
        )
        self._client = AsyncOpenAI(
            base_url=f"{self._base_url}/v1",
            api_key=api_key,
            timeout=timeout,
        )

    def _inject_tools_into_system(
        self,
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> str:
        """Append ReAct-style tool schema to the system prompt."""
        tool_desc = json.dumps(tools, indent=2)
        react_instructions = (
            "Available tools:\n"
            f"{tool_desc}\n\n"
            'To call a tool, respond ONLY with a JSON object in this exact format:\n'
            '{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\n'
            "Do not include any other text when calling a tool."
        )
        if system:
            return f"{system}\n\n{react_instructions}"
        return react_instructions

    def _parse_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Extract tool calls from plain-text responses (non-native tool calling)."""
        matches = _TOOL_JSON_RE.findall(text)
        tool_calls: list[ToolCall] = []
        for match in matches:
            try:
                obj = json.loads(match)
                if "tool" in obj and "args" in obj:
                    import uuid as _uuid

                    tool_calls.append(
                        ToolCall(
                            id=_uuid.uuid4().hex[:8],
                            name=str(obj["tool"]),
                            args=dict(obj["args"]) if isinstance(obj["args"], dict) else {},
                        )
                    )
            except json.JSONDecodeError:
                continue
        return tool_calls

    def _build_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Build the messages list, injecting tools into system if needed."""
        effective_system = system
        if tools and not self._capabilities.supports_tool_calling:
            effective_system = self._inject_tools_into_system(system, tools)

        result: list[dict[str, Any]] = []
        system_prepended = False
        # In-message system content collected when the model lacks a system role;
        # folded into the user-turn prepend below instead of a system message.
        folded_system: list[str] = []

        if effective_system and self._capabilities.supports_system_prompt:
            result.append({"role": "system", "content": effective_system})
            system_prepended = True

        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                if not self._capabilities.supports_system_prompt:
                    # Model has no system role — fold this content into the
                    # user-turn prepend rather than injecting a system message.
                    content = str(msg.get("content", ""))
                    if content:
                        folded_system.append(content)
                elif not system_prepended:
                    result.insert(0, {"role": "system", "content": msg.get("content", "")})
                    system_prepended = True
                continue
            if role == "tool":
                # Local servers reject bare "tool" roles (no native tool_calls
                # turn precedes them here) — fold the result into a user turn.
                result.append({
                    "role": "user",
                    "content": (
                        f"[Tool result {msg.get('tool_use_id', '')}]: "
                        f"{msg.get('content', '')}"
                    ),
                })
                continue
            result.append({"role": role, "content": msg.get("content", "")})

        # Prepend system content as a leading user turn when the model lacks a
        # system role — combine the explicit/effective system with any in-message
        # system content folded in above.
        if not self._capabilities.supports_system_prompt:
            parts: list[str] = []
            if effective_system:
                parts.append(effective_system)
            parts.extend(folded_system)
            if parts:
                prepend = "\n\n".join(parts)
                result.insert(0, {"role": "user", "content": prepend})
                result.insert(1, {"role": "assistant", "content": "Understood."})

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
        """Send a completion request to the OpenAI-compatible endpoint."""
        built_messages = self._build_messages(messages, system, tools)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": built_messages,
            **kwargs,
        }

        use_native_tools = (
            tools is not None and self._capabilities.supports_tool_calling
        )
        if use_native_tools and tools:
            create_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters") or t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]

        try:
            response = await self._client.chat.completions.create(**create_kwargs)
        except OpenAIRateLimitError as exc:
            raise LLMError(
                f"Rate limit from {self._base_url}: {exc}",
                failure_class=FailureClass.LLM_RATE_LIMIT,
            ) from exc
        except APITimeoutError as exc:
            raise LLMError(
                f"Timeout from {self._base_url}: {exc}",
                failure_class=FailureClass.LLM_TIMEOUT,
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                f"Connection error to {self._base_url}: {exc}",
                failure_class=FailureClass.LLM_ERROR,
            ) from exc
        except APIStatusError as exc:
            raise LLMError(
                f"API error from {self._base_url} ({exc.status_code}): {exc}",
                failure_class=FailureClass.LLM_ERROR,
            ) from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls: list[ToolCall] = []

        if use_native_tools and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        elif tools and not self._capabilities.supports_tool_calling:
            tool_calls = self._parse_tool_calls_from_text(content)

        usage = response.usage
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            provider=self.provider_name,
            cached=False,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text deltas from the OpenAI-compatible endpoint."""
        max_tokens = kwargs.pop("max_tokens", 4096)
        system: str | None = kwargs.pop("system", None)
        built_messages = self._build_messages(messages, system, None)

        try:
            async with await self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=built_messages,
                stream=True,
                **kwargs,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except OpenAIRateLimitError as exc:
            raise LLMError(
                f"Rate limit from {self._base_url} during stream: {exc}",
                failure_class=FailureClass.LLM_RATE_LIMIT,
            ) from exc
        except APITimeoutError as exc:
            raise LLMError(
                f"Timeout from {self._base_url} during stream: {exc}",
                failure_class=FailureClass.LLM_TIMEOUT,
            ) from exc
        except APIStatusError as exc:
            fc = (
                FailureClass.LLM_RATE_LIMIT
                if getattr(exc, "status_code", None) == 429
                else FailureClass.LLM_ERROR
            )
            raise LLMError(
                f"Streaming error from {self._base_url} ({exc.status_code}): {exc}",
                failure_class=fc,
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                f"Connection error to {self._base_url} during stream: {exc}",
                failure_class=FailureClass.LLM_ERROR,
            ) from exc

    async def health_check(self) -> bool:
        """Return True if the local model server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try /health first (vLLM / SGLang)
                try:
                    r = await client.get(f"{self._base_url}/health")
                    if r.status_code < 500:
                        return True
                except httpx.RequestError:
                    pass
                # Fall back to /v1/models (Ollama / llama.cpp)
                r = await client.get(f"{self._base_url}/v1/models")
                return r.status_code < 500
        except Exception as exc:
            logger.warning("Local LLM health check failed (%s): %s", self._base_url, exc)
            return False
