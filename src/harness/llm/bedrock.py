"""AWS Bedrock LLM providers.

Two adapters, because Bedrock hosts two request shapes:

- :class:`BedrockClaudeProvider` — Claude models on Bedrock via the Anthropic
  SDK's ``AsyncAnthropicBedrock`` client (the ``anthropic[bedrock]`` extra).
  Model IDs carry the ``anthropic.`` provider prefix, e.g.
  ``anthropic.claude-opus-4-8``. It subclasses :class:`AnthropicProvider`, so
  message mapping, tool conversion, and response parsing are identical to the
  first-party Anthropic path — only the client and the cache-control handling
  differ.
- :class:`BedrockConverseProvider` — every other Bedrock model (Llama, Mistral,
  DeepSeek, Titan, Cohere, …) via the unified ``bedrock-runtime`` **Converse**
  API over boto3.

Both resolve AWS credentials from the standard chain (env vars, shared profile,
assumed role, instance metadata) and require the ``bedrock`` extra:
``pip install agent-haas[bedrock]``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from harness.core.context import LLMResponse
from harness.core.errors import FailureClass, LLMError
from harness.llm.anthropic import AnthropicProvider

logger = logging.getLogger(__name__)


class BedrockClaudeProvider(AnthropicProvider):
    """Claude on Amazon Bedrock via ``AsyncAnthropicBedrock``.

    Use ``anthropic.``-prefixed model IDs (e.g. ``anthropic.claude-opus-4-8``).
    Reuses all of :class:`AnthropicProvider` except the client construction and
    prompt-cache control blocks (Bedrock support for ``cache_control`` is
    region/model dependent, so we omit it to avoid 400s).
    """

    provider_name: str = "bedrock"

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4-6",
        *,
        aws_region: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_session_token: str | None = None,
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        # AnthropicProvider.__init__ expects an api_key; Bedrock uses AWS creds
        # instead, so pass an empty key and store the AWS config ourselves.
        super().__init__(api_key="", model=model, max_retries=max_retries, timeout=timeout)
        self._aws_region = aws_region
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key
        self._aws_session_token = aws_session_token

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from anthropic import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AsyncAnthropicBedrock,
            )
            from anthropic import (
                RateLimitError as AnthropicRateLimitError,
            )
        except ImportError as exc:
            raise ImportError(
                "anthropic[bedrock] is required for BedrockClaudeProvider.\n"
                "Install: pip install 'anthropic[bedrock]'\n"
                "     or: pip install agent-haas[bedrock]"
            ) from exc
        client_kwargs: dict[str, Any] = {
            "max_retries": self._max_retries,
            "timeout": self._timeout,
        }
        if self._aws_region:
            client_kwargs["aws_region"] = self._aws_region
        if self._aws_access_key:
            client_kwargs["aws_access_key"] = self._aws_access_key
        if self._aws_secret_key:
            client_kwargs["aws_secret_key"] = self._aws_secret_key
        if self._aws_session_token:
            client_kwargs["aws_session_token"] = self._aws_session_token
        self._client = AsyncAnthropicBedrock(**client_kwargs)
        self._exc_rate_limit = AnthropicRateLimitError
        self._exc_timeout = APITimeoutError
        self._exc_connection = APIConnectionError
        self._exc_status = APIStatusError

    def _build_system_block(self, system: str) -> list[dict[str, Any]]:
        # No cache_control on Bedrock — keep the system block plain.
        return [{"type": "text", "text": system}]

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Reuse the parent's OpenAI→Anthropic conversion, then strip any
        # cache_control the parent attached (unsupported on Bedrock).
        converted = super()._build_tools(tools)
        for t in converted:
            t.pop("cache_control", None)
        return converted


class BedrockConverseProvider:
    """General Bedrock models via the boto3 ``bedrock-runtime`` Converse API.

    Serves Llama, Mistral, DeepSeek, Titan, Cohere, and other Bedrock models with
    one unified call. Text + system prompts are supported; tool use is not mapped
    here (route tool requests to a Claude/cloud provider instead).
    """

    provider_name: str = "bedrock-converse"

    def __init__(
        self,
        model: str,
        *,
        aws_region: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_session_token: str | None = None,
    ) -> None:
        self.model = model
        self._aws_region = aws_region
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key
        self._aws_session_token = aws_session_token
        self._client: Any = None
        self._exc_client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for BedrockConverseProvider.\n"
                "Install: pip install boto3\n"
                "     or: pip install agent-haas[bedrock]"
            ) from exc
        session_kwargs: dict[str, Any] = {}
        if self._aws_region:
            session_kwargs["region_name"] = self._aws_region
        if self._aws_access_key:
            session_kwargs["aws_access_key_id"] = self._aws_access_key
        if self._aws_secret_key:
            session_kwargs["aws_secret_access_key"] = self._aws_secret_key
        if self._aws_session_token:
            session_kwargs["aws_session_token"] = self._aws_session_token
        self._client = boto3.client("bedrock-runtime", **session_kwargs)
        self._exc_client = ClientError

    @staticmethod
    def _to_converse_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def _push(role: str, text: str) -> None:
            if not text:
                return
            # Converse requires alternating roles — merge consecutive turns.
            if out and out[-1]["role"] == role:
                out[-1]["content"][0]["text"] += "\n" + text
            else:
                out.append({"role": role, "content": [{"text": text}]})

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "tool":
                # Fold tool results into a user turn instead of dropping them.
                _push("user", f"[Tool result {m.get('tool_use_id', '')}]: {content}")
                continue
            if role not in ("user", "assistant"):
                continue  # system handled separately
            text = content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            _push(role, text)
        return out

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._ensure_client()
        if tools:
            logger.debug("BedrockConverseProvider ignores tools; route tool calls to Claude.")
        if system is None:
            for msg in messages:
                if msg.get("role") == "system":
                    system = str(msg.get("content", ""))
                    break

        request: dict[str, Any] = {
            "modelId": self.model,
            "messages": self._to_converse_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            request["system"] = [{"text": system}]

        try:
            response = await asyncio.to_thread(self._client.converse, **request)
        except self._exc_client as exc:
            code = exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            if code in ("ThrottlingException", "TooManyRequestsException"):
                fc = FailureClass.LLM_RATE_LIMIT
            elif code in ("ModelTimeoutException",):
                fc = FailureClass.LLM_TIMEOUT
            else:
                fc = FailureClass.LLM_ERROR
            raise LLMError(
                f"Bedrock Converse error ({code or 'unknown'}): {exc}",
                failure_class=fc,
                context={"model": self.model},
            ) from exc
        except Exception as exc:  # network/credential errors
            raise LLMError(
                f"Bedrock Converse call failed: {exc}",
                failure_class=FailureClass.LLM_ERROR,
                context={"model": self.model},
            ) from exc

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = response.get("usage", {})
        return LLMResponse(
            content=text,
            tool_calls=[],
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            model=self.model,
            provider=self.provider_name,
        )

    async def stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        """Bedrock Converse streaming — yields text deltas."""
        self._ensure_client()
        system = kwargs.get("system")
        max_tokens = kwargs.get("max_tokens", 1024)
        request: dict[str, Any] = {
            "modelId": self.model,
            "messages": self._to_converse_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            request["system"] = [{"text": system}]
        response = await asyncio.to_thread(self._client.converse_stream, **request)
        for event in response.get("stream", []):
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield delta["text"]

    async def health_check(self) -> bool:
        try:
            self._ensure_client()
        except Exception:
            return False
        return self._client is not None
