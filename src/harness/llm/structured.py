"""Structured-output router that returns validated Pydantic models from any LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from harness.core.context import LLMResponse
from harness.core.errors import FailureClass, LLMError
from harness.core.protocols import LLMProvider

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Extract the first JSON block from text, stripping markdown fences."""
    # Try fenced code block first
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _try_parse(text: str, response_model: type[_T]) -> _T:
    """Attempt to parse text as JSON and validate against response_model."""
    raw = _extract_json(text)
    data = json.loads(raw)
    return response_model.model_validate(data)


class StructuredOutputRouter:
    """Wraps any LLMProvider to return validated Pydantic model instances."""

    def __init__(self, repair_model_name: str = "claude-sonnet-4-6") -> None:
        self._repair_model_name = repair_model_name

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[_T],
        provider: LLMProvider,
        max_retries: int = 3,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> _T:
        """Return a validated instance of response_model from the LLM response."""

        # Strategy 1: Try instructor if the provider is Anthropic or has tool calling
        try:
            result = await self._try_instructor(
                messages, response_model, provider, max_tokens, system
            )
            if result is not None:
                return result
        except Exception as exc:
            logger.debug("instructor strategy failed: %s", exc)

        # Strategy 2: Repair loop with JSON extraction
        last_text = ""
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                schema_str = json.dumps(response_model.model_json_schema(), indent=2)
                prompt_messages = list(messages)

                if attempt == 0:
                    # First attempt: ask for JSON matching the schema
                    json_instruction = (
                        f"Respond with a valid JSON object matching this schema exactly. "
                        f"Output ONLY the JSON — no markdown, no explanation:\n{schema_str}"
                    )
                    if system:
                        effective_system = f"{system}\n\n{json_instruction}"
                    else:
                        effective_system = json_instruction
                else:
                    # Repair attempt: show the bad JSON and ask for a fix
                    repair_msg = (
                        f"The previous JSON was invalid. Error: {last_exc}\n\n"
                        f"Bad JSON:\n{last_text}\n\n"
                        f"Fix it to match this schema and output ONLY valid JSON:\n{schema_str}"
                    )
                    prompt_messages = prompt_messages + [
                        {"role": "assistant", "content": last_text},
                        {"role": "user", "content": repair_msg},
                    ]
                    effective_system = system

                response: LLMResponse = await provider.complete(
                    prompt_messages,
                    max_tokens=max_tokens,
                    system=effective_system,
                )
                last_text = response.content

                return _try_parse(last_text, response_model)

            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                logger.debug(
                    "Structured output attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                continue
            except LLMError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.debug("Unexpected error in structured output: %s", exc)
                continue

        raise LLMError(
            f"Failed to produce valid {response_model.__name__} after {max_retries} attempts. "
            f"Last error: {last_exc}. Last text: {last_text[:200]}",
            failure_class=FailureClass.LLM_PARSE_ERROR,
            context={
                "model": response_model.__name__,
                "attempts": max_retries,
                "last_text_snippet": last_text[:200],
            },
        )

    async def _try_instructor(
        self,
        messages: list[dict[str, Any]],
        response_model: type[_T],
        provider: LLMProvider,
        max_tokens: int,
        system: str | None,
    ) -> _T | None:
        """Attempt structured extraction via the instructor library."""
        try:
            import instructor
        except ImportError:
            return None

        from harness.llm.anthropic import AnthropicProvider
        from harness.llm.local import OpenAICompatProvider

        if isinstance(provider, AnthropicProvider):
            try:
                import anthropic

                # The SDK client is lazy-loaded — ensure it exists before
                # handing it to instructor (instructor.from_anthropic(None) raises).
                provider._ensure_client()
                patched = instructor.from_anthropic(provider._client)
                build_kwargs: dict[str, Any] = {
                    "model": provider.model,
                    "max_tokens": max_tokens,
                    "messages": [
                        m for m in messages if m.get("role") != "system"
                    ],
                    "response_model": response_model,
                }
                effective_system = system
                if effective_system is None:
                    for msg in messages:
                        if msg.get("role") == "system":
                            effective_system = str(msg.get("content", ""))
                            break
                if effective_system:
                    build_kwargs["system"] = effective_system
                return await patched.messages.create(**build_kwargs)
            except Exception as exc:
                logger.debug("instructor+anthropic failed: %s", exc)
                return None

        if isinstance(provider, OpenAICompatProvider):
            try:
                patched = instructor.from_openai(provider._client)
                return await patched.chat.completions.create(
                    model=provider.model,
                    max_tokens=max_tokens,
                    messages=[m for m in messages if not (
                        m.get("role") == "system" and system is not None
                    )],
                    response_model=response_model,
                )
            except Exception as exc:
                logger.debug("instructor+openai_compat failed: %s", exc)
                return None

        return None
