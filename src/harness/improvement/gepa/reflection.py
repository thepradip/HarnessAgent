"""Reflection LM + sync/async bridge for GEPA.

GEPA's ``optimize()`` is a synchronous engine: it calls the adapter and the
reflection ("teacher") LM with plain blocking calls.  HarnessAgent, however, is
fully asynchronous — the LLM providers and the Evaluator are coroutines.

The bridge here lets GEPA run inside a worker thread (see
:func:`GepaPatchGenerator.generate`) while still driving the project's async
machinery: every blocking call schedules its coroutine back onto the original
event loop via :func:`asyncio.run_coroutine_threadsafe` and waits for the result.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Callable that runs a coroutine on the owning event loop from a worker thread
# and blocks until it resolves.
CoroRunner = Callable[[Coroutine[Any, Any, Any]], Any]


def make_coro_runner(loop: asyncio.AbstractEventLoop) -> CoroRunner:
    """Return a blocking runner that executes coroutines on ``loop``.

    The returned callable is meant to be invoked from a *different* thread than
    the one running ``loop`` (GEPA's worker thread). Calling it from within the
    loop's own thread would deadlock, so we guard against that.

    Args:
        loop: The event loop that owns the async resources (Evaluator, LLM).

    Returns:
        A callable ``run(coro) -> result`` that blocks until ``coro`` completes.
    """

    def run(coro: Coroutine[Any, Any, T]) -> T:
        running = None
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError(
                "make_coro_runner() must be called from a worker thread, not the "
                "event loop thread — GEPA optimization should run via asyncio.to_thread."
            )
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    return run


# System instruction for the reflection LM. GEPA's reflection prompt template
# already carries the task-specific framing; this just anchors the role.
_REFLECT_SYSTEM = (
    "You are an expert prompt engineer improving the system prompt of an AI agent. "
    "Analyze the provided execution feedback and propose a revised prompt that fixes "
    "the observed failures while preserving everything that already works. "
    "Return only the improved text requested, with no preamble."
)


class ProviderReflectionLM:
    """Adapts a HarnessAgent LLM provider to GEPA's reflection-LM protocol.

    GEPA calls ``reflection_lm(prompt) -> str`` synchronously, where ``prompt`` is
    either a string or a list of chat messages. This wrapper forwards the call to
    the async provider's ``complete()`` via the supplied :class:`CoroRunner`.

    Args:
        llm_provider: A provider exposing
            ``async complete(messages, *, max_tokens, system=None) -> LLMResponse``
            with a ``.content`` string attribute.
        run_coro:     Blocking runner from :func:`make_coro_runner`.
        max_tokens:   Max tokens for each reflection generation.
    """

    def __init__(
        self,
        llm_provider: Any,
        run_coro: CoroRunner,
        max_tokens: int = 4096,
    ) -> None:
        self._llm = llm_provider
        self._run_coro = run_coro
        self._max_tokens = max_tokens

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)

        async def _complete() -> str:
            response = await self._llm.complete(
                messages=messages,
                max_tokens=self._max_tokens,
                system=_REFLECT_SYSTEM,
            )
            return str(getattr(response, "content", "") or "")

        result: str = self._run_coro(_complete())
        return result
