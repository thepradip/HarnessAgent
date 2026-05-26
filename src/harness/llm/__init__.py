"""LLM provider adapters, router, structured output, and semantic cache.

Providers are NOT imported at module level — the anthropic / openai SDKs are
optional extras and should not be required just to ``import harness``.

Use lazy access::

    from harness.llm import AnthropicProvider   # SDK imported on first use

Or import directly::

    from harness.llm.anthropic import AnthropicProvider
"""

from __future__ import annotations

from harness.llm.cache import SemanticLLMCache
from harness.llm.router import LLMRouter
from harness.llm.structured import StructuredOutputRouter

__all__ = [
    "LLMRouter",
    "AnthropicProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "OpenAICompatProvider",
    "StructuredOutputRouter",
    "SemanticLLMCache",
]


def __getattr__(name: str) -> object:
    """Lazy-load providers so SDK packages are not imported at module load time."""
    if name == "AnthropicProvider":
        from harness.llm.anthropic import AnthropicProvider
        return AnthropicProvider
    if name in ("OpenAIProvider", "AzureOpenAIProvider"):
        import importlib
        mod = importlib.import_module("harness.llm.openai_provider")
        return getattr(mod, name)
    if name == "OpenAICompatProvider":
        from harness.llm.local import OpenAICompatProvider
        return OpenAICompatProvider
    raise AttributeError(f"module 'harness.llm' has no attribute {name!r}")
