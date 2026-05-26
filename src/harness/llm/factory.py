"""Build a fully configured LLMRouter from settings.

Usage:
    from harness.llm.factory import build_router
    router = build_router(get_config())

The router is priority-ordered:
  1. Anthropic Claude (if ANTHROPIC_API_KEY set)
  2. OpenAI models (if OPENAI_API_KEY set) — each model registered separately
  3. Local models (vLLM / SGLang / llama.cpp) — if base_url env vars set
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from harness.llm.router import LLMRouter, LLMRouterConfig

if TYPE_CHECKING:
    from harness.core.config import Settings

logger = logging.getLogger(__name__)

# Context window sizes per model family (tokens)
_OPENAI_CONTEXT: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.5": 128_000,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5.2": 128_000,
    "gpt-5.5": 128_000,
    "o1": 128_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 128_000,
    "o3-mini": 128_000,
    "o4-mini": 128_000,
}


def build_router(config: "Settings") -> LLMRouter:
    """Create an LLMRouter from environment configuration.

    Priority order (lower number = tried first):
        0   — Claude (primary, highest quality)
        10  — GPT-4o / GPT-5 (strong fallback)
        20  — GPT-4o-mini / o-series (cost-optimised)
        100 — vLLM / SGLang / llama.cpp (local / offline)
    """
    router_config = LLMRouterConfig(
        circuit_failure_threshold=5,
        circuit_recovery_timeout=60.0,
        circuit_success_threshold=2,
    )
    router = LLMRouter(config=router_config)

    # Lazy provider imports — SDKs are optional extras
    from harness.llm.anthropic import AnthropicProvider
    from harness.llm.hermes import HermesXMLProvider
    from harness.llm.local import ModelCapabilities, OpenAICompatProvider
    from harness.llm.openai_provider import AzureOpenAIProvider, OpenAIProvider

    # ------------------------------------------------------------------
    # Azure OpenAI (priority 0 — takes precedence when configured)
    # ------------------------------------------------------------------
    _az_key = getattr(config, "azure_openai_api_key", "") or ""
    _az_ep  = getattr(config, "azure_openai_endpoint", "") or ""
    if isinstance(_az_key, str) and isinstance(_az_ep, str) and _az_key and _az_ep:
        deployment = config.azure_openai_deployment or "gpt-5.2"
        router.register(
            AzureOpenAIProvider(
                api_key=_az_key,
                azure_endpoint=_az_ep,
                deployment=deployment,
                api_version=getattr(config, "azure_openai_api_version", "2025-04-01-preview") or "2025-04-01-preview",
            ),
            priority=0,
            context_window=_OPENAI_CONTEXT.get(deployment, 128_000),
        )
        logger.info("Registered Azure OpenAI provider: deployment=%s", deployment)
    else:
        logger.debug("AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not set — Azure disabled")

    # ------------------------------------------------------------------
    # Anthropic Claude
    # ------------------------------------------------------------------
    if config.anthropic_api_key:
        model = config.default_model or "claude-sonnet-4-6"
        router.register(
            AnthropicProvider(api_key=config.anthropic_api_key, model=model),
            priority=0,
            context_window=200_000,
        )
        logger.info("Registered Anthropic provider: %s", model)
    else:
        logger.warning("ANTHROPIC_API_KEY not set — Anthropic provider disabled")

    # ------------------------------------------------------------------
    # OpenAI — register each requested model as a separate provider entry
    # ------------------------------------------------------------------
    if config.openai_api_key:
        openai_models: list[tuple[str, int, int]] = []  # (model, priority, context_window)

        # Parse OPENAI_MODELS env var if set, else register the default
        raw = getattr(config, "openai_models", None) or ""
        if raw:
            for entry in raw.split(","):
                m = entry.strip()
                if m:
                    openai_models.append((m, 10, _OPENAI_CONTEXT.get(m, 128_000)))
        else:
            # Default: register gpt-4o-mini as a capable, cheap fallback
            openai_models = [
                ("gpt-4o-mini", 20, 128_000),
            ]

        for model_id, priority, ctx in openai_models:
            router.register(
                OpenAIProvider(api_key=config.openai_api_key, model=model_id),
                priority=priority,
                context_window=ctx,
            )
            logger.info("Registered OpenAI provider: %s (priority=%d)", model_id, priority)
    else:
        logger.info("OPENAI_API_KEY not set — OpenAI provider disabled")

    # ------------------------------------------------------------------
    # vLLM (self-hosted)
    # ------------------------------------------------------------------
    vllm_url = getattr(config, "vllm_base_url", None) or ""
    if vllm_url:
        vllm_model = getattr(config, "vllm_model", "mistralai/Mistral-7B-Instruct-v0.3")
        router.register(
            OpenAICompatProvider(
                base_url=vllm_url,
                model=vllm_model,
                capabilities=ModelCapabilities(
                    supports_tool_calling=True,
                    supports_system_prompt=True,
                    context_window=32_768,
                ),
            ),
            priority=100,
            context_window=32_768,
        )
        logger.info("Registered vLLM provider: %s @ %s", vllm_model, vllm_url)

    # ------------------------------------------------------------------
    # SGLang (self-hosted)
    # ------------------------------------------------------------------
    sglang_url = getattr(config, "sglang_base_url", None) or ""
    if sglang_url:
        sglang_model = getattr(config, "sglang_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        router.register(
            OpenAICompatProvider(
                base_url=sglang_url,
                model=sglang_model,
                capabilities=ModelCapabilities(
                    supports_tool_calling=False,   # use ReAct injection
                    supports_system_prompt=True,
                    context_window=8_192,
                ),
            ),
            priority=110,
            context_window=8_192,
        )
        logger.info("Registered SGLang provider: %s @ %s", sglang_model, sglang_url)

    # ------------------------------------------------------------------
    # llama.cpp HTTP server (CPU / Metal)
    # ------------------------------------------------------------------
    llamacpp_url = getattr(config, "llamacpp_base_url", None) or ""
    if llamacpp_url:
        router.register(
            OpenAICompatProvider(
                base_url=llamacpp_url,
                model="local-model",
                capabilities=ModelCapabilities(
                    supports_tool_calling=False,
                    supports_system_prompt=False,
                    context_window=4_096,
                ),
            ),
            priority=120,
            context_window=4_096,
        )
        logger.info("Registered llama.cpp provider @ %s", llamacpp_url)

    # ------------------------------------------------------------------
    # Hermes / Qwen XML (SGLang or vLLM with Hermes-2-Pro / Qwen models)
    # ------------------------------------------------------------------
    hermes_url = getattr(config, "hermes_base_url", None) or ""
    if hermes_url:
        hermes_model = getattr(
            config, "hermes_model", "NousResearch/Hermes-2-Pro-Llama-3-8B"
        )
        hermes_ctx = int(getattr(config, "hermes_context_window", 8_192))
        router.register(
            HermesXMLProvider(
                base_url=hermes_url,
                model=hermes_model,
                context_window=hermes_ctx,
            ),
            priority=105,
            context_window=hermes_ctx,
        )
        logger.info(
            "Registered HermesXML provider: %s @ %s (ctx=%d)",
            hermes_model,
            hermes_url,
            hermes_ctx,
        )

    if not router._config.providers:
        raise RuntimeError(
            "No LLM providers configured. Set one of:\n"
            "  AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT  (for Azure GPT-5.2)\n"
            "  ANTHROPIC_API_KEY                             (for Claude)\n"
            "  OPENAI_API_KEY                                (for OpenAI)\n"
            "in your .env file."
        )

    return router
