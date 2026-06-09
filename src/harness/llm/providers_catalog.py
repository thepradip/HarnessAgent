"""Declarative catalog of OpenAI-API-compatible LLM vendors.

DeepSeek, Together AI, Fireworks, Groq, OpenRouter, Mistral, and xAI all speak
the OpenAI Chat Completions API, so each is served by the existing
``OpenAIProvider`` with a different ``base_url`` — no per-vendor adapter code.

This module is pure data plus a resolver: :func:`resolve_compat_vendors` reads
the environment and returns one :class:`CompatRegistration` per (vendor, model)
whose API key is set. ``llm/factory.py`` turns each into a registered provider.

Each vendor is enabled by setting its ``<KEY>`` env var. Optional overrides:
``<VENDOR>_BASE_URL`` (custom/proxy endpoint) and ``<VENDOR>_MODELS``
(comma-separated model ids, replacing the defaults).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class VendorSpec:
    """Static description of an OpenAI-compatible vendor."""
    name: str                                   # provider_name (also tier-key prefix)
    base_url: str
    env_key: str                                # env var holding the API key
    # (model_id, tier) defaults; tier ∈ cheap | standard | premium
    default_models: tuple[tuple[str, str], ...]
    priority: int = 50                          # router priority band (after cloud primaries)


# Sensible, current defaults per vendor. Models are overridable via <VENDOR>_MODELS.
OPENAI_COMPAT_VENDORS: tuple[VendorSpec, ...] = (
    VendorSpec(
        "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
        (("deepseek-chat", "cheap"), ("deepseek-reasoner", "premium")),
    ),
    VendorSpec(
        "together", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
        (("meta-llama/Llama-3.3-70B-Instruct-Turbo", "standard"),),
    ),
    VendorSpec(
        "fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
        (("accounts/fireworks/models/llama-v3p3-70b-instruct", "standard"),),
    ),
    VendorSpec(
        "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
        (("llama-3.3-70b-versatile", "cheap"),),
    ),
    VendorSpec(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
        (("openai/gpt-4o-mini", "cheap"),),
    ),
    VendorSpec(
        "mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
        (("mistral-large-latest", "premium"), ("mistral-small-latest", "cheap")),
    ),
    VendorSpec(
        "xai", "https://api.x.ai/v1", "XAI_API_KEY",
        (("grok-2-latest", "standard"),),
    ),
)


@dataclass(frozen=True)
class CompatRegistration:
    """One concrete provider the factory should register."""
    vendor: str
    model: str
    api_key: str
    base_url: str
    tier: str
    priority: int


def resolve_compat_vendors(
    getenv: Callable[[str], str | None] = os.environ.get,
) -> list[CompatRegistration]:
    """Return registrations for every configured OpenAI-compatible vendor.

    A vendor is included only when its API-key env var is set. ``<VENDOR>_MODELS``
    overrides the default model list (tier defaults to ``standard`` for overrides
    unless the model also appears in the vendor's defaults).
    """
    out: list[CompatRegistration] = []
    for spec in OPENAI_COMPAT_VENDORS:
        key = (getenv(spec.env_key) or "").strip()
        if not key:
            continue
        base = (getenv(f"{spec.name.upper()}_BASE_URL") or "").strip() or spec.base_url
        default_tier = dict(spec.default_models)
        models_env = (getenv(f"{spec.name.upper()}_MODELS") or "").strip()
        if models_env:
            models = [(m.strip(), default_tier.get(m.strip(), "standard"))
                      for m in models_env.split(",") if m.strip()]
        else:
            models = list(spec.default_models)
        for model, tier in models:
            out.append(CompatRegistration(spec.name, model, key, base, tier, spec.priority))
    return out
