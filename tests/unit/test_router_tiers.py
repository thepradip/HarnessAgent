"""Unit tests for cost-aware tiered routing in LLMRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from harness.core.context import LLMResponse
from harness.llm.router import LLMRouter, LLMRouterConfig


def _make_provider(name, model, healthy=True):
    p = AsyncMock()
    p.provider_name = name
    p.model = model
    resp = LLMResponse(content=f"from {name}", tool_calls=[], input_tokens=1,
                       output_tokens=1, model=model, provider=name)
    p.complete = AsyncMock(return_value=resp)
    p.health_check = AsyncMock(return_value=healthy)
    return p


def _router(scorer=None, tenant_tiers=None):
    cfg = LLMRouterConfig(scorer=scorer, tenant_tiers=tenant_tiers or {})
    return LLMRouter(config=cfg)


@pytest.mark.asyncio
async def test_no_tier_preserves_priority_order():
    # Backward compatibility: no tier, no scorer -> priority order unchanged.
    r = _router()
    r.register(_make_provider("cheap_p", "c"), priority=0, tier="cheap")
    r.register(_make_provider("prem_p", "p"), priority=1, tier="premium")
    resp = await r.complete([{"role": "user", "content": "hi"}], max_tokens=10)
    assert resp.provider == "cheap_p"  # lowest priority wins


@pytest.mark.asyncio
async def test_explicit_tier_prefers_matching_provider():
    r = _router()
    r.register(_make_provider("cheap_p", "c"), priority=0, tier="cheap")
    r.register(_make_provider("prem_p", "p"), priority=1, tier="premium")
    # Even though cheap_p has higher priority, an explicit premium tier wins.
    resp = await r.complete([{"role": "user", "content": "hi"}], max_tokens=10, tier="premium")
    assert resp.provider == "prem_p"


@pytest.mark.asyncio
async def test_scorer_picks_tier_by_complexity():
    # A scorer that returns "premium" should route to the premium provider.
    scorer = type("S", (), {"score": lambda self, *a, **k: "premium"})()
    r = _router(scorer=scorer)
    r.register(_make_provider("cheap_p", "c"), priority=0, tier="cheap")
    r.register(_make_provider("prem_p", "p"), priority=1, tier="premium")
    resp = await r.complete([{"role": "user", "content": "hard"}], max_tokens=10)
    assert resp.provider == "prem_p"


@pytest.mark.asyncio
async def test_tier_falls_back_when_target_unhealthy():
    r = _router()
    r.register(_make_provider("cheap_p", "c"), priority=0, tier="cheap")
    r.register(_make_provider("prem_p", "p", healthy=False), priority=1, tier="premium")
    # premium target is unhealthy -> falls back through remaining providers.
    resp = await r.complete([{"role": "user", "content": "x"}], max_tokens=10, tier="premium")
    assert resp.provider == "cheap_p"


@pytest.mark.asyncio
async def test_unknown_tier_falls_back_to_full_order():
    r = _router()
    r.register(_make_provider("a", "a"), priority=0, tier="cheap")
    r.register(_make_provider("b", "b"), priority=1, tier="standard")
    # No provider in "premium" -> use full priority order, don't error.
    resp = await r.complete([{"role": "user", "content": "x"}], max_tokens=10, tier="premium")
    assert resp.provider == "a"


@pytest.mark.asyncio
async def test_per_tenant_map_selects_model_across_vendors():
    # acme maps premium -> the deepseek-reasoner model key; routing must honor it.
    tenant_tiers = {"acme": {"premium": ["deepseek:deepseek-reasoner"]}}
    r = _router(tenant_tiers=tenant_tiers)
    r.register(_make_provider("openai", "gpt-4o-mini"), priority=0, tier="cheap")
    r.register(_make_provider("deepseek", "deepseek-reasoner"), priority=5, tier="premium")
    resp = await r.complete(
        [{"role": "user", "content": "x"}], max_tokens=10, tier="premium", tenant_id="acme"
    )
    assert resp.provider == "deepseek"


@pytest.mark.asyncio
async def test_scorer_exception_does_not_break_request():
    boom = type("S", (), {"score": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x"))})()
    r = _router(scorer=boom)
    r.register(_make_provider("only", "m"), priority=0, tier="standard")
    resp = await r.complete([{"role": "user", "content": "x"}], max_tokens=10)
    assert resp.provider == "only"
