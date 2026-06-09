"""Health-aware, context-window-aware LLM provider router for HarnessAgent."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from harness.core.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from harness.core.context import LLMResponse
from harness.core.errors import CircuitOpenError, FailureClass, LLMError
from harness.core.protocols import LLMProvider

logger = logging.getLogger(__name__)

# Errors that should trigger fallback to the next provider
_RETRYABLE = frozenset({
    FailureClass.LLM_RATE_LIMIT,
    FailureClass.LLM_TIMEOUT,
    FailureClass.LLM_ERROR,
})

# Errors worth retrying on the same provider (with backoff) before falling back
_BACKOFF_RETRYABLE = frozenset({
    FailureClass.LLM_RATE_LIMIT,
    FailureClass.LLM_TIMEOUT,
})

_BACKOFF_DELAYS = (1.0, 2.0, 4.0)  # seconds; each multiplied by ±20% jitter


@dataclass
class ProviderEntry:
    """A provider registered with the router."""
    priority: int
    provider: LLMProvider
    context_window: int = 200_000
    enabled: bool = True
    tier: str = "standard"          # cheap | standard | premium — cost/capability band
    cost: float = 0.0               # informational relative cost weight (USD/M output)


@dataclass
class LLMRouterConfig:
    """Configuration for LLMRouter.

    Cost-aware routing (all optional — when unset the router behaves exactly as
    a priority-ordered router):

    - ``scorer``: a :class:`~harness.llm.complexity.ComplexityScorer`. When set
      and no explicit ``tier`` is passed to ``complete()``, the router scores the
      request to pick a tier.
    - ``tenant_tiers``: per-tenant ``{tier -> [provider_key, ...]}`` maps, where a
      provider key is ``"provider_name:model"`` (or a bare model id). Lets each
      tenant supply its own model per tier across vendors.
    - ``default_tiers``: fallback ``{tier -> [provider_key, ...]}`` used when a
      tenant has no entry. When neither map is set, tiering falls back to each
      provider's ``ProviderEntry.tier`` tag.
    """
    providers: list[ProviderEntry] = field(default_factory=list)
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    circuit_success_threshold: int = 2
    scorer: Any | None = None       # ComplexityScorer | None
    tenant_tiers: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    default_tiers: dict[str, list[str]] = field(default_factory=dict)


class LLMRouter:
    """Routes LLM completion requests across multiple providers with circuit breaking.

    Optional semantic caching: pass a ``SemanticLLMCache`` to avoid redundant
    LLM calls for semantically similar queries (cosine similarity ≥ threshold).
    """

    def __init__(
        self,
        config: LLMRouterConfig | None = None,
        registry: CircuitBreakerRegistry | None = None,
        cache: Any | None = None,
        health_ttl: float = 5.0,
    ) -> None:
        self._config = config or LLMRouterConfig()
        self._registry = registry or CircuitBreakerRegistry()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cache = cache  # SemanticLLMCache | None
        self._health_cache: dict[str, tuple[bool, float]] = {}
        self._health_ttl = health_ttl

    def register(
        self,
        provider: LLMProvider,
        priority: int = 0,
        context_window: int = 200_000,
        tier: str = "standard",
        cost: float = 0.0,
    ) -> None:
        """Add a provider to the router.

        ``tier`` tags the provider's cost/capability band (``cheap`` / ``standard``
        / ``premium``); it is used by complexity-based routing when no explicit
        per-tenant tier map is configured. ``cost`` is an informational weight.
        """
        self._config.providers.append(
            ProviderEntry(
                priority=priority,
                provider=provider,
                context_window=context_window,
                tier=tier,
                cost=cost,
            )
        )
        self._config.providers.sort(key=lambda e: e.priority)

    def _get_breaker(self, provider: LLMProvider) -> CircuitBreaker:
        """Return or create the circuit breaker for a provider (synchronous)."""
        key = f"{provider.provider_name}:{provider.model}"
        if key not in self._breakers:
            self._breakers[key] = self._registry.get_or_create(
                name=key,
                failure_threshold=self._config.circuit_failure_threshold,
                recovery_timeout=self._config.circuit_recovery_timeout,
                success_threshold=self._config.circuit_success_threshold,
            )
        return self._breakers[key]

    def _sorted_providers(self) -> list[ProviderEntry]:
        return [e for e in sorted(self._config.providers, key=lambda e: e.priority) if e.enabled]

    def _resolve_tier(
        self,
        messages: list[dict[str, Any]],
        *,
        tier: str | None,
        system: str | None,
        tools: list[dict[str, Any]] | None,
        required_context: int,
        max_tokens: int,
    ) -> str | None:
        """Pick the target tier: explicit arg wins, else the scorer, else None."""
        if tier is not None:
            return tier
        scorer = self._config.scorer
        if scorer is None:
            return None
        try:
            return scorer.score(
                messages,
                system=system,
                tools=tools,
                required_context=required_context,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # never let routing-policy errors break a request
            logger.debug("Complexity scorer failed (ignoring tier): %s", exc)
            return None

    def _ordered_entries(
        self, tenant_id: str | None, tier: str | None
    ) -> list[ProviderEntry]:
        """Enabled providers, target-tier first then the rest as fallback.

        Tier membership comes from the tenant's tier map (or the default map) when
        configured, falling back to each entry's own ``tier`` tag. Health, context
        window, and circuit-breaker handling are unchanged — this only reorders.
        """
        entries = self._sorted_providers()
        if tier is None:
            return entries

        tier_map = self._config.tenant_tiers.get(tenant_id or "") or self._config.default_tiers
        preferred_keys = set(tier_map.get(tier, [])) if tier_map else None

        def in_tier(e: ProviderEntry) -> bool:
            if preferred_keys is not None:
                key = f"{e.provider.provider_name}:{e.provider.model}"
                return key in preferred_keys or e.provider.model in preferred_keys
            return e.tier == tier

        preferred = [e for e in entries if in_tier(e)]
        if not preferred:
            return entries  # nothing matched the tier — fall back to full order
        rest = [e for e in entries if not in_tier(e)]
        return preferred + rest

    async def _is_healthy(self, provider: LLMProvider) -> bool:
        """Return cached health status; re-check after _health_ttl seconds."""
        key = f"{provider.provider_name}:{provider.model}"
        cached = self._health_cache.get(key)
        if cached is not None and time.monotonic() < cached[1]:
            return cached[0]
        try:
            result = await provider.health_check()
        except Exception:
            result = False
        self._health_cache[key] = (result, time.monotonic() + self._health_ttl)
        return result

    async def _try_with_backoff(
        self,
        provider: LLMProvider,
        breaker: CircuitBreaker,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> LLMResponse:
        """Attempt provider.complete() up to 4 times (initial + 3 retries) with
        exponential backoff for rate-limit and timeout errors."""
        last: Exception | None = None
        for attempt, delay in enumerate([0.0, *_BACKOFF_DELAYS]):
            if delay:
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
            try:
                async with breaker.call():
                    return await provider.complete(messages, **kw)
            except CircuitOpenError:
                raise  # circuit opened mid-retry; let router move to next provider
            except LLMError as exc:
                if exc.failure_class not in _BACKOFF_RETRYABLE:
                    raise
                last = exc
                logger.debug(
                    "backoff attempt %d/3 for %s:%s — %s",
                    attempt + 1, provider.provider_name, provider.model, exc.failure_class,
                )
        raise last  # type: ignore[misc]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        required_context: int = 0,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        skip_cache: bool = False,
        tenant_id: str | None = None,
        tier: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Route a completion request, falling back on retryable errors.

        Cost-aware routing
        -------------------
        When ``tier`` is given it selects the target cost/capability band
        directly. Otherwise, if a complexity ``scorer`` is configured, the
        request is scored to pick a tier. ``tenant_id`` selects that tenant's
        tier→model map (so different tenants can be served by different models).
        Providers in the target tier are tried first; the router falls back
        through the remaining providers by priority on any retryable error.

        Cache behaviour
        ---------------
        When a ``SemanticLLMCache`` is attached, the cache is checked before
        routing.  A cache hit returns immediately with ``cached=True``.
        After a successful provider call the response is stored in the cache.
        Pass ``skip_cache=True`` to bypass for calls that must be fresh
        (e.g. tool-use steps where exact output matters).
        """
        # ── Semantic cache lookup ───────────────────────────────────────
        if self._cache is not None and not skip_cache and not tools:
            try:
                cached_text = await self._cache.get(messages)
                if cached_text is not None:
                    logger.debug("LLM cache hit — skipping provider call")
                    try:
                        from harness.observability.metrics import get_prometheus_metrics
                        m = get_prometheus_metrics()
                        if m and hasattr(m, "llm_cache_hits_total"):
                            m.llm_cache_hits_total.labels(provider="cache").inc()
                    except Exception:
                        pass
                    return LLMResponse(
                        content=cached_text,
                        tool_calls=[],
                        input_tokens=0,
                        output_tokens=0,
                        model="cached",
                        provider="cache",
                        cached=True,
                    )
            except Exception as exc:
                logger.debug("Cache lookup failed (continuing without cache): %s", exc)

        last_exc: Exception | None = None

        target_tier = self._resolve_tier(
            messages,
            tier=tier,
            system=system,
            tools=tools,
            required_context=required_context,
            max_tokens=max_tokens,
        )
        if target_tier is not None:
            logger.debug("Routing tier=%s tenant=%s", target_tier, tenant_id)

        for entry in self._ordered_entries(tenant_id, target_tier):
            provider = entry.provider

            # Skip providers whose context window is too small
            if required_context > 0 and required_context > entry.context_window:
                logger.debug(
                    "Skipping %s:%s — required_context %d > window %d",
                    provider.provider_name, provider.model,
                    required_context, entry.context_window,
                )
                continue

            # Skip providers that fail a health check (result is TTL-cached)
            if not await self._is_healthy(provider):
                logger.debug("Skipping %s:%s — health check failed",
                             provider.provider_name, provider.model)
                continue

            breaker = self._get_breaker(provider)

            try:
                response = await self._try_with_backoff(
                    provider,
                    breaker,
                    messages,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    **kwargs,
                )
                # ── Store in cache (only text responses, not tool calls) ─
                if (self._cache is not None and not skip_cache
                        and not tools and response.content):
                    try:
                        await self._cache.set(messages, response.content)
                    except Exception as exc:
                        logger.debug("Cache store failed: %s", exc)
                return response
            except CircuitOpenError as exc:
                logger.warning("Circuit open for %s:%s, trying next",
                               provider.provider_name, provider.model)
                last_exc = exc
                continue
            except LLMError as exc:
                if exc.failure_class in _RETRYABLE:
                    logger.warning(
                        "All retries exhausted for %s:%s (%s), trying next provider",
                        provider.provider_name, provider.model, exc.failure_class,
                    )
                    last_exc = exc
                    continue
                raise

        raise LLMError(
            f"All providers exhausted. Last error: {last_exc}",
            failure_class=FailureClass.LLM_ERROR,
        ) from last_exc

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
        tier: str | None = None,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from the first available provider (tier-aware)."""
        target_tier = self._resolve_tier(
            messages, tier=tier, system=system, tools=tools,
            required_context=0, max_tokens=kwargs.get("max_tokens", 0),
        )
        stream_kwargs = dict(kwargs)
        if system is not None:
            stream_kwargs["system"] = system
        if tools is not None:
            stream_kwargs["tools"] = tools
        for entry in self._ordered_entries(tenant_id, target_tier):
            provider = entry.provider
            if not await self._is_healthy(provider):
                continue

            breaker = self._get_breaker(provider)
            try:
                async with breaker.call():
                    async for token in provider.stream(messages, **stream_kwargs):
                        yield token
                    return
            except (CircuitOpenError, LLMError):
                continue

        raise LLMError("No providers available for streaming", failure_class=FailureClass.LLM_ERROR)

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered providers concurrently."""
        async def _check(entry: ProviderEntry) -> tuple[str, bool]:
            key = f"{entry.provider.provider_name}:{entry.provider.model}"
            try:
                return key, await entry.provider.health_check()
            except Exception:
                return key, False

        results = await asyncio.gather(*[_check(e) for e in self._sorted_providers()])
        return dict(results)
