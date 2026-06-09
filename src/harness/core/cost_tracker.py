"""LLM cost tracking with per-tenant budget enforcement for HarnessAgent."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

from harness.core.errors import FailureClass, RateLimitError

logger = logging.getLogger(__name__)

# Cost per 1,000,000 tokens in USD
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "claude-sonnet-4-6": {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5":  {"input": 0.25,  "output": 1.25},
    "claude-opus-4-7":   {"input": 15.0,  "output": 75.0},
    # OpenAI GPT-4o family
    "gpt-4o":            {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4.5":           {"input": 75.0,  "output": 150.0},
    # OpenAI GPT-5 family (announced pricing; update when GA)
    "gpt-5":             {"input": 2.50,  "output": 10.0},
    "gpt-5-mini":        {"input": 0.40,  "output": 1.60},
    # OpenAI o-series reasoning models
    "o1":                {"input": 15.0,  "output": 60.0},
    "o1-mini":           {"input": 1.10,  "output": 4.40},
    "o1-preview":        {"input": 15.0,  "output": 60.0},
    "o3":                {"input": 10.0,  "output": 40.0},
    "o3-mini":           {"input": 1.10,  "output": 4.40},
    "o4-mini":           {"input": 1.10,  "output": 4.40},
    # Local / self-hosted — no API cost
    "local":             {"input": 0.0,   "output": 0.0},
    "vllm":              {"input": 0.0,   "output": 0.0},
    "ollama":            {"input": 0.0,   "output": 0.0},
    "llamacpp":          {"input": 0.0,   "output": 0.0},
    "sglang":            {"input": 0.0,   "output": 0.0},
}

_COST_LEDGER_PREFIX = "harness:cost_ledger"
_TENANT_SPEND_PREFIX = "harness:tenant_spend"


def _model_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute the USD cost for a given model and token counts."""
    costs = MODEL_COSTS.get(model)
    if costs is None:
        # Try prefix matching (e.g. "claude-sonnet-4-6-20250101" -> "claude-sonnet-4-6")
        for key in MODEL_COSTS:
            if model.startswith(key):
                costs = MODEL_COSTS[key]
                break
    if costs is None:
        costs = {"input": 0.0, "output": 0.0}
    per_m = 1_000_000.0
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / per_m


@dataclass
class RunCost:
    """Cost record for a single agent run."""

    run_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = ""


class CostTracker:
    """Records LLM usage costs and enforces per-tenant USD budgets via Redis."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        budget_usd_per_tenant: float = 100.0,
    ) -> None:
        self._redis = redis_client
        self._budget = budget_usd_per_tenant

    def _ledger_key(self, run_id: str) -> str:
        """Return the Redis key for per-run cost records."""
        return f"{_COST_LEDGER_PREFIX}:{run_id}"

    def _spend_key(self, tenant_id: str, window: str) -> str:
        """Return the Redis key for accumulated tenant spend within a window."""
        now = datetime.now(timezone.utc)
        if window == "month":
            period = f"{now.year}-{now.month:02d}"
        elif window == "day":
            period = f"{now.year}-{now.month:02d}-{now.day:02d}"
        else:
            period = window
        return f"{_TENANT_SPEND_PREFIX}:{tenant_id}:{period}"

    async def record(
        self,
        run_id: str,
        tenant_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> RunCost:
        """Record costs for a run and accumulate tenant spend counters."""
        cost = _model_cost_usd(model, input_tokens, output_tokens)
        run_cost = RunCost(
            run_id=run_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            tenant_id=tenant_id,
        )

        pipe = self._redis.pipeline(transaction=False)
        # Store the run-level cost record (TTL 90 days)
        ledger_key = self._ledger_key(run_id)
        pipe.hset(
            ledger_key,
            mapping={
                "run_id": run_id,
                "tenant_id": tenant_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": str(cost),
                "timestamp": run_cost.timestamp.isoformat(),
            },
        )
        pipe.expire(ledger_key, 86400 * 90)

        # Accumulate monthly spend
        month_key = self._spend_key(tenant_id, "month")
        pipe.incrbyfloat(month_key, cost)
        pipe.expire(month_key, 86400 * 32)

        # Accumulate daily spend
        day_key = self._spend_key(tenant_id, "day")
        pipe.incrbyfloat(day_key, cost)
        pipe.expire(day_key, 86400 * 2)

        await pipe.execute()
        logger.debug(
            "Recorded cost: run_id=%s tenant=%s model=%s cost=$%.6f",
            run_id,
            tenant_id,
            model,
            cost,
        )
        return run_cost

    async def get_tenant_spend(
        self,
        tenant_id: str,
        window: str = "month",
    ) -> float:
        """Return total USD spend for a tenant in the given time window."""
        key = self._spend_key(tenant_id, window)
        raw = await self._redis.get(key)
        if raw is None:
            return 0.0
        return float(raw)

    async def check_budget(self, tenant_id: str) -> bool:
        """Return True if tenant is under budget; raise RateLimitError if over."""
        spend = await self.get_tenant_spend(tenant_id, window="month")
        if spend >= self._budget:
            raise RateLimitError(
                f"Tenant '{tenant_id}' has exceeded monthly cost budget "
                f"(${spend:.4f} >= ${self._budget:.2f})",
                retry_after=0.0,
                failure_class=FailureClass.LLM_RATE_LIMIT,
                context={
                    "tenant_id": tenant_id,
                    "spend_usd": spend,
                    "budget_usd": self._budget,
                },
            )
        return True

    async def get_run_cost(self, run_id: str) -> RunCost | None:
        """Retrieve the cost record for a specific run, or None if not found."""
        data = await self._redis.hgetall(self._ledger_key(run_id))
        if not data:
            return None

        # Normalise keys/values: production clients use decode_responses=True
        # (str keys), but a raw client returns bytes. Handle both.
        def _norm(value: Any) -> Any:
            return value.decode() if isinstance(value, bytes) else value

        record = {_norm(k): _norm(v) for k, v in data.items()}
        return RunCost(
            run_id=record["run_id"],
            tenant_id=record["tenant_id"],
            model=record["model"],
            input_tokens=int(record["input_tokens"]),
            output_tokens=int(record["output_tokens"]),
            cost_usd=float(record["cost_usd"]),
            timestamp=datetime.fromisoformat(record["timestamp"]),
        )
