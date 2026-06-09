"""Unit tests for CostTracker, incl. the decode_responses regression (item 3)."""

from __future__ import annotations

import pytest

from harness.core.cost_tracker import CostTracker


@pytest.mark.asyncio
async def test_get_run_cost_with_decode_responses_client(redis_client):
    """get_run_cost must work against a decode_responses=True client.

    Regression: the old code indexed hgetall with byte keys (data[b"run_id"]),
    which raised KeyError on every record stored by a production client that
    decodes responses to str.
    """
    tracker = CostTracker(redis_client=redis_client, budget_usd_per_tenant=100.0)

    await tracker.record(
        run_id="run-decode",
        tenant_id="tenant-a",
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
    )

    cost = await tracker.get_run_cost("run-decode")
    assert cost is not None
    assert cost.run_id == "run-decode"
    assert cost.tenant_id == "tenant-a"
    assert cost.model == "claude-sonnet-4-6"
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500
    assert cost.cost_usd > 0.0


@pytest.mark.asyncio
async def test_get_run_cost_missing_returns_none(redis_client):
    tracker = CostTracker(redis_client=redis_client)
    assert await tracker.get_run_cost("nope") is None
