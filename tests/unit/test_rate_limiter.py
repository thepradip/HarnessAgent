"""Unit tests for RateLimiter using fakeredis."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from harness.core.rate_limiter import RateLimiter


@pytest_asyncio.fixture
async def rate_limiter(redis_client):
    """Rate limiter backed by fakeredis, limit of 5 req/60s."""
    return RateLimiter(redis_client=redis_client, default_rpm=5, window_seconds=60)


@pytest.mark.asyncio
async def test_allows_requests_under_limit(rate_limiter):
    for i in range(5):
        result = await rate_limiter.check(tenant_id="tenant-a", resource="api")
        assert result.allowed is True, f"Request {i} should be allowed"
        assert result.retry_after == 0.0


@pytest.mark.asyncio
async def test_blocks_over_limit(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-b", resource="api")

    result = await rate_limiter.check(tenant_id="tenant-b", resource="api")
    assert result.allowed is False
    assert result.retry_after > 0


@pytest.mark.asyncio
async def test_sliding_window_clears_old_requests(redis_client):
    rl = RateLimiter(redis_client=redis_client, default_rpm=3, window_seconds=1)

    for _ in range(3):
        await rl.check(tenant_id="tenant-window", resource="api")

    result = await rl.check(tenant_id="tenant-window", resource="api")
    assert result.allowed is False

    await asyncio.sleep(1.1)

    result = await rl.check(tenant_id="tenant-window", resource="api")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_different_tenants_independent(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-x", resource="api")

    result = await rate_limiter.check(tenant_id="tenant-x", resource="api")
    assert result.allowed is False

    result = await rate_limiter.check(tenant_id="tenant-y", resource="api")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_different_resources_independent(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-r", resource="search")

    result = await rate_limiter.check(tenant_id="tenant-r", resource="search")
    assert result.allowed is False

    result = await rate_limiter.check(tenant_id="tenant-r", resource="ingest")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_returns_retry_after_when_blocked(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-retry", resource="llm")

    result = await rate_limiter.check(tenant_id="tenant-retry", resource="llm")
    assert result.allowed is False
    assert isinstance(result.retry_after, float)
    assert 0.0 < result.retry_after <= 60.0


# ---------------------------------------------------------------------------
# Regression tests for the deny-path rollback fix (item 4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denied_rollback_does_not_drop_concurrent_entry(redis_client):
    """A denied request must roll back ONLY its own entry, not a neighbour's.

    Regression: the old code used zremrangebyscore(now, now+0.001) which could
    erase a concurrent entry that landed at the same timestamp.
    """
    rl = RateLimiter(redis_client=redis_client, default_rpm=2, window_seconds=60)

    r1 = await rl.check(tenant_id="t-roll", resource="api")
    r2 = await rl.check(tenant_id="t-roll", resource="api")
    assert r1.allowed and r2.allowed

    key = rl._key("t-roll", "api")
    assert await redis_client.zcard(key) == 2

    denied = await rl.check(tenant_id="t-roll", resource="api")
    assert denied.allowed is False

    assert await redis_client.zcard(key) == 2, "rollback erased a recorded entry"


@pytest.mark.asyncio
async def test_unique_members_no_collision_same_timestamp(redis_client, monkeypatch):
    """Two requests at the same time.time() must produce distinct members.

    Regression: id(object()) of a freed temp object could collide, so the
    second zadd would overwrite the first entry.
    """
    import harness.core.rate_limiter as rl_mod

    rl = RateLimiter(redis_client=redis_client, default_rpm=10, window_seconds=60)
    monkeypatch.setattr(rl_mod.time, "time", lambda: 1000.0)

    await rl.check(tenant_id="t-uniq", resource="api")
    await rl.check(tenant_id="t-uniq", resource="api")

    key = rl._key("t-uniq", "api")
    assert await redis_client.zcard(key) == 2, "same-timestamp members collided"


@pytest.mark.asyncio
async def test_explicit_zero_limit_denies(redis_client):
    """An explicit limit=0 must deny, not fall back to default_rpm."""
    rl = RateLimiter(redis_client=redis_client, default_rpm=60, window_seconds=60)
    result = await rl.check(tenant_id="t-zero", resource="api", limit=0)
    assert result.allowed is False
