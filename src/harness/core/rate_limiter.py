"""Redis sliding-window rate limiter and FastAPI middleware for HarnessAgent."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from harness.core.errors import RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate-limit check."""

    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: float


def _parse_entry_cost(member: bytes | str) -> float:
    """Parse the cost encoded in a sorted-set member string.

    Members are stored as '{timestamp}:{id}:{cost}'.
    Returns 1.0 for legacy entries that lack the cost field.
    """
    try:
        raw = member.decode() if isinstance(member, bytes) else member
        parts = raw.split(":")
        if len(parts) >= 3:
            return float(parts[2])
    except (ValueError, AttributeError, IndexError):
        pass
    return 1.0


class RateLimiter:
    """Sliding-window rate limiter backed by Redis sorted sets."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        default_rpm: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self._redis = redis_client
        self.default_rpm = default_rpm
        self.window_seconds = window_seconds

    def _key(self, tenant_id: str, resource: str) -> str:
        """Build the Redis sorted-set key for a tenant+resource pair."""
        return f"harness:rate_limit:{tenant_id}:{resource}"

    async def check(
        self,
        tenant_id: str,
        resource: str = "api",
        cost: float = 1.0,
        limit: int | None = None,
    ) -> RateLimitResult:
        """Check the rate limit and record the request if allowed."""
        # An explicit limit of 0 means "deny everything"; only fall back to the
        # default when no limit was supplied at all.
        effective_limit = self.default_rpm if limit is None else limit
        now = time.time()
        window_start = now - self.window_seconds
        key = self._key(tenant_id, resource)
        reset_at = datetime.fromtimestamp(now + self.window_seconds, tz=timezone.utc)

        # Build the member string ONCE so the rollback can target this exact
        # entry. uuid4 guarantees uniqueness even for concurrent calls in the
        # same process (id(object()) of a freed temp object can collide).
        member = f"{now}:{uuid4().hex}:{cost}"

        pipe = self._redis.pipeline(transaction=True)
        # Remove expired entries
        pipe.zremrangebyscore(key, "-inf", window_start)
        # Fetch current window entries with their scores
        pipe.zrange(key, 0, -1, withscores=True)
        # Add the new request — encode cost in the member so we can sum it later
        pipe.zadd(key, {member: now})
        # Set TTL on the key
        pipe.expire(key, self.window_seconds * 2)
        results: list[Any] = await pipe.execute()

        # results[1] is list of (member, score) pairs — sum actual costs
        window_entries: list[tuple[bytes, float]] = results[1]
        current_cost = sum(_parse_entry_cost(m) for m, _ in window_entries)

        # Block if adding this request's cost would exceed the limit
        if current_cost + cost > effective_limit:
            # Undo exactly the entry we added — never a score range, which can
            # erase a concurrent request that landed at the same timestamp.
            await self._redis.zrem(key, member)
            oldest_score = window_entries[0][1] if window_entries else now
            retry_after = max(0.0, (oldest_score + self.window_seconds) - now)
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        remaining = max(0, int(effective_limit - (current_cost + cost)))
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=0.0,
        )

    async def require(
        self,
        tenant_id: str,
        resource: str = "api",
        cost: float = 1.0,
        limit: int | None = None,
    ) -> RateLimitResult:
        """Check the rate limit and raise RateLimitError if denied."""
        result = await self.check(tenant_id, resource, cost, limit)
        if not result.allowed:
            raise RateLimitError(
                f"Rate limit exceeded for tenant '{tenant_id}' on resource '{resource}'",
                retry_after=result.retry_after,
                context={
                    "tenant_id": tenant_id,
                    "resource": resource,
                    "retry_after": result.retry_after,
                    "reset_at": result.reset_at.isoformat(),
                },
            )
        return result


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces per-tenant rate limits via RateLimiter."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: RateLimiter,
        tenant_header: str = "X-Tenant-ID",
        default_tenant: str = "anonymous",
    ) -> None:
        super().__init__(app)
        self._limiter = rate_limiter
        self._tenant_header = tenant_header
        self._default_tenant = default_tenant

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Enforce the rate limit before passing the request downstream."""
        tenant_id = request.headers.get(self._tenant_header, self._default_tenant)
        try:
            result = await self._limiter.require(tenant_id, resource="api")
        except RateLimitError as exc:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": str(exc),
                    "retry_after": exc.retry_after,
                },
                headers={
                    "Retry-After": str(int(exc.retry_after)),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = result.reset_at.isoformat()
        return response
