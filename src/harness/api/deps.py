"""FastAPI dependency injection for HarnessAgent."""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from harness.core.config import DEFAULT_JWT_SECRET, get_config

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# Redis key prefix for API key records
_APIKEY_PREFIX = "harness:apikey"


def _hash_api_key(raw_key: str) -> str:
    """Return a truncated SHA-256 hash of the raw key for Redis lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Infrastructure dependencies
# ---------------------------------------------------------------------------


async def get_agent_factory(request: Request) -> Any:
    """Return the real agent factory from app.state (None in read-only mode)."""
    return getattr(request.app.state, "agent_factory", None)


async def get_redis(request: Request) -> Any:
    """Return the Redis client stored in app state.

    Raises:
        HTTPException 503 if Redis is not available.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis not available",
        )
    return redis


async def get_runner(request: Request) -> Any:
    """Return the singleton AgentRunner stored in app state."""
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runner not initialised",
        )
    return runner


async def get_memory_manager(request: Request) -> Any:
    """Return the singleton MemoryManager stored in app state."""
    mm = getattr(request.app.state, "memory_manager", None)
    if mm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not initialised",
        )
    return mm


async def get_event_bus(request: Request) -> Any:
    """Return the singleton EventBus stored in app state."""
    bus = getattr(request.app.state, "event_bus", None)
    if bus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event bus not initialised",
        )
    return bus


async def get_hitl_manager(request: Request) -> Any:
    """Return the HITLManager from app state."""
    hitl = getattr(request.app.state, "hitl_manager", None)
    if hitl is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HITL manager not initialised",
        )
    return hitl


async def get_prompt_manager(request: Request) -> Any:
    """Return the PromptManager from app state."""
    pm = getattr(request.app.state, "prompt_manager", None)
    if pm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prompt manager not initialised",
        )
    return pm


async def get_hermes_loop(request: Request) -> Any:
    """Return the HermesLoop from app state."""
    hermes = getattr(request.app.state, "hermes_loop", None)
    if hermes is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hermes loop not initialised",
        )
    return hermes


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------


def _decode_jwt(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException 401 on invalid or expired token.
        ImportError if python-jose is not installed (server misconfiguration —
            must surface as a 500, not a client-facing 401).
    """
    # Deliberately outside the try: a missing JWT library is a deployment
    # problem, not bad client credentials.
    from jose import JWTError, jwt  # type: ignore

    cfg = get_config()

    # Defense in depth: never accept tokens signed with the placeholder
    # secret in prod (the config validator should already have refused to
    # start, but a stale cached Settings must not weaken auth).
    if cfg.environment == "prod" and cfg.jwt_secret_key == DEFAULT_JWT_SECRET:
        logger.error("Rejecting JWT auth: default jwt_secret_key in prod")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT authentication is not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            cfg.jwt_secret_key,
            algorithms=["HS256"],
        )
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _resolve_api_key(request: Request, raw_key: str) -> Optional[str]:
    """Look up a raw API key in Redis and return the tenant_id, or None."""
    try:
        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            return None
        key_hash = _hash_api_key(raw_key)
        stored = await redis.hgetall(f"{_APIKEY_PREFIX}:{key_hash}")
        if not stored:
            return None
        tenant_id = stored.get("tenant_id") or stored.get(b"tenant_id")
        if isinstance(tenant_id, bytes):
            tenant_id = tenant_id.decode()
        # Update last_used timestamp (fire-and-forget)
        import time
        try:
            await redis.hset(f"{_APIKEY_PREFIX}:{key_hash}", "last_used", str(time.time()))
        except Exception:
            pass
        return str(tenant_id) if tenant_id else None
    except Exception as exc:
        logger.debug("API key lookup failed: %s", exc)
        return None


async def create_api_key(
    request: Request,
    tenant_id: str,
    name: str = "default",
) -> str:
    """Create a new API key for *tenant_id* and store its hash in Redis.

    Args:
        request:   FastAPI request (for Redis access).
        tenant_id: The tenant this key belongs to.
        name:      Human-readable label for the key.

    Returns:
        The raw API key string (shown once, never stored in plain text).
    """
    import time
    raw_key = secrets.token_urlsafe(32)
    key_hash = _hash_api_key(raw_key)
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise RuntimeError("Redis not available")
    await redis.hset(
        f"{_APIKEY_PREFIX}:{key_hash}",
        mapping={
            "tenant_id": tenant_id,
            "name": name,
            "created_at": str(time.time()),
            "last_used": "",
        },
    )
    # No TTL — API keys are permanent until revoked
    logger.info("Created API key '%s' for tenant %s", name, tenant_id)
    return raw_key


async def revoke_api_key(request: Request, raw_key: str) -> bool:
    """Delete an API key from Redis.  Returns True if it existed."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return False
    key_hash = _hash_api_key(raw_key)
    deleted = await redis.delete(f"{_APIKEY_PREFIX}:{key_hash}")
    return bool(deleted)


async def get_current_tenant(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """Extract the tenant_id from a JWT bearer token OR an X-API-Key header.

    Auth precedence: API key → JWT → dev default.

    Args:
        request:   FastAPI request (needed for Redis API-key lookup).
        token:     JWT bearer token (optional).
        x_api_key: Raw API key from ``X-API-Key`` header (optional).

    Returns:
        tenant_id string.

    Raises:
        HTTPException 401 if credentials are missing (in prod) or invalid.
    """
    cfg = get_config()

    # 1. API key (service accounts, CI/CD pipelines)
    if x_api_key:
        tenant_id = await _resolve_api_key(request, x_api_key)
        if tenant_id:
            return tenant_id
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. JWT bearer token
    if token:
        payload = _decode_jwt(token)
        tenant_id = payload.get("tenant_id") or payload.get("sub")
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing tenant_id claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return str(tenant_id)

    # 3. Dev mode fallback
    if cfg.environment == "dev":
        return "default"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated — provide Bearer token or X-API-Key header",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    """Extract the full user info from JWT or API key.

    Returns a minimal user dict if no credentials are provided in dev mode.

    Returns:
        Dict with at minimum: sub, tenant_id, role.

    Raises:
        HTTPException 401 if invalid.
    """
    cfg = get_config()

    if x_api_key:
        tenant_id = await _resolve_api_key(request, x_api_key)
        if tenant_id:
            return {"sub": f"apikey:{tenant_id}", "tenant_id": tenant_id, "role": "service"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token:
        return _decode_jwt(token)

    if cfg.environment == "dev":
        return {"sub": "dev-user", "tenant_id": "default", "role": "admin"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
