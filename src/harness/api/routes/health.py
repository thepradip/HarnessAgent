"""Health, readiness, liveness, and metrics routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from harness.core.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter()

_VERSION = "0.1.0"


async def _check_redis(request: Request) -> bool:
    """Ping Redis and return True if healthy."""
    try:
        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            return False
        await redis.ping()
        return True
    except Exception:
        return False


async def _check_vector_db() -> bool:
    """Return True if the configured vector backend is reachable."""
    try:
        cfg = get_config()
        if cfg.vector_backend == "chroma":
            # We use embedded Chroma (a local persistent path), not a server,
            # so probing http://localhost:8000 would always report degraded.
            # Healthy = the configured persistence directory is present/creatable.
            import os
            path = cfg.chroma_path
            return os.path.isdir(path) or os.path.isdir(os.path.dirname(path) or ".")
        elif cfg.vector_backend == "qdrant":
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{cfg.qdrant_url}/healthz")
                return resp.status_code == 200
        return True
    except Exception:
        return False


async def _check_graph_db() -> bool:
    """Return True if the graph backend is reachable."""
    try:
        cfg = get_config()
        if cfg.graph_backend == "neo4j":
            import httpx
            # Neo4j browser health endpoint
            base_url = cfg.neo4j_url.replace("bolt://", "http://").replace("7687", "7474")
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{base_url}/")
                return resp.status_code < 500
        return True  # networkx is always in-memory
    except Exception:
        return False


def _check_llm() -> bool:
    """Return True if an LLM provider is configured.

    NOTE: ``/health`` is unauthenticated, so we deliberately do NOT make a live
    call to api.anthropic.com here — that would let any anonymous caller drive
    external traffic (and metered cost) by hammering the health endpoint. We
    only check that a key is configured; reachability is verified lazily on the
    first real LLM call (and surfaced via circuit-breaker metrics).
    """
    try:
        cfg = get_config()
        return bool(getattr(cfg, "anthropic_api_key", "") or getattr(cfg, "openai_api_key", ""))
    except Exception:
        return False


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Aggregate health check for all service dependencies.

    Returns:
        200 with status="ok" if all services healthy.
        200 with status="degraded" if any service is unhealthy (but process is alive).
    """
    redis_ok = await _check_redis(request)
    vector_ok = await _check_vector_db()
    graph_ok = await _check_graph_db()
    llm_ok = _check_llm()

    services = {
        "redis": redis_ok,
        "vector_db": vector_ok,
        "graph_db": graph_ok,
        "llm": llm_ok,
    }

    overall = "ok" if all(services.values()) else "degraded"
    status_code = status.HTTP_200_OK

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "services": services,
            "version": _VERSION,
        },
    )


@router.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    """Kubernetes readiness probe.

    The service is ready when Redis (our primary state store) is reachable.

    Returns:
        200 if ready, 503 if not ready.
    """
    redis_ok = await _check_redis(request)
    if not redis_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "reason": "Redis not available"},
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"ready": True},
    )


@router.get("/health/live")
async def liveness() -> JSONResponse:
    """Kubernetes liveness probe.

    Always returns 200 as long as the process is alive.

    Returns:
        200 always.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"alive": True},
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint in text exposition format.

    Returns:
        text/plain with Prometheus metrics.
    """
    try:
        from prometheus_client import REGISTRY, generate_latest  # type: ignore

        output = generate_latest(REGISTRY)
        return PlainTextResponse(
            content=output.decode("utf-8") if isinstance(output, bytes) else output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except ImportError:
        return PlainTextResponse(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )
    except Exception as exc:
        logger.error("Metrics generation failed: %s", exc)
        return PlainTextResponse(
            content=f"# Error generating metrics: {exc}\n",
            media_type="text/plain",
        )
