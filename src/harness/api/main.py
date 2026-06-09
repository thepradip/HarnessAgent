"""FastAPI application factory for HarnessAgent."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import REGISTRY

from harness.core.config import get_config
from harness.core.errors import (
    BudgetExceeded,
    CircuitOpenError,
    FailureClass,
    HarnessError,
    LLMError,
    SafetyViolation,
    ToolError,
)

logger = logging.getLogger(__name__)


def _failure_class_to_status(failure_class: Any) -> int:
    """Map a FailureClass to an appropriate HTTP status code."""
    if failure_class is None:
        return 500
    fc_str = str(failure_class)
    client_errors = {
        FailureClass.TOOL_SCHEMA_ERROR.value,
        FailureClass.TOOL_NOT_FOUND.value,
        FailureClass.SAFETY_INPUT.value,
        FailureClass.SAFETY_STEP.value,
        FailureClass.SAFETY_OUTPUT.value,
        FailureClass.BUDGET_STEPS.value,
        FailureClass.BUDGET_TOKENS.value,
        FailureClass.BUDGET_TIME.value,
    }
    if fc_str in client_errors:
        return 400
    if fc_str in (FailureClass.LLM_RATE_LIMIT.value, FailureClass.BUDGET_COST.value):
        return 429
    return 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init and cleanup."""
    cfg = get_config()

    # Initialize Redis
    try:
        redis_client = aioredis.from_url(
            cfg.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await redis_client.ping()
        app.state.redis = redis_client
        logger.info("Redis connected: %s", cfg.redis_url)
    except Exception as exc:
        logger.warning("Redis not available at startup: %s", exc)
        app.state.redis = None

    # Per-tenant API rate limiter (read by RateLimitMiddleware). Requires Redis;
    # when Redis is down the middleware fails open.
    app.state.rate_limiter = None
    if cfg.rate_limit_enabled and app.state.redis is not None:
        try:
            from harness.core.rate_limiter import RateLimiter
            app.state.rate_limiter = RateLimiter(
                app.state.redis, default_rpm=cfg.rate_limit_rpm
            )
            logger.info("Rate limiter enabled (%d rpm/tenant)", cfg.rate_limit_rpm)
        except Exception as exc:
            logger.warning("Rate limiter init failed: %s", exc)

    # Prometheus metrics (create a fresh registry if running in tests)
    try:
        from prometheus_client import Counter
        app.state.metrics = {
            "requests_total": Counter(
                "harness_requests_total",
                "Total HTTP requests",
                ["method", "path", "status"],
                registry=REGISTRY,
            )
        }
    except Exception as exc:
        logger.debug("Prometheus metrics init: %s", exc)
        app.state.metrics = {}

    # Wire the real agent factory so POST /runs can execute agents directly
    try:
        from harness.workers.agent_worker import build_agent_factory
        app.state.agent_factory = build_agent_factory(cfg)
        logger.info("Agent factory initialised (%s)", cfg.environment)
    except Exception as exc:
        logger.warning("Agent factory unavailable — runs will execute via worker only: %s", exc)
        app.state.agent_factory = None

    # Single shared TraceRecorder (own connection pool). Reused across requests
    # so we don't open/leak a fresh pool (max 20 conns) per trace query.
    try:
        from harness.observability.trace_recorder import TraceRecorder
        app.state.trace_recorder = TraceRecorder.create(redis_url=cfg.redis_url)
        logger.info("TraceRecorder initialised")
    except Exception as exc:
        logger.warning("TraceRecorder unavailable: %s", exc)
        app.state.trace_recorder = None

    logger.info("HarnessAgent API started (env=%s)", cfg.environment)
    yield

    # Shutdown cleanup
    if getattr(app.state, "trace_recorder", None) is not None:
        try:
            await app.state.trace_recorder.close()
            logger.info("TraceRecorder connection closed")
        except Exception as exc:
            logger.debug("TraceRecorder close failed: %s", exc)

    if hasattr(app.state, "redis") and app.state.redis is not None:
        await app.state.redis.aclose()
        logger.info("Redis connection closed")

    # Flush + stop OTel exporter threads if a tracer was installed.
    try:
        from harness.observability.tracer import shutdown_tracer_provider
        shutdown_tracer_provider()
    except Exception as exc:
        logger.debug("Tracer shutdown failed: %s", exc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured FastAPI instance.
    """
    cfg = get_config()

    app = FastAPI(
        title="HarnessAgent",
        version="0.1.0",
        description="Production-grade multi-agent orchestration harness",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.environment == "dev" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (request ID + tenant)
    from harness.api.middleware import RequestIDMiddleware, TenantMiddleware
    app.add_middleware(TenantMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Per-tenant rate limiting. The limiter itself is created in lifespan (it
    # needs the live Redis client) and read from app.state per request; the
    # middleware fails open when it's absent.
    if cfg.rate_limit_enabled:
        from harness.core.rate_limiter import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(HarnessError)
    async def harness_error_handler(request: Request, exc: HarnessError) -> JSONResponse:
        status = _failure_class_to_status(exc.failure_class)
        return JSONResponse(
            status_code=status,
            content={
                "error": str(exc),
                "failure_class": str(exc.failure_class),
                "error_id": exc.error_id,
                "type": type(exc).__name__,
            },
            headers={"X-Error-ID": exc.error_id},
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "error": str(exc),
                "failure_class": str(exc.failure_class),
                "error_id": exc.error_id,
                "type": "LLMError",
            },
        )

    @app.exception_handler(ToolError)
    async def tool_error_handler(request: Request, exc: ToolError) -> JSONResponse:
        status = 400 if exc.failure_class == FailureClass.TOOL_SCHEMA_ERROR else 500
        return JSONResponse(
            status_code=status,
            content={
                "error": str(exc),
                "failure_class": str(exc.failure_class),
                "error_id": exc.error_id,
                "tool_name": exc.tool_name,
                "type": "ToolError",
            },
        )

    @app.exception_handler(SafetyViolation)
    async def safety_error_handler(request: Request, exc: SafetyViolation) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "failure_class": str(exc.failure_class),
                "error_id": exc.error_id,
                "guard_source": exc.guard_source,
                "type": "SafetyViolation",
            },
        )

    @app.exception_handler(BudgetExceeded)
    async def budget_error_handler(request: Request, exc: BudgetExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "error": str(exc),
                "failure_class": str(exc.failure_class),
                "error_id": exc.error_id,
                "type": "BudgetExceeded",
            },
        )

    @app.exception_handler(CircuitOpenError)
    async def circuit_error_handler(request: Request, exc: CircuitOpenError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "failure_class": str(exc.failure_class),
                "error_id": exc.error_id,
                "service_name": exc.service_name,
                "type": "CircuitOpenError",
            },
            headers={"Retry-After": "30"},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex
        logger.exception("Unhandled exception %s: %s", error_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_id": error_id,
                "type": type(exc).__name__,
            },
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    from harness.api.routes.evals import router as evals_router
    from harness.api.routes.feedback import router as feedback_router
    from harness.api.routes.health import router as health_router
    from harness.api.routes.improvement import router as improvement_router
    from harness.api.routes.memory import router as memory_router
    from harness.api.routes.runs import router as runs_router
    from harness.api.routes.steps import router as steps_router
    from harness.api.routes.traces import router as traces_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(runs_router, prefix="/runs", tags=["Runs"])
    app.include_router(steps_router, prefix="/runs", tags=["Steps"])
    app.include_router(traces_router, prefix="/runs", tags=["Traces"])
    app.include_router(feedback_router, prefix="/runs", tags=["Feedback"])
    app.include_router(memory_router, prefix="/memory", tags=["Memory"])
    app.include_router(evals_router, prefix="/evals", tags=["Evals"])
    app.include_router(improvement_router, tags=["Improvement"])

    # ------------------------------------------------------------------
    # Operator UI
    # ------------------------------------------------------------------
    ui_path = Path(__file__).resolve().parents[3] / "ui" / "dashboard.html"

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(ui_path)

    @app.get("/ui", include_in_schema=False)
    async def dashboard_alias() -> FileResponse:
        return FileResponse(ui_path)

    return app
