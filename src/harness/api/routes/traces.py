"""Trace query API — GET /runs/{run_id}/trace and GET /runs/spans/{span_id}."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from harness.api.deps import get_current_tenant, get_redis
from harness.core.config import get_config
from harness.orchestrator.runner import AgentRunner

logger = logging.getLogger(__name__)
router = APIRouter()


def _runner(redis):
    return AgentRunner(
        redis=redis,
        agent_factory=lambda agent_type: None,
    )


def _get_recorder(request: Request):
    """Return the shared TraceRecorder from app.state, creating one if needed.

    The recorder owns its own Redis connection pool; reusing the single
    lifespan-managed instance avoids leaking a pool (max 20 conns) per request.
    The lazy fallback keeps lightweight/test apps that skip lifespan working.
    """
    recorder = getattr(request.app.state, "trace_recorder", None)
    if recorder is None:
        from harness.observability.trace_recorder import TraceRecorder
        cfg = get_config()
        recorder = TraceRecorder.create(redis_url=cfg.redis_url)
        request.app.state.trace_recorder = recorder
    return recorder


@router.get("/{run_id}/trace")
async def get_run_trace(
    run_id: str,
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> dict:
    """
    Return the full span hierarchy for a run as a trace tree.

    Response shape
    --------------
    {
      "trace_id":            "...",
      "run_id":              "...",
      "agent_type":          "sql",
      "status":              "ok",
      "start_time":          "2024-01-01T00:00:00+00:00",
      "end_time":            "...",
      "duration_ms":         1234.5,
      "total_input_tokens":  1000,
      "total_output_tokens": 500,
      "total_cost_usd":      0.0025,
      "span_count":          6,
      "spans": [
        {
          "span_id":        "abc123",
          "parent_span_id": null,
          "kind":           "run",
          "name":           "run:sql_agent",
          "status":         "ok",
          "duration_ms":    1234.5,
          "input_tokens":   0,
          "output_tokens":  0,
          ...
        },
        ...
      ]
    }

    Raises
    ------
    404  Run has no recorded trace (trace_recorder not wired, or run too old).
    """
    record = await _runner(redis).get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )
    if record.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    recorder = _get_recorder(request)

    trace = await recorder.get_trace(run_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trace found for run {run_id}. "
                   "Traces are available for 48 h after run completion.",
        )

    return trace.to_dict()


@router.get("/spans/{span_id}")
async def get_span(
    span_id: str,
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> dict:
    """
    Return a single span by span_id.

    Raises 404 if the span is not found, has expired, or belongs to another
    tenant (404 on mismatch so span_ids cannot be enumerated cross-tenant).
    """
    recorder = _get_recorder(request)

    span = await recorder.get_span(span_id)
    # Spans recorded without an AgentContext carry tenant_id="" — treat those
    # conservatively as not visible to any tenant rather than world-readable.
    if span is None or not span.tenant_id or span.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Span not found: {span_id}",
        )
    return span.to_dict()
