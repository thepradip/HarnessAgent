"""Server-Sent Events route for real-time step streaming.

This endpoint consumes the SAME Redis *stream* (``harness:events:{run_id}``)
that the agent event sink (``harness.workers.agent_worker._RedisStreamEventSink``)
and ``harness.observability.event_bus.EventBus`` write to via ``XADD``.  It uses
the same blocking ``XREAD`` loop as ``runs.py:stream_run_events`` so there is a
single source of truth for run events — no separate pub/sub channel.

It surfaces step-level events (step_start/step_end/tool_call/llm_call/
token_delta/feedback/run_end) and forwards every event, including the
``token_delta`` events emitted by the token-streaming feature.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from starlette.responses import StreamingResponse

from harness.api.deps import get_current_tenant, get_redis

logger = logging.getLogger(__name__)

router = APIRouter()

# Events that mean the run has reached a terminal state and the stream
# should close.  ``run_end`` is the step-level terminal marker; the others
# mirror the run-status terminal states emitted on the same stream.
_TERMINAL_EVENTS = (
    "run_end",
    "completed",
    "failed",
    "cancelled",
    "budget_exceeded",
)


async def _event_generator(
    run_id: str,
    redis: Any,
    request: Request,
    timeout: float,
) -> AsyncGenerator[str, None]:
    """Async generator that yields Server-Sent Events for a run.

    Reads the Redis *stream* for this run via blocking ``XREAD`` (the same
    transport used by ``/runs/{id}/stream``) and yields JSON-serialised
    StepEvent payloads.  A terminal event (run_end/completed/failed/
    cancelled/budget_exceeded) or a client disconnect closes the stream.

    Args:
        run_id:  The run to stream events for.
        redis:   Async Redis client.
        request: The HTTP request (used for disconnect detection).
        timeout: Maximum seconds to keep the connection open.

    Yields:
        SSE-formatted strings.
    """
    stream_key = f"harness:events:{run_id}"
    last_id = "0"
    deadline = asyncio.get_event_loop().time() + timeout

    yield "retry: 2000\n\n"  # tell client to reconnect after 2s on disconnect

    try:
        while asyncio.get_event_loop().time() < deadline:
            if await request.is_disconnected():
                logger.debug("SSE: client disconnected for run %s", run_id)
                return

            try:
                # XREAD with 1s block — yields when events arrive or times out
                entries = await redis.xread({stream_key: last_id}, count=50, block=1000)
            except Exception as exc:
                logger.debug("SSE stream read error for run %s: %s", run_id, exc)
                await asyncio.sleep(0.5)
                continue

            if not entries:
                continue

            for _stream_name, messages in entries:
                for msg_id, fields in messages:
                    last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                    # Fields may be bytes when the client was created without
                    # decode_responses; normalise and decode JSON values.
                    payload: dict[str, Any] = {}
                    for k, v in fields.items():
                        key = k if isinstance(k, str) else k.decode()
                        val = v if isinstance(v, str) else v.decode()
                        try:
                            payload[key] = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            payload[key] = val

                    data = json.dumps(payload, ensure_ascii=False)
                    yield f"data: {data}\n\n"

                    event_type = payload.get("event_type", "")
                    if event_type in _TERMINAL_EVENTS:
                        yield "data: [DONE]\n\n"
                        return

        # Deadline reached without a terminal event.
        yield f"data: {json.dumps({'event_type': 'timeout', 'run_id': run_id})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.error("SSE generator error for run %s: %s", run_id, exc)
        yield f"data: {json.dumps({'event_type': 'error', 'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"


@router.get("/{run_id}/steps")
async def stream_run_steps(
    run_id: str,
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
    timeout: float = Query(default=300.0, ge=1.0, le=3600.0),
) -> StreamingResponse:
    """Stream real-time step events for a run via Server-Sent Events.

    The stream reads the Redis *stream* keyed by run_id
    (``harness:events:{run_id}``).  The agent event sink XADDs StepEvent JSON
    objects to this stream as the agent executes; this endpoint forwards every
    event (step_start/step_end/tool_call/llm_call/token_delta/feedback/run_end).
    A ``[DONE]`` sentinel is sent when the run reaches a terminal state or the
    connection times out.

    Args:
        run_id:    The run to stream.
        request:   FastAPI request object (for disconnect detection).
        tenant_id: Extracted from JWT.
        redis:     Redis client.
        timeout:   Maximum seconds to keep the connection open (default 300).

    Returns:
        text/event-stream response.

    Raises:
        404 if the run is not found (or the run record is corrupt — fail closed).
        403 if the run belongs to another tenant.
    """
    # Validate run exists and belongs to tenant.
    run_key = f"harness:run:{run_id}"
    raw = await redis.get(run_key)
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )

    try:
        run_data = json.loads(raw if isinstance(raw, str) else raw.decode())
        record_tenant = run_data.get("tenant_id")
    except (ValueError, KeyError, AttributeError):
        # Corrupt or non-JSON run record — fail closed (404) rather than
        # skipping the ownership check and serving the stream anyway.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )
    if record_tenant != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return StreamingResponse(
        _event_generator(run_id, redis, request, timeout),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )
