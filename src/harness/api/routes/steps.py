"""Server-Sent Events route for real-time step streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status

from harness.api.deps import get_current_tenant, get_redis

logger = logging.getLogger(__name__)

router = APIRouter()

_SSE_HEARTBEAT_INTERVAL = 15  # seconds between keepalive pings
_SSE_TIMEOUT = 300            # max seconds to hold an SSE connection


async def _event_generator(
    run_id: str,
    redis: Any,
    tenant_id: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Async generator that yields Server-Sent Events for a run.

    Subscribes to the Redis pub/sub channel for this run and yields
    JSON-serialised StepEvent payloads.  Sends a [DONE] sentinel when the
    run completes or when the client disconnects.

    Args:
        run_id:    The run to stream events for.
        redis:     Async Redis client.
        tenant_id: For access-control validation.
        request:   The HTTP request (used for disconnect detection).

    Yields:
        SSE-formatted strings.
    """
    channel = f"harness:events:{run_id}"
    pubsub = redis.pubsub()

    try:
        await pubsub.subscribe(channel)
        logger.debug("SSE: subscribed to channel %s", channel)

        elapsed = 0.0
        while elapsed < _SSE_TIMEOUT:
            # Check for client disconnect
            if await request.is_disconnected():
                logger.debug("SSE: client disconnected for run %s", run_id)
                break

            # Poll for messages with a short timeout
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                message = None

            if message is not None and message.get("type") == "message":
                data = message.get("data", "")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                # Yield the SSE event
                yield f"data: {data}\n\n"

                # Check if this is a terminal event
                try:
                    parsed = json.loads(data)
                    event_type = parsed.get("event_type", "")
                    if event_type in ("completed", "failed", "cancelled"):
                        yield f"data: [DONE]\n\n"
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

            else:
                # Send heartbeat comment to keep connection alive
                elapsed += 1.0
                if int(elapsed) % _SSE_HEARTBEAT_INTERVAL == 0:
                    yield ": heartbeat\n\n"

        else:
            # Timeout reached
            yield f"data: {json.dumps({'event_type': 'timeout', 'run_id': run_id})}\n\n"
            yield f"data: [DONE]\n\n"

    except Exception as exc:
        logger.error("SSE generator error for run %s: %s", run_id, exc)
        yield f"data: {json.dumps({'event_type': 'error', 'error': str(exc)})}\n\n"
        yield f"data: [DONE]\n\n"
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
        except Exception:
            pass


@router.get("/{run_id}/steps")
async def stream_run_steps(
    run_id: str,
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
):
    """Stream real-time step events for a run via Server-Sent Events.

    The stream uses a Redis pub/sub channel keyed by run_id.  Workers
    publish StepEvent JSON objects to this channel as the agent executes.
    A ``[DONE]`` sentinel is sent when the run completes or the connection
    times out.

    Args:
        run_id:    The run to stream.
        request:   FastAPI request object (for disconnect detection).
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        EventSourceResponse with text/event-stream content type.

    Raises:
        404 if the run is not found.
        503 if the SSE library is not available.
    """
    # Validate run exists and belongs to tenant
    run_key = f"harness:run:{run_id}"
    raw = await redis.get(run_key)
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )

    try:
        import json as _json
        run_data = _json.loads(raw if isinstance(raw, str) else raw.decode())
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

    # Try to use sse-starlette for proper SSE support
    try:
        from sse_starlette.sse import EventSourceResponse  # type: ignore

        async def generator():
            async for chunk in _event_generator(run_id, redis, tenant_id, request):
                # sse-starlette expects dict or string; strip the "data: " prefix
                if chunk.startswith("data: "):
                    payload = chunk[6:].rstrip("\n")
                    if payload == "[DONE]":
                        yield {"data": "[DONE]", "event": "done"}
                    else:
                        yield {"data": payload}
                elif chunk.startswith(": "):
                    yield {"comment": chunk[2:].rstrip("\n")}

        return EventSourceResponse(generator())

    except ImportError:
        # Fall back to a plain StreamingResponse
        from starlette.responses import StreamingResponse

        return StreamingResponse(
            _event_generator(run_id, redis, tenant_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
