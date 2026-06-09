"""Agent run management API routes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from harness.api.deps import get_agent_factory, get_current_tenant, get_redis
from harness.core.config import get_config
from harness.orchestrator.runner import AgentRunner, RunRecord

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    """Request body for creating a new agent run."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_type": "sql",
                "task": "List all tables in the database",
                "metadata": {"priority": "high"},
            }
        }
    )

    agent_type: str = Field(..., description="Runnable native agent type: sql or code")
    task: str = Field(..., description="The task for the agent to execute", min_length=1)
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class RunRecordResponse(BaseModel):
    """API response model for a RunRecord."""

    run_id: str
    tenant_id: str
    agent_type: str
    task: str
    status: str
    result: dict | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    hitl_pending: bool = False
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_record(cls, record: RunRecord) -> RunRecordResponse:
        return cls(
            run_id=record.run_id,
            tenant_id=record.tenant_id,
            agent_type=record.agent_type,
            task=record.task,
            status=record.status,
            result=record.result,
            created_at=record.created_at.isoformat(),
            started_at=record.started_at.isoformat() if record.started_at else None,
            completed_at=record.completed_at.isoformat() if record.completed_at else None,
            hitl_pending=record.hitl_pending,
            metadata=record.metadata,
        )


_ALLOWED_AGENT_TYPES = {
    "sql",
    "code",
}


def _get_runner(redis: Any, agent_factory: Any = None) -> AgentRunner:
    """Build an AgentRunner for this request.

    Uses the real agent factory from ``app.state.agent_factory`` when
    available (set at startup in ``lifespan()``), so the API can execute
    runs directly without delegating to the RQ worker.

    Falls back to an informative error factory in environments where the
    full agent stack is not wired (e.g. read-only API replicas).
    """
    from harness.orchestrator.runner import AgentRunner

    def _stub_factory(agent_type: str):
        raise RuntimeError(
            f"Agent factory not available for type '{agent_type}'. "
            "Ensure the API was started with the full harness stack, "
            "or submit the run to the RQ worker queue."
        )

    factory = agent_factory or _stub_factory
    return AgentRunner(redis=redis, agent_factory=factory)


def _enqueue_run(run_id: str, agent_type: str) -> str:
    """Enqueue a run on the RQ queue consumed by harness.workers.agent_worker.

    Uses a synchronous Redis connection (RQ is sync-only). The connection is
    closed before returning so each enqueue does not leak a connection. This
    function is blocking and MUST be run off the event loop (see create_run).
    """
    import redis as sync_redis
    from rq import Queue

    from harness.workers.agent_worker import process_run_job

    cfg = get_config()
    conn = sync_redis.from_url(cfg.redis_url, decode_responses=False)
    try:
        queue = Queue(agent_type, connection=conn)
        job = queue.enqueue(process_run_job, run_id)
        return job.id
    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("Failed to close sync Redis connection for enqueue", exc_info=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED, response_model=RunRecordResponse)
async def create_run(
    body: CreateRunRequest,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
    agent_factory: Any = Depends(get_agent_factory),
) -> RunRecordResponse:
    """Create a new agent run and enqueue it for a worker.

    Args:
        body:      Run creation payload.
        tenant_id: Extracted from JWT.
        redis:     Redis client from app state.

    Returns:
        201 with the created RunRecord.

    Raises:
        400 if agent_type is not recognised.
    """
    if body.agent_type not in _ALLOWED_AGENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent_type '{body.agent_type}'. Allowed: {sorted(_ALLOWED_AGENT_TYPES)}",
        )

    runner = _get_runner(redis, agent_factory)
    record = await runner.create_run(
        tenant_id=tenant_id,
        agent_type=body.agent_type,
        task=body.task,
        metadata=body.metadata,
    )

    # Enqueue in RQ for the worker to pick up. The worker listens on RQ queues
    # named after agent types, so a plain Redis list would leave runs stranded.
    # _enqueue_run is blocking (sync Redis + RQ), so run it off the event loop.
    try:
        job_id = await asyncio.to_thread(_enqueue_run, record.run_id, body.agent_type)
        record.metadata = {**record.metadata, "queue": body.agent_type, "job_id": job_id}
        await runner.update_run(record)
        logger.info("Enqueued run %s as RQ job %s on queue %s", record.run_id, job_id, body.agent_type)
    except Exception as exc:
        # Enqueue failed: the run would otherwise sit in 'pending' forever with
        # no worker. Mark it failed and surface a 503 so the client can retry.
        logger.error("Failed to enqueue run %s: %s", record.run_id, exc)
        try:
            record.status = "failed"
            record.result = {"error": f"Failed to enqueue run: {exc}"}
            await runner.update_run(record)
        except Exception:  # pragma: no cover - best-effort bookkeeping
            logger.debug("Failed to mark run %s as failed after enqueue error", record.run_id, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue run for processing; please retry.",
        ) from exc

    return RunRecordResponse.from_record(record)


@router.get("/{run_id}", response_model=RunRecordResponse)
async def get_run(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> RunRecordResponse:
    """Retrieve a run by ID.

    Args:
        run_id:    The run identifier.
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        200 with RunRecord.

    Raises:
        404 if run not found.
        403 if run belongs to a different tenant.
    """
    runner = _get_runner(redis)
    record = await runner.get_run(run_id)

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

    return RunRecordResponse.from_record(record)


@router.get("", response_model=list[RunRecordResponse])
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> list[RunRecordResponse]:
    """List runs for the authenticated tenant.

    Args:
        limit:     Maximum number of runs to return (1-100).
        offset:    Pagination offset.
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        200 with list of RunRecords, newest first.
    """
    runner = _get_runner(redis)
    records = await runner.list_runs(
        tenant_id=tenant_id, limit=limit, offset=offset
    )
    return [RunRecordResponse.from_record(r) for r in records]


@router.get("/{run_id}/stream")
async def stream_run_events(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
    timeout: float = Query(default=300.0, ge=1.0, le=3600.0),
) -> StreamingResponse:
    """Stream run events as Server-Sent Events (SSE).

    Subscribes to the Redis event stream for *run_id* and forwards events
    as SSE until the run completes or the timeout expires.

    Each event is a JSON-encoded StepEvent payload::

        data: {"event_type": "token_delta", "payload": {"delta": "Hello"}, ...}

    Args:
        run_id:  The run to stream events from.
        timeout: Maximum seconds to keep the connection open (default 300).

    Returns:
        text/event-stream response.

    Raises:
        404 if run not found.
        403 if run belongs to another tenant.
    """
    runner = _get_runner(redis)
    record = await runner.get_run(run_id)

    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run not found: {run_id}")
    if record.tenant_id != tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    async def _event_generator():
        stream_key = f"harness:events:{run_id}"
        last_id = "0"
        deadline = asyncio.get_event_loop().time() + timeout

        yield "retry: 2000\n\n"  # tell client to reconnect after 2s on disconnect

        while asyncio.get_event_loop().time() < deadline:
            try:
                # XREAD with 1s block — yields when events arrive or times out
                entries = await redis.xread({stream_key: last_id}, count=50, block=1000)
            except Exception as exc:
                logger.debug("SSE stream read error for run %s: %s", run_id, exc)
                await asyncio.sleep(0.5)
                continue

            if entries:
                for _stream_name, messages in entries:
                    for msg_id, fields in messages:
                        last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                        # Fields may be bytes
                        payload = {}
                        for k, v in fields.items():
                            key = k if isinstance(k, str) else k.decode()
                            val = v if isinstance(v, str) else v.decode()
                            try:
                                payload[key] = json.loads(val)
                            except (json.JSONDecodeError, TypeError):
                                payload[key] = val

                        data = json.dumps(payload, ensure_ascii=False)
                        yield f"data: {data}\n\n"

                        # Stop streaming when run reaches a terminal state
                        event_type = payload.get("event_type", "")
                        if event_type in ("completed", "failed", "cancelled", "budget_exceeded"):
                            yield "data: {\"event_type\": \"stream_end\"}\n\n"
                            return

            # Check if run finished (handles case where events were missed)
            rec = await runner.get_run(run_id)
            if rec and rec.status in ("completed", "failed", "cancelled"):
                yield "data: {\"event_type\": \"stream_end\"}\n\n"
                return

        yield "data: {\"event_type\": \"stream_timeout\"}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
async def cancel_run(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> None:
    """Cancel a run that is pending or running.

    Args:
        run_id:    The run to cancel.
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        204 on success.

    Raises:
        404 if run not found.
        403 if run belongs to another tenant.
    """
    runner = _get_runner(redis)
    record = await runner.get_run(run_id)

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

    await runner.cancel_run(run_id)
