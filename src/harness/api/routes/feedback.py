"""REST endpoints for real-time agent feedback."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from harness.api.deps import get_current_tenant, get_redis
from harness.feedback.channel import FeedbackChannel, FeedbackEvent

logger = logging.getLogger(__name__)
router = APIRouter()


async def _require_owned_run(run_id: str, tenant_id: str, redis: Any) -> None:
    """Load the RunRecord and enforce tenant ownership (mirrors runs.py).

    Raises:
        HTTPException 404 if the run does not exist.
        HTTPException 403 if the run belongs to another tenant.
    """
    from harness.orchestrator.runner import AgentRunner

    runner = AgentRunner(redis=redis, agent_factory=lambda agent_type: None)
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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    type: Literal["correction", "hint", "score", "stop", "redirect"] = "hint"
    content: str = Field(default="", max_length=4000)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = Field(default="human", max_length=64)
    priority: int = Field(default=2, ge=1, le=3)


class FeedbackResponse(BaseModel):
    feedback_id: str
    run_id: str
    type: str
    content: str
    score: float | None
    source: str
    priority: int
    created_at: str
    applied: bool

    @classmethod
    def from_event(cls, ev: FeedbackEvent) -> "FeedbackResponse":
        return cls(
            feedback_id=ev.feedback_id,
            run_id=ev.run_id,
            type=ev.type,
            content=ev.content,
            score=ev.score,
            source=ev.source,
            priority=ev.priority,
            created_at=ev.created_at.isoformat(),
            applied=ev.applied,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{run_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send real-time feedback to a running agent",
    description="""
Publish a feedback event into the agent's channel. The agent reads and applies
it before its next LLM call.

**Feedback types:**
- `correction` — inject a correction into agent context (high urgency)
- `hint` — inject soft guidance (lower priority)
- `score` — report a quality score; hints are injected when score < 0.40
- `stop` — signal the agent to stop cleanly after the current step
- `redirect` — replace the agent's remaining task description
""",
)
async def post_feedback(
    run_id: str,
    body: FeedbackRequest,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> FeedbackResponse:
    await _require_owned_run(run_id, tenant_id, redis)
    channel = FeedbackChannel(redis)
    event = FeedbackEvent(
        run_id=run_id,
        type=body.type,
        content=body.content,
        score=body.score,
        source=body.source,
        priority=body.priority,
    )
    entry_id = await channel.publish(run_id, event)
    if not entry_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to publish feedback — Redis unavailable",
        )
    logger.info(
        "Feedback %s/%s published to run=%s by tenant=%s",
        event.type, event.feedback_id[:8], run_id[:8], tenant_id,
    )
    return FeedbackResponse.from_event(event)


@router.get(
    "/{run_id}/feedback",
    response_model=list[FeedbackResponse],
    summary="List feedback events sent to a run",
)
async def list_feedback(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> list[FeedbackResponse]:
    await _require_owned_run(run_id, tenant_id, redis)
    channel = FeedbackChannel(redis)
    events = await channel.history(run_id, count=100)
    return [FeedbackResponse.from_event(ev) for ev in events]


@router.delete(
    "/{run_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
    summary="Clear feedback history for a run",
)
async def clear_feedback(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> Response:
    await _require_owned_run(run_id, tenant_id, redis)
    channel = FeedbackChannel(redis)
    await channel.clear(run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
