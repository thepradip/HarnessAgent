"""Improvement, HITL, and Prompts API routes."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from harness.api.deps import (
    get_current_tenant,
    get_current_user,
    get_hitl_manager,
    get_prompt_manager,
    get_redis,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PatchResponse(BaseModel):
    patch_id: str
    agent_type: str
    target: str
    op: str
    path: str
    value: str
    rationale: str
    proposed_by: str
    proposed_at: str
    score: Optional[float]
    status: str
    based_on_errors: list[str]

    @classmethod
    def from_patch(cls, p: Any) -> "PatchResponse":
        return cls(
            patch_id=p.patch_id,
            agent_type=p.agent_type,
            target=p.target,
            op=p.op,
            path=p.path,
            value=p.value,
            rationale=p.rationale,
            proposed_by=p.proposed_by,
            proposed_at=p.proposed_at.isoformat(),
            score=p.score,
            status=p.status,
            based_on_errors=p.based_on_errors,
        )


class PatchOutcomeResponse(BaseModel):
    patch_id: str
    baseline_score: float
    patched_score: float
    improvement: float
    accepted: bool
    eval_summary: str


class ErrorRecordResponse(BaseModel):
    record_id: str
    agent_type: str
    task: str
    failure_class: str
    error_message: str
    stack_trace: str
    created_at: str

    @classmethod
    def from_record(cls, r: Any) -> "ErrorRecordResponse":
        return cls(
            record_id=r.record_id,
            agent_type=r.agent_type,
            task=r.task,
            failure_class=r.failure_class,
            error_message=r.error_message,
            stack_trace=r.stack_trace,
            created_at=r.created_at.isoformat(),
        )


class ApprovalRequestResponse(BaseModel):
    request_id: str
    run_id: str
    tenant_id: str
    tool_name: str
    tool_args: dict
    reason: str
    status: str
    created_at: str
    expires_at: str
    resolved_at: Optional[str]
    resolved_by: Optional[str]

    @classmethod
    def from_request(cls, r: Any) -> "ApprovalRequestResponse":
        return cls(
            request_id=r.request_id,
            run_id=r.run_id,
            tenant_id=r.tenant_id,
            tool_name=r.tool_name,
            tool_args=r.tool_args,
            reason=r.reason,
            status=r.status,
            created_at=r.created_at.isoformat(),
            expires_at=r.expires_at.isoformat(),
            resolved_at=r.resolved_at.isoformat() if r.resolved_at else None,
            resolved_by=r.resolved_by,
        )


class ResolveRequest(BaseModel):
    resolved_by: str = Field(..., description="Username or system ID of resolver")


class PromptVersionResponse(BaseModel):
    version_id: str
    agent_type: str
    content: str
    version_number: int
    active: bool
    score: Optional[float]
    patch_id: Optional[str]
    created_by: str
    created_at: str
    tags: list[str]
    metadata: dict

    @classmethod
    def from_version(cls, v: Any) -> "PromptVersionResponse":
        return cls(
            version_id=v.version_id,
            agent_type=v.agent_type,
            content=v.content,
            version_number=v.version_number,
            active=v.active,
            score=v.score,
            patch_id=v.patch_id,
            created_by=v.created_by,
            created_at=v.created_at.isoformat(),
            tags=v.tags,
            metadata=v.metadata,
        )


# ---------------------------------------------------------------------------
# Improvement Routes
# ---------------------------------------------------------------------------


def _get_error_collector(redis: Any) -> Any:
    """Build an ErrorCollector from the Redis client."""
    from harness.improvement.error_collector import ErrorCollector
    return ErrorCollector(redis=redis)


def _get_patch_store(redis: Any) -> Any:
    """Return a simple dict-backed patch store for demo purposes."""
    # In production this would be a Redis-backed store
    return _RedisPatchStore(redis)


class _RedisPatchStore:
    """Minimal Redis-backed patch store."""
    _PREFIX = "harness:patch"

    def __init__(self, redis: Any) -> None:
        self._r = redis

    async def save(self, patch: Any) -> None:
        await self._r.set(f"{self._PREFIX}:{patch.patch_id}", patch.to_json())

    async def get(self, patch_id: str) -> Optional[Any]:
        from harness.improvement.patch_generator import Patch
        raw = await self._r.get(f"{self._PREFIX}:{patch_id}")
        if not raw:
            return None
        return Patch.from_json(raw if isinstance(raw, str) else raw.decode())

    async def list(self, agent_type: Optional[str] = None, patch_status: Optional[str] = None) -> list[Any]:
        from harness.improvement.patch_generator import Patch
        patches = []
        pattern = f"{self._PREFIX}:*"
        async for key in self._r.scan_iter(match=pattern, count=200):
            raw = await self._r.get(key)
            if raw:
                try:
                    p = Patch.from_json(raw if isinstance(raw, str) else raw.decode())
                    if agent_type and p.agent_type != agent_type:
                        continue
                    if patch_status and p.status != patch_status:
                        continue
                    patches.append(p)
                except Exception:
                    pass
        return sorted(patches, key=lambda x: x.proposed_at, reverse=True)


@router.get("/improvement/patches", response_model=list[PatchResponse])
async def list_patches(
    agent_type: Optional[str] = Query(default=None),
    patch_status: Optional[str] = Query(default="pending", alias="status"),
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> list[PatchResponse]:
    """List improvement patches, optionally filtered by agent_type and status."""
    store = _get_patch_store(redis)
    patches = await store.list(agent_type=agent_type, patch_status=patch_status)
    return [PatchResponse.from_patch(p) for p in patches]


@router.post("/improvement/patches/{patch_id}/approve", response_model=PatchResponse)
async def approve_patch(
    patch_id: str,
    tenant_id: str = Depends(get_current_tenant),
    user: dict = Depends(get_current_user),
    redis: Any = Depends(get_redis),
    pm: Any = Depends(get_prompt_manager),
) -> PatchResponse:
    """Approve a patch and apply it to the active prompt."""
    store = _get_patch_store(redis)
    patch = await store.get(patch_id)
    if patch is None:
        raise HTTPException(status_code=404, detail=f"Patch not found: {patch_id}")

    # Apply the patch
    try:
        await pm.apply_patch(patch)
        patch.status = "applied"
        await store.save(patch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to apply patch: {exc}")

    return PatchResponse.from_patch(patch)


@router.post("/improvement/patches/{patch_id}/reject", response_model=PatchResponse)
async def reject_patch(
    patch_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> PatchResponse:
    """Reject a pending patch."""
    store = _get_patch_store(redis)
    patch = await store.get(patch_id)
    if patch is None:
        raise HTTPException(status_code=404, detail=f"Patch not found: {patch_id}")

    patch.status = "rejected"
    await store.save(patch)
    return PatchResponse.from_patch(patch)


@router.post("/improvement/cycle", response_model=PatchOutcomeResponse)
async def trigger_improvement_cycle(
    request: Request,
    agent_type: Optional[str] = Query(default=None),
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
    pm: Any = Depends(get_prompt_manager),
) -> PatchOutcomeResponse:
    """Manually trigger a Hermes improvement cycle for an agent type.

    The Hermes loop is stored on ``app.state.hermes_loop`` when the API is
    started with the full improvement stack. If it is not wired, we return a
    501 rather than a 200 that masks a no-op.
    """
    error_collector = _get_error_collector(redis)
    errors = await error_collector.get_recent(agent_type or "sql", limit=20)

    if len(errors) < 2:
        return PatchOutcomeResponse(
            patch_id="none",
            baseline_score=0.0,
            patched_score=0.0,
            improvement=0.0,
            accepted=False,
            eval_summary=f"Not enough errors to trigger cycle (found {len(errors)})",
        )

    hermes = getattr(request.app.state, "hermes_loop", None)
    if hermes is None:
        # Be honest: we cannot run a cycle without the Hermes loop wired.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Improvement cycle not available: Hermes loop is not configured "
                "on this API instance. Start the API with the full improvement "
                f"stack or run the cycle via the worker ({len(errors)} errors "
                f"pending for agent_type={agent_type or 'sql'})."
            ),
        )

    outcome = await hermes.run_cycle(agent_type=agent_type or "sql", errors=errors)
    return PatchOutcomeResponse(
        patch_id=getattr(outcome, "patch_id", "cycle"),
        baseline_score=getattr(outcome, "baseline_score", 0.0),
        patched_score=getattr(outcome, "patched_score", 0.0),
        improvement=getattr(outcome, "improvement", 0.0),
        accepted=getattr(outcome, "accepted", False),
        eval_summary=getattr(
            outcome, "eval_summary", f"Cycle run for {len(errors)} errors on agent_type={agent_type}"
        ),
    )


@router.get("/improvement/errors", response_model=list[ErrorRecordResponse])
async def list_errors(
    agent_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> list[ErrorRecordResponse]:
    """List recent error records."""
    ec = _get_error_collector(redis)
    if agent_type:
        records = await ec.get_recent(agent_type, limit=limit)
    else:
        # Gather from all agent types
        all_records = []
        for at in ["sql", "code", "research", "orchestrator"]:
            recs = await ec.get_recent(at, limit=limit)
            all_records.extend(recs)
        all_records.sort(key=lambda r: r.created_at, reverse=True)
        records = all_records[:limit]
    return [ErrorRecordResponse.from_record(r) for r in records]


@router.get("/improvement/failures/heatmap")
async def failure_heatmap(
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> dict:
    """Return a heatmap of failure_class counts per agent_type."""
    ec = _get_error_collector(redis)
    return await ec.failure_heatmap()


# ---------------------------------------------------------------------------
# HITL Routes
# ---------------------------------------------------------------------------


@router.get("/hitl/pending", response_model=list[ApprovalRequestResponse])
async def list_hitl_pending(
    tenant_id: str = Depends(get_current_tenant),
    hitl: Any = Depends(get_hitl_manager),
) -> list[ApprovalRequestResponse]:
    """List all pending HITL approval requests for the tenant."""
    requests = await hitl.list_pending(tenant_id=tenant_id)
    return [ApprovalRequestResponse.from_request(r) for r in requests]


async def _require_owned_hitl_request(
    hitl: Any, request_id: str, tenant_id: str
) -> None:
    """Enforce tenant ownership of a HITL request before resolution.

    Raises:
        HTTPException 404 if the request does not exist or belongs to another
            tenant (404 on mismatch so request_ids cannot be enumerated).
    """
    req = await hitl.get(request_id)
    if req is None or req.tenant_id != tenant_id:
        raise HTTPException(
            status_code=404, detail=f"HITL request not found: {request_id}"
        )


@router.post("/hitl/{request_id}/approve", response_model=ApprovalRequestResponse)
async def approve_hitl(
    request_id: str,
    body: ResolveRequest,
    tenant_id: str = Depends(get_current_tenant),
    hitl: Any = Depends(get_hitl_manager),
) -> ApprovalRequestResponse:
    """Approve a HITL request."""
    await _require_owned_hitl_request(hitl, request_id, tenant_id)
    try:
        req = await hitl.approve(request_id, resolved_by=body.resolved_by)
        return ApprovalRequestResponse.from_request(req)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"HITL request not found: {request_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/hitl/{request_id}/reject", response_model=ApprovalRequestResponse)
async def reject_hitl(
    request_id: str,
    body: ResolveRequest,
    tenant_id: str = Depends(get_current_tenant),
    hitl: Any = Depends(get_hitl_manager),
) -> ApprovalRequestResponse:
    """Reject a HITL request."""
    await _require_owned_hitl_request(hitl, request_id, tenant_id)
    try:
        req = await hitl.reject(request_id, resolved_by=body.resolved_by)
        return ApprovalRequestResponse.from_request(req)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"HITL request not found: {request_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Prompts Routes
# ---------------------------------------------------------------------------


@router.get("/prompts/{agent_type}/versions", response_model=list[PromptVersionResponse])
async def list_prompt_versions(
    agent_type: str,
    limit: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_current_tenant),
    pm: Any = Depends(get_prompt_manager),
) -> list[PromptVersionResponse]:
    """List all prompt versions for an agent type."""
    versions = await pm.list_versions(agent_type, limit=limit)
    return [PromptVersionResponse.from_version(v) for v in versions]


@router.post("/prompts/{agent_type}/promote/{version_id}", response_model=PromptVersionResponse)
async def promote_prompt(
    agent_type: str,
    version_id: str,
    tenant_id: str = Depends(get_current_tenant),
    pm: Any = Depends(get_prompt_manager),
) -> PromptVersionResponse:
    """Promote a specific prompt version to active."""
    try:
        version = await pm.promote(version_id)
        return PromptVersionResponse.from_version(version)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt version not found: {version_id}")


@router.post("/prompts/{agent_type}/rollback", response_model=PromptVersionResponse)
async def rollback_prompt(
    agent_type: str,
    steps: int = Query(default=1, ge=1, le=10),
    tenant_id: str = Depends(get_current_tenant),
    pm: Any = Depends(get_prompt_manager),
) -> PromptVersionResponse:
    """Roll back the active prompt N steps."""
    try:
        version = await pm.rollback(agent_type, steps=steps)
        return PromptVersionResponse.from_version(version)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
