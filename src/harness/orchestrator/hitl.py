"""Human-in-the-loop (HITL) approval request management."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HITL_KEY_PREFIX = "harness:hitl"
_HITL_PENDING_SET = "harness:hitl:pending"
_DEFAULT_TTL_SECONDS = 3600  # 1 hour


def _hitl_key(request_id: str) -> str:
    return f"{_HITL_KEY_PREFIX}:{request_id}"


@dataclass
class ApprovalRequest:
    """A pending human approval request.

    Attributes:
        request_id:  Unique identifier.
        run_id:      The run this request belongs to.
        tenant_id:   Owning tenant.
        tool_name:   The tool that requires approval.
        tool_args:   Arguments the tool would be called with.
        reason:      Why approval is needed.
        status:      pending | approved | rejected | expired.
        created_at:  UTC time of creation.
        expires_at:  UTC time after which the request auto-expires.
        resolved_at: UTC time of resolution (None until resolved).
        resolved_by: Who resolved the request.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = ""
    tenant_id: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    reason: str = ""
    status: str = "pending"  # pending | approved | rejected | expired
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(seconds=_DEFAULT_TTL_SECONDS)
    )
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """Return True if the request has passed its expiry time."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_resolved(self) -> bool:
        """Return True if the request has been approved or rejected."""
        return self.status in ("approved", "rejected")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalRequest":
        def _dt(v) -> Optional[datetime]:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(v)
            except (ValueError, TypeError):
                return None

        return cls(
            request_id=data.get("request_id", uuid.uuid4().hex),
            run_id=data.get("run_id", ""),
            tenant_id=data.get("tenant_id", ""),
            tool_name=data.get("tool_name", ""),
            tool_args=dict(data.get("tool_args", {})),
            reason=data.get("reason", ""),
            status=data.get("status", "pending"),
            created_at=_dt(data.get("created_at")) or datetime.now(timezone.utc),
            expires_at=_dt(data.get("expires_at"))
            or datetime.now(timezone.utc) + timedelta(seconds=_DEFAULT_TTL_SECONDS),
            resolved_at=_dt(data.get("resolved_at")),
            resolved_by=data.get("resolved_by"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ApprovalRequest":
        return cls.from_dict(json.loads(raw))


class HITLManager:
    """Manages HITL approval requests using Redis as persistent storage.

    Args:
        redis:                  Async Redis client.
        ttl_seconds:            Time-to-live for pending requests (default 3600s).
        event_bus:              Optional EventBus for broadcasting approval events.
        policy_store:           Optional PolicyStore. When provided and
                                ``learn_from_rejections`` is True, rejected tool
                                names are automatically added to the tenant's
                                ``blocked_tools`` list so the same tool is never
                                auto-approved again.
        learn_from_rejections:  Persist rejected tool names into HarnessPolicy.
    """

    def __init__(
        self,
        redis: Any,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        event_bus: Optional[Any] = None,
        policy_store: Optional[Any] = None,
        learn_from_rejections: bool = False,
        memory_manager: Optional[Any] = None,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._event_bus = event_bus
        self._policy_store = policy_store
        self._learn_from_rejections = learn_from_rejections
        self._memory_manager = memory_manager

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def request_approval(
        self,
        run_id: str,
        tenant_id: str,
        tool_name: str,
        tool_args: dict,
        reason: str = "",
    ) -> ApprovalRequest:
        """Create and persist a new approval request.

        Args:
            run_id:     The run requiring approval.
            tenant_id:  Owning tenant.
            tool_name:  The tool that needs approval.
            tool_args:  Arguments for the tool call.
            reason:     Explanation of why approval is needed.

        Returns:
            The created ApprovalRequest.
        """
        req = ApprovalRequest(
            run_id=run_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            tool_args=tool_args,
            reason=reason,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._ttl),
        )
        await self._persist(req)
        logger.info(
            "Created HITL request %s for run=%s, tool=%s",
            req.request_id,
            run_id,
            tool_name,
        )
        return req

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    async def approve(
        self, request_id: str, resolved_by: str = "human"
    ) -> ApprovalRequest:
        """Approve a pending request.

        Args:
            request_id:  The request to approve.
            resolved_by: Who approved (username or system identifier).

        Returns:
            The updated ApprovalRequest.

        Raises:
            KeyError: If the request is not found.
            ValueError: If the request is already resolved or expired.
        """
        req = await self.get(request_id)
        if req is None:
            raise KeyError(f"ApprovalRequest not found: {request_id}")
        if req.is_resolved:
            raise ValueError(f"Request {request_id} is already {req.status}")
        if req.is_expired:
            req.status = "expired"
            await self._persist(req)
            raise ValueError(f"Request {request_id} has expired")

        req.status = "approved"
        req.resolved_at = datetime.now(timezone.utc)
        req.resolved_by = resolved_by
        await self._persist(req)
        await self._remove_from_pending(request_id)

        logger.info("Approved HITL request %s by %s", request_id, resolved_by)
        await self._update_memory_from_decision(req, "approved", self._memory_manager)
        return req

    async def reject(
        self, request_id: str, resolved_by: str = "human"
    ) -> ApprovalRequest:
        """Reject a pending request.

        Args:
            request_id:  The request to reject.
            resolved_by: Who rejected.

        Returns:
            The updated ApprovalRequest.

        Raises:
            KeyError: If the request is not found.
            ValueError: If already resolved or expired.
        """
        req = await self.get(request_id)
        if req is None:
            raise KeyError(f"ApprovalRequest not found: {request_id}")
        if req.is_resolved:
            raise ValueError(f"Request {request_id} is already {req.status}")
        if req.is_expired:
            req.status = "expired"
            await self._persist(req)
            raise ValueError(f"Request {request_id} has expired")

        req.status = "rejected"
        req.resolved_at = datetime.now(timezone.utc)
        req.resolved_by = resolved_by
        await self._persist(req)
        await self._remove_from_pending(request_id)

        logger.info("Rejected HITL request %s by %s", request_id, resolved_by)

        # Learn from rejection — add tool to tenant's blocked list so it is
        # never auto-approved again in future runs for this tenant.
        if self._learn_from_rejections and self._policy_store is not None:
            await self._block_tool_from_rejection(req)

        # Persist rejection into memory so agent retrieves it as future context
        await self._update_memory_from_decision(req, "rejected", self._memory_manager)

        return req

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, request_id: str) -> Optional[ApprovalRequest]:
        """Retrieve an ApprovalRequest by ID."""
        raw = await self._redis.get(_hitl_key(request_id))
        if not raw:
            return None
        return ApprovalRequest.from_json(raw if isinstance(raw, str) else raw.decode())

    async def list_pending(self, tenant_id: Optional[str] = None) -> list[ApprovalRequest]:
        """Return all pending (non-expired) approval requests.

        Args:
            tenant_id: If provided, filter by tenant.

        Returns:
            List of pending ApprovalRequest objects.
        """
        pending: list[ApprovalRequest] = []
        pattern = f"{_HITL_KEY_PREFIX}:*"
        async for key in self._redis.scan_iter(match=pattern, count=200):
            key_str = key if isinstance(key, str) else key.decode()
            # Skip the pending set key itself
            if key_str == _HITL_PENDING_SET:
                continue
            raw = await self._redis.get(key_str)
            if not raw:
                continue
            try:
                req = ApprovalRequest.from_json(
                    raw if isinstance(raw, str) else raw.decode()
                )
                if req.status != "pending":
                    continue
                if req.is_expired:
                    req.status = "expired"
                    await self._persist(req)
                    continue
                if tenant_id and req.tenant_id != tenant_id:
                    continue
                pending.append(req)
            except Exception as exc:
                logger.warning("Failed to parse HITL record at %s: %s", key_str, exc)

        return sorted(pending, key=lambda r: r.created_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _persist(self, req: ApprovalRequest) -> None:
        """Save an ApprovalRequest to Redis with TTL."""
        await self._redis.setex(
            _hitl_key(req.request_id),
            self._ttl * 2,  # keep longer than TTL for history
            req.to_json(),
        )

    # ------------------------------------------------------------------
    # Wait for decision  (called by BaseAgent._check_hitl)
    # ------------------------------------------------------------------

    async def await_decision(
        self,
        request_id: str,
        timeout: float = _DEFAULT_TTL_SECONDS,
        poll_interval: float = 2.0,
    ) -> str:
        """
        Block until the approval request is resolved or expires.

        Polls Redis every ``poll_interval`` seconds until the status
        transitions away from "pending" or ``timeout`` elapses.

        Returns
        -------
        str
            "approved" | "rejected" | "expired"

        Notes
        -----
        The caller (BaseAgent._check_hitl) raises HITLRejected when the
        return value is "rejected" or "expired".
        """
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            req = await self.get(request_id)

            if req is None:
                logger.warning("await_decision: request %s not found", request_id)
                return "expired"

            if req.status == "approved":
                logger.info("HITL request %s approved", request_id)
                return "approved"

            if req.status in ("rejected", "expired"):
                logger.info("HITL request %s %s", request_id, req.status)
                return req.status

            if req.is_expired:
                req.status = "expired"
                await self._persist(req)
                logger.info("HITL request %s auto-expired", request_id)
                return "expired"

            # Still pending — wait and poll again
            await asyncio.sleep(poll_interval)

        # Timeout exhausted — mark as expired
        req = await self.get(request_id)
        if req is not None and req.status == "pending":
            req.status = "expired"
            await self._persist(req)
            logger.warning(
                "HITL request %s timed out after %.0fs", request_id, timeout
            )
        return "expired"

    async def _update_memory_from_decision(
        self,
        req: ApprovalRequest,
        decision: str,
        memory_manager: Optional[Any] = None,
    ) -> None:
        """Persist the HITL decision into memory so future retrievals surface it.

        Approved tools build confidence; rejected tools become warnings that
        surface as context before the agent attempts the same action again.
        This closes the loop described in §5.2.5 of arXiv:2605.18747.
        """
        if memory_manager is None:
            return
        try:
            verb = "approved" if decision == "approved" else "rejected"
            content = (
                f"HITL {verb}: tool='{req.tool_name}' "
                f"args={req.tool_args} reason='{req.reason}' "
                f"resolved_by='{req.resolved_by}'"
            )
            await memory_manager.remember(
                text=content,
                metadata={
                    "hitl_decision": decision,
                    "tool_name": req.tool_name,
                    "run_id": req.run_id,
                    "request_id": req.request_id,
                },
                tenant_id=req.tenant_id,
            )
        except Exception as exc:
            logger.debug("_update_memory_from_decision failed: %s", exc)

    async def _block_tool_from_rejection(self, req: ApprovalRequest) -> None:
        """Persist a rejected tool into the tenant's HarnessPolicy blocked list.

        This closes the HITL feedback loop: a human rejection becomes durable
        policy so the same tool is never attempted again for this tenant without
        explicit policy revision.
        """
        if not req.tool_name or not req.tenant_id:
            return
        try:
            policy = await self._policy_store.get(req.tenant_id)
            if req.tool_name not in policy.blocked_tools:
                policy.blocked_tools = list(policy.blocked_tools) + [req.tool_name]
                await self._policy_store.set(policy)
                logger.info(
                    "HITL learn: added '%s' to blocked_tools for tenant '%s' (request %s)",
                    req.tool_name, req.tenant_id, req.request_id,
                )
        except Exception as exc:
            logger.warning(
                "HITL learn: failed to update policy for tenant '%s': %s",
                req.tenant_id, exc,
            )

    async def _remove_from_pending(self, request_id: str) -> None:
        """No-op if not using a separate pending set."""
        pass
