"""RunRecord and AgentRunner — orchestrates agent execution and state tracking."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import signal
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RUN_KEY_PREFIX = "harness:run"


def _run_key(run_id: str) -> str:
    return f"{_RUN_KEY_PREFIX}:{run_id}"


@dataclass
class RunRecord:
    """Persistent record of an agent run.

    Attributes:
        run_id:       Unique identifier.
        tenant_id:    Owning tenant.
        agent_type:   Which agent to execute.
        task:         The task string.
        status:       One of: pending, running, completed, failed, cancelled.
        result:       Serialised AgentResult once the run is done.
        created_at:   UTC time of creation.
        started_at:   UTC time when execution began (None until started).
        completed_at: UTC time when execution finished (None until done).
        hitl_pending: True when waiting for human approval.
        metadata:     Arbitrary extra data.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = ""
    agent_type: str = ""
    task: str = ""
    status: str = "pending"  # pending | running | completed | failed | cancelled
    result: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    hitl_pending: bool = False
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "hitl_pending": self.hitl_pending,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        def _dt(v) -> datetime | None:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(v)
            except (ValueError, TypeError):
                return None

        return cls(
            run_id=data.get("run_id", uuid.uuid4().hex),
            tenant_id=data.get("tenant_id", ""),
            agent_type=data.get("agent_type", ""),
            task=data.get("task", ""),
            status=data.get("status", "pending"),
            result=data.get("result"),
            created_at=_dt(data.get("created_at")) or datetime.now(UTC),
            started_at=_dt(data.get("started_at")),
            completed_at=_dt(data.get("completed_at")),
            hitl_pending=bool(data.get("hitl_pending", False)),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, raw: str) -> RunRecord:
        return cls.from_dict(json.loads(raw))


class AgentRunner:
    """Manages run lifecycle: create, execute, update, retrieve.

    Args:
        redis:          Async Redis client for run state persistence.
        agent_factory:  Callable(agent_type) -> agent instance.
        workspace_base: Base path for per-run workspace directories.
        event_bus:      Optional EventBus for broadcasting step events.
        error_collector: Optional ErrorCollector for recording failures.
    """

    def __init__(
        self,
        redis: Any,
        agent_factory: Any,
        workspace_base: str = "/workspaces",
        event_bus: Any | None = None,
        error_collector: Any | None = None,
    ) -> None:
        self._redis = redis
        self._agent_factory = agent_factory
        self._workspace_base = Path(workspace_base)
        self._event_bus = event_bus
        self._error_collector = error_collector
        self._shutting_down: bool = False
        self._inflight: set[str] = set()

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def setup_signal_handling(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown.

        Call once from within a running event loop (e.g. at app startup).
        On Windows, add_signal_handler is not supported — the call is a no-op.
        """
        try:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGTERM, self._request_shutdown)
            loop.add_signal_handler(signal.SIGINT, self._request_shutdown)
            logger.info("AgentRunner: SIGTERM/SIGINT handlers registered")
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("AgentRunner: could not register signal handlers: %s", exc)

    def _request_shutdown(self) -> None:
        """Set the shutdown flag — no new runs will be started."""
        if not self._shutting_down:
            logger.info(
                "AgentRunner: shutdown requested — %d run(s) in flight",
                len(self._inflight),
            )
        self._shutting_down = True

    async def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Set shutdown flag and wait up to drain_timeout for in-flight runs.

        Any run still active after the timeout is marked 'failed' in Redis
        with error_message='Run interrupted by graceful shutdown'.

        Args:
            drain_timeout: Seconds to wait before force-marking stragglers.
        """
        self._request_shutdown()

        if not self._inflight:
            logger.info("AgentRunner: no in-flight runs — clean shutdown")
            return

        logger.info(
            "AgentRunner: draining %d in-flight run(s) (timeout=%.1fs)...",
            len(self._inflight),
            drain_timeout,
        )
        loop = asyncio.get_event_loop()
        deadline = loop.time() + drain_timeout
        while self._inflight:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(1.0, remaining))

        if self._inflight:
            logger.warning(
                "AgentRunner: drain timeout — interrupting %d run(s): %s",
                len(self._inflight),
                list(self._inflight),
            )
            for run_id in list(self._inflight):
                try:
                    record = await self.get_run(run_id)
                    if record and record.status == "running":
                        record.status = "failed"
                        record.result = {
                            "run_id": run_id,
                            "output": "",
                            "steps": 0,
                            "tokens": 0,
                            "success": False,
                            "error_message": "Run interrupted by graceful shutdown",
                        }
                        record.completed_at = datetime.now(UTC)
                        await self.update_run(record)
                except Exception as exc:
                    logger.debug("Could not mark run %s interrupted: %s", run_id, exc)
            self._inflight.clear()
        else:
            logger.info("AgentRunner: all in-flight runs completed — clean shutdown")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_run(
        self,
        tenant_id: str,
        agent_type: str,
        task: str,
        metadata: dict | None = None,
    ) -> RunRecord:
        """Create and persist a new RunRecord with status=pending."""
        record = RunRecord(
            tenant_id=tenant_id,
            agent_type=agent_type,
            task=task,
            metadata=metadata or {},
        )
        await self._redis.set(_run_key(record.run_id), record.to_json())
        logger.info(
            "Created run %s (agent_type=%s, tenant=%s)", record.run_id, agent_type, tenant_id
        )
        return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a RunRecord by run_id."""
        raw = await self._redis.get(_run_key(run_id))
        if not raw:
            return None
        return RunRecord.from_json(raw if isinstance(raw, str) else raw.decode())

    async def list_runs(
        self,
        tenant_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RunRecord]:
        """Return paginated RunRecords for a tenant, newest first."""
        pattern = f"{_RUN_KEY_PREFIX}:*"
        all_records: list[RunRecord] = []
        async for key in self._redis.scan_iter(match=pattern, count=200):
            raw = await self._redis.get(key)
            if raw:
                try:
                    rec = RunRecord.from_json(raw if isinstance(raw, str) else raw.decode())
                    if rec.tenant_id == tenant_id:
                        all_records.append(rec)
                except Exception:
                    pass

        # Sort by created_at descending
        all_records.sort(key=lambda r: r.created_at, reverse=True)
        return all_records[offset: offset + limit]

    async def update_run(self, record: RunRecord) -> RunRecord:
        """Persist an updated RunRecord."""
        await self._redis.set(_run_key(record.run_id), record.to_json())
        return record

    async def cancel_run(self, run_id: str) -> RunRecord | None:
        """Mark a run as cancelled if it is still pending or running."""
        record = await self.get_run(run_id)
        if record is None:
            return None
        if record.status in ("completed", "failed", "cancelled"):
            return record
        record.status = "cancelled"
        record.completed_at = datetime.now(UTC)
        await self.update_run(record)
        logger.info("Cancelled run %s", run_id)
        return record

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_run(self, run_id: str) -> RunRecord:
        """Execute a run end-to-end.

        Fetches the RunRecord, builds the agent context, runs the agent,
        and updates the record with the final status and result.

        Args:
            run_id: The run to execute.

        Returns:
            The updated RunRecord.
        """
        record = await self.get_run(run_id)
        if record is None:
            raise KeyError(f"Run not found: {run_id}")
        if record.status == "cancelled":
            logger.info("Skipping cancelled run %s", run_id)
            return record

        # Reject new runs during shutdown
        if self._shutting_down:
            logger.warning("Rejecting run %s — shutdown in progress", run_id)
            record.status = "cancelled"
            record.completed_at = datetime.now(UTC)
            await self.update_run(record)
            return record

        # Mark as running and track as in-flight
        record.status = "running"
        record.started_at = datetime.now(UTC)
        await self.update_run(record)
        self._inflight.add(run_id)

        workspace = self._workspace_base / record.tenant_id / record.run_id
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            agent = self._agent_factory(record.agent_type)
            agent_result = await _run_agent(agent, record, workspace)

            record.status = "completed" if getattr(agent_result, "success", False) else "failed"
            record.result = _serialise_result(agent_result)

        except Exception as exc:
            logger.exception("Run %s raised unhandled exception: %s", run_id, exc)
            record.status = "failed"
            record.result = {
                "run_id": run_id,
                "output": "",
                "steps": 0,
                "tokens": 0,
                "success": False,
                "error_message": str(exc),
            }
            if self._error_collector is not None:
                try:
                    await self._error_collector.record(
                        agent_type=record.agent_type,
                        task=record.task,
                        failure_class="UNKNOWN",
                        error_message=str(exc),
                    )
                except Exception:
                    pass

        finally:
            self._inflight.discard(run_id)

        record.completed_at = datetime.now(UTC)

        # Don't clobber a concurrent cancel_run(): re-read the persisted
        # status and keep "cancelled" if it was set while the agent ran.
        persisted = await self.get_run(run_id)
        if persisted is not None and persisted.status == "cancelled":
            logger.info(
                "Run %s was cancelled during execution — keeping cancelled status",
                run_id,
            )
            return persisted

        await self.update_run(record)

        logger.info(
            "Run %s finished with status=%s", run_id, record.status
        )
        return record


def _serialise_result(result: Any) -> dict:
    """Convert an AgentResult to a plain dict for JSON storage."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "run_id": getattr(result, "run_id", ""),
        "output": getattr(result, "output", ""),
        "steps": getattr(result, "steps", 0),
        "tokens": getattr(result, "tokens", 0),
        "success": getattr(result, "success", False),
        "failure_class": getattr(result, "failure_class", None),
        "error_message": getattr(result, "error_message", None),
        "elapsed_seconds": getattr(result, "elapsed_seconds", 0.0),
        "cost_usd": getattr(result, "cost_usd", 0.0),
        "tool_calls": getattr(result, "tool_calls", 0),
        "tool_errors": getattr(result, "tool_errors", 0),
        "guardrail_hits": getattr(result, "guardrail_hits", 0),
        "handoff_count": getattr(result, "handoff_count", 0),
        "cache_hits": getattr(result, "cache_hits", 0),
        "cache_read_tokens": getattr(result, "cache_read_tokens", 0),
    }


async def _run_agent(agent: Any, record: RunRecord, workspace: Path) -> Any:
    """Run either a BaseAgent-style agent or a legacy keyword-style agent."""
    run = agent.run
    params = inspect.signature(run).parameters

    # BaseAgent.run(ctx) is the production path. Build the context here so
    # workers and schedulers invoke the same lifecycle used in direct tests.
    if "ctx" in params or (
        "tenant_id" not in params and "task" not in params and len(params) <= 1
    ):
        from harness.core.context import AgentContext

        metadata = dict(record.metadata or {})
        ctx = AgentContext(
            run_id=record.run_id,
            tenant_id=record.tenant_id,
            agent_type=record.agent_type,
            task=record.task,
            memory=getattr(agent, "_memory", None),
            workspace_path=workspace,
            max_steps=_metadata_int(metadata, "max_steps", 50),
            max_tokens=_metadata_int(metadata, "max_tokens", 100_000),
            timeout_seconds=_metadata_float(metadata, "timeout_seconds", 300.0),
            metadata=metadata,
        )
        result = run(ctx)
    else:
        result = run(
            tenant_id=record.tenant_id,
            task=record.task,
            run_id=record.run_id,
            workspace_path=workspace,
            metadata=record.metadata,
        )

    if asyncio.iscoroutine(result):
        return await result
    return result


def _metadata_int(metadata: dict, key: str, default: int) -> int:
    try:
        return int(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _metadata_float(metadata: dict, key: str, default: float) -> float:
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
