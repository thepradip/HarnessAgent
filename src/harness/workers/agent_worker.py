"""RQ-based agent worker for processing run jobs from the queue."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async job handler
# ---------------------------------------------------------------------------


async def process_run_job_async(run_id: str, config_dict: dict | None = None) -> None:
    """Process a single agent run job asynchronously.

    Initialises all required dependencies (Redis, memory manager, LLM router),
    delegates to AgentRunner.execute_run(), and updates the run record on
    completion or failure.

    Args:
        run_id:      The run identifier to execute.
        config_dict: Optional configuration overrides for this run.
    """
    config_dict = config_dict or {}

    # --- Import harness dependencies ---
    import redis.asyncio as aioredis

    from harness.core.config import get_config

    cfg = get_config()

    # Initialise Redis
    redis_client = aioredis.from_url(
        config_dict.get("redis_url", cfg.redis_url),
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=10,
    )

    try:
        await redis_client.ping()
        logger.info("Worker: Redis connected for run %s", run_id)
    except Exception as exc:
        logger.error("Worker: Redis connection failed: %s", exc)
        raise

    # Build an agent factory bound to this worker's live Redis connection
    def agent_factory(agent_type: str) -> Any:
        """Create the appropriate agent for the given type."""
        try:
            return _build_agent(
                agent_type, cfg, config_dict=config_dict, redis_client=redis_client
            )
        except ImportError as exc:
            logger.warning("Agent module not found for %s: %s", agent_type, exc)
            raise RuntimeError(f"Agent '{agent_type}' is not available: {exc}") from exc

    # Initialise error collector for failure recording
    from harness.improvement.error_collector import ErrorCollector
    error_collector = ErrorCollector(redis=redis_client)

    # Build the AgentRunner
    from harness.orchestrator.runner import AgentRunner
    runner = AgentRunner(
        redis=redis_client,
        agent_factory=agent_factory,
        workspace_base=config_dict.get("workspace_base_path", cfg.workspace_base_path),
        error_collector=error_collector,
    )

    # Execute the run
    try:
        record = await runner.execute_run(run_id)
        logger.info(
            "Worker: run %s completed with status=%s",
            run_id,
            record.status,
        )
    except KeyError as exc:
        logger.error("Worker: run %s not found: %s", run_id, exc)
        raise
    except Exception as exc:
        logger.exception("Worker: run %s raised unhandled exception: %s", run_id, exc)

        # Attempt to mark as failed in Redis
        try:
            from datetime import datetime

            from harness.orchestrator.runner import RunRecord, _run_key

            raw = await redis_client.get(_run_key(run_id))
            if raw:
                rec = RunRecord.from_json(raw if isinstance(raw, str) else raw.decode())
                rec.status = "failed"
                rec.completed_at = datetime.now(UTC)
                rec.result = {
                    "run_id": run_id,
                    "output": "",
                    "steps": 0,
                    "tokens": 0,
                    "success": False,
                    "error_message": str(exc),
                }
                await redis_client.set(_run_key(run_id), rec.to_json())
        except Exception as update_exc:
            logger.warning("Worker: failed to update run status: %s", update_exc)

        raise
    finally:
        await redis_client.aclose()


# ---------------------------------------------------------------------------
# Synchronous job entry point (called by RQ)
# ---------------------------------------------------------------------------


def process_run_job(run_id: str, config_dict: dict | None = None) -> None:
    """Synchronous wrapper called by RQ workers.

    RQ enqueues calls to this function. It bridges RQ's synchronous
    interface to the async implementation.

    Args:
        run_id:      The run identifier to execute.
        config_dict: Optional configuration overrides.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("RQ job: processing run %s", run_id)
    asyncio.run(process_run_job_async(run_id, config_dict or {}))


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def start_worker(
    queues: list[str] | None = None,
    redis_url: str | None = None,
) -> None:
    """Start an RQ worker listening on the specified queues.

    Args:
        queues:    Queue names to listen on.  Defaults to ["default", "agent"].
        redis_url: Redis URL.  Falls back to config.
    """
    import redis as sync_redis  # type: ignore
    from rq import Queue, Worker  # type: ignore

    from harness.core.config import get_config

    cfg = get_config()
    effective_redis_url = redis_url or cfg.redis_url
    effective_queues = queues or ["default", "agent", "sql", "code"]

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    conn = sync_redis.from_url(effective_redis_url, decode_responses=False)
    queue_objects = [Queue(q, connection=conn) for q in effective_queues]

    logger.info(
        "Starting RQ worker on queues: %s (redis=%s)",
        effective_queues,
        effective_redis_url,
    )

    worker = Worker(queue_objects, connection=conn)
    worker.work(with_scheduler=True)


# ---------------------------------------------------------------------------
# Plugin registry — third-party agent types
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: dict[str, type] = {}


def register_agent(agent_type: str, agent_class: type) -> None:
    """Register a custom agent class for a given agent_type string.

    Example
    -------
    from harness.workers.agent_worker import register_agent
    from mypackage import ResearchAgent

    register_agent("research", ResearchAgent)
    # Now POST /runs with agent_type="research" will use ResearchAgent
    """
    _AGENT_REGISTRY[agent_type] = agent_class
    logger.info("Registered custom agent: %s → %s", agent_type, agent_class.__name__)


def _build_agent(
    agent_type: str,
    cfg: Any,
    config_dict: dict | None = None,
    redis_client: Any = None,
) -> Any:
    """Create a production BaseAgent subclass with shared harness services.

    Args:
        agent_type:   Which agent to build ('sql', 'code', or a registered type).
        cfg:          Harness config (from get_config()).
        config_dict:  Optional per-run configuration overrides.
        redis_client: Async Redis client used for event streaming and cost
                      tracking.  Pass the live connection so events and
                      budgets are actually persisted.
    """
    from harness.core.cost_tracker import CostTracker
    from harness.llm.factory import build_router
    from harness.observability.trace_recorder import TraceRecorder
    from harness.safety.pipeline_factory import build_pipeline, get_default_config
    from harness.tools.code_tools import ApplyPatchTool, LintCodeTool, RunCodeTool
    from harness.tools.file_tools import ListWorkspaceTool, ReadFileTool, WriteFileTool
    from harness.tools.registry import ToolRegistry

    config_dict = config_dict or {}

    safety = build_pipeline(agent_type, get_default_config(agent_type))
    registry = ToolRegistry(safety_pipeline=safety)
    for tool in [
        ReadFileTool(),
        WriteFileTool(),
        ListWorkspaceTool(),
        RunCodeTool(),
        LintCodeTool(),
        ApplyPatchTool(),
    ]:
        registry.register(tool)

    sql_connection = (
        config_dict.get("sql_connection_string")
        or getattr(cfg, "sql_connection_string", "")
    )
    if sql_connection:
        from harness.tools.sql_tools import SQLConnectionConfig, build_sql_tools

        for tool in build_sql_tools(SQLConnectionConfig(connection_string=sql_connection)):
            registry.register(tool)

    common_kwargs = {
        "llm_router": build_router(cfg),
        "memory_manager": _NoopMemory(),
        "tool_registry": registry,
        "safety_pipeline": safety,
        "step_tracer": None,
        "mlflow_tracer": None,
        "failure_tracker": None,
        "audit_logger": None,
        "event_bus": _RedisStreamEventSink(redis_client),
        "cost_tracker": CostTracker(
            redis_client=redis_client,
            budget_usd_per_tenant=cfg.cost_budget_usd_per_tenant,
        ),
        "checkpoint_manager": _NoopCheckpointManager(),
        "message_bus": None,
        "trace_recorder": TraceRecorder.create(
            redis_url=config_dict.get("redis_url", cfg.redis_url),
        ),
    }

    if agent_type == "sql":
        from harness.agents.sql_agent import SQLAgent

        return SQLAgent(**common_kwargs)
    if agent_type == "code":
        from harness.agents.code_agent import CodeAgent

        return CodeAgent(**common_kwargs)
    # Plugin registry — third-party agents registered at startup
    if agent_type in _AGENT_REGISTRY:
        cls = _AGENT_REGISTRY[agent_type]
        return cls(**common_kwargs)

    raise ValueError(
        f"Unknown agent_type {agent_type!r}. "
        f"Supported: 'sql', 'code', or register a custom agent with "
        f"harness.workers.agent_worker.register_agent('{agent_type}', MyAgent)."
    )


def build_agent_factory(cfg: Any) -> Any:
    """Return a callable agent factory using the default harness config.

    Used by the API server at startup so it can execute runs directly
    without delegating to the RQ worker (useful in dev / single-process mode).

    Returns
    -------
    callable
        A function (agent_type: str) -> BaseAgent subclass instance.
    """
    import redis.asyncio as aioredis

    # Shared sync Redis client for the factory (used only for event sink + cost)
    _redis = aioredis.from_url(
        cfg.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
    )

    def _factory(agent_type: str) -> Any:
        return _build_agent(agent_type, cfg, redis_client=_redis)

    return _factory


class _NoopMemory:
    """Small memory shim for worker smoke runs when no memory stack is configured."""

    async def push_message(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def get_history(self, *args: Any, **kwargs: Any) -> list:
        return []

    async def fit_history(self, *args: Any, **kwargs: Any) -> Any:
        return type("FitHistoryResult", (), {"summary": None, "messages": []})()

    async def smart_retrieve(self, *args: Any, **kwargs: Any) -> str:
        return ""


class _NoopCheckpointManager:
    async def load(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def save(self, *args: Any, **kwargs: Any) -> None:
        return None


class _RedisStreamEventSink:
    """Event sink compatible with BaseAgent._emit_event and /runs/{id}/stream."""

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    async def publish(self, event: Any) -> None:
        await self._redis.xadd(
            f"harness:events:{event.run_id}",
            {
                "run_id": event.run_id,
                "step": str(event.step),
                "event_type": event.event_type,
                "payload": json.dumps(event.payload, default=str),
                "timestamp": event.timestamp.isoformat(),
            },
            maxlen=1000,
            approximate=True,
        )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    start_worker()
