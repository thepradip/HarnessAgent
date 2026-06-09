"""Failure tracking, classification, and sampling for HarnessAgent."""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Re-export FailureClass from errors so importers can get both from one place
from harness.core.errors import FailureClass  # noqa: F401, E402


@dataclass
class StepFailure:
    """Structured record of a single agent step failure."""

    run_id: str
    step_number: int
    failure_class: FailureClass
    message: str
    agent_type: str

    # Optional context fields
    tool_name: str | None = None
    mcp_server: str | None = None
    inter_agent_target: str | None = None
    model: str | None = None
    provider: str | None = None
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    mlflow_span_id: str | None = None
    otel_trace_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hermes_patch_proposed: str | None = None
    hermes_patch_applied: bool = False

    # Auto-generated unique ID
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_number": self.step_number,
            "failure_class": self.failure_class.value,
            "message": self.message,
            "agent_type": self.agent_type,
            "tool_name": self.tool_name,
            "mcp_server": self.mcp_server,
            "inter_agent_target": self.inter_agent_target,
            "model": self.model,
            "provider": self.provider,
            "context_snapshot": self.context_snapshot,
            "stack_trace": self.stack_trace,
            "mlflow_span_id": self.mlflow_span_id,
            "otel_trace_id": self.otel_trace_id,
            "timestamp": self.timestamp.isoformat(),
            "hermes_patch_proposed": self.hermes_patch_proposed,
            "hermes_patch_applied": self.hermes_patch_applied,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepFailure":
        d = dict(d)
        d["failure_class"] = FailureClass(d["failure_class"])
        ts = d.get("timestamp")
        if isinstance(ts, str):
            d["timestamp"] = datetime.fromisoformat(ts)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        run_id: str,
        step_number: int,
        agent_type: str,
        failure_class: FailureClass = FailureClass.UNKNOWN,
        **kwargs: Any,
    ) -> "StepFailure":
        """Convenience constructor capturing the current traceback."""
        return cls(
            run_id=run_id,
            step_number=step_number,
            failure_class=failure_class,
            message=str(exc),
            agent_type=agent_type,
            stack_trace=traceback.format_exc(),
            **kwargs,
        )


@dataclass
class FailureSummary:
    """Aggregated failure statistics for an agent type."""

    top_classes: list[tuple[str, int]]
    failure_by_tool: dict[str, int]
    total_failures: int
    window_hours: float
    agent_type: str | None


class FailureTracker:
    """
    Records, samples, and summarises agent step failures.

    Failures are:
    - Embedded and stored in the vector store (for semantic search/sampling).
    - Incremented in Prometheus counters.
    - Published to Redis stream ``harness:failures``.
    - Logged as structured JSON.
    """

    _STREAM_KEY = "harness:failures"
    _ERRORS_COLLECTION = "harness_errors"

    def __init__(
        self,
        vector_store: Any = None,
        embedder: Any = None,
        redis_client: Any = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def record(self, failure: StepFailure) -> None:
        """
        Persist a StepFailure across all sinks:
        1. Structured JSON log.
        2. Prometheus counter increment.
        3. Vector store upsert (for sampling).
        4. Redis stream publish.
        """
        failure_dict = failure.to_dict()

        # 1. Structured log
        logger.error(
            "Step failure recorded",
            extra={"failure": failure_dict},
        )

        # 2. Prometheus
        try:
            from harness.observability.metrics import failure_total

            failure_total.labels(
                failure_class=failure.failure_class.value,
                agent_type=failure.agent_type,
            ).inc()
        except Exception as exc:
            logger.debug("Prometheus increment failed: %s", exc)

        # 3. Vector store
        if self._vector_store is not None and self._embedder is not None:
            try:
                embed_text = f"{failure.failure_class.value} {failure.message} {failure.stack_trace[:500]}"
                embeddings = await self._embedder.embed([embed_text])
                metadata: dict[str, Any] = {
                    "run_id": failure.run_id,
                    "failure_class": failure.failure_class.value,
                    "agent_type": failure.agent_type,
                    "timestamp": failure.timestamp.isoformat(),
                }
                if failure.tool_name:
                    metadata["tool_name"] = failure.tool_name
                await self._vector_store.upsert(
                    id=failure.id,
                    text=embed_text,
                    metadata=metadata,
                    embedding=embeddings[0],
                )
            except Exception as exc:
                logger.debug("Vector store failure record failed: %s", exc)

        # 4. Redis stream
        if self._redis is not None:
            try:
                await self._redis.xadd(
                    self._STREAM_KEY,
                    {"data": json.dumps(failure_dict, default=str)},
                    maxlen=10_000,
                    approximate=True,
                )
            except Exception as exc:
                logger.debug("Redis stream publish failed: %s", exc)

    async def sample_batch(
        self,
        agent_type: str,
        failure_class: FailureClass | None = None,
        k: int = 10,
    ) -> list[StepFailure]:
        """
        Return up to ``k`` representative failure records via vector search.

        The query is constructed from agent_type + optional failure_class.
        """
        if self._vector_store is None:
            return []

        query = f"{agent_type} failure"
        if failure_class is not None:
            query = f"{agent_type} {failure_class.value} failure"

        filter_dict: dict[str, Any] = {"agent_type": agent_type}
        if failure_class is not None:
            filter_dict["failure_class"] = failure_class.value

        try:
            hits = await self._vector_store.query(
                text=query, k=k, filter=filter_dict
            )
        except Exception as exc:
            logger.warning("FailureTracker.sample_batch query failed: %s", exc)
            return []

        failures: list[StepFailure] = []
        for hit in hits:
            meta = hit.metadata
            try:
                failures.append(
                    StepFailure(
                        id=hit.id,
                        run_id=meta.get("run_id", ""),
                        step_number=meta.get("step_number", 0),
                        failure_class=FailureClass(
                            meta.get("failure_class", FailureClass.UNKNOWN.value)
                        ),
                        message=hit.text,
                        agent_type=meta.get("agent_type", agent_type),
                        timestamp=datetime.fromisoformat(
                            meta.get("timestamp", datetime.now(timezone.utc).isoformat())
                        ),
                    )
                )
            except Exception:
                pass

        return failures

    async def get_summary(
        self,
        agent_type: str | None = None,
        window_hours: float = 24,
    ) -> dict[str, Any]:
        """
        Return aggregated statistics from the Redis failure stream.

        Falls back to empty summary if Redis unavailable.
        """
        summary = FailureSummary(
            top_classes=[],
            failure_by_tool={},
            total_failures=0,
            window_hours=window_hours,
            agent_type=agent_type,
        )

        if self._redis is None:
            return {
                "top_classes": [],
                "failure_by_tool": {},
                "total_failures": 0,
                "window_hours": window_hours,
                "agent_type": agent_type,
            }

        try:
            # Read last 1000 entries from the stream
            entries = await self._redis.xrange(
                self._STREAM_KEY, count=1_000
            )

            class_counts: dict[str, int] = {}
            tool_counts: dict[str, int] = {}
            cutoff_ts = (
                datetime.now(timezone.utc).timestamp() - window_hours * 3600
            )
            total = 0

            for _stream_id, data in entries:
                try:
                    failure_data = json.loads(data.get("data", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue

                ts_str = failure_data.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                except (ValueError, TypeError):
                    # Unparseable timestamp: skip rather than counting the entry
                    # in-window, which would inflate the windowed summary stats.
                    continue
                if ts < cutoff_ts:
                    continue

                at = failure_data.get("agent_type", "")
                if agent_type and at != agent_type:
                    continue

                fc = failure_data.get("failure_class", "UNKNOWN")
                class_counts[fc] = class_counts.get(fc, 0) + 1

                tool = failure_data.get("tool_name")
                if tool:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

                total += 1

            summary.top_classes = sorted(
                class_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
            summary.failure_by_tool = tool_counts
            summary.total_failures = total

        except Exception as exc:
            logger.warning("FailureTracker.get_summary failed: %s", exc)

        return {
            "top_classes": summary.top_classes,
            "failure_by_tool": summary.failure_by_tool,
            "total_failures": summary.total_failures,
            "window_hours": summary.window_hours,
            "agent_type": summary.agent_type,
        }

    async def get_heatmap(self) -> dict[str, dict[str, int]]:
        """
        Return a 2D count matrix: {agent_type: {failure_class: count}}.

        Reads from the Redis failure stream.
        """
        heatmap: dict[str, dict[str, int]] = {}

        if self._redis is None:
            return heatmap

        try:
            entries = await self._redis.xrange(self._STREAM_KEY, count=5_000)
            for _stream_id, data in entries:
                try:
                    failure_data = json.loads(data.get("data", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue

                at = failure_data.get("agent_type", "unknown")
                fc = failure_data.get("failure_class", "UNKNOWN")

                if at not in heatmap:
                    heatmap[at] = {}
                heatmap[at][fc] = heatmap[at].get(fc, 0) + 1

        except Exception as exc:
            logger.warning("FailureTracker.get_heatmap failed: %s", exc)

        return heatmap
