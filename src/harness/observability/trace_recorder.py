"""
TraceRecorder — durable span lifecycle manager for HarnessAgent.

Each span is persisted to:
  1. Redis HASH  harness:span:{span_id}        (48 h TTL, for live query)
  2. Redis ZSET  harness:trace:{run_id}         (span_ids scored by start_time)
  3. JSONL file  logs/runs/{run_id}/trace.jsonl (append on span close)

The recorder tracks a per-run span stack so callers never need to manage
parent_span_id manually — it is always the top of the stack.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from harness.observability.trace_schema import (
    SpanKind,
    SpanStatus,
    TraceSpan,
    TraceView,
)

if TYPE_CHECKING:
    from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

_SPAN_PFX = "harness:span:"
_TRACE_PFX = "harness:trace:"
_SPAN_TTL = 172_800   # 48 h
_TRACE_TTL = 172_800


def _new_span_id() -> str:
    return format(random.getrandbits(64), "016x")


def _decode_redis_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    """Decode values written through Redis HSET back into native JSON types."""
    decoded: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, bytes):
            value = value.decode()
        if not isinstance(value, str):
            decoded[key] = value
            continue
        try:
            decoded[key] = json.loads(value)
        except json.JSONDecodeError:
            decoded[key] = value
    return decoded


class TraceRecorder:
    """
    Records, stores, and queries agent trace spans.

    Typical usage via context manager (recommended)
    -----------------------------------------------
    async with recorder.span(SpanKind.LLM, "llm:claude", ctx) as span_id:
        response = await llm.complete(...)
        recorder.set_llm_usage(span_id, response.input_tokens,
                                response.output_tokens, response.cached)

    Manual usage
    ------------
    span_id = await recorder.start_span(run_id, SpanKind.TOOL, "tool:sql", ctx)
    ...
    await recorder.end_span(run_id, span_id, output="42 rows", error=None)
    """

    def __init__(self, redis_url: str, log_dir: str | Path = "logs") -> None:
        self._redis_url = redis_url
        self._log_dir = Path(log_dir)
        self._pool: ConnectionPool | None = None
        self._client: aioredis.Redis | None = None
        # Per-run span stack: run_id → [span_id, ...]
        # Top of stack is the current parent for new spans in that run.
        self._stacks: dict[str, list[str]] = {}
        # Pending LLM usage that arrives after span creation
        self._pending_usage: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _redis(self) -> aioredis.Redis:
        if self._client is None:
            self._pool = aioredis.ConnectionPool.from_url(
                self._redis_url, max_connections=20, decode_responses=True
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    async def start_span(
        self,
        run_id: str,
        kind: SpanKind,
        name: str,
        ctx: AgentContext | None = None,
        input_preview: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Open a new span under the current parent for run_id.
        Returns the new span_id.
        """
        parent_span_id = (self._stacks.get(run_id) or [None])[-1]
        span_id = _new_span_id()

        trace_id = (ctx.trace_id if ctx else None) or run_id
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            run_id=run_id,
            kind=kind,
            name=name,
            status=SpanStatus.RUNNING,
            start_time=datetime.now(UTC),
            parent_span_id=parent_span_id,
            input_preview=input_preview[:500],
            agent_type=ctx.agent_type if ctx else "",
            tenant_id=ctx.tenant_id if ctx else "",
            step=ctx.step_count if ctx else 0,
            metadata=metadata or {},
        )

        await self._persist_span(span)
        self._stacks.setdefault(run_id, []).append(span_id)
        return span_id

    async def end_span(
        self,
        run_id: str,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        output_preview: str = "",
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> TraceSpan | None:
        """Close a span, compute duration, persist final state."""
        # Apply any pending usage registered via set_llm_usage()
        pending = self._pending_usage.pop(span_id, {})
        input_tokens = pending.get("input_tokens", input_tokens)
        output_tokens = pending.get("output_tokens", output_tokens)
        cost_usd = pending.get("cost_usd", cost_usd)
        cached = pending.get("cached", cached)

        r = await self._redis()
        raw = await r.hgetall(f"{_SPAN_PFX}{span_id}")
        if not raw:
            logger.debug("end_span: span %s not found in Redis", span_id)
            return None

        try:
            span = TraceSpan.from_dict(_decode_redis_mapping(raw))
        except Exception as exc:
            logger.debug("end_span: failed to deserialise span %s: %s", span_id, exc)
            return None

        span.finish(
            status=status,
            output_preview=output_preview,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached=cached,
        )

        await self._persist_span(span)
        await self._append_trace_jsonl(span)

        # Pop from stack (remove this span_id; may not be top if exception path)
        stack = self._stacks.get(run_id, [])
        if span_id in stack:
            stack.remove(span_id)

        # When the run's stack is empty (root span finished) drop the key so
        # _stacks does not grow unboundedly across many runs in a long-lived
        # recorder. NOTE: the list-based parent stack is not safe for truly
        # concurrent spans within one run — a fan-out of parallel spans can
        # interleave push/pop and mis-attribute parents. Spans within a run are
        # expected to be sequential; concurrent fan-out should use separate runs
        # (a contextvar-based parent would be the fix if that changes).
        if not stack:
            self._stacks.pop(run_id, None)

        return span

    def set_llm_usage(
        self,
        span_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> None:
        """Register LLM token usage so end_span can pick it up."""
        self._pending_usage[span_id] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "cached": cached,
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def span(
        self,
        run_id: str,
        kind: SpanKind,
        name: str,
        ctx: AgentContext | None = None,
        input_preview: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Async context manager that starts a span on enter and ends it on exit.
        Yields the span_id so callers can annotate it mid-flight.

        Example
        -------
        async with recorder.span(run_id, SpanKind.TOOL, "tool:sql", ctx,
                                  input_preview=str(args)) as sid:
            result = await registry.execute(ctx, call)
        """
        span_id = await self.start_span(
            run_id=run_id,
            kind=kind,
            name=name,
            ctx=ctx,
            input_preview=input_preview,
            metadata=metadata,
        )
        error: str | None = None
        final_status = SpanStatus.OK
        try:
            yield span_id
        except Exception as exc:
            error = str(exc)
            final_status = SpanStatus.ERROR
            raise
        finally:
            await self.end_span(
                run_id=run_id,
                span_id=span_id,
                status=final_status,
                error=error,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get_trace(self, run_id: str) -> TraceView | None:
        """Return all spans for run_id as a TraceView, sorted by start_time."""
        r = await self._redis()
        span_ids: list[str] = await r.zrange(f"{_TRACE_PFX}{run_id}", 0, -1)
        if not span_ids:
            return None

        spans: list[TraceSpan] = []
        for sid in span_ids:
            raw = await r.hgetall(f"{_SPAN_PFX}{sid}")
            if raw:
                try:
                    spans.append(TraceSpan.from_dict(_decode_redis_mapping(raw)))
                except Exception:
                    pass

        if not spans:
            return None

        spans.sort(key=lambda s: s.start_time)
        root = next((s for s in spans if s.parent_span_id is None), spans[0])

        return TraceView(
            trace_id=root.trace_id,
            run_id=run_id,
            agent_type=root.agent_type,
            status=root.status,
            start_time=root.start_time,
            end_time=root.end_time,
            duration_ms=root.duration_ms,
            total_input_tokens=sum(s.input_tokens for s in spans),
            total_output_tokens=sum(s.output_tokens for s in spans),
            total_cost_usd=sum(s.cost_usd for s in spans),
            span_count=len(spans),
            spans=spans,
        )

    async def get_span(self, span_id: str) -> TraceSpan | None:
        """Retrieve a single span by ID."""
        r = await self._redis()
        raw = await r.hgetall(f"{_SPAN_PFX}{span_id}")
        if not raw:
            return None
        try:
            return TraceSpan.from_dict(_decode_redis_mapping(raw))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_span(self, span: TraceSpan) -> None:
        try:
            r = await self._redis()
            span_key = f"{_SPAN_PFX}{span.span_id}"
            trace_key = f"{_TRACE_PFX}{span.run_id}"

            d = span.to_dict()
            # Redis HSET requires string values
            mapping = {k: (json.dumps(v) if not isinstance(v, str) else v)
                       for k, v in d.items() if v is not None}
            await r.hset(span_key, mapping=mapping)
            await r.expire(span_key, _SPAN_TTL)

            # Score by start_time as Unix timestamp for ordered retrieval
            score = span.start_time.timestamp()
            await r.zadd(trace_key, {span.span_id: score})
            await r.expire(trace_key, _TRACE_TTL)
        except Exception as exc:
            logger.debug("TraceRecorder._persist_span failed: %s", exc)

    def _trace_log_path(self, run_id: str) -> Path:
        p = self._log_dir / "runs" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p / "trace.jsonl"

    async def _append_trace_jsonl(self, span: TraceSpan) -> None:
        try:
            line = json.dumps(span.to_dict(), ensure_ascii=False)
            path = self._trace_log_path(span.run_id)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            logger.debug("TraceRecorder JSONL write failed: %s", exc)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, redis_url: str, log_dir: str | Path = "logs") -> TraceRecorder:
        return cls(redis_url=redis_url, log_dir=log_dir)
