"""Benchmark 4b: Span recording overhead — real localhost Redis.

Re-runs the same span overhead measurements as bench_span_overhead.py
but against a real Redis instance at redis://localhost:6379.

Compares results with the fakeredis baseline to quantify real I/O cost.

Run:
    redis-server --daemonize yes   # if not already running
    PYTHONPATH=src python benchmarks/bench_span_overhead_redis.py

Output:
    benchmarks/results/span_recording_overhead_redis.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from statistics import mean, median, stdev
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = "redis://localhost:6379"
ITERATIONS = 2_000
WARMUP = 200   # longer warmup to let real Redis connection stabilise

FAKEREDIS_P50_US = 871.7   # from bench_span_overhead.py results


def _percentile(sorted_data: list[float], pct: int) -> float:
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[idx]


def _stats(latencies: list[float], label: str) -> dict:
    s = sorted(latencies)
    p50 = _percentile(s, 50)
    p95 = _percentile(s, 95)
    p99 = _percentile(s, 99)
    avg = mean(s)
    sd = stdev(s) if len(s) > 1 else 0
    print(
        f"  {label:<40} p50={p50:>8.1f}µs  p95={p95:>8.1f}µs  "
        f"p99={p99:>8.1f}µs  avg={avg:>8.1f}µs  σ={sd:>8.1f}µs"
    )
    return {
        "label": label,
        "p50_us": round(p50, 1),
        "p95_us": round(p95, 1),
        "p99_us": round(p99, 1),
        "avg_us": round(avg, 1),
        "stdev_us": round(sd, 1),
        "n": len(s),
    }


async def _bare_noop() -> None:
    pass


async def _measure_baseline(n: int) -> list[float]:
    latencies: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        await _bare_noop()
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    return latencies


async def _measure_start_end(recorder, run_id, ctx, kind, n: int) -> list[float]:
    from harness.observability.trace_schema import SpanKind, SpanStatus
    latencies: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        span_id = await recorder.start_span(
            run_id=run_id, kind=kind, name=f"bench:llm:{i}", ctx=ctx,
            input_preview="SELECT * FROM orders WHERE id = 42",
        )
        await recorder.end_span(
            run_id=run_id, span_id=span_id, status=SpanStatus.OK,
            output_preview="42 rows returned",
            input_tokens=512, output_tokens=128, cost_usd=0.00015,
        )
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    return latencies


async def _measure_context_manager(recorder, run_id, ctx, kind, n: int) -> list[float]:
    latencies: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        async with recorder.span(run_id, kind, f"bench:tool:{i}", ctx,
                                 input_preview="noop_tool_call") as _span_id:
            pass
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    return latencies


async def run() -> None:
    import redis.asyncio as aioredis
    from harness.core.context import AgentContext
    from harness.observability.trace_recorder import TraceRecorder
    from harness.observability.trace_schema import SpanKind

    print(f"\nSpan Recording Overhead — real Redis ({REDIS_URL})")
    print(f"  {ITERATIONS} iterations, {WARMUP} warmup\n")

    # Verify Redis is reachable
    try:
        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        print("  Redis: connected ✓\n")
    except Exception as exc:
        print(f"  Redis not reachable: {exc}")
        print("  Run: redis-server --daemonize yes")
        sys.exit(1)

    log_dir = Path("/tmp/bench_spans_redis")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Real Redis — no client injection
    recorder = TraceRecorder(redis_url=REDIS_URL, log_dir=log_dir)

    ctx = AgentContext(
        run_id="bench-redis",
        tenant_id="bench",
        agent_type="sql",
        task="benchmark",
        memory=MagicMock(),
        workspace_path=log_dir,
        max_steps=100_000,
        max_tokens=1_000_000,
        timeout_seconds=3600.0,
    )
    run_id = f"bench-redis-{uuid.uuid4().hex[:8]}"
    kind = SpanKind.LLM

    # Warmup
    print(f"  Warming up ({WARMUP} iterations)…")
    await _measure_start_end(recorder, run_id + "-warmup", ctx, kind, WARMUP)
    print()

    print(f"  Running measurements ({ITERATIONS} iterations each)…\n")

    baseline = await _measure_baseline(ITERATIONS)
    raw = await _measure_start_end(recorder, run_id, ctx, kind, ITERATIONS)
    ctx_mgr = await _measure_context_manager(recorder, run_id + "-cm", ctx, kind, ITERATIONS)

    baseline_p50 = _percentile(sorted(baseline), 50)

    stats = {
        "baseline":      _stats(baseline, "Baseline: bare async no-op"),
        "start_end":     _stats(raw,       "start_span + end_span"),
        "ctx_manager":   _stats(ctx_mgr,   "async with recorder.span(...)"),
    }

    net_p50 = round(stats["start_end"]["p50_us"] - baseline_p50, 1)
    net_p50_cm = round(stats["ctx_manager"]["p50_us"] - baseline_p50, 1)

    print(f"\n  Net overhead (real Redis)")
    print(f"    start_span/end_span net p50 : {net_p50:>8.1f} µs")
    print(f"    context-manager net p50     : {net_p50_cm:>8.1f} µs")
    print(f"\n  vs fakeredis baseline")
    print(f"    fakeredis net p50           : {FAKEREDIS_P50_US:>8.1f} µs")
    delta = round(net_p50 - FAKEREDIS_P50_US, 1)
    sign = "+" if delta >= 0 else ""
    print(f"    real Redis delta            : {sign}{delta:>7.1f} µs  "
          f"({'real Redis slower — network stack' if delta > 0 else 'real Redis faster — OS cache'})")

    output = {
        "benchmark": "span_recording_overhead_redis",
        "redis_url": REDIS_URL,
        "summary": {
            "iterations": ITERATIONS,
            "backend": "real_redis_localhost",
            "baseline_p50_us": round(baseline_p50, 1),
            "start_end_span_net_p50_us": net_p50,
            "context_manager_net_p50_us": net_p50_cm,
            "fakeredis_net_p50_us": FAKEREDIS_P50_US,
            "delta_vs_fakeredis_us": delta,
        },
        "measurements": list(stats.values()),
    }
    out = RESULTS_DIR / "span_recording_overhead_redis.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"\n  Results → {out}")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(run())
