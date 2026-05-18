"""Benchmark 4: Trace span recording overhead — p50/p95/p99 latency.

Measures the wall-clock time added by the TraceRecorder on top of a bare
async no-op call. Uses fakeredis so no real Redis is needed.

Three measurements:
  1. Raw overhead — start_span + end_span round-trip
  2. Context-manager overhead — async with recorder.span(...) block
  3. LLM usage annotation — set_llm_usage on an open span

All timings are in microseconds (µs). Runs 2 000 iterations per measurement.

Run:
    PYTHONPATH=src python benchmarks/bench_span_overhead.py

Output:
    benchmarks/results/span_recording_overhead.json
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

ITERATIONS = 2_000
WARMUP = 100


def _percentile(sorted_data: list[float], p: float) -> float:
    idx = min(int(len(sorted_data) * p / 100), len(sorted_data) - 1)
    return sorted_data[idx]


async def _bare_noop() -> None:
    """Baseline: pure async function with no harness work."""
    pass


async def _measure_start_end(recorder, run_id, ctx, kind, n: int) -> list[float]:
    from harness.observability.trace_schema import SpanKind, SpanStatus

    latencies: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        span_id = await recorder.start_span(
            run_id=run_id,
            kind=kind,
            name=f"bench:llm:{i}",
            ctx=ctx,
            input_preview="SELECT * FROM orders WHERE id = 42",
        )
        await recorder.end_span(
            run_id=run_id,
            span_id=span_id,
            status=SpanStatus.OK,
            output_preview="42 rows returned",
            input_tokens=512,
            output_tokens=128,
            cost_usd=0.00015,
        )
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    return latencies


async def _measure_context_manager(recorder, run_id, ctx, kind, n: int) -> list[float]:
    latencies: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        async with recorder.span(run_id, kind, f"bench:tool:{i}", ctx,
                                 input_preview="noop_tool_call") as _span_id:
            pass   # the code under test is just the harness overhead
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    return latencies


async def _measure_set_llm_usage(recorder, run_id, ctx, kind, n: int) -> list[float]:
    from harness.observability.trace_schema import SpanKind, SpanStatus

    # Pre-open n spans, then measure set_llm_usage per span
    span_ids = []
    for i in range(n):
        sid = await recorder.start_span(run_id, kind, f"bench:llm:usage:{i}", ctx)
        span_ids.append(sid)

    latencies: list[float] = []
    for sid in span_ids:
        t0 = time.perf_counter()
        recorder.set_llm_usage(sid, input_tokens=512, output_tokens=128, cost_usd=0.00015)
        latencies.append((time.perf_counter() - t0) * 1_000_000)

    # Close them all
    for sid in span_ids:
        await recorder.end_span(run_id, sid, status=SpanStatus.OK)

    return latencies


async def _measure_baseline(n: int) -> list[float]:
    latencies: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        await _bare_noop()
        latencies.append((time.perf_counter() - t0) * 1_000_000)
    return latencies


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
        "n": len(latencies),
    }


async def run() -> None:
    import fakeredis.aioredis as fake_aioredis  # type: ignore

    from harness.core.context import AgentContext
    from harness.observability.trace_recorder import TraceRecorder
    from harness.observability.trace_schema import SpanKind

    print(f"\nSpan Recording Overhead Benchmark ({ITERATIONS} iterations each)")
    print(f"  Backend: fakeredis (in-process, no real I/O)\n")

    # Wire up fakeredis
    fake_redis = fake_aioredis.FakeRedis(decode_responses=True)
    log_dir = Path("/tmp/bench_spans")
    log_dir.mkdir(parents=True, exist_ok=True)

    recorder = TraceRecorder(redis_url="redis://localhost:6379", log_dir=log_dir)
    # Inject the fake Redis client to avoid real connection
    recorder._client = fake_redis

    ctx = AgentContext(
        run_id="bench-overhead",
        tenant_id="bench",
        agent_type="sql",
        task="benchmark",
        memory=MagicMock(),
        workspace_path=Path("/tmp/bench_spans"),
        max_steps=100_000,
        max_tokens=1_000_000,
        timeout_seconds=3600.0,
    )
    run_id = "bench-overhead"
    kind = SpanKind.LLM

    # Warmup
    print("Warming up...")
    await _measure_start_end(recorder, run_id + "-warmup", ctx, kind, WARMUP)
    await _measure_context_manager(recorder, run_id + "-warmup", ctx, kind, WARMUP)

    print("Running measurements...\n")
    measurements = []

    # 1. Baseline
    base = await _measure_baseline(ITERATIONS)
    measurements.append(_stats(base, "Baseline: bare async no-op"))

    # 2. start_span + end_span
    se = await _measure_start_end(recorder, run_id, ctx, kind, ITERATIONS)
    measurements.append(_stats(se, "start_span + end_span"))

    # 3. Context manager
    cm = await _measure_context_manager(recorder, run_id, ctx, kind, ITERATIONS)
    measurements.append(_stats(cm, "async with recorder.span(...)"))

    # 4. set_llm_usage
    usage = await _measure_set_llm_usage(recorder, run_id, ctx, kind, ITERATIONS)
    measurements.append(_stats(usage, "set_llm_usage (sync annotation)"))

    # Compute net overhead vs baseline
    base_p50 = measurements[0]["p50_us"]
    print("\nNet harness overhead (minus baseline):")
    for m in measurements[1:]:
        net = m["p50_us"] - base_p50
        print(f"  {m['label']:<40} net p50 = {net:+.1f}µs")

    overhead_p50 = measurements[1]["p50_us"] - base_p50

    summary = {
        "iterations": ITERATIONS,
        "backend": "fakeredis",
        "baseline_p50_us": base_p50,
        "start_end_span_net_p50_us": round(measurements[1]["p50_us"] - base_p50, 1),
        "context_manager_net_p50_us": round(measurements[2]["p50_us"] - base_p50, 1),
        "set_llm_usage_net_p50_us": round(measurements[3]["p50_us"] - base_p50, 1),
    }

    print(f"\nSpan overhead (p50): ~{overhead_p50:.0f} µs = {overhead_p50/1000:.3f} ms")

    output = {
        "benchmark": "span_recording_overhead",
        "summary": summary,
        "measurements": measurements,
    }
    out_path = RESULTS_DIR / "span_recording_overhead.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")

    await fake_redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
