"""Run all HarnessAgent benchmarks and print a consolidated paper-ready summary.

Usage:
    PYTHONPATH=src python benchmarks/run_all.py
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
BENCHMARKS = [
    ("bench_graphrag",       "GraphRAG token efficiency"),
    ("bench_cache",          "Semantic cache hit rate"),
    ("bench_circuit_breaker","Circuit breaker reliability"),
    ("bench_span_overhead",  "Span recording overhead"),
    ("bench_hermes",         "Hermes self-improvement"),
    # Multi-agent-type benchmarks (require --data-dir for real datasets)
    # Run individually:
    #   python benchmarks/bench_agent.py --agent sql  --dataset bird --data-dir /path
    #   python benchmarks/bench_agent.py --agent code --dataset humaneval --data-dir /path
    #   python benchmarks/bench_agent.py --agent base --dataset gsm8k --data-dir /path
]

# Quick self-test benchmarks (no data download needed)
AGENT_SELF_TESTS = [
    ("sql",  "AriaSql self-test (no DB)"),
    ("code", "AriaCode self-test (no sandbox)"),
    ("base", "Base agent self-test (reasoning)"),
]


def _load_result(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name.replace('bench_', '')}.json"
    candidates = list(RESULTS_DIR.glob(f"*{name.replace('bench_', '')}*.json"))
    if candidates:
        return json.loads(candidates[0].read_text())
    return None


def main() -> None:
    import asyncio

    print("=" * 70)
    print("HarnessAgent Benchmark Suite")
    print("=" * 70)

    for module_name, label in BENCHMARKS:
        print(f"\n{'─'*70}")
        print(f"Running: {label}")
        print(f"{'─'*70}")
        try:
            mod = importlib.import_module(f"benchmarks.{module_name}")
            t0 = time.perf_counter()
            asyncio.run(mod.run())
            elapsed = time.perf_counter() - t0
            print(f"[done in {elapsed:.1f}s]")
        except Exception as exc:
            print(f"[FAILED: {exc}]")

    # Consolidated summary
    print("\n" + "=" * 70)
    print("CONSOLIDATED RESULTS (paper-ready)")
    print("=" * 70)

    # GraphRAG
    r = _load_result("bench_graphrag")
    if r:
        s = r["summary"]
        print(f"\n1. GraphRAG Token Efficiency")
        print(f"   Naive vector search avg tokens : {s['naive_avg_tokens']}")
        print(f"   GraphRAG avg tokens            : {s['graphrag_avg_tokens']}")
        print(f"   Token savings                  : {s['overall_savings_pct']}%")
        print(f"   Table coverage                 : {s['avg_coverage_pct']}%")

    # Cache
    r = _load_result("bench_cache")
    if r:
        best = r.get("best_operating_point", {})
        print(f"\n2. Semantic Cache ({r.get('embedder', '?')})")
        print(f"   Best threshold      : {best.get('threshold', '?')}")
        print(f"   True-positive rate  : {best.get('true_positive_rate', 0)*100:.1f}%")
        print(f"   False-positive rate : {best.get('false_positive_rate', 0)*100:.1f}%")
        print(f"   p50 lookup latency  : {best.get('paraphrase_latency_p50_ms', '?')} ms")

    # Circuit breaker
    r = _load_result("bench_circuit_breaker")
    if r:
        scenarios = {s["name"]: s for s in r.get("scenarios", [])}
        print(f"\n3. Circuit Breaker Reliability")
        for name, s in scenarios.items():
            print(f"   {name}: final={s['final_state']}  "
                  f"time_to_open={s['time_to_open_ms']}ms  "
                  f"false_pos={s['false_positive_trip']}")

    # Span overhead
    r = _load_result("bench_span_overhead")
    if r:
        s = r["summary"]
        print(f"\n4. Span Recording Overhead ({r['summary']['iterations']} iterations)")
        print(f"   Baseline p50                   : {s['baseline_p50_us']} µs")
        print(f"   start_span+end_span net p50    : {s['start_end_span_net_p50_us']} µs")
        print(f"   Context manager net p50        : {s['context_manager_net_p50_us']} µs")
        print(f"   set_llm_usage net p50          : {s['set_llm_usage_net_p50_us']} µs")

    # Hermes
    r = _load_result("bench_hermes")
    if r:
        s = r["summary"]
        print(f"\n5. Hermes Self-Improvement")
        print(f"   Pre-patch pass@1   : {s['pre_patch_pass1']*100:.1f}%")
        print(f"   Post-patch pass@1  : {s['post_patch_pass1']*100:.1f}%")
        print(f"   Improvement        : +{s['improvement_pct']}pp")
        print(f"   Converged at cycle : {s['converged_at_cycle']}")

    print("\nAll results in: benchmarks/results/\n")


if __name__ == "__main__":
    # Allow running individual benchmarks from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
