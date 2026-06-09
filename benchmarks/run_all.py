"""Run the HarnessAgent system benchmarks and emit a publishable report.

These are the *system* benchmarks — deterministic, reproducible, and require
NO API keys or external datasets (Redis is faked in-process). They measure the
harness's own machinery: span overhead, semantic-cache behavior, circuit-breaker
reliability, GraphRAG token efficiency, and the Hermes self-improvement loop.

Task / accuracy benchmarks (BIRD, GSM8K, HumanEval, tau-bench, AgencyBench)
need real LLM calls + dataset downloads and are listed in the report with their
exact run commands — they are NOT run here and their numbers are never invented.

Usage:
    PYTHONPATH=src python benchmarks/run_all.py

Outputs:
    benchmarks/results/<benchmark>.json   (per-benchmark, written by each run())
    benchmarks/results/REPORT.md          (consolidated, publishable)
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BENCH_DIR = ROOT / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
REPORT_PATH = RESULTS_DIR / "REPORT.md"


# --------------------------------------------------------------------------
# Metric extractors — read the ACTUAL schema each benchmark writes and return
# ordered (metric, value) rows for the report. Kept beside the schema so a
# benchmark output change is a one-line fix here.
# --------------------------------------------------------------------------

def _x_graphrag(d: dict) -> list[tuple[str, str]]:
    s = d["summary"]
    return [
        ("Naive vector-search avg tokens", f"{s['naive_avg_tokens']}"),
        ("GraphRAG avg tokens", f"{s['graphrag_avg_tokens']}"),
        ("Token savings", f"{s['overall_savings_pct']}%"),
        ("Schema-table coverage", f"{s['avg_coverage_pct']}%"),
        ("Queries", f"{s['query_count']} over {s['schema_tables']} tables"),
    ]


def _x_cache(d: dict) -> list[tuple[str, str]]:
    per = {round(r["threshold"], 2): r for r in d["per_threshold"]}
    default = per.get(0.97) or d["per_threshold"][-1]
    best = d["best_zero_fp_threshold"]
    return [
        ("Embedder", f"{d['embedder']} (real={d['real_embeddings']})"),
        ("Default threshold", f"{default['threshold']}"),
        ("Near-exact hit-rate @default", f"{default['near_exact_hit_rate']*100:.0f}%"),
        ("Minor-rephrase hit-rate @default", f"{default['minor_rephrase_hit_rate']*100:.0f}%"),
        ("Decoy false-positive @default", f"{default['decoy_hit_rate']*100:.0f}%"),
        ("Lookup p50 @default", f"{default['lookup_p50_ms']} ms"),
        ("Best zero-FP threshold", f"{best['threshold']} "
            f"(near-exact {best['near_exact_hit_rate']*100:.0f}%)"),
    ]


def _x_circuit(d: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for s in d["scenarios"]:
        opened = s.get("time_to_open_ms") is not None
        verdict = (
            f"opened in {s['time_to_open_ms']} ms → final {s['final_state']}"
            if opened else f"did not open (final {s['final_state']})"
        )
        if s.get("false_positive_trip"):
            verdict += "  ⚠ false-positive trip"
        rows.append((s["name"], verdict))
    return rows


def _x_span(d: dict) -> list[tuple[str, str]]:
    s = d["summary"]
    return [
        ("Iterations / backend", f"{s['iterations']} / {s['backend']}"),
        ("Baseline p50", f"{s['baseline_p50_us']} µs"),
        ("start_span+end_span net p50", f"{s['start_end_span_net_p50_us']} µs"),
        ("Context-manager net p50", f"{s['context_manager_net_p50_us']} µs"),
        ("set_llm_usage net p50", f"{s['set_llm_usage_net_p50_us']} µs"),
    ]


def _x_hermes(d: dict) -> list[tuple[str, str]]:
    s = d["summary"]
    conv = s["converged_at_cycle"]
    return [
        ("Seed failures", f"{s['seed_failure_count']}"),
        ("Pre-patch pass@1", f"{s['pre_patch_pass1']*100:.1f}%"),
        ("Post-patch pass@1", f"{s['post_patch_pass1']*100:.1f}%"),
        ("Improvement", f"+{s['improvement_pct']} pp"),
        ("Cycles run", f"{s['cycles_run']}"),
        ("Converged at cycle", f"{conv}" if conv is not None else "not converged"),
        ("Patches applied", f"{s['patches_applied']}"),
    ]


# module file, result json, label, extractor, note
SYSTEM_BENCHMARKS = [
    ("bench_graphrag.py",        "graphrag_token_efficiency.json", "GraphRAG token efficiency", _x_graphrag,
     "Measures schema-token reduction vs naive vector retrieval on a synthetic 10-table schema."),
    ("bench_cache.py",           "semantic_cache_hit_rate.json",   "Semantic cache hit-rate",   _x_cache,
     "Real all-MiniLM-L6-v2 embeddings over curated near-exact/rephrase/paraphrase/decoy query tiers."),
    ("bench_circuit_breaker.py", "circuit_breaker_reliability.json","Circuit-breaker reliability", _x_circuit,
     "Failure-injection scenarios against the real CircuitBreaker; verifies it opens under failure and recovers."),
    ("bench_span_overhead.py",   "span_recording_overhead.json",   "Span recording overhead",   _x_span,
     "Wall-clock cost the TraceRecorder adds over a bare async no-op (fakeredis backend; real Redis adds network)."),
    ("bench_hermes.py",          "hermes_self_improvement.json",   "Hermes self-improvement (synthetic)", _x_hermes,
     "SYNTHETIC loop-mechanics demo: mock LLM + mock agent runner modeling a 10%→95% golden fix. "
     "Exercises detect→propose→evaluate→auto-apply→converge; the improvement magnitude is a property of "
     "the model, NOT a measured LLM gain. Real self-improvement: run bench_hermes_real.py with a live LLM."),
]

# Task / accuracy benchmarks — require API keys + datasets. NOT run here.
TASK_BENCHMARKS = [
    ("SQL exec accuracy (BIRD)", "bench_agent.py --agent sql --dataset bird --data-dir <path>"),
    ("Code pass@1 (HumanEval)",  "bench_agent.py --agent code --dataset humaneval --data-dir <path>"),
    ("Reasoning (GSM8K)",        "bench_agent.py --agent base --dataset gsm8k --data-dir <path>"),
    ("BIRD (live LLM)",          "bench_bird_real.py --data-dir <path>"),
    ("Hermes (live LLM)",        "bench_hermes_real.py"),
    ("Safety trajectories (tau-bench)", "bench_taubench.py"),
    ("AgencyBench ablation",     "bench_agencybench_ablation.py"),
]


def _load_module(filename: str):
    path = BENCH_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_bench_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / type-hint resolution in the module
    # can find their own module via sys.modules (otherwise asdict / get_type_hints
    # on nested dataclasses raises 'NoneType' has no attribute '__dict__').
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    print("=" * 70)
    print("HarnessAgent — System Benchmark Suite")
    print("=" * 70)

    suite_start = time.time()
    statuses: dict[str, dict] = {}

    for filename, result_file, label, _extractor, _note in SYSTEM_BENCHMARKS:
        print(f"\n{'─'*70}\nRunning: {label}\n{'─'*70}")
        entry = {"label": label, "result_file": result_file, "status": "failed",
                 "elapsed_s": 0.0, "error": None}
        try:
            mod = _load_module(filename)
            t0 = time.perf_counter()
            asyncio.run(mod.run())
            entry["elapsed_s"] = round(time.perf_counter() - t0, 1)
            rpath = RESULTS_DIR / result_file
            # Fresh = the run() we just invoked rewrote the file.
            if rpath.exists() and rpath.stat().st_mtime >= suite_start:
                entry["status"] = "fresh"
            elif rpath.exists():
                entry["status"] = "stale"  # ran but didn't update its output
            print(f"[{entry['status']} in {entry['elapsed_s']}s]")
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED: {entry['error']}]")
            traceback.print_exc()
        statuses[filename] = entry

    _print_console_summary(statuses)
    _write_report(statuses, suite_start)
    print(f"\nReport written to {REPORT_PATH}")

    failed = [e["label"] for e in statuses.values() if e["status"] == "failed"]
    if failed:
        print(f"\n⚠ {len(failed)} benchmark(s) FAILED: {', '.join(failed)}")
        sys.exit(1)


def _print_console_summary(statuses: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("CONSOLIDATED RESULTS")
    print("=" * 70)
    for filename, result_file, label, extractor, _note in SYSTEM_BENCHMARKS:
        st = statuses[filename]
        print(f"\n{label}  [{st['status']}]")
        if st["status"] in ("failed",):
            print(f"   (no fresh result — {st['error']})")
            continue
        try:
            d = json.loads((RESULTS_DIR / result_file).read_text())
            for metric, value in extractor(d):
                print(f"   {metric:38s}: {value}")
        except Exception as exc:
            print(f"   (could not read result: {exc})")


def _write_report(statuses: dict[str, dict], suite_start: float) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# HarnessAgent Benchmark Report\n")
    lines.append(f"_Generated {now} · Python {platform.python_version()} · "
                 f"{platform.system()} {platform.machine()}_\n")
    lines.append(
        "\n**System benchmarks** below are deterministic and reproducible with "
        "`PYTHONPATH=src python benchmarks/run_all.py` — no API keys or datasets "
        "required (Redis is faked in-process). **Task/accuracy benchmarks** "
        "require live LLM calls and dataset downloads; their commands are listed "
        "at the end and must be run with your own keys/data.\n")

    lines.append("\n## System benchmarks (reproducible)\n")
    any_fresh = False
    for filename, result_file, label, extractor, note in SYSTEM_BENCHMARKS:
        st = statuses[filename]
        badge = {"fresh": "✅ fresh", "stale": "⚠️ stale", "failed": "❌ failed"}.get(
            st["status"], st["status"])
        lines.append(f"\n### {label}  ({badge}, {st['elapsed_s']}s)\n")
        lines.append(f"\n_{note}_\n")
        if st["status"] == "failed":
            lines.append(f"\n> Run failed: `{st['error']}`\n")
            continue
        try:
            d = json.loads((RESULTS_DIR / result_file).read_text())
            rows = extractor(d)
            lines.append("\n| Metric | Value |")
            lines.append("\n|---|---|")
            for metric, value in rows:
                lines.append(f"\n| {metric} | {value} |")
            lines.append("\n")
            if st["status"] == "fresh":
                any_fresh = True
        except Exception as exc:
            lines.append(f"\n> Could not read result: `{exc}`\n")

    lines.append("\n## Task / accuracy benchmarks (require API keys + datasets)\n")
    lines.append(
        "\nNot run by this script — run individually with your own credentials and "
        "datasets. Prior result JSONs (if any) live in `benchmarks/results/` but are "
        "only valid for the exact model + dataset version they were produced with.\n")
    lines.append("\n| Benchmark | Command (prefix `PYTHONPATH=src python benchmarks/`) |")
    lines.append("\n|---|---|")
    for label, cmd in TASK_BENCHMARKS:
        lines.append(f"\n| {label} | `{cmd}` |")
    lines.append("\n")

    if not any_fresh:
        lines.append("\n> ⚠️ No system benchmark produced a fresh result this run.\n")

    REPORT_PATH.write_text("".join(lines))


if __name__ == "__main__":
    main()
