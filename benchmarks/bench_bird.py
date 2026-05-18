"""BIRD Benchmark — evaluate a SQL agent on the BIRD-SQL dataset.

Tests the harness SQL agent (or any custom agent) against BIRD queries using:
  - SQLSandbox for deterministic execution-based scoring
  - SQLVerifier for step-by-step verification (schema → syntax → execution → result → quality)
  - AgentEvalReport for results breakdown by hardness
  - RLVR loop for prompt improvement from low-advantage episodes

Compatible agent: HarnessAgent SQLAgent (agent_type="sql") or any callable
    fn(task: str, db_path: str) -> str   (returns generated SQL)

Dataset: BIRD-SQL (https://bird-bench.github.io)
    bird_dir must contain:
      dev/dev.json
      dev/databases/{db_id}/{db_id}.sqlite

Usage (no LLM key required — uses mock agent by default):
    PYTHONPATH=src python benchmarks/bench_bird.py \\
        --bird-dir /path/to/bird \\
        --n-samples 50 \\
        --output benchmarks/results/bird_results.json

Usage with real agent:
    PYTHONPATH=src python benchmarks/bench_bird.py \\
        --bird-dir /path/to/bird \\
        --agent ariasql \\
        --n-samples 50

Output:
    benchmarks/results/bird_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bench_bird")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    question: str
    db_id: str
    hardness: str
    gold_sql: str
    generated_sql: str
    overall_reward: float
    verdict: str
    step_scores: dict[str, float]  # step_name → score
    feedback: str
    execution_time_ms: float
    cached: bool


@dataclass
class BIRDReport:
    n_cases: int
    n_evaluated: int
    overall_pass_rate: float
    execution_accuracy: float                      # exact result-set match rate
    by_hardness: dict[str, dict[str, float]]       # hardness → {pass_rate, exec_acc, count}
    step_pass_rates: dict[str, float]              # step_name → fraction passed
    failure_distribution: dict[str, int]
    avg_reward: float
    avg_exec_time_ms: float
    cases: list[CaseResult]
    run_id: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Mock SQL agent (deterministic — no LLM key required)
# ---------------------------------------------------------------------------

async def mock_sql_agent(task: str, db_path: str | None) -> str:
    """
    Minimal mock agent that tries to generate plausible SQL for common patterns.
    Always produces valid SELECT — used when no real agent is configured.
    """
    task_lower = task.lower()

    # Pattern: count queries
    if any(w in task_lower for w in ("how many", "count", "number of", "total number")):
        # Find a likely table name from the task
        nouns = [w for w in task_lower.split() if len(w) > 3 and w.isalpha()]
        table = nouns[-1] if nouns else "records"
        return f"SELECT COUNT(*) FROM {table}"

    # Pattern: list / show queries
    if any(w in task_lower for w in ("list", "show", "display", "find", "get")):
        nouns = [w for w in task_lower.split() if len(w) > 3 and w.isalpha()]
        table = nouns[-1] if nouns else "data"
        return f"SELECT * FROM {table} LIMIT 10"

    # Pattern: max/min
    if "maximum" in task_lower or "highest" in task_lower or "max" in task_lower:
        return "SELECT MAX(value) FROM records"
    if "minimum" in task_lower or "lowest" in task_lower or "min" in task_lower:
        return "SELECT MIN(value) FROM records"

    # Fallback
    return "SELECT 1"


# ---------------------------------------------------------------------------
# AriaSql agent stub (extend this with real implementation)
# ---------------------------------------------------------------------------

async def ariasql_agent(task: str, db_path: str | None) -> str:
    """
    AriaSql agent using HarnessAgent's AriaSql class.

    Uses SchemaStore (L3 context) + SQLVerifier self-correction.
    No LLM key required when _ARIASQL_INSTANCE is None (falls back to mock).

    To use a real LLM:
        Set ANTHROPIC_API_KEY or OPENAI_API_KEY in environment,
        or call _init_ariasql_agent() before running the benchmark.
    """
    global _ARIASQL_INSTANCE
    if _ARIASQL_INSTANCE is None:
        _ARIASQL_INSTANCE = _build_ariasql_instance()

    if _ARIASQL_INSTANCE is None:
        return await mock_sql_agent(task, db_path)

    return await _ARIASQL_INSTANCE.generate_sql(task, db_path=db_path)


# Singleton — initialised lazily on first ariasql call
_ARIASQL_INSTANCE: Any = None


def _build_ariasql_instance() -> Any:
    """Build AriaSql with available LLM provider. Returns None if no provider."""
    try:
        import os
        sys.path.insert(0, str(ROOT / "src"))
        from harness.agents.nexus_sql import NexusSql
        import fakeredis.aioredis as fakeredis
        from harness.memory.context_engineering import SchemaStore
        from harness.improvement.rlvr.verifiers import SQLVerifier

        # Build schema store (in-memory fakeredis for benchmark)
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = SchemaStore.__new__(SchemaStore)
        store._redis_url = "redis://unused"
        store._ttl = 86400
        store._client = redis

        # Try to get an LLM provider
        llm = _try_build_llm()
        if llm is None:
            logger.info("AriaSql: no LLM provider available — using mock agent")
            return None

        verifier = SQLVerifier(llm=llm, schema_store=store)
        agent = NexusSql(
            llm_provider=llm,
            schema_store=store,
            verifier=verifier,
            max_retries=2,
            correction_threshold=0.60,
        )
        logger.info("AriaSql: agent ready (with verifier + schema store)")
        return agent
    except Exception as exc:
        logger.warning("AriaSql: failed to build agent: %s", exc)
        return None


def _try_build_llm() -> Any:
    """Build an LLM router using whatever key is already in .env / harness config."""
    try:
        from harness.core.config import get_config
        from harness.llm.factory import build_router
        cfg = get_config()
        return build_router(cfg)   # LLMRouter implements .complete() — works as LLM
    except Exception as exc:
        logger.debug("Could not build LLM from harness config: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENTS: dict[str, Callable] = {
    "mock":    mock_sql_agent,
    "ariasql": ariasql_agent,
}


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def evaluate_case(
    case_id: str,
    question: str,
    db_id: str,
    db_path: str,
    hardness: str,
    gold_sql: str,
    agent_fn: Callable,
    schema_store: Any | None,
    llm: Any | None,
) -> CaseResult:
    from harness.eval.sandbox import SQLSandbox
    from harness.improvement.rlvr.verifiers import SQLVerifier

    t0 = time.monotonic()

    # 1. Run agent to get generated SQL
    try:
        generated_sql = await agent_fn(question, db_path)
    except Exception as exc:
        generated_sql = f"-- agent error: {exc}"

    # 2. Build sandbox and verifier for this db
    sandbox = SQLSandbox(db_path=db_path)
    verifier = SQLVerifier(sandbox=sandbox, llm=llm, schema_store=schema_store)

    # 3. Verify
    try:
        vr = await verifier.verify(
            task=question,
            action=generated_sql,
            result=None,
            gold=gold_sql,
            db_id=db_id,
        )
    except Exception as exc:
        logger.warning("Verifier failed for %s: %s", case_id, exc)
        from harness.improvement.rlvr.verifiers import VerificationResult, VerificationStep
        vr = VerificationResult(
            overall_reward=0.0, verdict="incorrect",
            steps=[VerificationStep("error", False, 0.0, str(exc))],
            feedback_for_agent=str(exc),
        )

    elapsed_ms = (time.monotonic() - t0) * 1000

    return CaseResult(
        case_id=case_id,
        question=question,
        db_id=db_id,
        hardness=hardness,
        gold_sql=gold_sql,
        generated_sql=generated_sql,
        overall_reward=vr.overall_reward,
        verdict=vr.verdict,
        step_scores={s.name: s.score for s in vr.steps},
        feedback=vr.feedback_for_agent,
        execution_time_ms=elapsed_ms,
        cached=vr.cached,
    )


async def run_benchmark(
    bird_dir: str,
    split: str,
    n_samples: int,
    agent_fn: Callable,
    llm: Any | None,
    concurrency: int,
) -> BIRDReport:
    from harness.eval.benchmark_loaders import load_bird

    run_id = uuid.uuid4().hex[:12]
    t_start = time.monotonic()

    logger.info("Loading BIRD %s split (n_samples=%d)…", split, n_samples)
    try:
        dataset = load_bird(bird_dir, split=split, n_samples=n_samples)
    except FileNotFoundError as exc:
        logger.error("BIRD dataset not found: %s", exc)
        raise

    logger.info("Loaded %d cases", len(dataset.cases))

    # Build schema store from sqlite files (cache schema per db_id)
    schema_store = _build_schema_store(dataset)

    semaphore = asyncio.Semaphore(concurrency)

    async def _eval_with_sem(case):
        async with semaphore:
            db_path = case.db_path or ""
            return await evaluate_case(
                case_id=case.case_id,
                question=case.task,
                db_id=case.metadata.get("db_id", ""),
                db_path=db_path,
                hardness=case.hardness or "unknown",
                gold_sql=case.expected_output or "",
                agent_fn=agent_fn,
                schema_store=schema_store,
                llm=llm,
            )

    results = await asyncio.gather(
        *[_eval_with_sem(c) for c in dataset.cases],
        return_exceptions=True,
    )

    # Aggregate
    case_results: list[CaseResult] = []
    for c, r in zip(dataset.cases, results):
        if isinstance(r, Exception):
            logger.warning("Case %s raised: %s", c.case_id, r)
            case_results.append(CaseResult(
                case_id=c.case_id, question=c.task,
                db_id=c.metadata.get("db_id", ""), hardness=c.hardness or "unknown",
                gold_sql=c.expected_output or "", generated_sql="",
                overall_reward=0.0, verdict="incorrect",
                step_scores={}, feedback=str(r),
                execution_time_ms=0.0, cached=False,
            ))
        else:
            case_results.append(r)

    elapsed = time.monotonic() - t_start
    report = _build_report(case_results, run_id, elapsed)

    _print_report(report)
    return report


def _build_schema_store(dataset: Any) -> Any | None:
    """Pre-load SQLite schemas into an in-memory SchemaStore."""
    try:
        import asyncio
        import fakeredis.aioredis as fakeredis
        from harness.memory.context_engineering import SchemaStore

        redis = fakeredis.FakeRedis(decode_responses=True)
        store = SchemaStore.__new__(SchemaStore)
        store._redis_url = "redis://unused"
        store._ttl = 3600
        store._client = redis

        db_ids_loaded: set[str] = set()

        async def _load_all():
            for case in dataset.cases:
                db_id = case.metadata.get("db_id", "")
                db_path = case.db_path or ""
                if db_id and db_path and db_id not in db_ids_loaded:
                    try:
                        await store.store_from_sqlite(db_id, db_path)
                        db_ids_loaded.add(db_id)
                    except Exception as exc:
                        logger.debug("Schema load failed for %s: %s", db_id, exc)

        asyncio.get_event_loop().run_until_complete(_load_all())
        logger.info("Schema store: loaded %d databases", len(db_ids_loaded))
        return store
    except Exception as exc:
        logger.debug("Schema store unavailable: %s", exc)
        return None


def _build_report(
    results: list[CaseResult],
    run_id: str,
    elapsed: float,
) -> BIRDReport:
    n = len(results)
    if n == 0:
        return BIRDReport(0, 0, 0.0, 0.0, {}, {}, {}, 0.0, 0.0, [], run_id, elapsed)

    passed = [r for r in results if r.verdict == "correct"]
    pass_rate = len(passed) / n

    # Execution accuracy = fraction where result_check passed
    exec_acc_count = sum(
        1 for r in results if r.step_scores.get("result_check", 0.0) >= 0.99
    )
    exec_acc = exec_acc_count / n

    # By hardness
    by_hardness: dict[str, dict] = {}
    for r in results:
        h = r.hardness
        if h not in by_hardness:
            by_hardness[h] = {"count": 0, "passed": 0, "exec_acc": 0}
        by_hardness[h]["count"] += 1
        if r.verdict == "correct":
            by_hardness[h]["passed"] += 1
        if r.step_scores.get("result_check", 0.0) >= 0.99:
            by_hardness[h]["exec_acc"] += 1
    for h, v in by_hardness.items():
        c = max(v["count"], 1)
        by_hardness[h]["pass_rate"] = round(v["passed"] / c, 4)
        by_hardness[h]["exec_acc_rate"] = round(v["exec_acc"] / c, 4)

    # Step pass rates
    all_steps: dict[str, list[float]] = {}
    for r in results:
        for step, score in r.step_scores.items():
            all_steps.setdefault(step, []).append(score)
    step_pass_rates = {
        step: round(sum(1 for s in scores if s >= 0.8) / len(scores), 4)
        for step, scores in all_steps.items()
    }

    # Failure distribution (from feedback)
    failures: dict[str, int] = {}
    for r in results:
        if r.verdict != "correct" and r.feedback:
            key = r.feedback.split(":")[0].strip()[:40]
            failures[key] = failures.get(key, 0) + 1

    avg_reward = sum(r.overall_reward for r in results) / n
    avg_exec_ms = sum(r.execution_time_ms for r in results) / n

    return BIRDReport(
        n_cases=n,
        n_evaluated=n,
        overall_pass_rate=round(pass_rate, 4),
        execution_accuracy=round(exec_acc, 4),
        by_hardness=by_hardness,
        step_pass_rates=step_pass_rates,
        failure_distribution=dict(sorted(failures.items(), key=lambda x: -x[1])[:10]),
        avg_reward=round(avg_reward, 4),
        avg_exec_time_ms=round(avg_exec_ms, 1),
        cases=results,
        run_id=run_id,
        elapsed_seconds=round(elapsed, 2),
    )


def _print_report(report: BIRDReport) -> None:
    print("\n" + "=" * 60)
    print(f"BIRD Benchmark Results  (run_id={report.run_id})")
    print("=" * 60)
    print(f"Cases evaluated : {report.n_evaluated}")
    print(f"Overall pass rate: {report.overall_pass_rate:.1%}")
    print(f"Execution accuracy: {report.execution_accuracy:.1%}")
    print(f"Avg reward      : {report.avg_reward:.3f}")
    print(f"Elapsed         : {report.elapsed_seconds:.1f}s")
    print()
    print("By hardness:")
    for h in ("easy", "medium", "hard", "extra-hard", "unknown"):
        stats = report.by_hardness.get(h)
        if stats:
            print(f"  {h:12s}  pass={stats['pass_rate']:.1%}  "
                  f"exec_acc={stats['exec_acc_rate']:.1%}  n={stats['count']}")
    print()
    print("Step pass rates:")
    for step, rate in sorted(report.step_pass_rates.items()):
        bar = "█" * int(rate * 20)
        print(f"  {step:20s}  {rate:.1%}  {bar}")
    if report.failure_distribution:
        print()
        print("Top failure categories:")
        for cat, count in list(report.failure_distribution.items())[:5]:
            print(f"  [{count:3d}] {cat}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(report: BIRDReport, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise (dataclasses aren't JSON-serialisable by default)
    data = {
        "run_id": report.run_id,
        "n_cases": report.n_cases,
        "n_evaluated": report.n_evaluated,
        "overall_pass_rate": report.overall_pass_rate,
        "execution_accuracy": report.execution_accuracy,
        "avg_reward": report.avg_reward,
        "avg_exec_time_ms": report.avg_exec_time_ms,
        "elapsed_seconds": report.elapsed_seconds,
        "by_hardness": report.by_hardness,
        "step_pass_rates": report.step_pass_rates,
        "failure_distribution": report.failure_distribution,
        "cases": [
            {
                "case_id": c.case_id,
                "question": c.question,
                "db_id": c.db_id,
                "hardness": c.hardness,
                "gold_sql": c.gold_sql,
                "generated_sql": c.generated_sql,
                "overall_reward": c.overall_reward,
                "verdict": c.verdict,
                "step_scores": c.step_scores,
                "feedback": c.feedback,
                "execution_time_ms": c.execution_time_ms,
                "cached": c.cached,
            }
            for c in report.cases
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Tokens per query (input / output) based on typical BIRD schema + question
# Schema block ~300 tokens, question ~30 tokens, SQL output ~30 tokens
# Verifier adds ~150 input + 100 output
_TOKENS_IN_PER_QUERY = 450    # input tokens per query (no retry)
_TOKENS_OUT_PER_QUERY = 130   # output tokens per query (no retry)
_RETRY_MULTIPLIER = 1.8       # avg multiplier when self-correction triggers ~40% of time

# Pricing per 1M tokens (as of May 2026)
_PRICING: dict[str, dict] = {
    "gpt-5.5":            {"in": 2.50,  "out": 10.00, "label": "GPT-5.5 (your key) ←"},
    "claude-haiku-4-5":   {"in": 0.80,  "out": 4.00,  "label": "Claude Haiku 4.5"},
    "claude-sonnet-4-6":  {"in": 3.00,  "out": 15.00, "label": "Claude Sonnet 4.6"},
    "claude-opus-4-7":    {"in": 15.00, "out": 75.00, "label": "Claude Opus 4.7"},
    "gpt-4o-mini":        {"in": 0.15,  "out": 0.60,  "label": "GPT-4o mini"},
    "gpt-4o":             {"in": 2.50,  "out": 10.00, "label": "GPT-4o"},
}


def print_cost_estimate(n_samples: int) -> None:
    """Print token and cost estimates before running."""
    tokens_in  = _TOKENS_IN_PER_QUERY  * _RETRY_MULTIPLIER * n_samples
    tokens_out = _TOKENS_OUT_PER_QUERY * _RETRY_MULTIPLIER * n_samples

    print("\n" + "=" * 62)
    print(f"Cost estimate for {n_samples} BIRD questions (with self-correction)")
    print("=" * 62)
    print(f"  Input tokens  : ~{tokens_in:,.0f}  ({_TOKENS_IN_PER_QUERY:.0f} × {_RETRY_MULTIPLIER}× × {n_samples})")
    print(f"  Output tokens : ~{tokens_out:,.0f}  ({_TOKENS_OUT_PER_QUERY:.0f} × {_RETRY_MULTIPLIER}× × {n_samples})")
    print()
    print(f"  {'Model':<35}  {'Cost':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    for model, p in _PRICING.items():
        cost = (tokens_in / 1_000_000 * p["in"]) + (tokens_out / 1_000_000 * p["out"])
        marker = " ←" if model == "claude-haiku-4-5" else ""
        print(f"  {p['label']:<35}  ${cost:>6.3f}{marker}")
    print()
    print("  Note: 4 of 5 verifier steps are FREE (no LLM).")
    print("        Only quality_check uses the LLM.")
    print("        Verifier cache cuts cost ~40% on repeated runs.")
    print("=" * 62 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIRD benchmark for HarnessAgent SQL agents")
    p.add_argument("--bird-dir", default="", help="Path to BIRD dataset root directory")
    p.add_argument("--split", default="dev", choices=["train", "dev"],
                   help="Dataset split (default: dev)")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of questions to evaluate (default: 50)")
    p.add_argument("--agent", default="mock", choices=list(AGENTS.keys()),
                   help="SQL agent to benchmark (default: mock)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Concurrent evaluations (default: 4)")
    p.add_argument("--output", default="benchmarks/results/bird_results.json",
                   help="Output file path")
    p.add_argument("--estimate", action="store_true",
                   help="Print cost estimate and exit without running")
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> BIRDReport:
    print_cost_estimate(args.n_samples)

    if args.estimate:
        import sys as _sys; _sys.exit(0)

    agent_fn = AGENTS[args.agent]

    if not args.bird_dir:
        logger.warning(
            "No --bird-dir provided. Running self-test with synthetic data."
        )
        return await _self_test(agent_fn, args.output)

    report = await run_benchmark(
        bird_dir=args.bird_dir,
        split=args.split,
        n_samples=args.n_samples,
        agent_fn=agent_fn,
        llm=None,  # no LLM quality check by default
        concurrency=args.concurrency,
    )
    save_results(report, args.output)
    return report


async def _self_test(agent_fn: Callable, output: str) -> BIRDReport:
    """Run against a tiny in-memory SQLite database — no BIRD download needed."""
    import sqlite3
    import tempfile

    logger.info("Self-test mode: generating synthetic BIRD-like questions")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/employees.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                salary REAL,
                active INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT,
                budget REAL
            )
        """)
        conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", [
            (1, "Alice", "Engineering", 90000, 1),
            (2, "Bob",   "Sales",       70000, 1),
            (3, "Carol", "Engineering", 85000, 0),
            (4, "Dave",  "HR",          60000, 1),
            (5, "Eve",   "Sales",       75000, 1),
        ])
        conn.executemany("INSERT INTO departments VALUES (?,?,?)", [
            (1, "Engineering", 500000),
            (2, "Sales",       300000),
            (3, "HR",          200000),
        ])
        conn.commit()
        conn.close()

        from harness.eval.sandbox import SQLSandbox
        from harness.improvement.rlvr.verifiers import SQLVerifier

        questions = [
            ("How many active employees are there?",
             "SELECT COUNT(*) FROM employees WHERE active = 1", "easy"),
            ("What is the average salary across all employees?",
             "SELECT AVG(salary) FROM employees", "easy"),
            ("List all employees in the Engineering department.",
             "SELECT name FROM employees WHERE department = 'Engineering'", "easy"),
            ("Which department has the highest budget?",
             "SELECT name FROM departments ORDER BY budget DESC LIMIT 1", "medium"),
            ("How many employees are in each department?",
             "SELECT department, COUNT(*) FROM employees GROUP BY department", "medium"),
            ("What is the total salary cost for active employees by department?",
             "SELECT department, SUM(salary) FROM employees WHERE active=1 GROUP BY department", "hard"),
            ("Find employees whose salary is above the average salary of their department.",
             "SELECT e.name FROM employees e WHERE e.salary > "
             "(SELECT AVG(e2.salary) FROM employees e2 WHERE e2.department = e.department)", "extra-hard"),
        ]

        results = []
        for i, (question, gold_sql, hardness) in enumerate(questions):
            r = await evaluate_case(
                case_id=f"selftest_{i:02d}",
                question=question,
                db_id="employees",
                db_path=db_path,
                hardness=hardness,
                gold_sql=gold_sql,
                agent_fn=agent_fn,
                schema_store=None,
                llm=None,
            )
            results.append(r)
            logger.info(
                "  [%d/%d] %-60s  reward=%.2f  verdict=%s",
                i + 1, len(questions), question[:60],
                r.overall_reward, r.verdict,
            )

    report = _build_report(results, "selftest", 0.0)
    _print_report(report)
    save_results(report, output)
    return report


def main() -> None:
    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
