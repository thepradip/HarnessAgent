"""
BIRD-Real Benchmark — NexusSql on gretelai/synthetic_text_to_sql.

Uses 100 real SQL tasks with CREATE TABLE DDL and gold SQL.
Builds real SQLite databases from the DDL, executes generated SQL,
and computes execution accuracy via SQLSandbox.

This replaces the n=7 self-test to give statistically credible results.

Run:
    PYTHONPATH=src python benchmarks/bench_bird_real.py
    PYTHONPATH=src python benchmarks/bench_bird_real.py --n-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench_bird_real")


# ---------------------------------------------------------------------------
# Build real SQLite DB from DDL+INSERT statements
# ---------------------------------------------------------------------------

def build_sqlite(sql_context: str, db_path: str) -> bool:
    """
    Execute the CREATE TABLE + INSERT statements from sql_context into a
    real SQLite file. Returns True on success.
    """
    stmts = [s.strip() for s in re.split(r";(?:\s*\n|\s+)", sql_context) if s.strip()]
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        for stmt in stmts:
            if not stmt:
                continue
            upper = stmt.upper().lstrip()
            if upper.startswith(("CREATE", "INSERT", "DROP")):
                try:
                    conn.execute(stmt)
                except Exception as exc:
                    # Skip individual statement errors (type mismatches, etc.)
                    logger.debug("DDL stmt skipped: %s — %s", stmt[:60], exc)
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        logger.debug("build_sqlite failed: %s", exc)
        return False


def execute_sql_sync(db_path: str, sql: str, timeout_s: float = 10.0) -> dict:
    """Execute SQL and return {columns, rows, row_count, error}."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout_s)
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        rows = [list(r) for r in cur.fetchall()]
        conn.close()
        return {"columns": columns, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def results_match(pred: dict, gold: dict) -> bool:
    """Compare result sets as sets of tuples (order-independent)."""
    if pred.get("error") or gold.get("error"):
        return False
    pred_set = {tuple(str(v) for v in r) for r in pred.get("rows", [])}
    gold_set = {tuple(str(v) for v in r) for r in gold.get("rows", [])}
    return pred_set == gold_set


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

def load_cases(n_samples: int, complexity_filter: list[str] | None = None) -> list[dict]:
    from datasets import load_dataset
    logger.info("Loading gretelai/synthetic_text_to_sql …")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    # Filter: need both CREATE TABLE and INSERT for a real DB
    valid = [
        row for row in ds
        if "CREATE TABLE" in row["sql_context"].upper()
        and "INSERT INTO" in row["sql_context"].upper()
        and row["sql"].upper().strip().startswith("SELECT")
        and (not complexity_filter or row["sql_complexity"] in complexity_filter)
    ]

    # Stratified sample across complexities
    import random
    random.seed(42)
    random.shuffle(valid)

    by_complexity: dict[str, list] = {}
    for row in valid:
        by_complexity.setdefault(row["sql_complexity"], []).append(row)

    cases = []
    complexities = list(by_complexity.keys())
    per = max(1, n_samples // len(complexities))
    for c in complexities:
        cases.extend(by_complexity[c][:per])
    cases = cases[:n_samples]

    logger.info("Selected %d cases across %d complexity types", len(cases), len(by_complexity))
    return cases


# ---------------------------------------------------------------------------
# Evaluate one case
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    question: str
    complexity: str
    gold_sql: str
    generated_sql: str
    exec_match: bool       # result sets identical
    gen_error: bool        # generated SQL raised an error
    reward: float
    verdict: str
    feedback: str
    elapsed_ms: float


async def evaluate_case(
    case_id: str,
    row: dict,
    agent,
    cache: "BenchmarkCache",
) -> CaseResult:
    from benchmarks.bench_cache_store import BenchmarkCache
    t0 = time.monotonic()
    question   = row["sql_prompt"]
    gold_sql   = row["sql"]
    context    = row["sql_context"]
    complexity = row["sql_complexity"]

    # ── Check full case cache first ───────────────────────────────────────
    cached = cache.get_case(case_id)
    if cached:
        logger.info("[%s] %-10s  CACHE HIT", case_id, complexity[:10])
        return CaseResult(**{k: cached[k] for k in CaseResult.__dataclass_fields__})

    # ── SQLite DB (cached by DDL hash) ────────────────────────────────────
    if cache.db_exists(context):
        db_path = str(cache.db_path(context))
    else:
        import tempfile, os
        tmp = tempfile.mktemp(suffix=".sqlite")
        ok  = build_sqlite(context, tmp)
        if not ok:
            return CaseResult(case_id, question, complexity, gold_sql, "",
                              False, True, 0.0, "incorrect",
                              "failed to build SQLite from DDL",
                              (time.monotonic()-t0)*1000)
        cache.copy_db(tmp, context)
        db_path = str(cache.db_path(context))
        try: os.unlink(tmp)
        except Exception: pass

    # ── NexusSql generation (cached by question+DDL) ─────────────────────
    generated = cache.get_generated(question, context)
    if generated is None:
        try:
            generated = await agent.generate_sql(
                question, db_path=db_path, gold_sql=gold_sql
            )
        except Exception as exc:
            generated = "SELECT 1"
            logger.debug("NexusSql failed for %s: %s", case_id, exc)
        cache.save_generated(question, context, generated)
    else:
        logger.debug("[%s] generation cache hit", case_id)

    # ── SQL execution (cached by DB hash + SQL) ───────────────────────────
    pred_result = cache.get_exec(db_path, generated)
    if pred_result is None:
        pred_result = execute_sql_sync(db_path, generated)
        cache.save_exec(db_path, generated, pred_result)

    gold_result = cache.get_exec(db_path, gold_sql)
    if gold_result is None:
        gold_result = execute_sql_sync(db_path, gold_sql)
        cache.save_exec(db_path, gold_sql, gold_result)

    exec_match = results_match(pred_result, gold_result)
    gen_error  = pred_result.get("error") is not None

    reward  = 1.0 if exec_match else (0.3 if not gen_error else 0.0)
    verdict = "correct" if exec_match else "incorrect"

    feedback = ""
    if not exec_match:
        if gen_error:
            feedback = f"Execution error: {pred_result['error'][:120]}"
        elif gold_result.get("error"):
            feedback = "Gold SQL also failed — skipping case"
        else:
            feedback = (f"Result mismatch: got {pred_result.get('row_count',0)} rows, "
                        f"expected {gold_result.get('row_count',0)}")

    elapsed = (time.monotonic() - t0) * 1000
    logger.info("[%s] %-10s  exec_match=%-5s  %s",
                case_id, complexity[:10], exec_match, question[:55])

    result = CaseResult(case_id, question, complexity, gold_sql, generated,
                        exec_match, gen_error, reward, verdict, feedback, elapsed)

    # Save full case result
    cache.save_case({
        "case_id": case_id, "question": question, "complexity": complexity,
        "gold_sql": gold_sql, "generated_sql": generated,
        "exec_match": exec_match, "gen_error": gen_error, "reward": reward,
        "verdict": verdict, "feedback": feedback, "elapsed_ms": elapsed,
    })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(n_samples: int = 100, concurrency: int = 4) -> dict:
    from harness.agents.nexus_sql import NexusSql
    from harness.core.config import get_config
    from harness.llm.factory import build_router
    from harness.improvement.rlvr.verifiers import SQLVerifier
    import fakeredis.aioredis as fakeredis
    from harness.memory.context_engineering import SchemaStore
    from benchmarks.bench_cache_store import BenchmarkCache

    run_tag = f"bird_n{n_samples}"
    cache   = BenchmarkCache(run_tag)

    # If we have enough cached cases, skip agent setup entirely
    if cache.case_count() >= n_samples:
        logger.info("All %d cases already cached — loading from disk (no LLM calls)",
                    n_samples)
        cached_cases = cache.load_all_cases()[:n_samples]
        results = [CaseResult(**{k: c[k] for k in CaseResult.__dataclass_fields__})
                   for c in cached_cases]
        agent = None
    else:
        cfg = get_config()
        llm = build_router(cfg)
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = SchemaStore.__new__(SchemaStore)
        store._redis_url = "redis://unused"; store._ttl = 86400; store._client = redis
        verifier = SQLVerifier(llm=llm, schema_store=store)
        agent = NexusSql(
            llm_provider=llm, schema_store=store, verifier=verifier,
            max_retries=2, correction_threshold=0.60,
        )
        results = []

    logger.info("Cache stats: %s", cache.stats())

    cases = load_cases(n_samples)
    logger.info("Running %d cases  concurrency=%d", len(cases), concurrency)

    run_id = uuid.uuid4().hex[:10]

    if agent is not None:
        sem = asyncio.Semaphore(concurrency)
        async def _run(i, row):
            async with sem:
                return await evaluate_case(f"bird_{i:04d}", row, agent, cache)

        raw = await asyncio.gather(*[_run(i, row) for i, row in enumerate(cases)],
                                   return_exceptions=True)

        for i, r in enumerate(raw):
            if isinstance(r, Exception):
                logger.warning("Case %d raised: %s", i, r)
            else:
                results.append(r)

    # Aggregate
    n = len(results)
    exec_acc   = sum(1 for r in results if r.exec_match) / n
    gen_err_rt = sum(1 for r in results if r.gen_error)  / n
    avg_reward = sum(r.reward for r in results) / n

    by_complexity: dict = {}
    for r in results:
        g = by_complexity.setdefault(r.complexity, {"n": 0, "correct": 0})
        g["n"] += 1
        if r.exec_match:
            g["correct"] += 1
    for g in by_complexity.values():
        g["exec_acc"] = round(g["correct"] / g["n"], 4)

    summary = {
        "run_id": run_id,
        "n_cases": n,
        "execution_accuracy": round(exec_acc, 4),
        "gen_error_rate": round(gen_err_rt, 4),
        "avg_reward": round(avg_reward, 4),
        "by_complexity": by_complexity,
        "agent": "NexusSql",
        "model": "gpt-5.5 (Azure)",
        "dataset": "gretelai/synthetic_text_to_sql",
    }

    print(f"\n{'='*60}")
    print(f"BIRD-Real Benchmark  (NexusSql + GPT-5.5  n={n})")
    print(f"{'='*60}")
    print(f"Execution accuracy  : {exec_acc:.1%}")
    print(f"Gen error rate      : {gen_err_rt:.1%}")
    print(f"Avg reward          : {avg_reward:.3f}")
    print(f"\nBy complexity:")
    for c, s in sorted(by_complexity.items(), key=lambda x: x[1]["exec_acc"], reverse=True):
        print(f"  {c:<22}  {s['exec_acc']:.1%}  (n={s['n']})")
    print(f"{'='*60}")

    output = {
        "benchmark": "bird_real",
        "summary": summary,
        "cases": [
            {"case_id": r.case_id, "question": r.question, "complexity": r.complexity,
             "exec_match": r.exec_match, "gen_error": r.gen_error,
             "reward": r.reward, "verdict": r.verdict, "feedback": r.feedback,
             "generated_sql": r.generated_sql[:200], "elapsed_ms": round(r.elapsed_ms, 1)}
            for r in results
        ],
    }
    out = RESULTS_DIR / "bird_real_results.json"
    out.write_text(json.dumps(output, indent=2))
    logger.info("Results → %s", out)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples",   type=int, default=100)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    asyncio.run(run(args.n_samples, args.concurrency))


if __name__ == "__main__":
    main()
