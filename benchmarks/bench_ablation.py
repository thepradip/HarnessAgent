"""
Ablation study: NexusSql self-correction loop contribution.

Compares three conditions on the same 50 SQL tasks:
  A. No verifier, no correction  (baseline LLM only)
  B. Verifier, no correction     (verify but don't retry)
  C. Verifier + correction       (full NexusSql pipeline)

Uses the BenchmarkCache to reuse pre-built SQLite DBs and execution
results — no LLM calls for gold SQL execution.

Reports per-condition exec accuracy and the marginal gain from each component.

Run:
    PYTHONPATH=src python benchmarks/bench_ablation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench_ablation")

N_TASKS     = 50
CONCURRENCY = 4


# ---------------------------------------------------------------------------
# Helpers (shared with bench_bird_real)
# ---------------------------------------------------------------------------

def _exec_sql(db_path: str, sql: str) -> dict:
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(sql)
        cols = [d[0] for d in (cur.description or [])]
        rows = [list(r) for r in cur.fetchall()]
        conn.close()
        return {"columns": cols, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def _match(pred: dict, gold: dict) -> bool:
    if pred.get("error") or gold.get("error"):
        return False
    return ({tuple(str(v) for v in r) for r in pred.get("rows", [])} ==
            {tuple(str(v) for v in r) for r in gold.get("rows", [])})


def _build_db(sql_context: str, db_path: str) -> bool:
    stmts = [s.strip() for s in re.split(r";(?:\s*\n|\s+)", sql_context) if s.strip()]
    try:
        conn = sqlite3.connect(db_path)
        for stmt in stmts:
            if stmt.upper().lstrip().startswith(("CREATE", "INSERT")):
                try:
                    conn.execute(stmt)
                except Exception:
                    pass
        conn.commit(); conn.close(); return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

async def _generate_sql(
    task: str, db_path: str, llm: any, condition: str,
    shared_store: any = None,   # pre-warmed schema store — same for B and C
    shared_verifier: any = None,
) -> str:
    """Generate SQL under the given condition."""
    from harness.agents.nexus_sql import NexusSql

    if condition == "A":
        # No schema context, no verifier, no retry — pure LLM
        agent = NexusSql(llm_provider=llm, schema_store=None, verifier=None,
                         max_retries=0, correction_threshold=1.1)
    elif condition == "B":
        # Schema context + verifier, no retry loop
        agent = NexusSql(llm_provider=llm, schema_store=shared_store,
                         verifier=shared_verifier,
                         max_retries=0, correction_threshold=1.1)
    else:  # C — full NexusSql pipeline
        agent = NexusSql(llm_provider=llm, schema_store=shared_store,
                         verifier=shared_verifier,
                         max_retries=2, correction_threshold=0.60)

    try:
        return await agent.generate_sql(task, db_path=db_path)
    except Exception as exc:
        logger.debug("generate failed: %s", exc)
        return "SELECT 1"


async def run_condition(
    condition: str,
    tasks: list[dict],
    llm: any,
    bird_cache: any,
    shared_store: any = None,
    shared_verifier: any = None,
) -> dict:
    """Run one condition on all tasks. Returns {exec_acc, gen_error_rate, results}."""
    import tempfile
    sem = asyncio.Semaphore(CONCURRENCY)

    results = []

    async def _eval(i, row):
        async with sem:
            question = row["sql_prompt"]
            context  = row["sql_context"]
            gold_sql = row["sql"]

            # Reuse persistent cached DB
            if bird_cache.db_exists(context):
                db_path = str(bird_cache.db_path(context))
            else:
                import os
                tmp = tempfile.mktemp(suffix=".sqlite")
                if not _build_db(context, tmp):
                    return {"match": False, "error": True}
                bird_cache.copy_db(tmp, context)
                db_path = str(bird_cache.db_path(context))
                try: os.unlink(tmp)
                except Exception: pass

            generated = await _generate_sql(
                question, db_path, llm, condition,
                shared_store=shared_store,
                shared_verifier=shared_verifier,
            )

            pred = _exec_sql(db_path, generated)
            gold = bird_cache.get_exec(db_path, gold_sql)
            if gold is None:
                gold = _exec_sql(db_path, gold_sql)
                bird_cache.save_exec(db_path, gold_sql, gold)

            match = _match(pred, gold)
            error = pred.get("error") is not None

            logger.info("[%s-%02d] condition=%s  match=%-5s  %s",
                        condition, i, condition, match, question[:50])
            return {"match": match, "error": error}

    raw = await asyncio.gather(*[_eval(i, row) for i, row in enumerate(tasks)],
                               return_exceptions=True)
    for r in raw:
        if isinstance(r, dict):
            results.append(r)

    n = len(results)
    exec_acc = sum(1 for r in results if r["match"]) / n if n else 0
    gen_err  = sum(1 for r in results if r["error"])  / n if n else 0
    return {"condition": condition, "exec_acc": round(exec_acc, 4),
            "gen_error_rate": round(gen_err, 4), "n": n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run() -> None:
    from datasets import load_dataset
    from harness.core.config import get_config
    from harness.llm.factory import build_router
    sys.path.insert(0, str(ROOT))
    from benchmarks.bench_cache_store import BenchmarkCache
    import random

    cfg = get_config()
    llm = build_router(cfg)
    bird_cache = BenchmarkCache("bird_n50")

    logger.info("Loading tasks …")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    valid = [
        row for row in ds
        if "CREATE TABLE" in row["sql_context"].upper()
        and "INSERT INTO" in row["sql_context"].upper()
        and row["sql"].upper().strip().startswith("SELECT")
    ]
    random.seed(42)
    random.shuffle(valid)
    by_c: dict = {}
    for row in valid:
        by_c.setdefault(row["sql_complexity"], []).append(row)
    tasks = []
    per = max(1, N_TASKS // len(by_c))
    for rows in by_c.values():
        tasks.extend(rows[:per])
    tasks = tasks[:N_TASKS]
    logger.info("Tasks: %d", len(tasks))

    # Pre-warm shared schema store with all task DBs — same as main benchmark
    from harness.memory.context_engineering import SchemaStore
    from harness.improvement.rlvr.verifiers import SQLVerifier
    import fakeredis.aioredis as fakeredis
    import tempfile, os

    redis = fakeredis.FakeRedis(decode_responses=True)
    shared_store = SchemaStore.__new__(SchemaStore)
    shared_store._redis_url = "redis://unused"; shared_store._ttl = 86400
    shared_store._client = redis

    logger.info("Pre-warming schema store for %d tasks …", len(tasks))
    for row in tasks:
        ctx = row["sql_context"]
        if bird_cache.db_exists(ctx):
            db_path = str(bird_cache.db_path(ctx))
        else:
            tmp = tempfile.mktemp(suffix=".sqlite")
            if _build_db(ctx, tmp):
                bird_cache.copy_db(tmp, ctx)
                db_path = str(bird_cache.db_path(ctx))
                try: os.unlink(tmp)
                except Exception: pass
            else:
                continue
        db_id = Path(db_path).stem
        try:
            await shared_store.store_from_sqlite(db_id, db_path)
        except Exception:
            pass
    logger.info("Schema store warmed: %d DBs indexed", len(await redis.keys("harness:schema:*")))

    shared_verifier = SQLVerifier(llm=llm, schema_store=shared_store)

    logger.info("─── Condition A: LLM only (no schema, no verifier, no correction) ───")
    result_a = await run_condition("A", tasks, llm, bird_cache,
                                   shared_store=None, shared_verifier=None)

    logger.info("─── Condition B: Schema + verifier, no correction ───")
    result_b = await run_condition("B", tasks, llm, bird_cache,
                                   shared_store=shared_store,
                                   shared_verifier=shared_verifier)

    logger.info("─── Condition C: Full NexusSql (schema + verifier + correction) ───")
    result_c = await run_condition("C", tasks, llm, bird_cache,
                                   shared_store=shared_store,
                                   shared_verifier=shared_verifier)

    print(f"\n{'='*60}")
    print(f"Ablation Study — NexusSql self-correction (n={N_TASKS})")
    print(f"{'='*60}")
    for r in (result_a, result_b, result_c):
        label = {"A": "LLM only          ", "B": "Verifier, no retry", "C": "Full NexusSql     "}[r["condition"]]
        print(f"  {label}  exec_acc={r['exec_acc']:.1%}  gen_err={r['gen_error_rate']:.1%}")

    gain_verifier   = round(result_b["exec_acc"] - result_a["exec_acc"], 4)
    gain_correction = round(result_c["exec_acc"] - result_b["exec_acc"], 4)
    gain_total      = round(result_c["exec_acc"] - result_a["exec_acc"], 4)
    print(f"\n  Verifier contribution    : +{gain_verifier:.1%}")
    print(f"  Self-correction gain     : +{gain_correction:.1%}")
    print(f"  Total gain (A→C)         : +{gain_total:.1%}")
    print(f"{'='*60}")

    output = {
        "benchmark": "ablation_nexussql_v2",
        "n_tasks": N_TASKS,
        "conditions": [result_a, result_b, result_c],
        "gains": {
            "verifier_only":    gain_verifier,
            "self_correction":  gain_correction,
            "total_a_to_c":     gain_total,
        },
        "agent": "NexusSql",
        "model": "gpt-5.5 (Azure)",
        "dataset": "gretelai/synthetic_text_to_sql",
    }
    out = RESULTS_DIR / "ablation_nexussql.json"
    out.write_text(json.dumps(output, indent=2))
    logger.info("Results → %s", out)


if __name__ == "__main__":
    asyncio.run(run())
