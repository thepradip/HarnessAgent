"""
Hermes-Real Benchmark — real SQL execution, real LLM, real patch generation.

Replaces the mock agent runner with:
  - Real SQLite databases (built from gretelai/synthetic_text_to_sql DDL)
  - Real NexusSql SQL generation (GPT-5.5 via Azure)
  - Real SQLSandbox execution for pass@1 measurement
  - Real HermesLoop patch generation via LLM

Methodology:
  1. Sample 50 SQL tasks stratified by complexity
  2. Measure pre-patch pass@1 using NexusSql with baseline prompt
  3. Record failures → seed ErrorCollector
  4. Run HermesLoop (real LLM) → generate + evaluate → apply patch
  5. Measure post-patch pass@1 on same task set
  6. Report improvement

Run:
    PYTHONPATH=src python benchmarks/bench_hermes_real.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench_hermes_real")

N_TASKS     = 50   # total tasks for pre/post measurement
N_EVAL      = 20   # held-out eval set for Hermes patch scoring
CONCURRENCY = 4


# ---------------------------------------------------------------------------
# SQLite helpers (shared with bench_bird_real)
# ---------------------------------------------------------------------------

def build_sqlite(sql_context: str, db_path: str) -> bool:
    stmts = [s.strip() for s in re.split(r";(?:\s*\n|\s+)", sql_context) if s.strip()]
    try:
        conn = sqlite3.connect(db_path)
        for stmt in stmts:
            if stmt.upper().lstrip().startswith(("CREATE", "INSERT")):
                try:
                    conn.execute(stmt)
                except Exception:
                    pass
        conn.commit(); conn.close()
        return True
    except Exception:
        return False


def exec_sql(db_path: str, sql: str) -> dict:
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(sql)
        cols  = [d[0] for d in (cur.description or [])]
        rows  = [list(r) for r in cur.fetchall()]
        conn.close()
        return {"columns": cols, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def results_match(pred: dict, gold: dict) -> bool:
    if pred.get("error") or gold.get("error"):
        return False
    ps = {tuple(str(v) for v in r) for r in pred.get("rows", [])}
    gs = {tuple(str(v) for v in r) for r in gold.get("rows", [])}
    return ps == gs


# ---------------------------------------------------------------------------
# Load tasks
# ---------------------------------------------------------------------------

def load_tasks(n: int, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    import random
    logger.info("Loading gretelai/synthetic_text_to_sql …")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    valid = [
        row for row in ds
        if "CREATE TABLE" in row["sql_context"].upper()
        and "INSERT INTO" in row["sql_context"].upper()
        and row["sql"].upper().strip().startswith("SELECT")
    ]
    random.seed(seed)
    random.shuffle(valid)
    # Stratify
    by_c: dict[str, list] = {}
    for row in valid:
        by_c.setdefault(row["sql_complexity"], []).append(row)
    tasks = []
    per = max(1, n // len(by_c))
    for rows in by_c.values():
        tasks.extend(rows[:per])
    return tasks[:n]


# ---------------------------------------------------------------------------
# Evaluate one task with a given NexusSql agent
# ---------------------------------------------------------------------------

async def eval_task(row: dict, agent: Any, idx: int,
                    bird_cache: Any, hermes_cache: Any,
                    phase: str = "pre") -> dict:
    question = row["sql_prompt"]
    context  = row["sql_context"]
    gold_sql = row["sql"]
    case_id  = f"hermes_{phase}_{idx:04d}"

    # Check hermes case cache first (keyed by phase to separate pre/post)
    cached = hermes_cache.get_case(case_id)
    if cached:
        logger.debug("[%s] hermes case cache hit", case_id)
        return {"success": cached["success"], "error": cached["error"],
                "sql": cached["sql"]}

    # Build DB — reuse bird_cache persistent DB if available
    if bird_cache.db_exists(context):
        db_path = str(bird_cache.db_path(context))
    else:
        import tempfile, os
        tmp = tempfile.mktemp(suffix=".sqlite")
        if not build_sqlite(context, tmp):
            return {"success": False, "error": "ddl_failed", "sql": ""}
        bird_cache.copy_db(tmp, context)
        db_path = str(bird_cache.db_path(context))
        try: os.unlink(tmp)
        except Exception: pass

    # For pre-patch phase, check bird generation cache (same question)
    generated = None
    if phase == "pre":
        generated = bird_cache.get_generated(question, context)

    if generated is None:
        try:
            generated = await agent.generate_sql(question, db_path=db_path)
        except Exception as exc:
            generated = "SELECT 1"
            logger.debug("generate_sql failed: %s", exc)
        if phase == "pre":
            bird_cache.save_generated(question, context, generated)

    # Execute — check execution cache
    pred = bird_cache.get_exec(db_path, generated)
    if pred is None:
        pred = exec_sql(db_path, generated)
        bird_cache.save_exec(db_path, generated, pred)

    gold = bird_cache.get_exec(db_path, gold_sql)
    if gold is None:
        gold = exec_sql(db_path, gold_sql)
        bird_cache.save_exec(db_path, gold_sql, gold)

    match = results_match(pred, gold)
    error = pred.get("error") or ("mismatch" if not match else "")
    result = {"success": match, "error": error, "sql": generated}

    hermes_cache.save_case({**result, "case_id": case_id,
                            "question": question[:200], "phase": phase})
    return result


# ---------------------------------------------------------------------------
# Real PromptStore backed by a mutable dict
# ---------------------------------------------------------------------------

class RealPromptStore:
    BASELINE = (
        "You are NexusSql, an expert SQL agent.\n"
        "Write correct SELECT queries. Use exact column and table names from the schema.\n"
        "Add LIMIT 100 when no aggregation is used."
    )

    def __init__(self) -> None:
        self._prompts: dict[str, str] = {"sql": self.BASELINE}
        self._versions: list[str] = [self.BASELINE]
        self.applied: list[str] = []

    async def get(self, agent_type: str) -> str:
        return self._prompts.get(agent_type, "")

    async def get_prompt(self, agent_type: str) -> str:
        return self._prompts.get(agent_type, "")

    async def apply_patch(self, patch: Any) -> str:
        cur = self._prompts.get(patch.agent_type, "")
        if patch.op == "append":
            new = cur + "\n" + patch.value
        elif patch.op == "prepend":
            new = patch.value + "\n" + cur
        elif patch.op == "replace" and patch.path and patch.path in cur:
            new = cur.replace(patch.path, patch.value, 1)
        else:
            new = cur + "\n" + patch.value
        self._prompts[patch.agent_type] = new
        self._versions.append(new)
        self.applied.append(patch.patch_id)
        version_id = f"v{len(self._versions)}"
        logger.info("Patch applied → prompt grew by %d chars (version %s)",
                    len(new) - len(cur), version_id)
        return version_id

    async def rollback(self, agent_type: str, version_id: str) -> None:
        idx = int(version_id.lstrip("v")) - 1
        if 0 <= idx < len(self._versions):
            self._prompts[agent_type] = self._versions[idx]


# ---------------------------------------------------------------------------
# NexusSql factory with overridable prompt
# ---------------------------------------------------------------------------

def make_agent(llm: Any, store: Any, verifier: Any, prompt_store: RealPromptStore) -> Any:
    """Build a NexusSql agent whose system prompt is read from prompt_store each call."""
    from harness.agents.nexus_sql import NexusSql

    class PatchableNexusSql(NexusSql):
        async def _call_llm(self, prompt_content: str) -> str:
            system = await prompt_store.get("sql")
            try:
                response = await self._llm.complete(
                    messages=[{"role": "user", "content": prompt_content}],
                    max_tokens=512,
                    system=system,
                    temperature=0.0,
                    skip_cache=False,
                )
                from harness.agents.nexus_sql import _extract_sql
                return _extract_sql(response.content)
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                return "SELECT 1"

    return PatchableNexusSql(
        llm_provider=llm, schema_store=store,
        verifier=verifier, max_retries=1, correction_threshold=0.65,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run() -> None:
    import fakeredis.aioredis as fakeredis
    from unittest.mock import MagicMock

    from harness.core.config import get_config
    from harness.llm.factory import build_router
    from harness.improvement.error_collector import ErrorCollector
    from harness.improvement.evaluator import EvalResult, Evaluator
    from harness.improvement.hermes import HermesLoop
    from harness.improvement.patch_generator import PatchGenerator
    from harness.improvement.rlvr.verifiers import SQLVerifier
    from harness.memory.context_engineering import SchemaStore
    from benchmarks.bench_cache_store import BenchmarkCache

    # Reuse cached BIRD generations/executions if available
    bird_cache   = BenchmarkCache(f"bird_n{N_TASKS + N_EVAL}")
    hermes_cache = BenchmarkCache("hermes_real")

    cfg = get_config()
    llm = build_router(cfg)

    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://unused"; store._ttl = 86400; store._client = fake_redis

    verifier    = SQLVerifier(llm=llm, schema_store=store)
    prompt_store = RealPromptStore()
    collector    = ErrorCollector(redis=fake_redis)

    tasks = load_tasks(N_TASKS + N_EVAL)
    train_tasks = tasks[:N_TASKS]
    eval_tasks  = tasks[N_TASKS : N_TASKS + N_EVAL]
    logger.info("Train: %d  Eval: %d  BIRD-cache: %s",
                len(train_tasks), len(eval_tasks), bird_cache.stats())

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _eval(i, row, phase="pre"):
        async with sem:
            return await eval_task(row, make_agent(llm, store, verifier, prompt_store)
                                   if phase == "post" else agent_pre,
                                   i, bird_cache, hermes_cache, phase)

    # ------------------------------------------------------------------
    # Step 1: Pre-patch pass@1
    # ------------------------------------------------------------------
    logger.info("─── Pre-patch pass@1 (%d tasks) ───", len(train_tasks))
    agent_pre = make_agent(llm, store, verifier, prompt_store)

    pre_raw = await asyncio.gather(*[_eval(i, r, "pre") for i, r in enumerate(train_tasks)],
                                   return_exceptions=True)
    pre_results = [r for r in pre_raw if isinstance(r, dict)]
    pre_pass1   = sum(1 for r in pre_results if r["success"]) / len(pre_results)
    failures    = [r for r in pre_results if not r["success"]]
    logger.info("Pre-patch pass@1 = %.1f%%  (%d failures)", pre_pass1 * 100, len(failures))

        # ------------------------------------------------------------------
        # Step 2: Seed ErrorCollector with real failures
        # ------------------------------------------------------------------
        for i, (row, res) in enumerate(zip(train_tasks, pre_results)):
            if not res["success"]:
                await collector.record(
                    agent_type="sql",
                    task=row["sql_prompt"],
                    failure_class="EXEC_MISMATCH",
                    error_message=res.get("error", "result mismatch")[:200],
                    context_snapshot={
                        "complexity": row["sql_complexity"],
                        "generated_sql": res.get("sql", "")[:150],
                        "gold_sql": row["sql"][:150],
                    },
                )
        error_count = await collector.count("sql")
        logger.info("Seeded %d real failures into ErrorCollector", error_count)

        # ------------------------------------------------------------------
        # Step 3: Hermes — real LLM patch generation
        # ------------------------------------------------------------------
        generator = PatchGenerator(llm_provider=llm, prompt_manager=prompt_store)

        class _RealEvaluator:
            """Replay eval_tasks with patched prompt and measure pass@1."""
            async def score(self, patch: Any, test_cases: list[Any], agent_type: str = "sql") -> EvalResult:
                eval_raw = await asyncio.gather(
                    *[_eval(i, row, "eval") for i, row in enumerate(eval_tasks)],
                    return_exceptions=True,
                )
                eval_res = [r for r in eval_raw if isinstance(r, dict)]
                successes = sum(1 for r in eval_res if r["success"])
                n = len(eval_res)
                logger.info("Eval pass@1 = %.1f%% (%d/%d)", successes/n*100, successes, n)
                return EvalResult(
                    patch_id=patch.patch_id,
                    test_cases=n,
                    successes=successes,
                    failures=n - successes,
                    avg_steps_delta=0.0,
                    avg_tokens_delta=0.0,
                )

        metrics = MagicMock()
        metrics.hermes_patches_total = MagicMock()
        metrics.hermes_patches_total.labels = MagicMock(return_value=MagicMock(inc=MagicMock()))

        config = MagicMock()
        config.hermes_min_errors_to_trigger = 3
        config.hermes_patch_score_threshold  = 0.60
        config.hermes_auto_apply             = True
        config.hermes_max_errors_to_sample   = min(8, error_count)

        hermes = HermesLoop(
            collector=collector,
            generator=generator,
            evaluator=_RealEvaluator(),
            prompt_store=prompt_store,
            metrics=metrics,
            config=config,
        )

        logger.info("─── Running Hermes cycles (real LLM) ───")
        cycle_results = []
        converged_at  = None
        for cycle in range(1, 4):
            t0 = time.monotonic()
            outcome = await hermes.run_cycle("sql")
            elapsed = (time.monotonic() - t0) * 1000
            applied = outcome.applied if outcome else False
            score   = outcome.eval_result.score if (outcome and outcome.eval_result) else None
            reason  = outcome.reason if outcome else "skipped"
            cycle_results.append({
                "cycle": cycle, "elapsed_ms": round(elapsed, 1),
                "patch_applied": applied, "patch_score": score, "reason": reason,
            })
            logger.info("Cycle %d: score=%s  applied=%s  %.0fms",
                        cycle, f"{score:.3f}" if score else "—", applied, elapsed)
            if applied and converged_at is None:
                converged_at = cycle

        # ------------------------------------------------------------------
        # Step 4: Post-patch pass@1 on train set
        # ------------------------------------------------------------------
        logger.info("─── Post-patch pass@1 (%d tasks) ───", len(train_tasks))
        post_raw = await asyncio.gather(*[_eval(i, r, "post") for i, r in enumerate(train_tasks)],
                                        return_exceptions=True)
        post_results = [r for r in post_raw if isinstance(r, dict)]
        post_pass1   = sum(1 for r in post_results if r["success"]) / len(post_results)
        improvement  = post_pass1 - pre_pass1
        logger.info("Post-patch pass@1 = %.1f%%  (+%.1fpp)", post_pass1 * 100, improvement * 100)

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "n_train_tasks":   len(train_tasks),
        "n_eval_tasks":    len(eval_tasks),
        "pre_patch_pass1": round(pre_pass1,  4),
        "post_patch_pass1":round(post_pass1, 4),
        "improvement_abs": round(improvement, 4),
        "improvement_pct": round(improvement * 100, 1),
        "seed_failures":   error_count,
        "cycles_run":      len(cycle_results),
        "converged_at_cycle": converged_at,
        "patches_applied":    len(prompt_store.applied),
        "agent":   "NexusSql",
        "model":   "gpt-5.5 (Azure)",
        "dataset": "gretelai/synthetic_text_to_sql",
        "eval_method": "real SQLite execution + result-set comparison",
    }

    print(f"\n{'='*60}")
    print(f"Hermes-Real Benchmark  (NexusSql + GPT-5.5)")
    print(f"{'='*60}")
    print(f"Pre-patch pass@1   : {pre_pass1:.1%}")
    print(f"Post-patch pass@1  : {post_pass1:.1%}")
    print(f"Improvement        : +{improvement*100:.1f}pp")
    print(f"Converged at cycle : {converged_at}")
    print(f"Patches applied    : {len(prompt_store.applied)}")
    print(f"{'='*60}")

    output = {
        "benchmark": "hermes_real",
        "summary": summary,
        "cycles": cycle_results,
    }
    out = RESULTS_DIR / "hermes_real_results.json"
    out.write_text(json.dumps(output, indent=2))
    logger.info("Results → %s", out)

    await fake_redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
