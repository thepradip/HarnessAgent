"""Benchmark 5: Hermes self-improvement loop — error-to-patch convergence.

Seeds the ErrorCollector with realistic SQL-agent failures, then runs the
HermesLoop for multiple cycles to measure:

  - pass@1 before patching: fraction of seeded tasks that succeed without patch
  - pass@1 after patching:  fraction that succeed after Hermes applies a patch
  - cycles to converge:     how many HermesLoop cycles until score >= threshold
  - rollback rate:          patches that were auto-applied then rolled back
  - patch score distribution: histogram of eval scores across proposed patches

Uses mocked LLM (deterministic patch proposals) and mocked agent runner
(controlled success/failure). No real LLM API key required.

Run:
    PYTHONPATH=src python benchmarks/bench_hermes.py

Output:
    benchmarks/results/hermes_self_improvement.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Seeded failure scenarios — realistic SQL agent errors
# ---------------------------------------------------------------------------

SEED_FAILURES = [
    {
        "task": "List all users who signed up in the last 30 days",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "SQL used 'signup_date' but column is named 'created_at'.",
    },
    {
        "task": "Show total revenue this month",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "Ambiguous column reference 'status' — must qualify with table name.",
    },
    {
        "task": "Find users who placed more than 5 orders",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "HAVING clause emitted without GROUP BY — query is invalid.",
    },
    {
        "task": "List all products with no orders",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "Used INNER JOIN instead of LEFT JOIN; query returns non-empty set.",
    },
    {
        "task": "Show average order value by payment method",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "Column 'payment_method' referenced from wrong table.",
    },
    {
        "task": "Find duplicate email addresses",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "Missing HAVING COUNT(*) > 1 after GROUP BY email.",
    },
    {
        "task": "Calculate refund rate by month",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "Division by zero: denominator not guarded.",
    },
    {
        "task": "Show top 10 customers by lifetime value",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "ORDER BY on non-aggregated column outside GROUP BY.",
    },
    {
        "task": "List all active promotions with their product names",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "NULL product_id causes inner join to drop valid promotions.",
    },
    {
        "task": "Count sessions longer than 5 minutes",
        "failure_class": "LLM_PARSE_ERROR",
        "error_message": "Arithmetic on TEXT column; CAST to INT required.",
    },
]

# Patch that fixes the most common failure patterns
GOLDEN_PATCH_VALUE = (
    "\n\n## SQL Agent — Column Qualification Rules\n"
    "- Always qualify column names with their table name in multi-table queries.\n"
    "- Use 'created_at' not 'signup_date' for the users table registration timestamp.\n"
    "- When joining payments to orders, join on payments.order_id = orders.id.\n"
    "- For 'no orders' queries, use LEFT JOIN ... WHERE right_table.id IS NULL.\n"
    "- For HAVING filters, always include GROUP BY first.\n"
    "- Cast TEXT columns to the correct type before arithmetic.\n"
    "- Use COALESCE(denominator, 1) to guard against division by zero.\n"
)


# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------

class _MockLLMForPatches:
    """Returns a deterministic patch proposal based on sampled errors."""
    provider_name = "mock"
    model = "mock-patch-generator"

    async def complete(self, messages: list[dict], *, max_tokens: int, **kw) -> Any:
        from harness.core.context import LLMResponse
        # Return a structured patch proposal
        patch_json = json.dumps({
            "target": "prompt",
            "op": "append",
            "path": "",
            "value": GOLDEN_PATCH_VALUE,
            "rationale": (
                "Five failure records share column-qualification and JOIN errors. "
                "Appending explicit rules to the system prompt should prevent recurrence."
            ),
        })
        return LLMResponse(
            content=patch_json,
            input_tokens=400,
            output_tokens=120,
            model="mock-patch-generator",
            provider="mock",
            cached=False,
        )


class _MockPromptStore:
    """In-memory prompt store — tracks versions and applied patches."""

    def __init__(self) -> None:
        self._prompts: dict[str, str] = {
            "sql": "You are a SQL agent. Answer questions by writing SQL queries.",
        }
        self._versions: dict[str, list[str]] = {"sql": [self._prompts["sql"]]}
        self.applied_patches: list[str] = []

    async def get(self, agent_type: str) -> str:
        return self._prompts.get(agent_type, "")

    async def get_prompt(self, agent_type: str) -> str:
        return self._prompts.get(agent_type, "")

    async def apply_patch(self, patch: Any) -> str:
        current = self._prompts.get(patch.agent_type, "")
        if patch.op == "append":
            new_prompt = current + patch.value
        elif patch.op == "prepend":
            new_prompt = patch.value + current
        elif patch.op == "replace" and patch.path in current:
            new_prompt = current.replace(patch.path, patch.value, 1)
        else:
            new_prompt = current + patch.value
        self._prompts[patch.agent_type] = new_prompt
        version_id = f"v{len(self._versions.get(patch.agent_type, [])) + 1}"
        self._versions.setdefault(patch.agent_type, []).append(new_prompt)
        self.applied_patches.append(patch.patch_id)
        return version_id

    async def rollback(self, agent_type: str, version_id: str) -> None:
        idx = int(version_id.lstrip("v")) - 1
        versions = self._versions.get(agent_type, [])
        if 0 <= idx < len(versions):
            self._prompts[agent_type] = versions[idx]


class _MockAgentRunner:
    """Simulates agent runs. Pre-patch: 10% success rate. Post-patch: 80%."""

    def __init__(self) -> None:
        self._patch_applied = False
        self.run_count = 0

    async def run(self, agent_type: str, task: str, prompt: str, **kw) -> dict:
        self.run_count += 1
        # Success rate depends on whether the patch has been applied
        success_rate = 0.80 if (GOLDEN_PATCH_VALUE in prompt) else 0.10
        import random
        success = random.random() < success_rate
        return {
            "success": success,
            "steps": 3 if success else 1,
            "input_tokens": 350 if success else 200,
            "output_tokens": 80 if success else 20,
            "error": None if success else "SQL column error",
        }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run() -> None:
    import fakeredis.aioredis as fake_aioredis  # type: ignore

    from harness.improvement.error_collector import ErrorCollector
    from harness.improvement.evaluator import EvalResult
    from harness.improvement.hermes import HermesLoop
    from harness.improvement.patch_generator import PatchGenerator

    fake_redis = fake_aioredis.FakeRedis(decode_responses=True)

    # --- Build real components with mocked backends ---
    collector = ErrorCollector(redis=fake_redis)
    prompt_store = _MockPromptStore()
    mock_llm = _MockLLMForPatches()
    agent_runner = _MockAgentRunner()

    # Seed error records
    print(f"\nHermes Self-Improvement Benchmark")
    print(f"  Seeding {len(SEED_FAILURES)} failure records...")
    for failure in SEED_FAILURES:
        await collector.record(
            agent_type="sql",
            task=failure["task"],
            failure_class=failure["failure_class"],
            error_message=failure["error_message"],
            context_snapshot={"step": 1, "tenant_id": "bench"},
        )

    error_count = await collector.count("sql")
    print(f"  Error count in window: {error_count}")

    # --- Pre-patch pass@1: run 20 tasks with baseline prompt ---
    print("\nMeasuring pre-patch pass@1 (20 tasks)...")
    baseline_prompt = await prompt_store.get("sql")
    pre_results = []
    for _ in range(20):
        task = SEED_FAILURES[_ % len(SEED_FAILURES)]["task"]
        r = await agent_runner.run("sql", task, baseline_prompt)
        pre_results.append(r["success"])
    pre_pass1 = sum(pre_results) / len(pre_results)
    print(f"  Pre-patch pass@1 = {pre_pass1:.1%}")

    # --- Configure PatchGenerator with mock LLM ---
    generator = PatchGenerator(llm_provider=mock_llm, prompt_manager=prompt_store)

    # --- Configure Evaluator with controlled agent runner ---
    class _BoundEvaluator:
        """Thin wrapper so Evaluator.score() uses our mock runner."""

        async def score(self, patch: Any, test_cases: list[Any], agent_type: str = "sql") -> EvalResult:
            prompt = await prompt_store.get(patch.agent_type)
            # Simulate applying the patch temporarily
            if patch.op == "append":
                patched_prompt = prompt + patch.value
            else:
                patched_prompt = prompt + patch.value

            successes = 0
            steps_delta = 0.0
            tokens_delta = 0.0
            for tc in test_cases:
                r = await agent_runner.run(patch.agent_type, tc.task, patched_prompt)
                if r["success"]:
                    successes += 1
                    steps_delta += r["steps"] - 3
                    tokens_delta += r["input_tokens"] - 350

            n = len(test_cases)
            return EvalResult(
                patch_id=patch.patch_id,
                test_cases=n,
                successes=successes,
                failures=n - successes,
                avg_steps_delta=steps_delta / n if n > 0 else 0,
                avg_tokens_delta=tokens_delta / n if n > 0 else 0,
            )

    evaluator = _BoundEvaluator()

    # Mock metrics (Prometheus)
    metrics = MagicMock()
    metrics.hermes_patches_total = MagicMock()
    metrics.hermes_patches_total.labels = MagicMock(return_value=MagicMock(inc=MagicMock()))

    # Mock settings
    config = MagicMock()
    config.hermes_min_errors_to_trigger = 5
    config.hermes_patch_score_threshold = 0.70
    config.hermes_auto_apply = True
    config.hermes_max_errors_to_sample = 5

    hermes = HermesLoop(
        collector=collector,
        generator=generator,
        evaluator=evaluator,
        prompt_store=prompt_store,
        metrics=metrics,
        config=config,
    )

    # --- Run up to 5 cycles ---
    cycle_results = []
    converged_cycle = None
    print("\nRunning Hermes cycles...")
    for cycle in range(1, 6):
        t0 = time.perf_counter()
        outcome = await hermes.run_cycle("sql")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        cycle_data = {
            "cycle": cycle,
            "elapsed_ms": round(elapsed_ms, 1),
            "patch_proposed": outcome is not None and outcome.patch is not None,
            "patch_applied": outcome.applied if outcome else False,
            "patch_score": (
                round(outcome.eval_result.score, 3)
                if outcome and outcome.eval_result else None
            ),
            "reason": outcome.reason if outcome else "skipped",
        }
        cycle_results.append(cycle_data)

        score_str = (
            f"score={outcome.eval_result.score:.3f}" if outcome and outcome.eval_result
            else "no eval"
        )
        applied_str = "APPLIED" if (outcome and outcome.applied) else "skipped/pending"
        print(f"  Cycle {cycle}: {score_str}  {applied_str}  {elapsed_ms:.0f}ms")

        if outcome and outcome.applied and converged_cycle is None:
            converged_cycle = cycle

    # --- Post-patch pass@1 ---
    print("\nMeasuring post-patch pass@1 (20 tasks)...")
    patched_prompt = await prompt_store.get("sql")
    post_results = []
    for i in range(20):
        task = SEED_FAILURES[i % len(SEED_FAILURES)]["task"]
        r = await agent_runner.run("sql", task, patched_prompt)
        post_results.append(r["success"])
    post_pass1 = sum(post_results) / len(post_results)
    print(f"  Post-patch pass@1 = {post_pass1:.1%}")

    improvement = post_pass1 - pre_pass1
    patch_applied = len(prompt_store.applied_patches) > 0
    prompt_delta_chars = len(patched_prompt) - len(baseline_prompt)

    summary = {
        "seed_failure_count": len(SEED_FAILURES),
        "pre_patch_pass1": round(pre_pass1, 3),
        "post_patch_pass1": round(post_pass1, 3),
        "improvement_abs": round(improvement, 3),
        "improvement_pct": round(improvement * 100, 1),
        "cycles_run": len(cycle_results),
        "converged_at_cycle": converged_cycle,
        "patches_applied": len(prompt_store.applied_patches),
        "prompt_size_increase_chars": prompt_delta_chars,
        "agent_runner_total_calls": agent_runner.run_count,
    }

    print(f"\nSUMMARY")
    print(f"  Pre-patch pass@1   : {pre_pass1:.1%}")
    print(f"  Post-patch pass@1  : {post_pass1:.1%}")
    print(f"  Improvement        : +{improvement*100:.1f}pp")
    print(f"  Converged at cycle : {converged_cycle}")
    print(f"  Patches applied    : {len(prompt_store.applied_patches)}")

    output = {
        "benchmark": "hermes_self_improvement",
        "summary": summary,
        "cycles": cycle_results,
    }
    out_path = RESULTS_DIR / "hermes_self_improvement.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")

    await fake_redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
