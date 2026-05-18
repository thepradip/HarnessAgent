"""
General agent benchmark — works with any HarnessAgent agent type and dataset.

Supported agents:
  sql      AriaSql       (BIRD / Spider / any SQL dataset)
  code     AriaCode      (HumanEval / any code dataset)
  base     Direct LLM    (GSM8K / general Q&A)

Supported datasets:
  bird     BIRD-SQL benchmark
  spider   Spider SQL benchmark
  humaneval OpenAI HumanEval
  gsm8k    Grade school math
  jsonl    Any JSONL file (task + gold fields)
  csv      Any CSV file

Usage:
  # Self-test (no dataset download required)
  python benchmarks/bench_agent.py --agent sql
  python benchmarks/bench_agent.py --agent code
  python benchmarks/bench_agent.py --agent base

  # Real datasets
  python benchmarks/bench_agent.py --agent sql   --dataset bird     --data-dir /path/to/bird
  python benchmarks/bench_agent.py --agent code  --dataset humaneval --data-dir /path/to/humaneval.jsonl
  python benchmarks/bench_agent.py --agent base  --dataset gsm8k    --data-dir /path/to/gsm8k_test.jsonl
  python benchmarks/bench_agent.py --agent base  --dataset jsonl    --data-dir /path/to/custom.jsonl

  # Estimate cost before running
  python benchmarks/bench_agent.py --agent code --dataset humaneval --n-samples 50 --estimate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
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
logger = logging.getLogger("bench_agent")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    task: str
    agent_type: str
    hardness: str
    gold: str
    generated: str            # generated SQL, code, or text response
    overall_reward: float
    verdict: str
    step_scores: dict[str, float]
    feedback: str
    execution_time_ms: float


@dataclass
class BenchReport:
    agent_type: str
    dataset: str
    n_cases: int
    overall_pass_rate: float
    execution_accuracy: float   # fraction where result matched gold exactly
    avg_reward: float
    by_hardness: dict[str, dict]
    step_pass_rates: dict[str, float]
    failure_distribution: dict[str, int]
    cases: list[CaseResult]
    run_id: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

_LLM: Any = None   # lazy singleton


def _get_llm() -> Any | None:
    global _LLM
    if _LLM is None:
        try:
            from harness.core.config import get_config
            from harness.llm.factory import build_router
            _LLM = build_router(get_config())
        except Exception as exc:
            logger.debug("LLM unavailable: %s", exc)
    return _LLM


def _get_verifier(agent_type: str) -> Any | None:
    llm = _get_llm()
    if llm is None:
        return None
    try:
        from harness.improvement.rlvr.verifiers import get_verifier
        return get_verifier(agent_type, llm=llm)
    except Exception:
        return None


async def _sql_agent(task: str, metadata: dict) -> str:
    from harness.agents.nexus_sql import NexusSql
    llm = _get_llm()
    if llm is None:
        return _mock_sql(task)
    v = _get_verifier("sql")
    agent = NexusSql(llm_provider=llm, verifier=v)
    return await agent.generate_sql(
        task,
        db_path=metadata.get("db_path"),
        db_id=metadata.get("db_id"),
        gold_sql=metadata.get("gold"),
    )


async def _code_agent(task: str, metadata: dict) -> str:
    from harness.agents.aria_code import AriaCode
    llm = _get_llm()
    if llm is None:
        return _mock_code(task)
    v = _get_verifier("code")
    agent = AriaCode(llm_provider=llm, verifier=v)
    return await agent.generate_code(
        task,
        expected_output=metadata.get("expected_output"),
        gold_code=metadata.get("gold"),
    )


async def _base_agent(task: str, metadata: dict) -> str:
    llm = _get_llm()
    if llm is None:
        return _mock_base(task)
    try:
        resp = await llm.complete(
            messages=[{"role": "user", "content": task}],
            max_tokens=512,
            system="You are a helpful assistant. Answer concisely and correctly.",
            temperature=0.0,
            skip_cache=False,
        )
        return resp.content.strip()
    except Exception as exc:
        logger.warning("base agent failed: %s", exc)
        return ""


def _mock_sql(task: str) -> str:
    t = task.lower()
    if any(w in t for w in ("how many", "count", "number")):
        return "SELECT COUNT(*) FROM records"
    if any(w in t for w in ("average", "avg", "mean")):
        return "SELECT AVG(value) FROM records"
    return "SELECT * FROM records LIMIT 10"


def _mock_code(task: str) -> str:
    return "def solution():\n    pass  # TODO: implement"


def _mock_base(task: str) -> str:
    return "(no LLM configured)"


AGENTS: dict[str, Callable] = {
    "sql":  _sql_agent,
    "code": _code_agent,
    "base": _base_agent,
}


# ---------------------------------------------------------------------------
# Verifier dispatch
# ---------------------------------------------------------------------------

async def _verify_case(
    task: str,
    generated: str,
    gold: str | None,
    agent_type: str,
    metadata: dict,
) -> tuple[float, str, dict[str, float]]:
    """Run the domain verifier; returns (reward, feedback, step_scores)."""
    verifier = _get_verifier(agent_type)
    if verifier is None:
        # Rule-based fallback when no LLM is available
        from harness.eval.scorers import score_exact_match
        score = score_exact_match(generated, gold) if gold else 0.5
        return score, "", {"exact_match": score}

    try:
        kwargs: dict[str, Any] = dict(
            task=task, action=generated, result=None, gold=gold
        )
        if agent_type == "sql":
            kwargs["db_id"] = metadata.get("db_id", "")
            sandbox_path = metadata.get("db_path")
            if sandbox_path:
                from harness.eval.sandbox import SQLSandbox
                kwargs.setdefault("sandbox", SQLSandbox(db_path=sandbox_path))
        elif agent_type == "code":
            kwargs["expected_output"] = metadata.get("expected_output")

        vr = await verifier.verify(**kwargs)
        step_scores = {s.name: s.score for s in vr.steps}
        return vr.overall_reward, vr.feedback_for_agent, step_scores
    except Exception as exc:
        logger.debug("verify failed: %s", exc)
        return 0.5, str(exc), {}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_dataset(dataset: str, data_dir: str, n_samples: int, agent_type: str) -> list[dict]:
    """Load cases as plain dicts: {case_id, task, gold, hardness, metadata}."""
    from harness.eval.benchmark_loaders import (
        load_bird, load_csv, load_gsm8k, load_humaneval, load_jsonl, load_spider
    )

    loaders = {
        "bird":      lambda: load_bird(data_dir, n_samples=n_samples),
        "spider":    lambda: load_spider(data_dir, n_samples=n_samples),
        "humaneval": lambda: load_humaneval(data_dir, n_samples=n_samples),
        "gsm8k":     lambda: load_gsm8k(data_dir, n_samples=n_samples),
        "jsonl":     lambda: load_jsonl(data_dir, n_samples=n_samples, agent_type=agent_type),
        "csv":       lambda: load_csv(data_dir, n_samples=n_samples, agent_type=agent_type),
    }

    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose: {list(loaders)}")

    ds = loaders[dataset]()
    cases = []
    for c in ds.cases:
        cases.append({
            "case_id": c.case_id,
            "task": c.task,
            "gold": c.expected_output or "",
            "hardness": c.hardness or "unknown",
            "metadata": {
                "db_id": c.metadata.get("db_id", ""),
                "db_path": c.db_path or "",
                "expected_output": c.expected_output or "",
                "gold": c.expected_output or "",
                **c.metadata,
            },
        })
    return cases


# ---------------------------------------------------------------------------
# Self-test datasets
# ---------------------------------------------------------------------------

_SELF_TESTS: dict[str, list[dict]] = {
    "sql": [
        {"case_id": "sql_01", "task": "How many employees are there?",
         "gold": "SELECT COUNT(*) FROM employees", "hardness": "easy"},
        {"case_id": "sql_02", "task": "What is the average salary?",
         "gold": "SELECT AVG(salary) FROM employees", "hardness": "easy"},
        {"case_id": "sql_03", "task": "List employees by department.",
         "gold": "SELECT name, department FROM employees ORDER BY department", "hardness": "medium"},
    ],
    "code": [
        {"case_id": "code_01",
         "task": "Write a Python function `reverse_string(s)` that reverses a string.",
         "gold": "def reverse_string(s): return s[::-1]",
         "hardness": "easy",
         "metadata": {"expected_output": "olleh"}},
        {"case_id": "code_02",
         "task": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
         "gold": "def is_palindrome(s): return s == s[::-1]",
         "hardness": "easy"},
        {"case_id": "code_03",
         "task": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number.",
         "gold": "def fibonacci(n): a,b=0,1\n  for _ in range(n): a,b=b,a+b\n  return a",
         "hardness": "medium"},
    ],
    "base": [
        {"case_id": "base_01", "task": "What is 2 + 2?", "gold": "4", "hardness": "easy"},
        {"case_id": "base_02", "task": "What is the capital of France?", "gold": "Paris", "hardness": "easy"},
        {"case_id": "base_03", "task": "If John has 5 apples and gives 2 away, how many does he have?",
         "gold": "3", "hardness": "easy"},
    ],
}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

async def evaluate_case(
    case: dict,
    agent_fn: Callable,
    agent_type: str,
) -> CaseResult:
    t0 = time.monotonic()
    metadata = case.get("metadata", {})
    gold = case.get("gold", "")
    task = case["task"]

    try:
        generated = await agent_fn(task, metadata)
    except Exception as exc:
        generated = ""
        logger.warning("Agent failed for %s: %s", case["case_id"], exc)

    reward, feedback, step_scores = await _verify_case(
        task, generated, gold, agent_type, metadata
    )
    elapsed_ms = (time.monotonic() - t0) * 1000

    return CaseResult(
        case_id=case["case_id"],
        task=task,
        agent_type=agent_type,
        hardness=case.get("hardness", "unknown"),
        gold=gold,
        generated=generated,
        overall_reward=reward,
        verdict="correct" if reward >= 0.85 else "partial" if reward >= 0.40 else "incorrect",
        step_scores=step_scores,
        feedback=feedback,
        execution_time_ms=elapsed_ms,
    )


async def run_benchmark(
    agent_type: str,
    cases: list[dict],
    concurrency: int,
) -> tuple[list[CaseResult], float]:
    agent_fn = AGENTS[agent_type]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()

    async def _run(case):
        async with sem:
            return await evaluate_case(case, agent_fn, agent_type)

    results = await asyncio.gather(*[_run(c) for c in cases], return_exceptions=True)
    elapsed = time.monotonic() - t0

    case_results: list[CaseResult] = []
    for case, r in zip(cases, results):
        if isinstance(r, Exception):
            logger.warning("Case %s raised: %s", case["case_id"], r)
            case_results.append(CaseResult(
                case_id=case["case_id"], task=case["task"],
                agent_type=agent_type, hardness=case.get("hardness","unknown"),
                gold=case.get("gold",""), generated="",
                overall_reward=0.0, verdict="incorrect",
                step_scores={}, feedback=str(r),
                execution_time_ms=0.0,
            ))
        else:
            case_results.append(r)

    return case_results, elapsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    agent_type: str,
    dataset: str,
    results: list[CaseResult],
    elapsed: float,
    run_id: str,
) -> BenchReport:
    n = len(results)
    if n == 0:
        return BenchReport(agent_type, dataset, 0, 0.0, 0.0, 0.0, {}, {}, {}, [], run_id, elapsed)

    correct = [r for r in results if r.verdict == "correct"]
    exec_exact = [r for r in results if r.step_scores.get("result_check", r.step_scores.get("output_check", r.step_scores.get("answer_check", 0.0))) >= 0.99]

    by_hardness: dict[str, dict] = {}
    for r in results:
        h = r.hardness
        g = by_hardness.setdefault(h, {"count": 0, "passed": 0, "exec": 0})
        g["count"] += 1
        if r.verdict == "correct":
            g["passed"] += 1
        if r in exec_exact:
            g["exec"] += 1
    for h, v in by_hardness.items():
        c = max(v["count"], 1)
        by_hardness[h]["pass_rate"]      = round(v["passed"] / c, 4)
        by_hardness[h]["exec_acc_rate"]  = round(v["exec"] / c, 4)

    all_steps: dict[str, list[float]] = {}
    for r in results:
        for step, score in r.step_scores.items():
            all_steps.setdefault(step, []).append(score)
    step_pass_rates = {
        s: round(sum(1 for x in scores if x >= 0.8) / len(scores), 4)
        for s, scores in all_steps.items()
    }

    failures: dict[str, int] = {}
    for r in results:
        if r.verdict != "correct" and r.feedback:
            key = r.feedback.split(":")[0].strip()[:40]
            failures[key] = failures.get(key, 0) + 1

    return BenchReport(
        agent_type=agent_type,
        dataset=dataset,
        n_cases=n,
        overall_pass_rate=round(len(correct) / n, 4),
        execution_accuracy=round(len(exec_exact) / n, 4),
        avg_reward=round(sum(r.overall_reward for r in results) / n, 4),
        by_hardness=by_hardness,
        step_pass_rates=step_pass_rates,
        failure_distribution=dict(sorted(failures.items(), key=lambda x: -x[1])[:10]),
        cases=results,
        run_id=run_id,
        elapsed_seconds=round(elapsed, 2),
    )


def print_report(report: BenchReport) -> None:
    print("\n" + "=" * 60)
    print(f"Benchmark: {report.dataset.upper()}  Agent: {report.agent_type}  (run={report.run_id})")
    print("=" * 60)
    print(f"Cases          : {report.n_cases}")
    print(f"Pass rate      : {report.overall_pass_rate:.1%}")
    print(f"Exec accuracy  : {report.execution_accuracy:.1%}")
    print(f"Avg reward     : {report.avg_reward:.3f}")
    print(f"Elapsed        : {report.elapsed_seconds:.1f}s")
    print()
    if report.by_hardness:
        print("By hardness:")
        for h in ("easy", "medium", "hard", "extra-hard", "unknown"):
            s = report.by_hardness.get(h)
            if s:
                print(f"  {h:12s}  pass={s['pass_rate']:.1%}  exec={s['exec_acc_rate']:.1%}  n={s['count']}")
        print()
    if report.step_pass_rates:
        print("Step pass rates:")
        for step, rate in sorted(report.step_pass_rates.items()):
            bar = "█" * int(rate * 20)
            print(f"  {step:22s}  {rate:.1%}  {bar}")
    if report.failure_distribution:
        print()
        print("Top failures:")
        for cat, count in list(report.failure_distribution.items())[:5]:
            print(f"  [{count:3d}] {cat}")
    print("=" * 60)


def save_report(report: BenchReport, output: str) -> None:
    data = {
        "run_id": report.run_id,
        "agent_type": report.agent_type,
        "dataset": report.dataset,
        "n_cases": report.n_cases,
        "overall_pass_rate": report.overall_pass_rate,
        "execution_accuracy": report.execution_accuracy,
        "avg_reward": report.avg_reward,
        "elapsed_seconds": report.elapsed_seconds,
        "by_hardness": report.by_hardness,
        "step_pass_rates": report.step_pass_rates,
        "failure_distribution": report.failure_distribution,
        "cases": [
            {"case_id": c.case_id, "task": c.task[:200],
             "hardness": c.hardness, "verdict": c.verdict,
             "overall_reward": c.overall_reward,
             "step_scores": c.step_scores,
             "feedback": c.feedback,
             "generated": c.generated[:300]}
            for c in report.cases
        ],
    }
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HarnessAgent general benchmark")
    p.add_argument("--agent", default="base",
                   choices=list(AGENTS), help="Agent type (default: base)")
    p.add_argument("--dataset", default="",
                   help="Dataset name: bird|spider|humaneval|gsm8k|jsonl|csv")
    p.add_argument("--data-dir", default="",
                   help="Path to dataset file or directory")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of cases to evaluate (default: 10)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Concurrent agent calls (default: 4)")
    p.add_argument("--output", default="",
                   help="Output JSON path (default: results/{agent}_{dataset}.json)")
    p.add_argument("--estimate", action="store_true",
                   help="Show cost estimate only, do not run")
    return p.parse_args()


async def _main(args: argparse.Namespace) -> BenchReport:
    agent_type = args.agent
    dataset = args.dataset or agent_type
    run_id = uuid.uuid4().hex[:12]
    output = args.output or str(RESULTS_DIR / f"{agent_type}_{dataset}.json")

    if args.estimate:
        _print_estimate(agent_type, args.n_samples)
        import sys as _sys; _sys.exit(0)

    _print_estimate(agent_type, args.n_samples)

    # Load cases
    if args.data_dir and args.dataset:
        logger.info("Loading %s dataset from %s", args.dataset, args.data_dir)
        try:
            cases = _load_dataset(args.dataset, args.data_dir, args.n_samples, agent_type)
        except Exception as exc:
            logger.error("Dataset load failed: %s", exc)
            raise
    else:
        logger.info("No --data-dir provided — running self-test for agent_type=%s", agent_type)
        cases = _SELF_TESTS.get(agent_type, _SELF_TESTS["base"])
        for c in cases:
            if "metadata" not in c:
                c["metadata"] = {}

    logger.info("Running %d cases  agent=%s  concurrency=%d", len(cases), agent_type, args.concurrency)
    results, elapsed = await run_benchmark(agent_type, cases, args.concurrency)

    for r in results:
        logger.info(
            "  [%s] reward=%.2f  verdict=%-9s  %s",
            r.case_id, r.overall_reward, r.verdict, r.task[:60],
        )

    report = build_report(agent_type, dataset, results, elapsed, run_id)
    print_report(report)
    save_report(report, output)
    return report


# Cost estimates per LLM call (input / output tokens)
_EST: dict[str, tuple[int, int]] = {
    "sql":  (450, 130),
    "code": (600, 400),
    "base": (200, 150),
}

_PRICING = {
    "gpt-5.5":           {"in": 2.50,  "out": 10.00},
    "claude-haiku-4-5":  {"in": 0.80,  "out":  4.00},
    "claude-sonnet-4-6": {"in": 3.00,  "out": 15.00},
    "gpt-4o-mini":       {"in": 0.15,  "out":  0.60},
}


def _print_estimate(agent_type: str, n: int) -> None:
    in_t, out_t = _EST.get(agent_type, (400, 200))
    # avg 1.8× multiplier for self-correction
    total_in  = in_t  * 1.8 * n
    total_out = out_t * 1.8 * n
    print(f"\nCost estimate: {agent_type} agent  ×{n} cases")
    print(f"  Input ~{total_in:,.0f} tokens  Output ~{total_out:,.0f} tokens")
    for model, p in _PRICING.items():
        cost = total_in / 1e6 * p["in"] + total_out / 1e6 * p["out"]
        print(f"  {model:<22}  ${cost:.3f}")
    print()


def main() -> None:
    args = parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
