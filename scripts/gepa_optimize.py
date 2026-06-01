#!/usr/bin/env python
"""Optimize agent prompt components on a benchmark with GEPA (gold-scored).

Benchmark-agnostic experiment driver:

  1. Load a benchmark via the existing loaders (GSM8K / HumanEval / Spider / BIRD).
  2. Build the production agent stack (AgentRunner + agent factory + EvalRunner).
  3. Seed the requested prompt components (system_prompt / planner_prompt /
     context_summary) from their current values.
  4. Run GEPA against the gold-labeled dataset using the benchmark's scorer.
  5. Report baseline vs optimized score on a held-out split and write the evolved
     components to disk.

Benchmark notes (why the default is GSM8K, not BIRD):
  - gsm8k     reasoning; gold = final number → exact-match is a TRUE correctness
              signal with no sandbox. Best default.
  - humaneval code generation; for real pass@1 use execution scoring (a sandbox);
              the default exact-match only checks textual similarity.
  - spider/bird  text-to-SQL; the AST-equivalence scorer is a proxy — real scoring
              needs DB execution. BIRD in particular is messy/real-world DBs.

Requires a real agent runtime (LLM credentials, Redis). Offline job, not the
request path.

Examples
--------
    python scripts/gepa_optimize.py --benchmark gsm8k --data-dir /data/gsm8k.jsonl \
        --n-samples 40 --budget 80
    python scripts/gepa_optimize.py --benchmark humaneval --data-dir /data/humaneval.jsonl \
        --components system_prompt,context_summary --mlflow
    python scripts/gepa_optimize.py --benchmark bird --data-dir /data/bird --scorer sql_equiv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("gepa_optimize")

# Default seed for the context-compression component (mirrors BaseAgent._fit_history).
_DEFAULT_CONTEXT_SUMMARY = (
    "Summarize the following conversation history concisely in 3-5 sentences, "
    "preserving key decisions, findings, and tool call outcomes:"
)

# Per-benchmark defaults: agent_type, default scorer, and the default split.
BENCHMARKS: dict[str, dict[str, Any]] = {
    "gsm8k": {"agent_type": "code", "scorer": "exact", "split": "test"},
    "humaneval": {"agent_type": "code", "scorer": "execution", "split": None},
    "spider": {"agent_type": "sql", "scorer": "sql_equiv", "split": "dev"},
    "bird": {"agent_type": "sql", "scorer": "sql_equiv", "split": "dev"},
}

SCORERS = ("exact", "sql_equiv", "execution")


def pick_scorer(name: str) -> Callable[..., Any]:
    """Return an EvalRunner-compatible scorer callable for ``name``.

    - ``exact``:     substring/exact-match on the gold text.
    - ``sql_equiv``: SQL AST equivalence (ignores formatting/aliasing).
    - ``execution``: HumanEval-style pass@1 — run candidate code against the case's
                     unit tests in a sandbox (case-aware, async). Real correctness.
    """
    from harness.eval.scorers import score_exact_match, score_sql_equivalence

    if name == "sql_equiv":
        return lambda output, expected: score_sql_equivalence(output or "", expected or "")
    if name == "exact":
        return lambda output, expected: score_exact_match(output or "", expected or "")
    if name == "execution":
        from harness.eval.sandbox import CodeSandbox
        from harness.eval.scorers import score_code_execution

        sandbox = CodeSandbox()

        async def _execution_scorer(output: str, expected: Any, case: Any) -> float:
            result = await score_code_execution(output, case, sandbox)
            return float(result.score)

        return _execution_scorer
    raise ValueError(f"unknown scorer {name!r} (choose: {', '.join(SCORERS)})")


def load_benchmark(name: str, data_dir: str, split: str | None, n_samples: int | None) -> Any:
    """Load ``name`` via the matching benchmark loader, normalizing signatures."""
    from harness.eval import benchmark_loaders as bl

    eff_split: Any = split or BENCHMARKS[name]["split"]
    if name == "spider":
        return bl.load_spider(data_dir, split=eff_split or "dev", n_samples=n_samples)
    if name == "bird":
        return bl.load_bird(data_dir, split=eff_split or "dev", n_samples=n_samples)
    if name == "humaneval":
        return bl.load_humaneval(data_dir, n_samples=n_samples)  # no split
    if name == "gsm8k":
        return bl.load_gsm8k(data_dir, split=eff_split or "test", n_samples=n_samples)
    raise ValueError(f"unknown benchmark {name!r} (choose: {', '.join(BENCHMARKS)})")


async def build_seed_prompts(
    components: list[str], agent_type: str, prompt_manager: Any
) -> dict[str, str]:
    """Build the seed component texts GEPA starts evolving from."""
    seed: dict[str, str] = {}
    for component in components:
        if component == "system_prompt":
            seed["system_prompt"] = await prompt_manager.get_prompt(agent_type)
        elif component == "planner_prompt":
            from harness.orchestrator.planner import _PLAN_PROMPT_TEMPLATE

            seed["planner_prompt"] = _PLAN_PROMPT_TEMPLATE
        elif component == "context_summary":
            seed["context_summary"] = _DEFAULT_CONTEXT_SUMMARY
        else:
            raise ValueError(
                f"unknown component {component!r} "
                "(choose: system_prompt, planner_prompt, context_summary)"
            )
    return seed


async def _aggregate_score(
    eval_runner: Any, dataset: Any, seed: dict[str, str], scorer: Any, tenant: str
) -> float:
    """Run one eval pass with ``seed`` injected and return mean per-case score."""
    from dataclasses import replace

    from harness.core.prompt_overrides import OVERRIDES_KEY
    from harness.eval.datasets import EvalDataset

    cases = [
        replace(c, metadata={**(c.metadata or {}), OVERRIDES_KEY: seed})
        for c in dataset.cases
    ]
    ds = EvalDataset(name="baseline", agent_type=dataset.agent_type, cases=cases)
    report = await eval_runner.run(ds, tenant_id=tenant, scorer=scorer)
    scores = list((report.scores or {}).values())
    return sum(scores) / len(scores) if scores else 0.0


async def run(args: argparse.Namespace) -> int:
    import redis.asyncio as aioredis

    from harness.core.config import get_config
    from harness.eval.runner import EvalRunner
    from harness.improvement.gepa import optimize_prompts_on_dataset
    from harness.orchestrator.runner import AgentRunner
    from harness.prompts.manager import PromptManager
    from harness.prompts.store import PromptStore
    from harness.workers.agent_worker import build_agent_factory

    cfg = get_config()
    spec = BENCHMARKS[args.benchmark]
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    scorer = pick_scorer(args.scorer or spec["scorer"])

    # 1. Load benchmark
    dataset = load_benchmark(args.benchmark, args.data_dir, args.split, args.n_samples)
    agent_type = getattr(dataset, "agent_type", None) or spec["agent_type"]
    logger.info(
        "Loaded %s: %d case(s), agent_type=%s", args.benchmark, len(dataset.cases), agent_type
    )
    if not dataset.cases:
        logger.error("No cases loaded — check --data-dir / --split.")
        return 2

    train, val = _split(dataset, args.val_frac)

    # 2. Agent stack
    redis_client = aioredis.from_url(cfg.redis_url, encoding="utf-8", decode_responses=True)
    prompt_manager = PromptManager(store=PromptStore(redis=redis_client))
    agent_runner = AgentRunner(
        redis=redis_client,
        agent_factory=build_agent_factory(cfg),
        workspace_base=cfg.workspace_base_path,
    )
    eval_runner = EvalRunner(agent_runner, llm_provider=None)

    # 3. Seed components + reflection LM
    seed = await build_seed_prompts(components, agent_type, prompt_manager)
    logger.info("Optimizing components: %s", list(seed))
    llm = _build_llm(cfg, args)

    # 4. Baseline score (seed prompts) on the val split
    baseline = await _aggregate_score(eval_runner, val, seed, scorer, args.tenant)
    logger.info("Baseline mean score on %d val case(s): %.3f", len(val.cases), baseline)

    # 5. Optimize on the train split
    result = await optimize_prompts_on_dataset(
        eval_runner=eval_runner,
        dataset=train,
        valset=val,
        llm_provider=llm,
        seed_prompts=seed,
        budget=args.budget,
        scorer=scorer,
        tenant_id=args.tenant,
        concurrency=args.concurrency,
        use_mlflow=args.mlflow,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri if args.mlflow else None,
        mlflow_experiment_name=cfg.mlflow_experiment_name if args.mlflow else None,
    )

    # 6. Optimized score on the same val split
    optimized = await _aggregate_score(eval_runner, val, result.components, scorer, args.tenant)

    # 7. Report + persist
    print(f"\n==== GEPA x {args.benchmark} ====")
    print(f"components      : {list(seed)}")
    print(f"train/val cases : {len(train.cases)} / {len(val.cases)}")
    print(f"metric calls    : {result.total_metric_calls}")
    print(f"baseline score  : {baseline:.3f}")
    print(f"optimized score : {optimized:.3f}  (delta {optimized - baseline:+.3f})")
    print(f"improved        : {result.improved}")

    payload = {
        "benchmark": args.benchmark,
        "agent_type": agent_type,
        "baseline_score": baseline,
        "optimized_score": optimized,
        "improved": result.improved,
        "total_metric_calls": result.total_metric_calls,
        "components": result.components,
        "seed": result.seed,
    }
    with Path(args.output).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"evolved prompts -> {args.output}")

    await redis_client.aclose()
    return 0


def _split(dataset: Any, val_frac: float) -> tuple[Any, Any]:
    """Deterministic head/tail split into (train, val)."""
    from harness.eval.datasets import EvalDataset

    n = len(dataset.cases)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    val_cases = dataset.cases[:n_val]
    train_cases = dataset.cases[n_val:] or dataset.cases
    name = dataset.name
    return (
        EvalDataset(name=f"{name}-train", agent_type=dataset.agent_type, cases=train_cases),
        EvalDataset(name=f"{name}-val", agent_type=dataset.agent_type, cases=val_cases or train_cases),
    )


def _build_llm(cfg: Any, args: argparse.Namespace) -> Any:
    """Build the reflection (teacher) LM provider from config."""
    from harness.llm.anthropic import AnthropicProvider

    return AnthropicProvider(
        api_key=cfg.anthropic_api_key,
        model=args.reflection_model or cfg.default_model,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GEPA prompt optimization on a benchmark.")
    p.add_argument("--benchmark", default="gsm8k", choices=sorted(BENCHMARKS))
    p.add_argument("--data-dir", required=True, help="Path/dir for the benchmark data.")
    p.add_argument("--split", default=None, help="Override the benchmark's default split.")
    p.add_argument("--n-samples", type=int, default=40, help="Cases to load (None=all).")
    p.add_argument("--val-frac", type=float, default=0.4, help="Fraction held out for scoring.")
    p.add_argument(
        "--components",
        default="system_prompt",
        help="Comma list: system_prompt,planner_prompt,context_summary",
    )
    p.add_argument(
        "--scorer", default=None, choices=list(SCORERS),
        help="Override the benchmark's default scorer (execution = HumanEval pass@1).",
    )
    p.add_argument("--budget", type=int, default=60, help="Max candidate evaluations.")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--reflection-model", default=None, help="Override teacher LM model.")
    p.add_argument("--tenant", default="gepa-opt")
    p.add_argument("--mlflow", action="store_true", help="Log the run to MLflow.")
    p.add_argument("--output", default="gepa_result.json")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
