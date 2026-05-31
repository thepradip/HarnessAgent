#!/usr/bin/env python
"""End-to-end demo: run the GEPA Hermes generator and track it in MLflow.

This exercises ``GepaPatchGenerator`` for real — real ``gepa.optimize``, the real
async/thread bridge, and GEPA's native MLflow logging — without needing agents,
Docker, or API keys. It substitutes:

  * a toy in-memory Evaluator that rewards prompts containing target rules, and
  * a deterministic "reflection LM" (a fake provider whose ``complete`` returns an
    improved prompt), driven through the genuine ``ProviderReflectionLM`` bridge.

It then reads the MLflow run back and prints the logged params/metrics, proving
the optimization was tracked.

Run:
    python scripts/gepa_mlflow_demo.py
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

from harness.improvement.error_collector import ErrorRecord
from harness.improvement.gepa import GepaPatchGenerator

# The toy task: a good SQL-agent prompt should mention these three rules.
TARGET_RULES = ["validate", "schema", "limit"]
SEED_PROMPT = "You are a SQL agent."
IMPROVED_PROMPT = (
    "You are a SQL agent. Always validate the query against the schema "
    "and add a LIMIT to result sets."
)

TRACKING_DB = Path(__file__).resolve().parent.parent / "gepa_mlflow_demo.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB}"
EXPERIMENT_NAME = "gepa-hermes-demo"


class ToyEvaluator:
    """Scores a candidate prompt by how many target rules it contains.

    Mirrors the real Evaluator's contract: ``async score(patch, test_cases,
    agent_type) -> EvalResult-like`` with ``.score`` and ``.successes``.
    """

    async def score(self, *, patch, test_cases, agent_type):
        text = patch.value.lower()
        present = sum(1 for rule in TARGET_RULES if rule in text)
        score = present / len(TARGET_RULES)
        return SimpleNamespace(
            score=score,
            successes=1 if score >= 0.99 else 0,
            test_cases=len(test_cases),
        )


class FakeReflectionProvider:
    """Stands in for an LLM provider: proposes the improved prompt on reflection."""

    async def complete(self, messages, *, max_tokens, system=None, **kwargs):
        return SimpleNamespace(content=IMPROVED_PROMPT)


class FakePromptManager:
    async def get_prompt(self, agent_type: str) -> str:
        return SEED_PROMPT


def _make_errors(n: int = 4) -> list[ErrorRecord]:
    return [
        ErrorRecord(
            record_id=uuid.uuid4().hex,
            agent_type="sql",
            task=f"Answer SQL question #{i}",
            failure_class="LLM_PARSE_ERROR",
            error_message="Returned an unvalidated query with no LIMIT.",
        )
        for i in range(n)
    ]


async def run_optimization() -> object:
    config = SimpleNamespace(
        hermes_gepa_budget=24,
        hermes_gepa_min_train=2,
        hermes_gepa_reflection_max_tokens=256,
        hermes_gepa_use_mlflow=True,
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_experiment_name=EXPERIMENT_NAME,
    )
    generator = GepaPatchGenerator(
        llm_provider=FakeReflectionProvider(),
        prompt_manager=FakePromptManager(),
        evaluator=ToyEvaluator(),
        config=config,
    )
    print(f"Seed prompt:     {SEED_PROMPT!r}")
    patch = await generator.generate("sql", _make_errors())
    if patch is None:
        print("No improved prompt produced.")
    else:
        print(f"Evolved prompt:  {patch.value!r}")
        print(f"Patch op/score:  op={patch.op}  score={patch.score}")
        print(f"Rationale:       {patch.rationale}")
    return patch


def report_mlflow() -> None:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("\nMLflow: no experiment found — tracking did not run.")
        return

    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
    print(f"\nMLflow experiment '{EXPERIMENT_NAME}' (id={exp.experiment_id})")
    print(f"  tracking_uri: {TRACKING_URI}")
    print(f"  runs logged:  {len(runs)}")
    if not runs:
        return

    run = runs[0]
    print(f"\nLatest run {run.info.run_id} (status={run.info.status}):")

    params = run.data.params
    print(f"  params ({len(params)}):")
    for key in sorted(params)[:8]:
        print(f"    {key} = {params[key]}")

    metrics = run.data.metrics
    print(f"  metrics ({len(metrics)}):")
    for key in sorted(metrics):
        history = client.get_metric_history(run.info.run_id, key)
        values = [round(h.value, 4) for h in history]
        shown = values if len(values) <= 6 else [values[0], "...", values[-1]]
        print(f"    {key}: {shown}")


async def _main() -> None:
    await run_optimization()
    report_mlflow()


if __name__ == "__main__":
    asyncio.run(_main())
