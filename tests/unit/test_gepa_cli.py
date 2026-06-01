"""Tests for the benchmark GEPA optimization CLI (scripts/gepa_optimize.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.core.prompt_overrides import OVERRIDES_KEY
from harness.eval.datasets import EvalCase, EvalDataset

# Load the script as a module (it lives under scripts/, not the package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gepa_optimize.py"
_spec = importlib.util.spec_from_file_location("gepa_optimize", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_pick_scorer_variants():
    assert callable(cli.pick_scorer("exact"))
    assert callable(cli.pick_scorer("sql_equiv"))
    assert callable(cli.pick_scorer("execution"))
    with pytest.raises(ValueError):
        cli.pick_scorer("nope")


def test_pick_scorer_sql_equiv_scores_equivalent_sql():
    scorer = cli.pick_scorer("sql_equiv")
    assert scorer("SELECT 1", "SELECT 1") == 1.0


def test_benchmarks_registry_defaults():
    # gsm8k is the default benchmark and uses a true (exact-answer) scorer.
    assert cli.build_parser().parse_args(["--data-dir", "/x"]).benchmark == "gsm8k"
    assert cli.BENCHMARKS["gsm8k"]["scorer"] == "exact"
    # HumanEval defaults to real pass@1 execution scoring.
    assert cli.BENCHMARKS["humaneval"]["scorer"] == "execution"
    assert cli.BENCHMARKS["bird"]["agent_type"] == "sql"
    assert set(cli.BENCHMARKS) == {"gsm8k", "humaneval", "spider", "bird"}


def test_load_benchmark_dispatches(monkeypatch):
    calls = {}

    def rec(name):
        def _f(*args, **kwargs):
            calls[name] = (args, kwargs)
            return EvalDataset(name=name, agent_type="x", cases=[])
        return _f

    monkeypatch.setattr("harness.eval.benchmark_loaders.load_gsm8k", rec("gsm8k"))
    monkeypatch.setattr("harness.eval.benchmark_loaders.load_humaneval", rec("humaneval"))
    monkeypatch.setattr("harness.eval.benchmark_loaders.load_bird", rec("bird"))

    cli.load_benchmark("gsm8k", "/d", None, 5)
    cli.load_benchmark("humaneval", "/d", None, 5)
    cli.load_benchmark("bird", "/d", "dev", 5)

    assert calls["gsm8k"][1]["split"] == "test"     # benchmark default applied
    assert "split" not in calls["humaneval"][1]      # humaneval loader takes no split
    assert calls["bird"][1]["split"] == "dev"


@pytest.mark.asyncio
async def test_build_seed_prompts_each_component():
    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="SYS")
    seed = await cli.build_seed_prompts(
        ["system_prompt", "planner_prompt", "context_summary"], "code", pm
    )
    assert seed["system_prompt"] == "SYS"
    assert "decomposition" in seed["planner_prompt"].lower()
    assert "summarize" in seed["context_summary"].lower()


@pytest.mark.asyncio
async def test_build_seed_prompts_unknown_component():
    with pytest.raises(ValueError):
        await cli.build_seed_prompts(["bogus"], "code", MagicMock())


def test_split_holds_out_val():
    ds = EvalDataset(
        name="d",
        agent_type="code",
        cases=[EvalCase(case_id=f"c{i}", agent_type="code", task="t") for i in range(10)],
    )
    train, val = cli._split(ds, 0.4)
    assert len(val.cases) == 4
    assert len(train.cases) == 6


def test_parser_defaults():
    args = cli.build_parser().parse_args(["--data-dir", "/data"])
    assert args.benchmark == "gsm8k"
    assert args.scorer is None        # resolved from the benchmark default at runtime
    assert args.components == "system_prompt"


# ---------------------------------------------------------------------------
# End-to-end run() with all externals monkeypatched (gsm8k default)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_end_to_end(monkeypatch, tmp_path):
    def fake_load_gsm8k(path, split="test", n_samples=None):
        return EvalDataset(
            name="gsm8k",
            agent_type="code",
            cases=[
                EvalCase(case_id=f"g{i}", agent_type="code", task=f"q{i}", expected_output="42")
                for i in range(5)
            ],
        )

    class FakeEvalRunner:
        def __init__(self, agent_runner, llm_provider=None):
            pass

        async def run(self, dataset, tenant_id="eval", scorer=None, **kwargs):
            # A case passes iff the injected system prompt says "validate".
            scores = {}
            for case in dataset.cases:
                sp = (case.metadata or {}).get(OVERRIDES_KEY, {}).get("system_prompt", "")
                scores[case.case_id] = 1.0 if "validate" in sp.lower() else 0.0
            return SimpleNamespace(scores=scores, errors={}, diagnostics=None)

    async def fake_optimize(**kwargs):
        from harness.improvement.gepa.eval_optimizer import EvalOptimizationResult

        return EvalOptimizationResult(
            components={"system_prompt": "Always validate the final answer."},
            seed=kwargs["seed_prompts"],
            best_score=1.0,
            improved=True,
            total_metric_calls=7,
        )

    fake_redis = MagicMock()
    fake_redis.aclose = AsyncMock()
    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="You are a math agent.")  # seed lacks "validate"

    monkeypatch.setattr("harness.eval.benchmark_loaders.load_gsm8k", fake_load_gsm8k)
    monkeypatch.setattr("harness.eval.runner.EvalRunner", FakeEvalRunner)
    monkeypatch.setattr("harness.improvement.gepa.optimize_prompts_on_dataset", fake_optimize)
    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **k: fake_redis)
    monkeypatch.setattr(
        "harness.workers.agent_worker.build_agent_factory", lambda cfg: (lambda at: None)
    )
    monkeypatch.setattr("harness.llm.anthropic.AnthropicProvider", lambda **k: MagicMock())
    monkeypatch.setattr("harness.prompts.manager.PromptManager", lambda store: pm)
    monkeypatch.setattr("harness.prompts.store.PromptStore", lambda redis: MagicMock())

    out = tmp_path / "result.json"
    args = cli.build_parser().parse_args(
        ["--benchmark", "gsm8k", "--data-dir", "/x", "--n-samples", "5",
         "--budget", "4", "--output", str(out)]
    )
    rc = await cli.run(args)

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["benchmark"] == "gsm8k"
    assert payload["baseline_score"] == 0.0
    assert payload["optimized_score"] == 1.0
    assert payload["improved"] is True
    fake_redis.aclose.assert_awaited_once()
