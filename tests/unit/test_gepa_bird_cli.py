"""Tests for the BIRD GEPA optimization CLI (scripts/gepa_bird_optimize.py)."""

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
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gepa_bird_optimize.py"
_spec = importlib.util.spec_from_file_location("gepa_bird_optimize", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_pick_scorer_variants():
    assert callable(cli.pick_scorer("exact"))
    assert callable(cli.pick_scorer("sql_equiv"))
    with pytest.raises(ValueError):
        cli.pick_scorer("nope")


def test_pick_scorer_sql_equiv_scores_equivalent_sql():
    scorer = cli.pick_scorer("sql_equiv")
    assert scorer("SELECT 1", "SELECT 1") == 1.0


@pytest.mark.asyncio
async def test_build_seed_prompts_each_component():
    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="SYS")
    seed = await cli.build_seed_prompts(
        ["system_prompt", "planner_prompt", "context_summary"], "sql", pm
    )
    assert seed["system_prompt"] == "SYS"
    assert "decomposition" in seed["planner_prompt"].lower()
    assert "summarize" in seed["context_summary"].lower()


@pytest.mark.asyncio
async def test_build_seed_prompts_unknown_component():
    with pytest.raises(ValueError):
        await cli.build_seed_prompts(["bogus"], "sql", MagicMock())


def test_split_holds_out_val():
    ds = EvalDataset(
        name="d",
        agent_type="sql",
        cases=[EvalCase(case_id=f"c{i}", agent_type="sql", task="t") for i in range(10)],
    )
    train, val = cli._split(ds, 0.4)
    assert len(val.cases) == 4
    assert len(train.cases) == 6


def test_parser_defaults():
    args = cli.build_parser().parse_args(["--bird-dir", "/data/bird"])
    assert args.split == "dev"
    assert args.scorer == "sql_equiv"
    assert args.components == "system_prompt"


# ---------------------------------------------------------------------------
# End-to-end run() with all externals monkeypatched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_end_to_end(monkeypatch, tmp_path):
    # Fake BIRD dataset.
    def fake_load_bird(bird_dir, split="dev", n_samples=None):
        return EvalDataset(
            name="bird",
            agent_type="sql",
            cases=[
                EvalCase(case_id=f"b{i}", agent_type="sql", task=f"q{i}", expected_output="SELECT 1")
                for i in range(5)
            ],
        )

    # Fake EvalRunner: a case passes iff the injected system prompt says "validate".
    class FakeEvalRunner:
        def __init__(self, agent_runner, llm_provider=None):
            pass

        async def run(self, dataset, tenant_id="eval", scorer=None, **kwargs):
            scores = {}
            for case in dataset.cases:
                sp = (case.metadata or {}).get(OVERRIDES_KEY, {}).get("system_prompt", "")
                scores[case.case_id] = 1.0 if "validate" in sp.lower() else 0.0
            return SimpleNamespace(scores=scores, errors={}, diagnostics=None)

    async def fake_optimize(**kwargs):
        return cli_eval_result(
            components={"system_prompt": "You are a SQL agent. Always validate."},
            seed=kwargs["seed_prompts"],
        )

    def cli_eval_result(components, seed):
        from harness.improvement.gepa.eval_optimizer import EvalOptimizationResult

        return EvalOptimizationResult(
            components=components, seed=seed, best_score=1.0, improved=True, total_metric_calls=7
        )

    fake_redis = MagicMock()
    fake_redis.aclose = AsyncMock()

    monkeypatch.setattr("harness.eval.benchmark_loaders.load_bird", fake_load_bird)
    monkeypatch.setattr("harness.eval.runner.EvalRunner", FakeEvalRunner)
    monkeypatch.setattr("harness.improvement.gepa.optimize_prompts_on_dataset", fake_optimize)
    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **k: fake_redis)
    monkeypatch.setattr(
        "harness.workers.agent_worker.build_agent_factory", lambda cfg: (lambda at: None)
    )
    monkeypatch.setattr(
        "harness.llm.anthropic.AnthropicProvider", lambda **k: MagicMock()
    )

    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="You are a SQL agent.")  # seed lacks "validate"
    monkeypatch.setattr("harness.prompts.manager.PromptManager", lambda store: pm)
    monkeypatch.setattr("harness.prompts.store.PromptStore", lambda redis: MagicMock())

    out = tmp_path / "result.json"
    args = cli.build_parser().parse_args(
        ["--bird-dir", "/x", "--n-samples", "5", "--budget", "4", "--output", str(out)]
    )
    rc = await cli.run(args)

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["baseline_score"] == 0.0       # seed had no "validate"
    assert payload["optimized_score"] == 1.0      # evolved prompt does
    assert payload["improved"] is True
    fake_redis.aclose.assert_awaited_once()
