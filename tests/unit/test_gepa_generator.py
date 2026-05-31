"""Unit tests for the GEPA reflective prompt-evolution generator."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.improvement.error_collector import ErrorRecord
from harness.improvement.gepa import GepaPatchGenerator, build_patch_generator
from harness.improvement.gepa.adapter import COMPONENT, HarnessGepaAdapter
from harness.improvement.gepa.reflection import ProviderReflectionLM, make_coro_runner
from harness.improvement.patch_generator import PatchGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(task: str = "List all tables", agent_type: str = "sql") -> ErrorRecord:
    return ErrorRecord(
        record_id=uuid.uuid4().hex,
        agent_type=agent_type,
        task=task,
        failure_class="LLM_PARSE_ERROR",
        error_message="JSON parse failed",
        stack_trace="",
        created_at=datetime.now(UTC),
    )


class _FakeEvaluator:
    """Async Evaluator stub: scores a single-record batch via a scoring fn."""

    def __init__(self, scorer):
        self._scorer = scorer
        self.calls = 0

    async def score(self, *, patch, test_cases, agent_type):
        self.calls += 1
        record = test_cases[0]
        value = self._scorer(record, patch)
        return SimpleNamespace(
            score=value, successes=1 if value >= 0.5 else 0, test_cases=1
        )


def _sync_runner(coro):
    """Run a coroutine to completion on a throwaway loop (for adapter unit tests)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(budget=8, min_train=2, reflect_tokens=256):
    cfg = MagicMock()
    cfg.hermes_gepa_budget = budget
    cfg.hermes_gepa_min_train = min_train
    cfg.hermes_gepa_reflection_max_tokens = reflect_tokens
    # Mirror real defaults so MagicMock truthiness doesn't accidentally enable MLflow.
    cfg.hermes_gepa_use_mlflow = False
    cfg.mlflow_tracking_uri = None
    cfg.mlflow_experiment_name = None
    return cfg


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_heuristic_returns_patch_generator():
    gen = build_patch_generator(
        "heuristic", llm_provider=MagicMock(), prompt_manager=MagicMock()
    )
    assert isinstance(gen, PatchGenerator)


def test_factory_gepa_with_evaluator_returns_gepa_generator():
    gen = build_patch_generator(
        "gepa",
        llm_provider=MagicMock(),
        prompt_manager=MagicMock(),
        evaluator=MagicMock(),
        config=_make_config(),
    )
    assert isinstance(gen, GepaPatchGenerator)


def test_factory_gepa_without_evaluator_falls_back():
    gen = build_patch_generator(
        "gepa", llm_provider=MagicMock(), prompt_manager=MagicMock(), evaluator=None
    )
    assert isinstance(gen, PatchGenerator)


# ---------------------------------------------------------------------------
# Async / thread bridge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coro_runner_rejects_loop_thread():
    """The runner must refuse to run on its owning loop (would deadlock)."""
    loop = asyncio.get_running_loop()
    run = make_coro_runner(loop)

    async def _c():
        return 1

    coro = _c()
    with pytest.raises(RuntimeError):
        run(coro)
    coro.close()  # avoid "never awaited" warning


@pytest.mark.asyncio
async def test_reflection_lm_bridges_to_async_provider():
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=SimpleNamespace(content="NEW PROMPT"))
    run_coro = make_coro_runner(asyncio.get_running_loop())
    reflection = ProviderReflectionLM(llm, run_coro, max_tokens=100)

    # GEPA calls the reflection LM from its worker thread.
    out = await asyncio.to_thread(reflection, "please improve")

    assert out == "NEW PROMPT"
    llm.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_adapter_evaluate_returns_per_example_scores():
    # Score depends on whether the prompt contains "IMPROVED".
    def scorer(record, patch):
        return 0.9 if "IMPROVED" in patch.value else 0.2

    evaluator = _FakeEvaluator(scorer)
    adapter = HarnessGepaAdapter(evaluator, "sql", _sync_runner)
    batch = [_make_error("a"), _make_error("b"), _make_error("c")]

    result = adapter.evaluate(batch, {COMPONENT: "IMPROVED prompt"}, capture_traces=True)

    assert result.scores == [0.9, 0.9, 0.9]
    assert evaluator.calls == 3  # one score() call per record
    assert result.trajectories is not None
    assert len(result.trajectories) == 3
    assert result.trajectories[0]["success"] is True


def test_adapter_evaluate_failure_scores_zero_not_raises():
    async def _boom(*args, **kwargs):
        raise RuntimeError("agent blew up")

    evaluator = SimpleNamespace(score=_boom)
    adapter = HarnessGepaAdapter(evaluator, "sql", _sync_runner)

    result = adapter.evaluate([_make_error()], {COMPONENT: "p"}, capture_traces=False)

    assert result.scores == [0.0]  # failure -> 0.0, no exception
    assert result.trajectories is None


def test_adapter_reflective_dataset_has_feedback():
    evaluator = _FakeEvaluator(lambda r, p: 0.2)
    adapter = HarnessGepaAdapter(evaluator, "sql", _sync_runner)
    batch = [_make_error("List tables")]
    eval_batch = adapter.evaluate(batch, {COMPONENT: "p"}, capture_traces=True)

    ds = adapter.make_reflective_dataset({COMPONENT: "p"}, eval_batch, [COMPONENT])

    assert COMPONENT in ds
    assert len(ds[COMPONENT]) == 1
    record = ds[COMPONENT][0]
    assert "Feedback" in record and "FAILS" in record["Feedback"]
    assert "LLM_PARSE_ERROR" in record["Feedback"]


# ---------------------------------------------------------------------------
# GepaPatchGenerator.generate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_none_when_too_few_errors():
    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="BASE")
    gen = GepaPatchGenerator(
        llm_provider=MagicMock(),
        prompt_manager=pm,
        evaluator=MagicMock(),
        config=_make_config(min_train=5),
    )
    patch = await gen.generate("sql", [_make_error()])
    assert patch is None


@pytest.mark.asyncio
async def test_generate_returns_none_on_empty_errors():
    gen = GepaPatchGenerator(
        llm_provider=MagicMock(), prompt_manager=MagicMock(), evaluator=MagicMock()
    )
    assert await gen.generate("sql", []) is None


@pytest.mark.asyncio
async def test_generate_builds_set_patch_from_result(monkeypatch):
    """generate() orchestrates optimize in a thread and builds an op=set patch.

    The fake optimize calls adapter.evaluate from the worker thread, exercising
    the real run_coroutine_threadsafe bridge back to this event loop.
    """
    import gepa

    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="BASE PROMPT")
    evaluator = _FakeEvaluator(lambda r, p: 0.9 if "IMPROVED" in p.value else 0.1)
    patch_store = MagicMock()
    patch_store.save = AsyncMock()

    def fake_optimize(*, adapter, trainset, seed_candidate, **kwargs):
        # Exercise the genuine async/thread bridge from the worker thread.
        batch = adapter.evaluate(trainset, {COMPONENT: "IMPROVED"}, capture_traces=True)
        assert batch.scores  # bridge produced per-example scores
        return SimpleNamespace(
            best_candidate={COMPONENT: "IMPROVED PROMPT: validate inputs."},
            val_aggregate_subscores=[0.9],
            best_idx=0,
            total_metric_calls=3,
        )

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    gen = GepaPatchGenerator(
        llm_provider=MagicMock(),
        prompt_manager=pm,
        evaluator=evaluator,
        config=_make_config(),
        patch_store=patch_store,
    )
    errors = [_make_error("a"), _make_error("b"), _make_error("c")]
    patch = await gen.generate("sql", errors)

    assert patch is not None
    assert patch.op == "set"
    assert patch.value == "IMPROVED PROMPT: validate inputs."
    assert patch.proposed_by == "hermes-gepa"
    assert patch.score == 0.9
    assert len(patch.based_on_errors) == 3
    patch_store.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_returns_none_when_no_improvement(monkeypatch):
    import gepa

    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="BASE PROMPT")

    def fake_optimize(*, seed_candidate, **kwargs):
        # GEPA found nothing better than the seed.
        return SimpleNamespace(
            best_candidate={COMPONENT: "BASE PROMPT"},
            val_aggregate_subscores=[0.5],
            best_idx=0,
            total_metric_calls=3,
        )

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    gen = GepaPatchGenerator(
        llm_provider=MagicMock(),
        prompt_manager=pm,
        evaluator=_FakeEvaluator(lambda r, p: 0.5),
        config=_make_config(),
    )
    patch = await gen.generate("sql", [_make_error(), _make_error()])
    assert patch is None


@pytest.mark.asyncio
async def test_generate_survives_optimize_exception(monkeypatch):
    import gepa

    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="BASE PROMPT")

    def fake_optimize(**kwargs):
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr(gepa, "optimize", fake_optimize)

    gen = GepaPatchGenerator(
        llm_provider=MagicMock(),
        prompt_manager=pm,
        evaluator=_FakeEvaluator(lambda r, p: 0.5),
        config=_make_config(),
    )
    # Must not raise — a broken cycle should never crash Hermes.
    assert await gen.generate("sql", [_make_error(), _make_error()]) is None


@pytest.mark.asyncio
async def test_generate_real_gepa_evolves_prompt(monkeypatch):
    """Run the REAL gepa.optimize end-to-end and assert the prompt actually evolves.

    Catches API drift against the installed gepa version and exercises the real
    reflection/Pareto loop (no mocking of optimize).
    """
    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="You are a SQL agent.")

    # Reward prompts that mention validation; the reflection LM proposes exactly that.
    evaluator = _FakeEvaluator(
        lambda r, p: 1.0 if "validate" in p.value.lower() else 0.0
    )
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=SimpleNamespace(
            content="You are a SQL agent. Always validate SQL before execution."
        )
    )

    gen = GepaPatchGenerator(
        llm_provider=llm,
        prompt_manager=pm,
        evaluator=evaluator,
        config=_make_config(budget=12, min_train=2),
    )
    errors = [_make_error("q1"), _make_error("q2"), _make_error("q3")]

    patch = await gen.generate("sql", errors)

    assert patch is not None
    assert patch.op == "set"
    assert "validate" in patch.value.lower()  # the improved prompt was selected
    assert patch.proposed_by == "hermes-gepa"
    assert evaluator.calls > 0


@pytest.mark.asyncio
async def test_generate_logs_to_mlflow(tmp_path, monkeypatch):
    """GEPA's native MLflow tracking records the optimization run."""
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment = "gepa-test-tracking"

    pm = MagicMock()
    pm.get_prompt = AsyncMock(return_value="You are a SQL agent.")
    evaluator = _FakeEvaluator(
        lambda r, p: 1.0 if "validate" in p.value.lower() else 0.0
    )
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=SimpleNamespace(
            content="You are a SQL agent. Always validate SQL before execution."
        )
    )

    config = _make_config(budget=12, min_train=2)
    config.hermes_gepa_use_mlflow = True
    config.mlflow_tracking_uri = tracking_uri
    config.mlflow_experiment_name = experiment

    gen = GepaPatchGenerator(
        llm_provider=llm, prompt_manager=pm, evaluator=evaluator, config=config
    )
    patch = await gen.generate("sql", [_make_error("q1"), _make_error("q2")])
    assert patch is not None

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) >= 1
    # GEPA logs per-iteration optimization metrics.
    assert runs[0].data.metrics, "expected GEPA to log metrics to MLflow"
