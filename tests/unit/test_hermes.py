"""Unit tests for the Hermes self-improvement loop."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.improvement.error_collector import (
    ErrorCollector,
    ErrorRecord,
    _error_key,
)
from harness.improvement.evaluator import EvalResult, PatchEvaluator
from harness.improvement.hermes import HermesLoop, PatchOutcome
from harness.improvement.patch_generator import Patch


def _make_error(agent_type="sql"):
    return ErrorRecord(
        record_id=uuid.uuid4().hex,
        agent_type=agent_type,
        task="List all tables",
        failure_class="LLM_PARSE_ERROR",
        error_message="JSON parse failed",
        stack_trace="",
        created_at=datetime.now(timezone.utc),
    )


def _make_patch(agent_type="sql"):
    return Patch(
        patch_id=uuid.uuid4().hex,
        agent_type=agent_type,
        target="system_prompt",
        op="append",
        path="",
        value="Always validate SQL before execution.",
        rationale="Reduces parse errors.",
        status="pending",
    )


def _make_config(auto_apply=False, threshold=0.7, min_errors=5):
    cfg = MagicMock()
    cfg.hermes_auto_apply = auto_apply
    cfg.hermes_patch_score_threshold = threshold
    cfg.hermes_min_errors_to_trigger = min_errors
    cfg.hermes_interval_seconds = 3600.0
    return cfg


def _make_hermes(collector=None, generator=None, evaluator=None,
                 prompt_store=None, config=None):
    collector = collector or AsyncMock()
    generator = generator or AsyncMock()
    evaluator = evaluator or AsyncMock()
    prompt_store = prompt_store or AsyncMock()
    config = config or _make_config()
    metrics = MagicMock()
    return HermesLoop(
        collector=collector, generator=generator, evaluator=evaluator,
        prompt_store=prompt_store, metrics=metrics, config=config,
    )


@pytest.mark.asyncio
async def test_cycle_skips_when_too_few_errors():
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=2)
    hermes = _make_hermes(collector=collector, config=_make_config(min_errors=5))
    outcome = await hermes.run_cycle("sql")
    assert outcome is None


@pytest.mark.asyncio
async def test_cycle_generates_patch_from_errors():
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)
    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(score=0.8, test_cases=10, successes=8))
    hermes = _make_hermes(collector=collector, generator=generator,
                          evaluator=evaluator, config=_make_config(auto_apply=False))
    outcome = await hermes.run_cycle("sql")
    assert outcome is not None
    assert generator.generate.called


@pytest.mark.asyncio
async def test_cycle_applies_patch_when_auto_apply_true():
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()
    prompt_store = AsyncMock()
    prompt_store.apply_patch = AsyncMock(return_value=MagicMock())
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)
    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(score=0.9, test_cases=10, successes=9))
    hermes = _make_hermes(collector=collector, generator=generator, evaluator=evaluator,
                          prompt_store=prompt_store, config=_make_config(auto_apply=True, threshold=0.7))
    outcome = await hermes.run_cycle("sql")
    assert outcome is not None
    assert outcome.applied is True
    prompt_store.apply_patch.assert_called_once()


@pytest.mark.asyncio
async def test_cycle_queues_patch_when_auto_apply_false():
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()
    prompt_store = AsyncMock()
    prompt_store.apply_patch = AsyncMock()
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)
    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(score=0.9))
    hermes = _make_hermes(collector=collector, generator=generator, evaluator=evaluator,
                          prompt_store=prompt_store, config=_make_config(auto_apply=False))
    outcome = await hermes.run_cycle("sql")
    assert outcome is not None
    assert outcome.applied is False
    prompt_store.apply_patch.assert_not_called()


@pytest.mark.asyncio
async def test_cycle_rejects_patch_when_score_below_threshold():
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()
    prompt_store = AsyncMock()
    prompt_store.apply_patch = AsyncMock()
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)
    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(score=0.3))
    hermes = _make_hermes(collector=collector, generator=generator, evaluator=evaluator,
                          prompt_store=prompt_store, config=_make_config(auto_apply=True, threshold=0.7))
    outcome = await hermes.run_cycle("sql")
    assert outcome is not None
    assert outcome.applied is False
    prompt_store.apply_patch.assert_not_called()


@pytest.mark.asyncio
async def test_cycle_marginal_score_not_rejected_as_below_threshold():
    # auto_apply=True, score >= threshold but inside the regression-safety
    # margin (threshold <= score < threshold+0.15 and < 0.9). It must NOT be
    # rejected with a "Score < threshold" reason — that branch was a bug.
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()
    prompt_store = AsyncMock()
    prompt_store.apply_patch = AsyncMock()
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)
    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(score=0.75, test_cases=10, successes=8))
    hermes = _make_hermes(collector=collector, generator=generator, evaluator=evaluator,
                          prompt_store=prompt_store, config=_make_config(auto_apply=True, threshold=0.7))
    outcome = await hermes.run_cycle("sql")
    assert outcome is not None
    assert outcome.applied is False
    prompt_store.apply_patch.assert_not_called()
    assert "< threshold" not in outcome.reason
    assert patch.status == "pending"
    assert "margin" in outcome.reason.lower() or "manual review" in outcome.reason.lower()


@pytest.mark.asyncio
async def test_hermes_runs_rollback_check_before_cycle():
    """If online_monitor has a pending check, it should be evaluated first."""
    online_monitor = AsyncMock()
    online_monitor.check_and_maybe_rollback = AsyncMock(return_value=True)

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=2)  # below min_errors — cycle will skip

    hermes = _make_hermes(collector=collector, config=_make_config(min_errors=5))
    hermes._online_monitor = online_monitor

    outcome = await hermes.run_cycle("sql")

    online_monitor.check_and_maybe_rollback.assert_called_once_with(
        agent_type="sql",
        error_collector=collector,
        prompt_manager=hermes._prompt_store,
    )
    assert outcome is None  # skipped due to insufficient errors


@pytest.mark.asyncio
async def test_hermes_schedules_rollback_check_after_auto_apply():
    """After auto-apply, schedule_rollback_check must be called."""
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()

    prompt_store = AsyncMock()
    prompt_store.apply_patch = AsyncMock(return_value=MagicMock())
    prompt_store.get_version = AsyncMock(return_value=MagicMock(
        version_id="v_new", version_number=2
    ))

    online_monitor = AsyncMock()
    online_monitor.check_and_maybe_rollback = AsyncMock(return_value=False)
    online_monitor.schedule_rollback_check = AsyncMock()

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)

    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)

    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(
        score=0.9, test_cases=10, successes=9
    ))

    metrics = MagicMock()
    config = _make_config(auto_apply=True, threshold=0.7)
    hermes = HermesLoop(
        collector=collector, generator=generator, evaluator=evaluator,
        prompt_store=prompt_store, metrics=metrics, config=config,
        online_monitor=online_monitor,
    )

    outcome = await hermes.run_cycle("sql")

    assert outcome is not None
    assert outcome.applied is True
    online_monitor.schedule_rollback_check.assert_called_once()
    call_kwargs = online_monitor.schedule_rollback_check.call_args[1]
    assert call_kwargs["agent_type"] == "sql"
    assert call_kwargs["patch_id"] == patch.patch_id


@pytest.mark.asyncio
async def test_hermes_logs_cycle_to_mlflow():
    """MLflow tracer should be called at the end of every evaluated cycle."""
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()

    mlflow_tracer = AsyncMock()
    mlflow_tracer.log_hermes_cycle = AsyncMock()

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)

    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)

    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(
        score=0.5, test_cases=5, successes=3, failures=2
    ))

    metrics = MagicMock()
    config = _make_config(auto_apply=False)

    hermes = HermesLoop(
        collector=collector, generator=generator, evaluator=evaluator,
        prompt_store=AsyncMock(), metrics=metrics, config=config,
        mlflow_tracer=mlflow_tracer,
    )

    await hermes.run_cycle("sql")

    mlflow_tracer.log_hermes_cycle.assert_called_once()
    call_kwargs = mlflow_tracer.log_hermes_cycle.call_args[1]
    assert call_kwargs["agent_type"] == "sql"
    assert abs(call_kwargs["score"] - 0.5) < 0.01


@pytest.mark.asyncio
async def test_hermes_no_rollback_check_when_monitor_is_none():
    """HermesLoop with no online_monitor must work exactly as before."""
    errors = [_make_error() for _ in range(10)]
    patch = _make_patch()
    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    collector.get_recent = AsyncMock(return_value=errors)
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value=patch)
    evaluator = AsyncMock()
    evaluator.score = AsyncMock(return_value=MagicMock(score=0.8))

    hermes = _make_hermes(
        collector=collector, generator=generator, evaluator=evaluator,
        config=_make_config(auto_apply=False),
    )
    # online_monitor is None by default in _make_hermes

    outcome = await hermes.run_cycle("sql")
    assert outcome is not None
    assert outcome.patch is not None


def test_evaluator_scores_patch_correctly():
    result = EvalResult(
        patch_id="p1", test_cases=10, successes=8, failures=2,
        avg_steps_delta=2.0, avg_tokens_delta=100.0,
    )
    # score = 0.8 - 0.01*2 - 0.001*100 = 0.68
    assert abs(result.score - 0.68) < 0.01


@pytest.mark.asyncio
async def test_patch_generator_returns_valid_patch_json():
    from harness.improvement.patch_generator import PatchGenerator

    errors = [_make_error()]
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=MagicMock(
        content='{"op":"append","path":"","value":"Validate SQL first.","rationale":"reduce errors"}'
    ))
    mock_prompt_manager = AsyncMock()
    mock_prompt_manager.get_prompt = AsyncMock(return_value="Current prompt")

    generator = PatchGenerator(llm_provider=mock_llm, prompt_manager=mock_prompt_manager)
    patch = await generator.generate(agent_type="sql", errors=errors)

    assert isinstance(patch, Patch)
    assert patch.agent_type == "sql"
    assert patch.op in ("append", "replace", "remove", "add_example")


def _pe_pm_with_active(version_id="v-base"):
    pm = AsyncMock()
    pm.get_version = AsyncMock(return_value=MagicMock(version_id=version_id))
    pm.apply_patch = AsyncMock(return_value=MagicMock(version_id="v-new"))
    pm.promote = AsyncMock()
    pm.rollback = AsyncMock()
    return pm


def _pe_runner(baseline_sr, patched_sr):
    runner = AsyncMock()
    runner.run = AsyncMock(side_effect=[
        MagicMock(success_rate=baseline_sr),
        MagicMock(success_rate=patched_sr),
    ])
    return runner


@pytest.mark.asyncio
async def test_patch_evaluator_tie_restores_exact_baseline():
    # A tie (patched_sr == baseline_sr) must NOT leave the patch promoted; the
    # exact pre-existing version is restored via promote(baseline_id).
    pm = _pe_pm_with_active("v-base")
    pe = PatchEvaluator(eval_runner=_pe_runner(0.6, 0.6), prompt_manager=pm)
    score = await pe.score_patch(_make_patch(), dataset=MagicMock())
    assert score == pytest.approx(0.5)
    pm.promote.assert_called_once_with("v-base")
    pm.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_patch_evaluator_strict_improvement_keeps_patch():
    pm = _pe_pm_with_active("v-base")
    pe = PatchEvaluator(eval_runner=_pe_runner(0.6, 0.8), prompt_manager=pm)
    score = await pe.score_patch(_make_patch(), dataset=MagicMock())
    assert score > 0.5
    pm.promote.assert_not_called()
    pm.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_patch_evaluator_regression_restores_exact_baseline():
    pm = _pe_pm_with_active("v-base")
    pe = PatchEvaluator(eval_runner=_pe_runner(0.8, 0.5), prompt_manager=pm)
    score = await pe.score_patch(_make_patch(), dataset=MagicMock())
    assert score < 0.5
    pm.promote.assert_called_once_with("v-base")


@pytest.mark.asyncio
async def test_patch_evaluator_no_rollback_when_apply_fails_before_promote():
    # apply_patch raises before any new version is promoted — the active
    # version is untouched, so we must NOT promote/rollback anything.
    pm = _pe_pm_with_active("v-base")
    pm.apply_patch = AsyncMock(side_effect=RuntimeError("apply failed"))
    runner = AsyncMock()
    runner.run = AsyncMock(return_value=MagicMock(success_rate=0.7))
    pe = PatchEvaluator(eval_runner=runner, prompt_manager=pm)
    score = await pe.score_patch(_make_patch(), dataset=MagicMock())
    assert score == 0.0
    pm.promote.assert_not_called()
    pm.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_error_collector_trims_record_keys_not_just_index(redis_client):
    # Regression for orphaned record keys: when the index is trimmed past
    # max_records, the evicted record:<id> keys must be deleted too (and live
    # records carry a TTL), otherwise they leak forever.
    collector = ErrorCollector(redis_client, max_records=3)
    recs = []
    for _ in range(6):
        rec = await collector.record(
            agent_type="sql",
            task="t",
            failure_class="X",
            error_message="boom",
        )
        recs.append(rec)

    # Only the 3 newest remain indexed.
    assert await collector.count("sql") == 3

    # The 3 oldest record keys must have been deleted, not orphaned.
    oldest = recs[:3]
    for rec in oldest:
        assert await redis_client.get(_error_key(rec.record_id)) is None

    # Surviving records still carry a positive TTL.
    newest = recs[-1]
    assert await redis_client.get(_error_key(newest.record_id)) is not None
    assert await redis_client.ttl(_error_key(newest.record_id)) > 0
