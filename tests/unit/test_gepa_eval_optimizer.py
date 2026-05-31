"""Tests for the EvalDataset-backed (gold-scored) GEPA optimizer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.core.prompt_overrides import OVERRIDES_KEY, gepa_override
from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.runner import EvalRunner
from harness.improvement.gepa import optimize_prompts_on_dataset
from harness.improvement.gepa.eval_adapter import EvalDatasetGepaAdapter

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _PromptHonoringRunner:
    """Fake agent runner: the gold output is produced only when the injected
    system prompt contains the rule 'validate' — i.e. correctness depends on the
    candidate prompt, so GEPA must evolve toward it."""

    async def run(self, tenant_id=None, agent_type=None, task=None, metadata=None):
        overrides = (metadata or {}).get(OVERRIDES_KEY, {})
        sys_prompt = overrides.get("system_prompt", "")
        ok = "validate" in sys_prompt.lower()
        return SimpleNamespace(
            output="VALIDATED_OK" if ok else "missing",
            success=ok,
            steps=1,
            tokens=50,
            cost_usd=0.0,
            elapsed_seconds=0.01,
            tool_calls=0,
            tool_errors=0,
            guardrail_hits=0,
            handoff_count=0,
            cache_hits=0,
            cache_read_tokens=0,
            failure_class=None,
        )


def _dataset(n: int = 3) -> EvalDataset:
    return EvalDataset(
        name="d",
        agent_type="sql",
        cases=[
            EvalCase(
                case_id=f"c{i}",
                agent_type="sql",
                task=f"question {i}",
                expected_output="VALIDATED_OK",
            )
            for i in range(n)
        ],
    )


def _improving_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=SimpleNamespace(
            content="You are a SQL agent. Always validate the query before answering."
        )
    )
    return llm


def _sync_runner(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Override plumbing
# ---------------------------------------------------------------------------


def test_gepa_override_reads_injected_value():
    ctx = SimpleNamespace(metadata={OVERRIDES_KEY: {"system_prompt": "OVERRIDDEN"}})
    assert gepa_override(ctx, "system_prompt", "default") == "OVERRIDDEN"


def test_gepa_override_falls_back():
    assert gepa_override(SimpleNamespace(metadata={}), "system_prompt", "default") == "default"
    assert gepa_override(SimpleNamespace(metadata=None), "x", "default") == "default"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_eval_adapter_scores_per_case_against_gold():
    runner = EvalRunner(_PromptHonoringRunner())
    adapter = EvalDatasetGepaAdapter(runner, _sync_runner)
    cases = _dataset(3).cases

    good = adapter.evaluate(cases, {"system_prompt": "Always validate."}, capture_traces=True)
    bad = adapter.evaluate(cases, {"system_prompt": "Do nothing special."}, capture_traces=False)

    assert good.scores == [1.0, 1.0, 1.0]  # gold matched
    assert bad.scores == [0.0, 0.0, 0.0]   # gold missed
    assert good.trajectories is not None and good.trajectories[0]["passed"] is True


def test_eval_adapter_does_not_mutate_original_cases():
    runner = EvalRunner(_PromptHonoringRunner())
    adapter = EvalDatasetGepaAdapter(runner, _sync_runner)
    cases = _dataset(2).cases
    adapter.evaluate(cases, {"system_prompt": "Always validate."}, capture_traces=False)
    assert all(OVERRIDES_KEY not in (c.metadata or {}) for c in cases)


# ---------------------------------------------------------------------------
# End-to-end optimization (real gepa.optimize)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optimize_against_eval_dataset_improves_correctness():
    runner = EvalRunner(_PromptHonoringRunner())
    result = await optimize_prompts_on_dataset(
        eval_runner=runner,
        dataset=_dataset(3),
        llm_provider=_improving_llm(),
        seed_prompts={"system_prompt": "You are a SQL agent."},
        budget=12,
    )
    assert result.improved
    assert "validate" in result.components["system_prompt"].lower()
    assert result.best_score == 1.0


@pytest.mark.asyncio
async def test_optimize_multi_component_seed_roundtrips():
    """Multiple components are carried through; only system_prompt is wired to a
    runtime read here, but the candidate dict round-trips intact."""
    runner = EvalRunner(_PromptHonoringRunner())
    result = await optimize_prompts_on_dataset(
        eval_runner=runner,
        dataset=_dataset(2),
        llm_provider=_improving_llm(),
        seed_prompts={
            "system_prompt": "You are a SQL agent.",
            "handoff_prompt": "Hand off when stuck.",
        },
        budget=8,
    )
    assert set(result.components.keys()) == {"system_prompt", "handoff_prompt"}
    assert set(result.seed.keys()) == {"system_prompt", "handoff_prompt"}


@pytest.mark.asyncio
async def test_optimize_requires_seed():
    with pytest.raises(ValueError):
        await optimize_prompts_on_dataset(
            eval_runner=MagicMock(), dataset=_dataset(1), llm_provider=MagicMock(),
            seed_prompts={},
        )
