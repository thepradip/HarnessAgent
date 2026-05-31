"""Tests for the evaluator-backed HermesLoop assembly in the Hermes worker."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from harness.improvement.evaluator import Evaluator
from harness.improvement.gepa import GepaPatchGenerator
from harness.improvement.hermes import HermesLoop
from harness.improvement.patch_generator import PatchGenerator
from harness.workers.hermes_worker import build_hermes_loop


def _cfg():
    return SimpleNamespace(
        workspace_base_path="/tmp/ws",
        hermes_gepa_use_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name=None,
        hermes_patch_score_threshold=0.7,
        hermes_auto_apply=False,
    )


def _build(strategy: str) -> HermesLoop:
    llm = MagicMock()
    llm.complete = AsyncMock()
    return build_hermes_loop(
        redis_client=MagicMock(),
        cfg=_cfg(),
        llm_provider=llm,
        error_collector=MagicMock(),
        prompt_manager=MagicMock(),
        patch_store=MagicMock(),
        strategy=strategy,
        agent_factory=lambda agent_type: MagicMock(),  # stub: no real agents built
    )


def test_build_hermes_loop_wires_gepa_with_evaluator():
    hermes = _build("gepa")

    assert isinstance(hermes, HermesLoop)
    assert isinstance(hermes._evaluator, Evaluator)
    # GEPA generator is selected and shares the loop's evaluator as its metric.
    assert isinstance(hermes._generator, GepaPatchGenerator)
    assert hermes._generator._evaluator is hermes._evaluator
    # The loop applies patches through the prompt manager.
    assert hermes._prompt_store is not None
    assert hermes._online_monitor is not None


def test_build_hermes_loop_heuristic_strategy():
    hermes = _build("heuristic")

    assert isinstance(hermes, HermesLoop)
    assert isinstance(hermes._generator, PatchGenerator)
    assert not isinstance(hermes._generator, GepaPatchGenerator)
    # Evaluator is still wired so the loop can score heuristic patches too.
    assert isinstance(hermes._evaluator, Evaluator)


def test_build_hermes_loop_uses_default_factory_when_none(monkeypatch):
    """When no agent_factory is passed, the production builder is used."""
    called = {}

    def fake_build_agent_factory(cfg):
        called["cfg"] = cfg
        return lambda agent_type: MagicMock()

    monkeypatch.setattr(
        "harness.workers.agent_worker.build_agent_factory", fake_build_agent_factory
    )

    llm = MagicMock()
    llm.complete = AsyncMock()
    hermes = build_hermes_loop(
        redis_client=MagicMock(),
        cfg=_cfg(),
        llm_provider=llm,
        error_collector=MagicMock(),
        prompt_manager=MagicMock(),
        patch_store=MagicMock(),
        strategy="gepa",
    )
    assert isinstance(hermes, HermesLoop)
    assert "cfg" in called  # the default factory builder was invoked
