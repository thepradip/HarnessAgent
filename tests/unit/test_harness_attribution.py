"""Tests for harness component attribution (#3) and extended Hermes patches (#4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.eval.failure_taxonomy import (
    HarnessComponent,
    attribute_to_component,
)


# ===========================================================================
# HarnessComponent / attribute_to_component
# ===========================================================================

@pytest.mark.parametrize("fc,expected", [
    ("BUDGET_STEPS",       HarnessComponent.BUDGET),
    ("BUDGET_TOKENS",      HarnessComponent.BUDGET),
    ("BUDGET_TIME",        HarnessComponent.BUDGET),
    ("TOOL_NOT_FOUND",     HarnessComponent.TOOL),
    ("TOOL_SCHEMA_ERROR",  HarnessComponent.TOOL),
    ("TOOL_EXEC_ERROR",    HarnessComponent.TOOL),
    ("TOOL_TIMEOUT",       HarnessComponent.TOOL),
    ("SAFETY_INPUT",       HarnessComponent.SAFETY),
    ("SAFETY_STEP",        HarnessComponent.SAFETY),
    ("SAFETY_OUTPUT",      HarnessComponent.SAFETY),
    ("LLM_RATE_LIMIT",     HarnessComponent.LLM),
    ("LLM_TIMEOUT",        HarnessComponent.LLM),
    ("LLM_ERROR",          HarnessComponent.LLM),
    ("INTER_AGENT_REJECT", HarnessComponent.HITL),
    ("UNKNOWN",            HarnessComponent.UNKNOWN),
    (None,                 HarnessComponent.UNKNOWN),
    ("",                   HarnessComponent.UNKNOWN),
    ("something_random",   HarnessComponent.UNKNOWN),
])
def test_attribute_to_component(fc, expected):
    assert attribute_to_component(fc) == expected


def test_attribute_to_component_prefix_fallback():
    assert attribute_to_component("BUDGET_CUSTOM") == HarnessComponent.BUDGET
    assert attribute_to_component("TOOL_SOMETHING_NEW") == HarnessComponent.TOOL
    assert attribute_to_component("SAFETY_FUTURE") == HarnessComponent.SAFETY
    assert attribute_to_component("LLM_FUTURE") == HarnessComponent.LLM


def test_attribute_to_component_case_insensitive():
    assert attribute_to_component("budget_steps") == HarnessComponent.BUDGET
    assert attribute_to_component("Tool_Timeout") == HarnessComponent.TOOL


# ===========================================================================
# AgentEvalReport.component_attribution
# ===========================================================================

def _make_scores(verdict, harness_failure_class=None):
    from harness.eval.agent_scorer import AgentScores
    return AgentScores(
        correctness_score=1.0 if verdict == "PASS" else 0.0,
        quality_score=1.0,
        safety_score=1.0,
        overall_score=1.0 if verdict == "PASS" else 0.0,
        hardness="easy",
        nondeterministic_warning=False,
        verdict=verdict,
        harness_failure_class=harness_failure_class,
    )


def test_component_attribution_empty():
    from harness.eval.agent_report import AgentEvalReport
    report = AgentEvalReport("test", [])
    assert report.component_attribution() == {}


def test_component_attribution_all_pass():
    from harness.eval.agent_report import AgentEvalReport
    scores = [_make_scores("PASS") for _ in range(5)]
    report = AgentEvalReport("test", scores)
    assert report.component_attribution() == {}


def test_component_attribution_mixed():
    from harness.eval.agent_report import AgentEvalReport
    scores = [
        _make_scores("FAIL", "TOOL_TIMEOUT"),
        _make_scores("FAIL", "TOOL_EXEC_ERROR"),
        _make_scores("FAIL", "BUDGET_STEPS"),
        _make_scores("FAIL", "SAFETY_STEP"),
        _make_scores("PASS"),
    ]
    report = AgentEvalReport("test", scores)
    attr = report.component_attribution()
    assert attr["tool"] == 2
    assert attr["budget"] == 1
    assert attr["safety"] == 1
    assert "tool" in attr   # tool is top failure component


def test_component_attribution_unknown_failure_class():
    from harness.eval.agent_report import AgentEvalReport
    scores = [_make_scores("FAIL", "SOME_UNKNOWN_CLASS")]
    report = AgentEvalReport("test", scores)
    attr = report.component_attribution()
    assert attr.get("unknown", 0) == 1


def test_component_attribution_none_failure_class():
    from harness.eval.agent_report import AgentEvalReport
    scores = [_make_scores("FAIL", None)]
    report = AgentEvalReport("test", scores)
    attr = report.component_attribution()
    assert attr.get("unknown", 0) == 1


def test_component_attribution_in_markdown():
    from harness.eval.agent_report import AgentEvalReport
    scores = [
        _make_scores("FAIL", "TOOL_TIMEOUT"),
        _make_scores("FAIL", "BUDGET_STEPS"),
        _make_scores("PASS"),
    ]
    report = AgentEvalReport("test", scores)
    md = report.to_markdown()
    assert "Harness Component Attribution" in md
    assert "tool" in md
    assert "budget" in md


def test_component_attribution_in_json():
    import json
    from harness.eval.agent_report import AgentEvalReport
    scores = [_make_scores("FAIL", "SAFETY_STEP"), _make_scores("PASS")]
    report = AgentEvalReport("test", scores)
    data = json.loads(report.to_json())
    assert "component_attribution" in data
    assert data["component_attribution"]["safety"] == 1


def test_harness_failure_class_in_scores_json():
    from harness.eval.agent_scorer import AgentScores
    s = AgentScores(
        correctness_score=0.0, quality_score=1.0, safety_score=1.0,
        overall_score=0.0, hardness="medium", nondeterministic_warning=False,
        verdict="FAIL", harness_failure_class="TOOL_TIMEOUT",
    )
    import json
    d = json.loads(s.to_json())
    assert d["harness_failure_class"] == "TOOL_TIMEOUT"


# ===========================================================================
# PatchGenerator.generate_retry_patch
# ===========================================================================

def _make_error(failure_class, message):
    from harness.improvement.error_collector import ErrorRecord
    return ErrorRecord(
        agent_type="code",
        failure_class=failure_class,
        error_message=message,
        task="test task",
    )


@pytest.mark.asyncio
async def test_generate_retry_patch_identifies_slow_tool():
    from harness.improvement.patch_generator import PatchGenerator
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [
        _make_error("TOOL_TIMEOUT", "Tool 'run_python' timed out after 30.0s"),
        _make_error("TOOL_TIMEOUT", "Tool 'run_python' timed out after 30.0s"),
    ]
    patch = await gen.generate_retry_patch("code", errors)
    assert patch is not None
    assert patch.target == "retry_config"
    assert patch.path == "run_python"
    assert float(patch.value) == 60.0   # doubled from default 30s


@pytest.mark.asyncio
async def test_generate_retry_patch_caps_at_120s():
    from harness.improvement.patch_generator import PatchGenerator
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [_make_error("TOOL_TIMEOUT", "Tool 'slow_tool' timed out after 30.0s")]

    mock_registry = MagicMock()
    mock_tool = MagicMock()
    mock_tool.timeout_seconds = 90.0
    mock_registry.get = MagicMock(return_value=mock_tool)

    patch = await gen.generate_retry_patch("code", errors, tool_registry=mock_registry)
    assert float(patch.value) == 120.0  # capped at 120


@pytest.mark.asyncio
async def test_generate_retry_patch_no_timeout_errors_returns_none():
    from harness.improvement.patch_generator import PatchGenerator
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [_make_error("TOOL_EXEC_ERROR", "Tool raised exception")]
    patch = await gen.generate_retry_patch("code", errors)
    assert patch is None


@pytest.mark.asyncio
async def test_generate_retry_patch_unrecognised_message_returns_none():
    from harness.improvement.patch_generator import PatchGenerator
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [_make_error("TOOL_TIMEOUT", "some tool timed out (no quoted name)")]
    patch = await gen.generate_retry_patch("code", errors)
    assert patch is None


@pytest.mark.asyncio
async def test_generate_retry_patch_saved_to_store():
    from harness.improvement.patch_generator import PatchGenerator
    mock_store = AsyncMock()
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock(),
                         patch_store=mock_store)
    errors = [_make_error("TOOL_TIMEOUT", "Tool 'run_python' timed out after 30.0s")]
    await gen.generate_retry_patch("code", errors)
    mock_store.save.assert_called_once()


# ===========================================================================
# PatchGenerator.generate_permission_patch
# ===========================================================================

@pytest.mark.asyncio
async def test_generate_permission_patch_identifies_blocked_tool():
    from harness.improvement.patch_generator import PatchGenerator
    import json
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [
        _make_error("SAFETY_STEP", "Tool 'drop_table' is blocked by tenant policy"),
        _make_error("SAFETY_STEP", "Tool 'drop_table' is blocked by tenant policy"),
        _make_error("SAFETY_STEP", "Tool 'drop_table' blocked: policy violation"),
    ]
    patch = await gen.generate_permission_patch("sql", errors)
    assert patch is not None
    assert patch.target == "permission"
    assert patch.op == "set"
    value = json.loads(patch.value)
    assert value["add_to_blocked_tools"] == "drop_table"
    assert value["violation_count"] == 3


@pytest.mark.asyncio
async def test_generate_permission_patch_no_safety_errors_returns_none():
    from harness.improvement.patch_generator import PatchGenerator
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [_make_error("TOOL_EXEC_ERROR", "Tool failed")]
    patch = await gen.generate_permission_patch("code", errors)
    assert patch is None


@pytest.mark.asyncio
async def test_generate_permission_patch_generic_when_tool_unknown():
    from harness.improvement.patch_generator import PatchGenerator
    import json
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock())
    errors = [_make_error("SAFETY_OUTPUT", "Output blocked — PII detected")]
    patch = await gen.generate_permission_patch("code", errors)
    assert patch is not None
    value = json.loads(patch.value)
    assert "recommendation" in value


@pytest.mark.asyncio
async def test_generate_permission_patch_saved_to_store():
    from harness.improvement.patch_generator import PatchGenerator
    mock_store = AsyncMock()
    gen = PatchGenerator(llm_provider=MagicMock(), prompt_manager=MagicMock(),
                         patch_store=mock_store)
    errors = [_make_error("SAFETY_STEP", "Tool 'delete_file' blocked by tenant policy")]
    await gen.generate_permission_patch("code", errors)
    mock_store.save.assert_called_once()
