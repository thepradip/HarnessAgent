"""Tests for HumanEval-style code-execution (pass@1) scoring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.runner import EvalRunner, _invoke_scorer
from harness.eval.scorers import score_code_execution


class _FakeSandbox:
    """Records the assembled program and reports success/failure deterministically."""

    def __init__(self, succeed: bool):
        self._succeed = succeed
        self.last_program: str | None = None

    async def execute(self, action, language="python", **_):
        self.last_program = action
        return SimpleNamespace(
            success=self._succeed,
            error=None if self._succeed else "AssertionError: failed",
            raw_text="",
        )


def _humaneval_case(output_irrelevant: bool = False) -> EvalCase:
    return EvalCase(
        case_id="he1",
        agent_type="code",
        task="def add(a, b):\n    '''add two numbers'''\n",
        expected_output="    return a + b",
        sandbox_type="code",
        metadata={
            "entry_point": "add",
            "prompt": "def add(a, b):\n    '''add two numbers'''\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        },
    )


# ---------------------------------------------------------------------------
# score_code_execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_pass():
    sandbox = _FakeSandbox(succeed=True)
    result = await score_code_execution("def add(a, b):\n    return a + b", _humaneval_case(), sandbox)
    assert result.score == 1.0
    # The assembled program includes the test harness and the check() call.
    assert "def check(candidate)" in sandbox.last_program
    assert "check(add)" in sandbox.last_program


@pytest.mark.asyncio
async def test_execution_fail():
    sandbox = _FakeSandbox(succeed=False)
    result = await score_code_execution("def add(a, b):\n    return a - b", _humaneval_case(), sandbox)
    assert result.score == 0.0
    assert "failed" in result.details.lower()


@pytest.mark.asyncio
async def test_execution_prepends_prompt_for_bare_completion():
    """If the model returns only the body (no 'def add'), the prompt is prepended."""
    sandbox = _FakeSandbox(succeed=True)
    await score_code_execution("    return a + b", _humaneval_case(), sandbox)
    assert sandbox.last_program.startswith("def add(a, b):")


@pytest.mark.asyncio
async def test_execution_strips_markdown_fences():
    sandbox = _FakeSandbox(succeed=True)
    await score_code_execution("```python\ndef add(a, b):\n    return a + b\n```", _humaneval_case(), sandbox)
    assert "```" not in sandbox.last_program


@pytest.mark.asyncio
async def test_execution_no_test_harness_scores_zero():
    case = EvalCase(case_id="x", agent_type="code", task="t", metadata={"entry_point": "f"})
    result = await score_code_execution("def f(): pass", case, _FakeSandbox(succeed=True))
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_execution_sandbox_error_scores_zero():
    class _Boom:
        async def execute(self, *a, **k):
            raise RuntimeError("docker down")

    result = await score_code_execution("def add(): pass", _humaneval_case(), _Boom())
    assert result.score == 0.0
    assert "sandbox error" in result.details.lower()


# ---------------------------------------------------------------------------
# _invoke_scorer arity routing
# ---------------------------------------------------------------------------


def test_invoke_scorer_two_arg():
    case = _humaneval_case()
    assert _invoke_scorer(lambda o, e: 0.5, "o", "e", case) == 0.5


def test_invoke_scorer_three_arg_receives_case():
    case = _humaneval_case()
    seen = {}

    def scorer(output, expected, c):
        seen["case"] = c
        return 0.9

    assert _invoke_scorer(scorer, "o", "e", case) == 0.9
    assert seen["case"] is case


# ---------------------------------------------------------------------------
# End-to-end through EvalRunner (real _score / _invoke_scorer path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eval_runner_routes_case_to_execution_scorer():
    class _Runner:
        async def run(self, tenant_id=None, agent_type=None, task=None, metadata=None):
            # Agent "generates" a correct implementation.
            return SimpleNamespace(output="def add(a, b):\n    return a + b", success=True)

    sandbox = _FakeSandbox(succeed=True)

    async def scorer(output, expected, case):
        return (await score_code_execution(output, case, sandbox)).score

    dataset = EvalDataset(name="he", agent_type="code", cases=[_humaneval_case()])
    report = await EvalRunner(_Runner()).run(dataset, scorer=scorer)

    assert report.scores["he1"] == 1.0
    assert "check(add)" in sandbox.last_program  # the real path assembled + "ran" it


# ---------------------------------------------------------------------------
# Loader captures the test harness
# ---------------------------------------------------------------------------


def test_load_humaneval_captures_test_harness(tmp_path):
    from harness.eval.benchmark_loaders import load_humaneval

    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
    }
    path = tmp_path / "he.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    ds = load_humaneval(str(path))
    assert len(ds.cases) == 1
    meta = ds.cases[0].metadata
    assert meta["test"].startswith("def check")
    assert meta["entry_point"] == "add"
    assert meta["prompt"] == "def add(a, b):\n"
