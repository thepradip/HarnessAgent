"""Tests for PEV verifier protocol, implementations, and BaseAgent integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.verification.verifier import (
    CodeExitCodeVerifier,
    ExpectedOutputVerifier,
    NoopVerifier,
    VerificationResult,
    Verifier,
)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

def test_result_passed_auto_sets_score():
    r = VerificationResult(passed=True, verdict="correct", feedback="")
    assert r.score == 1.0


def test_result_failed_score_stays_zero():
    r = VerificationResult(passed=False, verdict="incorrect", feedback="bad")
    assert r.score == 0.0


def test_result_skipped_factory():
    r = VerificationResult.skipped()
    assert r.passed is True
    assert r.verdict == "skipped"
    assert r.score == 1.0


def test_result_correct_factory():
    r = VerificationResult.correct()
    assert r.passed is True
    assert r.score == 1.0


def test_result_incorrect_factory():
    r = VerificationResult.incorrect("something broke", score=0.1)
    assert r.passed is False
    assert r.score == 0.1
    assert "something broke" in r.feedback


def test_result_partial_factory():
    r = VerificationResult.partial("half done")
    assert r.passed is False
    assert r.score == 0.5


def test_verifier_protocol_check():
    assert isinstance(NoopVerifier(), Verifier)
    assert isinstance(CodeExitCodeVerifier(), Verifier)


# ---------------------------------------------------------------------------
# NoopVerifier
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_noop_verifier_always_passes():
    v = NoopVerifier()
    ctx = MagicMock()
    ctx.metadata = {}
    result = await v.verify(ctx, "anything", [])
    assert result.passed is True
    assert result.verdict == "skipped"


# ---------------------------------------------------------------------------
# CodeExitCodeVerifier
# ---------------------------------------------------------------------------

def _ctx_with_code_result(exit_code=0, stderr="", stdout="", timed_out=False):
    ctx = MagicMock()
    ctx.metadata = {
        "last_code_result": {
            "exit_code": exit_code,
            "stderr": stderr,
            "stdout": stdout,
            "timed_out": timed_out,
        }
    }
    ctx.run_id = "test-run"
    return ctx


@pytest.mark.asyncio
async def test_code_verifier_passes_on_exit_0():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(exit_code=0, stdout="hello")
    r = await v.verify(ctx, "hello", [])
    assert r.passed is True
    assert r.verdict == "correct"


@pytest.mark.asyncio
async def test_code_verifier_fails_on_nonzero_exit():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(exit_code=1, stderr="NameError: name 'x' is not defined")
    r = await v.verify(ctx, "", [])
    assert r.passed is False
    assert "exit 1" in r.feedback
    assert "NameError" in r.feedback


@pytest.mark.asyncio
async def test_code_verifier_fails_on_timeout():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(exit_code=-1, timed_out=True)
    r = await v.verify(ctx, "", [])
    assert r.passed is False
    assert "timed out" in r.feedback.lower()


@pytest.mark.asyncio
async def test_code_verifier_fails_on_oom():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(exit_code=137, timed_out=False, stderr="Killed")
    r = await v.verify(ctx, "", [])
    assert r.passed is False
    assert "OOM" in r.feedback or "137" in r.feedback


@pytest.mark.asyncio
async def test_code_verifier_fails_on_stderr_exception_despite_exit_0():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(
        exit_code=0,
        stderr="Traceback (most recent call last):\n  File 'x.py', line 1\nAssertionError",
    )
    r = await v.verify(ctx, "", [])
    assert r.passed is False
    assert "exception" in r.feedback.lower()


@pytest.mark.asyncio
async def test_code_verifier_passes_exit_0_clean():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(exit_code=0, stdout="42", stderr="")
    r = await v.verify(ctx, "42", [])
    assert r.passed is True


@pytest.mark.asyncio
async def test_code_verifier_skips_when_no_result():
    v = CodeExitCodeVerifier()
    ctx = MagicMock()
    ctx.metadata = {}
    r = await v.verify(ctx, "some output", [])
    assert r.passed is True
    assert r.verdict == "skipped"


@pytest.mark.asyncio
async def test_code_verifier_skips_when_no_metadata():
    v = CodeExitCodeVerifier()
    ctx = MagicMock(spec=[])  # no metadata attribute
    r = await v.verify(ctx, "output", [])
    assert r.passed is True
    assert r.verdict == "skipped"


@pytest.mark.asyncio
async def test_code_verifier_nonzero_no_stderr_flags_silent_failure():
    v = CodeExitCodeVerifier()
    ctx = _ctx_with_code_result(exit_code=2, stderr="", stdout="")
    r = await v.verify(ctx, "", [])
    assert r.passed is False
    assert "silent" in r.feedback.lower() or "exit" in r.feedback.lower()


# ---------------------------------------------------------------------------
# ExpectedOutputVerifier
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expected_output_substring_match():
    v = ExpectedOutputVerifier(expected="hello")
    ctx = _ctx_with_code_result(exit_code=0, stdout="hello world")
    r = await v.verify(ctx, "hello world", [])
    assert r.passed is True


@pytest.mark.asyncio
async def test_expected_output_substring_missing():
    v = ExpectedOutputVerifier(expected="goodbye")
    ctx = _ctx_with_code_result(exit_code=0, stdout="hello world")
    r = await v.verify(ctx, "hello world", [])
    assert r.passed is False
    assert "goodbye" in r.feedback


@pytest.mark.asyncio
async def test_expected_output_exact_match():
    v = ExpectedOutputVerifier(expected="42", exact=True)
    ctx = _ctx_with_code_result(exit_code=0, stdout="42\n")
    r = await v.verify(ctx, "42", [])
    assert r.passed is True


@pytest.mark.asyncio
async def test_expected_output_exact_mismatch():
    v = ExpectedOutputVerifier(expected="42", exact=True)
    ctx = _ctx_with_code_result(exit_code=0, stdout="42 is the answer")
    r = await v.verify(ctx, "", [])
    assert r.passed is False


@pytest.mark.asyncio
async def test_expected_output_case_insensitive():
    v = ExpectedOutputVerifier(expected="HELLO", case_sensitive=False)
    ctx = _ctx_with_code_result(exit_code=0, stdout="hello world")
    r = await v.verify(ctx, "", [])
    assert r.passed is True


# ---------------------------------------------------------------------------
# BaseAgent._verify_output integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_base_agent_verify_output_noop_when_no_verifier(agent_context):
    from harness.agents.base import BaseAgent
    agent = BaseAgent(
        llm_router=MagicMock(), memory_manager=None, tool_registry=None,
        safety_pipeline=None, step_tracer=None, mlflow_tracer=None,
        failure_tracker=None, audit_logger=None, event_bus=None,
        cost_tracker=None, checkpoint_manager=None,
    )
    ctx = agent_context()
    # No verifier in metadata — should skip
    result = await agent._verify_output(ctx, "output", [])
    assert result.passed is True
    assert result.verdict == "skipped"


@pytest.mark.asyncio
async def test_base_agent_verify_output_uses_configured_verifier(agent_context):
    from harness.agents.base import BaseAgent
    agent = BaseAgent(
        llm_router=MagicMock(), memory_manager=None, tool_registry=None,
        safety_pipeline=None, step_tracer=None, mlflow_tracer=None,
        failure_tracker=None, audit_logger=None, event_bus=None,
        cost_tracker=None, checkpoint_manager=None,
    )
    ctx = agent_context()
    ctx.metadata["last_code_result"] = {"exit_code": 1, "stderr": "Error", "stdout": "", "timed_out": False}
    ctx.metadata["verifier"] = CodeExitCodeVerifier()
    result = await agent._verify_output(ctx, "bad", [])
    assert result.passed is False


@pytest.mark.asyncio
async def test_base_agent_verify_output_survives_verifier_exception(agent_context):
    from harness.agents.base import BaseAgent
    agent = BaseAgent(
        llm_router=MagicMock(), memory_manager=None, tool_registry=None,
        safety_pipeline=None, step_tracer=None, mlflow_tracer=None,
        failure_tracker=None, audit_logger=None, event_bus=None,
        cost_tracker=None, checkpoint_manager=None,
    )
    ctx = agent_context()

    class BrokenVerifier:
        async def verify(self, ctx, output, history):
            raise RuntimeError("verifier crashed")

    ctx.metadata["verifier"] = BrokenVerifier()
    # Must not propagate the exception
    result = await agent._verify_output(ctx, "output", [])
    assert result.passed is True
    assert result.verdict == "skipped"


@pytest.mark.asyncio
async def test_base_agent_injects_feedback_on_verification_failure(agent_context):
    """When verification fails, feedback is injected and the loop continues.

    Flow:
      1. LLM → tool call (run_python, exit 1) → continue loop
      2. LLM → no tool call → verify (exit 1 → FAIL) → inject feedback
      3. LLM → tool call (run_python, exit 0) → continue loop
      4. LLM → no tool call → verify (exit 0 → PASS) → break
    """
    from harness.agents.base import BaseAgent
    from harness.core.context import LLMResponse, ToolCall, ToolResult

    call = ToolCall(id="c1", name="run_python", args={"code": "x = 1"})
    call2 = ToolCall(id="c2", name="run_python", args={"code": "x = 2"})

    responses = [
        LLMResponse(content="trying", tool_calls=[call],
                    input_tokens=10, output_tokens=10, model="m", provider="p"),
        LLMResponse(content="checking", tool_calls=[],       # verify fails here
                    input_tokens=10, output_tokens=10, model="m", provider="p"),
        LLMResponse(content="fixing", tool_calls=[call2],   # re-runs after feedback
                    input_tokens=10, output_tokens=10, model="m", provider="p"),
        LLMResponse(content="done", tool_calls=[],           # verify passes
                    input_tokens=10, output_tokens=10, model="m", provider="p"),
    ]

    router = MagicMock()
    router.complete = AsyncMock(side_effect=responses)

    mock_registry = MagicMock()
    mock_registry.to_anthropic_format = MagicMock(return_value=[])
    call_count = 0

    async def execute(ctx, c):
        nonlocal call_count
        call_count += 1
        exit_code = 1 if call_count == 1 else 0
        return ToolResult(data={
            "exit_code": exit_code,
            "stdout": "",
            "stderr": "NameError: bad" if exit_code else "",
            "timed_out": False,
        })

    mock_registry.execute = AsyncMock(side_effect=execute)

    checkpoint = MagicMock()
    checkpoint.load = AsyncMock(return_value=None)
    checkpoint.save = AsyncMock()
    cost = MagicMock()
    cost.record = AsyncMock(return_value=MagicMock(cost_usd=0.0))

    agent = BaseAgent(
        llm_router=router, memory_manager=None, tool_registry=mock_registry,
        safety_pipeline=None, step_tracer=None, mlflow_tracer=None,
        failure_tracker=None, audit_logger=None, event_bus=None,
        cost_tracker=cost, checkpoint_manager=checkpoint,
    )
    ctx = agent_context()
    ctx.metadata["verifier"] = CodeExitCodeVerifier()
    ctx.max_steps = 10

    result = await agent.run(ctx)
    assert result.success is True
    assert call_count == 2  # two tool executions — one failed, one passed


# ---------------------------------------------------------------------------
# HITL rejection → policy update
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hitl_rejection_updates_policy():
    import fakeredis.aioredis as fakeredis
    from harness.orchestrator.hitl import HITLManager
    from harness.safety.policies import HarnessPolicy, PolicyStore

    redis = fakeredis.FakeRedis(decode_responses=True)
    policy_store = PolicyStore(redis)

    # Initial policy — tool not blocked
    await policy_store.set(HarnessPolicy(tenant_id="acme", blocked_tools=[]))

    mgr = HITLManager(
        redis=redis,
        policy_store=policy_store,
        learn_from_rejections=True,
    )

    req = await mgr.request_approval(
        run_id="run1",
        tenant_id="acme",
        tool_name="drop_table",
        tool_args={"table": "users"},
        reason="risky op",
    )

    await mgr.reject(req.request_id, resolved_by="alice")

    # Policy should now block the tool
    updated_policy = await policy_store.get("acme")
    assert "drop_table" in updated_policy.blocked_tools


@pytest.mark.asyncio
async def test_hitl_rejection_does_not_duplicate_blocked_tool():
    import fakeredis.aioredis as fakeredis
    from harness.orchestrator.hitl import HITLManager
    from harness.safety.policies import HarnessPolicy, PolicyStore

    redis = fakeredis.FakeRedis(decode_responses=True)
    policy_store = PolicyStore(redis)
    await policy_store.set(HarnessPolicy(tenant_id="t1", blocked_tools=["drop_table"]))

    mgr = HITLManager(redis=redis, policy_store=policy_store, learn_from_rejections=True)
    req = await mgr.request_approval(
        run_id="r1", tenant_id="t1", tool_name="drop_table", tool_args={},
    )
    await mgr.reject(req.request_id)

    policy = await policy_store.get("t1")
    assert policy.blocked_tools.count("drop_table") == 1


@pytest.mark.asyncio
async def test_hitl_rejection_no_policy_store_no_error():
    import fakeredis.aioredis as fakeredis
    from harness.orchestrator.hitl import HITLManager

    redis = fakeredis.FakeRedis(decode_responses=True)
    # No policy_store, learn_from_rejections=True — should not crash
    mgr = HITLManager(redis=redis, learn_from_rejections=True)
    req = await mgr.request_approval(
        run_id="r1", tenant_id="t1", tool_name="some_tool", tool_args={},
    )
    await mgr.reject(req.request_id)  # must not raise


@pytest.mark.asyncio
async def test_hitl_rejection_learn_disabled_does_not_update_policy():
    import fakeredis.aioredis as fakeredis
    from harness.orchestrator.hitl import HITLManager
    from harness.safety.policies import HarnessPolicy, PolicyStore

    redis = fakeredis.FakeRedis(decode_responses=True)
    policy_store = PolicyStore(redis)
    await policy_store.set(HarnessPolicy(tenant_id="t1", blocked_tools=[]))

    # learn_from_rejections=False (default)
    mgr = HITLManager(redis=redis, policy_store=policy_store, learn_from_rejections=False)
    req = await mgr.request_approval(
        run_id="r1", tenant_id="t1", tool_name="risky_tool", tool_args={},
    )
    await mgr.reject(req.request_id)

    policy = await policy_store.get("t1")
    assert "risky_tool" not in policy.blocked_tools


@pytest.mark.asyncio
async def test_hitl_approval_does_not_update_policy():
    import fakeredis.aioredis as fakeredis
    from harness.orchestrator.hitl import HITLManager
    from harness.safety.policies import HarnessPolicy, PolicyStore

    redis = fakeredis.FakeRedis(decode_responses=True)
    policy_store = PolicyStore(redis)
    await policy_store.set(HarnessPolicy(tenant_id="t1", blocked_tools=[]))

    mgr = HITLManager(redis=redis, policy_store=policy_store, learn_from_rejections=True)
    req = await mgr.request_approval(
        run_id="r1", tenant_id="t1", tool_name="safe_tool", tool_args={},
    )
    await mgr.approve(req.request_id)

    policy = await policy_store.get("t1")
    assert "safe_tool" not in policy.blocked_tools
