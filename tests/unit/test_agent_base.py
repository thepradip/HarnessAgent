"""Unit tests for BaseAgent lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.agents.base import BaseAgent
from harness.core.context import AgentResult, LLMResponse, ToolCall, ToolResult


def _make_llm_response(content="Done", tool_calls=None, tokens=30):
    return LLMResponse(
        content=content, tool_calls=tool_calls or [],
        input_tokens=tokens // 2, output_tokens=tokens // 2,
        model="mock-model", provider="mock",
    )


def _mock_memory():
    m = AsyncMock()
    m.push_message = AsyncMock()
    m.get_history = AsyncMock(return_value=[])
    m.fit_history = AsyncMock(return_value=MagicMock(messages=[], truncated=False, summary=None))
    m.smart_retrieve = AsyncMock(return_value=MagicMock(
        graph_context="", vector_context=[], total_tokens_estimate=0
    ))
    return m


def _make_agent(router=None, tool_registry=None, memory=None,
                failure_tracker=None, cost_tracker=None, checkpoint_manager=None,
                trace_recorder=None):
    router = router or AsyncMock()
    if not hasattr(router.complete, 'return_value') or router.complete.return_value is None:
        router.complete = AsyncMock(return_value=_make_llm_response())

    tool_registry = tool_registry or AsyncMock()
    memory = memory or _mock_memory()

    failure_tracker = failure_tracker or AsyncMock()
    failure_tracker.record = AsyncMock()

    if cost_tracker is None:
        cost_tracker = AsyncMock()
        cost_tracker.record = AsyncMock(return_value=MagicMock(cost_usd=0.001))

    checkpoint_manager = checkpoint_manager or AsyncMock()
    checkpoint_manager.exists = AsyncMock(return_value=False)
    checkpoint_manager.load = AsyncMock(return_value=None)   # None = no checkpoint to restore
    checkpoint_manager.save = AsyncMock()

    return BaseAgent(
        llm_router=router,
        memory_manager=memory,
        tool_registry=tool_registry,
        safety_pipeline=None,
        step_tracer=None,
        mlflow_tracer=None,
        failure_tracker=failure_tracker,
        audit_logger=None,
        event_bus=None,
        cost_tracker=cost_tracker,
        checkpoint_manager=checkpoint_manager,
        message_bus=None,
        trace_recorder=trace_recorder,
    )


def _make_ctx(agent_context, memory=None):
    mem = memory or _mock_memory()
    return agent_context(memory=mem)


@pytest.mark.asyncio
async def test_run_completes_with_no_tool_calls(agent_context):
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("Final answer"))
    agent = _make_agent(router=router)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert isinstance(result, AgentResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_run_executes_tool_calls(agent_context):
    call = ToolCall(id="c1", name="echo", args={"message": "hi"})
    responses = [
        _make_llm_response("calling tool", tool_calls=[call]),
        _make_llm_response("Done"),
    ]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={"echo": "hi"}))
    agent = _make_agent(router=router, tool_registry=mock_registry)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is True
    mock_registry.execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_respects_step_budget(agent_context):
    call = ToolCall(id="c1", name="noop", args={})
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("looping", tool_calls=[call]))
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))
    agent = _make_agent(router=router, tool_registry=mock_registry)
    ctx = _make_ctx(agent_context)
    ctx.max_steps = 3
    result = await agent.run(ctx)
    assert result.success is False


@pytest.mark.asyncio
async def test_run_respects_token_budget(agent_context):
    call = ToolCall(id="c1", name="noop", args={})
    router = AsyncMock()
    router.complete = AsyncMock(return_value=LLMResponse(
        content="looping", tool_calls=[call],
        input_tokens=2500, output_tokens=2500, model="mock", provider="mock",
    ))
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))
    agent = _make_agent(router=router, tool_registry=mock_registry)
    ctx = _make_ctx(agent_context)
    ctx.max_tokens = 6000
    result = await agent.run(ctx)
    assert result.success is False


@pytest.mark.asyncio
async def test_safety_violation_stops_run(agent_context):
    guardrail_result = pytest.importorskip("guardrail.result", reason="guardrail not installed")
    Decision = guardrail_result.Decision
    GuardResult = guardrail_result.GuardResult
    mock_pipeline = MagicMock()
    blocked = GuardResult(decision=Decision.BLOCK, reason="PII", source="pii")
    allowed = GuardResult.allow(source="test")
    mock_pipeline.check_input = MagicMock(return_value=(allowed, {}))
    mock_pipeline.check_step = MagicMock(return_value=(allowed, {}))
    mock_pipeline.check_output = MagicMock(return_value=(blocked, {}))
    agent = _make_agent()
    agent._safety_pipeline = mock_pipeline
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is False


@pytest.mark.asyncio
async def test_checkpoint_saved_every_10_steps(agent_context):
    call = ToolCall(id="c1", name="noop", args={})
    responses = [_make_llm_response("step", tool_calls=[call])] * 10 + [_make_llm_response("done")]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))
    mock_checkpoint = AsyncMock()
    mock_checkpoint.exists = AsyncMock(return_value=False)
    mock_checkpoint.save = AsyncMock()
    agent = _make_agent(router=router, tool_registry=mock_registry, checkpoint_manager=mock_checkpoint)
    ctx = _make_ctx(agent_context)
    ctx.max_steps = 15
    await agent.run(ctx)
    assert mock_checkpoint.save.call_count >= 1


@pytest.mark.asyncio
async def test_resume_from_checkpoint(agent_context):
    mock_checkpoint = AsyncMock()
    mock_checkpoint.exists = AsyncMock(return_value=True)
    mock_checkpoint.load = AsyncMock(return_value=MagicMock(
        step_count=5, token_count=500, history_snapshot=[]
    ))
    mock_checkpoint.save = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("resumed"))
    agent = _make_agent(router=router, checkpoint_manager=mock_checkpoint)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)
    mock_checkpoint.load.assert_called_once_with(ctx.run_id, ctx.tenant_id)


@pytest.mark.asyncio
async def test_resume_restores_step_and_token_counts(agent_context):
    """Checkpoint step_count and token_count are applied to ctx before the loop runs."""
    mock_checkpoint = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("done"))
    agent = _make_agent(router=router, checkpoint_manager=mock_checkpoint)
    # Set AFTER _make_agent so the factory's default None load isn't used
    mock_checkpoint.load = AsyncMock(return_value=MagicMock(
        step_count=7, token_count=1000, history_snapshot=[]
    ))
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)
    # run adds 1 tick on top of the restored 7
    assert ctx.step_count >= 7
    assert ctx.token_count >= 1000


@pytest.mark.asyncio
async def test_resume_restores_history_into_loop(agent_context):
    """history_snapshot from checkpoint is used as the initial history list."""
    prior_history = [
        {"role": "user", "content": "prior task"},
        {"role": "assistant", "content": "prior answer"},
    ]
    mock_checkpoint = AsyncMock()
    captured_messages: list = []

    async def capturing_complete(messages, system, tools=None, max_tokens=256):
        captured_messages.extend(messages)
        return _make_llm_response("done")

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=capturing_complete)
    agent = _make_agent(router=router, checkpoint_manager=mock_checkpoint)
    # Set AFTER _make_agent so the factory's default None load isn't used
    mock_checkpoint.load = AsyncMock(return_value=MagicMock(
        step_count=2, token_count=100, history_snapshot=prior_history
    ))
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    # Prior history messages should appear in the first LLM call
    all_content = [m.get("content") for m in captured_messages]
    assert "prior answer" in all_content


@pytest.mark.asyncio
async def test_checkpoint_saved_on_failure(agent_context):
    """Checkpoint is written in finally even when an exception propagates."""
    mock_checkpoint = AsyncMock()
    mock_checkpoint.load = AsyncMock(return_value=None)
    mock_checkpoint.save = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("crash"))
    agent = _make_agent(router=router, checkpoint_manager=mock_checkpoint)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is False
    mock_checkpoint.save.assert_called()


@pytest.mark.asyncio
async def test_checkpoint_saved_on_success(agent_context):
    """Checkpoint is written in finally on a clean successful run."""
    mock_checkpoint = AsyncMock()
    mock_checkpoint.load = AsyncMock(return_value=None)
    mock_checkpoint.save = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("done"))
    agent = _make_agent(router=router, checkpoint_manager=mock_checkpoint)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is True
    mock_checkpoint.save.assert_called()


@pytest.mark.asyncio
async def test_failure_recorded_on_exception(agent_context):
    mock_failure_tracker = AsyncMock()
    mock_failure_tracker.record = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("Unexpected crash"))
    agent = _make_agent(router=router, failure_tracker=mock_failure_tracker)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is False
    mock_failure_tracker.record.assert_called_once()


@pytest.mark.asyncio
async def test_cost_tracked_per_llm_call(agent_context):
    mock_cost = AsyncMock()
    mock_cost.record = AsyncMock(return_value=MagicMock(cost_usd=0.005))
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("done"))
    agent = _make_agent(router=router, cost_tracker=mock_cost)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)
    mock_cost.record.assert_called()


@pytest.mark.asyncio
async def test_run_records_trace_spans(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    from harness.observability.trace_schema import SpanKind

    call = ToolCall(id="c1", name="echo", args={"message": "hi"})
    responses = [
        _make_llm_response("calling tool", tool_calls=[call], tokens=40),
        _make_llm_response("done", tokens=20),
    ]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)

    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={"echo": "hi"}))

    mock_cost = AsyncMock()
    mock_cost.record = AsyncMock(side_effect=[
        MagicMock(cost_usd=0.004),
        MagicMock(cost_usd=0.002),
    ])

    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    agent = _make_agent(
        router=router,
        tool_registry=mock_registry,
        cost_tracker=mock_cost,
        trace_recorder=recorder,
    )
    ctx = _make_ctx(agent_context)

    result = await agent.run(ctx)
    trace = await recorder.get_trace(ctx.run_id)

    assert result.success is True
    assert trace is not None
    assert trace.total_input_tokens == 30
    assert trace.total_output_tokens == 30
    assert trace.total_cost_usd == pytest.approx(0.006)

    kinds = [span.kind for span in trace.spans]
    assert SpanKind.RUN in kinds
    assert kinds.count(SpanKind.LLM) == 2
    assert SpanKind.TOOL in kinds


# ===========================================================================
# _check_policy — per-tenant policy enforcement
# ===========================================================================

def _make_policy(**kwargs):
    from harness.safety.policies import HarnessPolicy
    return HarnessPolicy(tenant_id="t1", **kwargs)


def _make_call(name: str):
    return ToolCall(id="x", name=name, args={})


@pytest.mark.asyncio
async def test_check_policy_no_policy_is_noop(agent_context):
    """No policy in metadata → no exception for any tool."""
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    # policy key absent — should not raise
    await agent._check_policy(ctx, _make_call("drop_table"))


@pytest.mark.asyncio
async def test_check_policy_blocked_tool_raises(agent_context):
    from harness.core.errors import SafetyViolation
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(blocked_tools=["drop_table"])
    with pytest.raises(SafetyViolation, match="blocked by tenant policy"):
        await agent._check_policy(ctx, _make_call("drop_table"))


@pytest.mark.asyncio
async def test_check_policy_allowed_tool_passes(agent_context):
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(blocked_tools=["drop_table"])
    await agent._check_policy(ctx, _make_call("read_file"))  # not blocked


@pytest.mark.asyncio
async def test_check_policy_code_exec_disabled_blocks_run_python(agent_context):
    from harness.core.errors import SafetyViolation
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(allow_code_execution=False)
    with pytest.raises(SafetyViolation, match="code execution"):
        await agent._check_policy(ctx, _make_call("run_python"))


@pytest.mark.asyncio
async def test_check_policy_code_exec_disabled_blocks_exec_prefix(agent_context):
    from harness.core.errors import SafetyViolation
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(allow_code_execution=False)
    with pytest.raises(SafetyViolation, match="code execution"):
        await agent._check_policy(ctx, _make_call("execute_shell"))


@pytest.mark.asyncio
async def test_check_policy_code_exec_enabled_allows_run_python(agent_context):
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(allow_code_execution=True)
    await agent._check_policy(ctx, _make_call("run_python"))


@pytest.mark.asyncio
async def test_check_policy_file_write_disabled_blocks_write_file(agent_context):
    from harness.core.errors import SafetyViolation
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(allow_file_write=False)
    with pytest.raises(SafetyViolation, match="file write"):
        await agent._check_policy(ctx, _make_call("write_file"))


@pytest.mark.asyncio
async def test_check_policy_file_write_disabled_blocks_apply_patch(agent_context):
    from harness.core.errors import SafetyViolation
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(allow_file_write=False)
    with pytest.raises(SafetyViolation, match="file write"):
        await agent._check_policy(ctx, _make_call("apply_patch"))


@pytest.mark.asyncio
async def test_check_policy_file_write_enabled_allows_write_file(agent_context):
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(allow_file_write=True)
    await agent._check_policy(ctx, _make_call("write_file"))


@pytest.mark.asyncio
async def test_check_policy_read_file_always_allowed(agent_context):
    """read_file is not code exec or file write — always passes."""
    agent = _make_agent()
    ctx = _make_ctx(agent_context)
    ctx.metadata["policy"] = _make_policy(
        allow_code_execution=False, allow_file_write=False
    )
    await agent._check_policy(ctx, _make_call("read_file"))
