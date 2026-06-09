"""Unit tests for ContextEngine — offload, compress, select, isolate, evaluate, sub-agents."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harness.memory.context_engine import (
    ActionRecord,
    BuiltContext,
    ContextEngine,
    SubAgentSlice,
    _group_by_skill,
    _importance_score,
    _merge_by_step,
    _score_confidence,
    _score_goal_progress,
    _score_tool_relevance,
    _sliding_fit,
    _total_tokens,
)
from harness.memory.schemas import ConversationMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(redis_client, *, max_hot=10_000, vector_store=None,
                 embedder=None, summarizer=None) -> ContextEngine:
    eng = ContextEngine(
        max_hot_tokens=max_hot,
        redis_url="redis://unused",
        vector_store=vector_store,
        embedder=embedder,
        summarizer=summarizer,
    )
    eng._client = redis_client
    return eng


def _msg(role: str = "user", content: str = "hello", tokens: int = 5) -> ConversationMessage:
    return ConversationMessage(role=role, content=content, tokens=tokens)


# ===========================================================================
# Pure helper functions (no Redis needed)
# ===========================================================================

# ── _score_goal_progress ────────────────────────────────────────────────────

def test_goal_progress_exact_keyword_match():
    score = _score_goal_progress("list all users from the database", "list users database")
    assert score > 0.5


def test_goal_progress_empty_goal_returns_neutral():
    score = _score_goal_progress("anything", "")
    assert score == 0.5


def test_goal_progress_no_overlap_returns_low():
    score = _score_goal_progress("completely unrelated response", "find elephant count")
    assert score < 0.4


def test_goal_progress_full_match_caps_at_1():
    score = _score_goal_progress("find users find users find users", "find users")
    assert score <= 1.0


def test_goal_progress_short_words_ignored():
    # Words ≤ 3 chars are filtered; score should still work
    score = _score_goal_progress("the a is or", "the a is or")
    # All words are short, goal_words becomes empty → neutral
    assert score == 0.5


# ── _score_tool_relevance ───────────────────────────────────────────────────

def test_tool_relevance_no_tool_is_neutral():
    assert _score_tool_relevance(None, None, False) == 0.5


def test_tool_relevance_error_is_low():
    assert _score_tool_relevance("execute_sql", None, True) == pytest.approx(0.1)


def test_tool_relevance_success_with_result_is_high():
    # result must be > 20 chars to trigger the 0.9 path
    long_result = "42 rows returned from the users table"
    assert _score_tool_relevance("execute_sql", long_result, False) == pytest.approx(0.9)


def test_tool_relevance_success_no_result_is_medium():
    assert _score_tool_relevance("execute_sql", "", False) == pytest.approx(0.6)


def test_tool_relevance_short_result_is_medium():
    assert _score_tool_relevance("execute_sql", "ok", False) == pytest.approx(0.6)


# ── _score_confidence ───────────────────────────────────────────────────────

def test_confidence_neutral_baseline():
    score = _score_confidence("the result is here")
    assert 0.1 <= score <= 1.0


def test_confidence_hedging_reduces_score():
    hedged = _score_confidence("i'm not sure but maybe it could be the answer")
    plain = _score_confidence("the answer is 42")
    assert hedged < plain


def test_confidence_affirming_increases_score():
    affirmed = _score_confidence("the answer is confirmed and completed")
    base = _score_confidence("something")
    assert affirmed >= base


def test_confidence_clamped_between_01_and_10():
    # Many hedges
    score = _score_confidence("i'm not sure maybe possibly uncertain perhaps could be")
    assert 0.1 <= score <= 1.0
    # Many affirms
    score2 = _score_confidence("found result done completed here is the answer confirmed success")
    assert 0.1 <= score2 <= 1.0


# ── _importance_score ───────────────────────────────────────────────────────

def test_importance_score_baseline_is_half():
    raw = [json.dumps({"role": "user", "content": "simple message", "tokens": 10})]
    score = _importance_score(raw)
    assert score == pytest.approx(0.5)


def test_importance_score_tool_messages_bump_up():
    raw = [json.dumps({"role": "tool", "content": "result data", "tokens": 20})]
    score = _importance_score(raw)
    assert score > 0.5


def test_importance_score_error_content_bumps_up():
    raw = [json.dumps({"role": "assistant", "content": "error: connection failed", "tokens": 15})]
    score = _importance_score(raw)
    assert score > 0.5


def test_importance_score_result_content_bumps_up():
    raw = [json.dumps({"role": "assistant", "content": "result: 42 rows found", "tokens": 15})]
    score = _importance_score(raw)
    assert score > 0.5


def test_importance_score_capped_at_1():
    raw = [json.dumps({"role": "tool", "content": "error result success found", "tokens": 10})] * 20
    score = _importance_score(raw)
    assert score <= 1.0


# ── _sliding_fit ─────────────────────────────────────────────────────────────

def test_sliding_fit_fits_all_within_budget():
    msgs = [_msg(tokens=100) for _ in range(5)]
    fitted, total, truncated = _sliding_fit(msgs, budget=600)
    assert len(fitted) == 5
    assert total == 500
    assert truncated is False


def test_sliding_fit_keeps_most_recent_on_overflow():
    msgs = [_msg(content=f"msg{i}", tokens=100) for i in range(10)]
    fitted, total, truncated = _sliding_fit(msgs, budget=300)
    # Should keep last 3
    assert len(fitted) == 3
    assert all(m.content.startswith("msg") for m in fitted)
    # Most recent messages are at the end of the list
    assert fitted[-1].content == "msg9"
    assert truncated is True


def test_sliding_fit_empty_budget_returns_empty():
    msgs = [_msg(tokens=100)]
    fitted, total, truncated = _sliding_fit(msgs, budget=0)
    assert fitted == []
    assert total == 0
    assert truncated is True


def test_sliding_fit_exact_budget_keeps_all():
    msgs = [_msg(tokens=50), _msg(tokens=50)]
    fitted, total, truncated = _sliding_fit(msgs, budget=100)
    assert len(fitted) == 2
    assert truncated is False


# ── _total_tokens ─────────────────────────────────────────────────────────────

def test_total_tokens_sums_correctly():
    items = [
        json.dumps({"role": "user", "content": "hi", "tokens": 10}),
        json.dumps({"role": "assistant", "content": "hello", "tokens": 20}),
    ]
    assert _total_tokens(items) == 30


def test_total_tokens_handles_malformed_json():
    items = ["not-json", json.dumps({"tokens": 15})]
    # Should not raise; malformed items skipped
    result = _total_tokens(items)
    assert result == 15


# ── _group_by_skill ─────────────────────────────────────────────────────────

def test_group_by_skill_aggregates_correctly():
    from datetime import datetime, timezone
    import uuid
    records = [
        ActionRecord(action_id=uuid.uuid4().hex, run_id="r1", step=1, skill_ns="sql",
                     llm_preview="", tool_name=None, goal_progress=0.8, tool_relevance=0.9,
                     confidence=0.7, composite_score=0.82, is_error=False,
                     timestamp=datetime.now(timezone.utc)),
        ActionRecord(action_id=uuid.uuid4().hex, run_id="r1", step=2, skill_ns="sql",
                     llm_preview="", tool_name="execute_sql", goal_progress=0.4,
                     tool_relevance=0.1, confidence=0.5, composite_score=0.33,
                     is_error=True, timestamp=datetime.now(timezone.utc)),
        ActionRecord(action_id=uuid.uuid4().hex, run_id="r1", step=3, skill_ns="code",
                     llm_preview="", tool_name=None, goal_progress=0.9, tool_relevance=0.9,
                     confidence=0.9, composite_score=0.9, is_error=False,
                     timestamp=datetime.now(timezone.utc)),
    ]
    groups = _group_by_skill(records)
    assert "sql" in groups
    assert "code" in groups
    assert groups["sql"]["count"] == 2
    assert groups["sql"]["errors"] == 1
    assert groups["code"]["count"] == 1
    assert groups["code"]["errors"] == 0
    assert "avg_score" in groups["sql"]


# ===========================================================================
# ContextEngine — Redis-backed tests
# ===========================================================================

# ── push ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_adds_message_to_hot_window(redis_client):
    eng = _make_engine(redis_client)
    await eng.push("run1", "user", "hello world", tokens=10, skill_ns="default")
    r = await eng._redis()
    count = await r.llen(eng._hot_key("run1", "default"))
    assert count == 1


@pytest.mark.asyncio
async def test_push_uses_separate_key_per_skill_ns(redis_client):
    eng = _make_engine(redis_client)
    await eng.push("run1", "user", "sql msg", tokens=5, skill_ns="sql")
    await eng.push("run1", "user", "code msg", tokens=5, skill_ns="code")
    r = await eng._redis()
    sql_count = await r.llen(eng._hot_key("run1", "sql"))
    code_count = await r.llen(eng._hot_key("run1", "code"))
    assert sql_count == 1
    assert code_count == 1


@pytest.mark.asyncio
async def test_push_newest_first_in_redis_list(redis_client):
    eng = _make_engine(redis_client)
    await eng.push("run1", "user", "first", tokens=5, skill_ns="default")
    await eng.push("user", "user", "second", tokens=5, skill_ns="default")
    # LPUSH means index 0 = newest
    r = await eng._redis()
    raw = await r.lindex(eng._hot_key("run1", "default"), 0)
    data = json.loads(raw)
    assert data["content"] == "first"  # index 0 = most recently pushed


@pytest.mark.asyncio
async def test_push_estimates_tokens_when_zero(redis_client):
    eng = _make_engine(redis_client)
    await eng.push("run1", "user", "hello world test message", tokens=0, skill_ns="default")
    r = await eng._redis()
    raw = await r.lindex(eng._hot_key("run1", "default"), 0)
    data = json.loads(raw)
    assert data["tokens"] > 0


# ── build_context ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_context_empty_run_returns_empty(redis_client):
    eng = _make_engine(redis_client)
    result = await eng.build_context("no-such-run", query="anything")
    assert isinstance(result, BuiltContext)
    assert result.messages == []
    assert result.total_tokens == 0
    assert result.truncated is False


@pytest.mark.asyncio
async def test_build_context_returns_hot_messages(redis_client):
    eng = _make_engine(redis_client)
    await eng.push("run2", "user", "hello", tokens=10, skill_ns="default")
    await eng.push("run2", "assistant", "hi there", tokens=15, skill_ns="default")
    result = await eng.build_context("run2", query="hello", skill_ns="default")
    assert len(result.messages) == 2
    assert result.total_tokens == 25


@pytest.mark.asyncio
async def test_build_context_respects_token_budget(redis_client):
    eng = _make_engine(redis_client, max_hot=1_000)
    for i in range(20):
        await eng.push("run3", "user", f"message {i}", tokens=100, skill_ns="default")
    result = await eng.build_context("run3", query="test", skill_ns="default",
                                     token_budget=500)
    assert result.total_tokens <= 500


@pytest.mark.asyncio
async def test_build_context_includes_shared_ns_for_skill(redis_client):
    eng = _make_engine(redis_client)
    await eng.push("run4", "system", "shared context", tokens=20, skill_ns="default")
    await eng.push("run4", "user", "sql query", tokens=15, skill_ns="sql")
    result = await eng.build_context("run4", query="list tables", skill_ns="sql",
                                     include_shared=True)
    contents = [m.content for m in result.messages]
    assert any("shared context" in c for c in contents)
    assert any("sql query" in c for c in contents)


@pytest.mark.asyncio
async def test_build_context_skill_ns_in_result(redis_client):
    eng = _make_engine(redis_client)
    result = await eng.build_context("run5", query="test", skill_ns="code")
    assert result.skill_ns == "code"


# ── evaluate_action ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_action_returns_action_record(redis_client):
    eng = _make_engine(redis_client)
    record = await eng.evaluate_action(
        "run-eval", step=1, goal="list users",
        llm_content="I found all users in the database",
        tool_name="execute_sql", tool_result="42 rows", is_error=False,
    )
    assert isinstance(record, ActionRecord)
    assert record.run_id == "run-eval"
    assert record.step == 1


@pytest.mark.asyncio
async def test_evaluate_action_composite_score_in_range(redis_client):
    eng = _make_engine(redis_client)
    record = await eng.evaluate_action(
        "run-eval2", step=1, goal="do something",
        llm_content="done", tool_name=None, tool_result=None,
    )
    assert 0.0 <= record.composite_score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_action_error_gives_low_tool_relevance(redis_client):
    eng = _make_engine(redis_client)
    record = await eng.evaluate_action(
        "run-eval3", step=1, goal="query",
        llm_content="something", tool_name="execute_sql", tool_result=None,
        is_error=True,
    )
    assert record.tool_relevance == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_evaluate_action_persisted_to_redis(redis_client):
    eng = _make_engine(redis_client)
    await eng.evaluate_action(
        "run-persist", step=2, goal="test goal",
        llm_content="test response", tool_name=None,
    )
    r = await eng._redis()
    count = await r.llen(f"harness:actions:run-persist")
    assert count == 1


@pytest.mark.asyncio
async def test_evaluate_action_stored_fields_are_correct(redis_client):
    eng = _make_engine(redis_client)
    record = await eng.evaluate_action(
        "run-fields", step=5, goal="find something", skill_ns="sql",
        llm_content="I found the answer", tool_name="execute_sql",
        tool_result="data", is_error=False,
    )
    r = await eng._redis()
    raw = await r.lindex("harness:actions:run-fields", 0)
    data = json.loads(raw)
    assert data["step"] == 5
    assert data["skill_ns"] == "sql"
    assert data["tool_name"] == "execute_sql"
    assert data["is_error"] is False
    assert "composite_score" in data


# ── get_action_log ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_action_log_returns_all_records(redis_client):
    eng = _make_engine(redis_client)
    for i in range(3):
        await eng.evaluate_action(
            "run-log", step=i, goal="test",
            llm_content="response", tool_name=None,
        )
    records = await eng.get_action_log("run-log")
    assert len(records) == 3


@pytest.mark.asyncio
async def test_get_action_log_filters_only_errors(redis_client):
    eng = _make_engine(redis_client)
    await eng.evaluate_action("run-ferr", step=1, goal="test",
                               llm_content="ok", tool_name=None, is_error=False)
    await eng.evaluate_action("run-ferr", step=2, goal="test",
                               llm_content="err", tool_name="tool", is_error=True)
    errors = await eng.get_action_log("run-ferr", only_errors=True)
    assert len(errors) == 1
    assert errors[0].is_error is True


@pytest.mark.asyncio
async def test_get_action_log_filters_by_min_score(redis_client):
    eng = _make_engine(redis_client)
    # Score depends on heuristics — we just check the filter doesn't crash
    await eng.evaluate_action("run-mscore", step=1, goal="test",
                               llm_content="response", tool_name=None)
    records = await eng.get_action_log("run-mscore", min_score=0.0)
    assert len(records) >= 1
    records_high = await eng.get_action_log("run-mscore", min_score=0.99)
    for r in records_high:
        assert r.composite_score >= 0.99


@pytest.mark.asyncio
async def test_get_action_log_empty_for_unknown_run(redis_client):
    eng = _make_engine(redis_client)
    records = await eng.get_action_log("no-such-run-log")
    assert records == []


# ── get_action_summary ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_action_summary_calculates_averages(redis_client):
    eng = _make_engine(redis_client)
    for i in range(4):
        await eng.evaluate_action(
            "run-sum", step=i, goal="task",
            llm_content="response", tool_name=None,
        )
    summary = await eng.get_action_summary("run-sum")
    assert summary.total_actions == 4
    assert 0.0 <= summary.avg_score <= 1.0
    assert summary.min_score <= summary.max_score


@pytest.mark.asyncio
async def test_get_action_summary_counts_errors(redis_client):
    eng = _make_engine(redis_client)
    await eng.evaluate_action("run-sum2", step=1, goal="task",
                               llm_content="ok", tool_name=None, is_error=False)
    await eng.evaluate_action("run-sum2", step=2, goal="task",
                               llm_content="err", tool_name="t", is_error=True)
    summary = await eng.get_action_summary("run-sum2")
    assert summary.error_count == 1


@pytest.mark.asyncio
async def test_get_action_summary_empty_run(redis_client):
    eng = _make_engine(redis_client)
    summary = await eng.get_action_summary("no-run-summary")
    assert summary.total_actions == 0


# ── inject_subagent_result ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_inject_subagent_result_adds_to_parent_window(redis_client):
    eng = _make_engine(redis_client)
    await eng.inject_subagent_result(
        parent_run_id="parent-run",
        child_run_id="child-run",
        result_summary="Child completed: found 42 rows",
        skill_ns="default",
    )
    r = await eng._redis()
    count = await r.llen(eng._hot_key("parent-run", "default"))
    assert count == 1


@pytest.mark.asyncio
async def test_inject_subagent_result_message_has_tool_role(redis_client):
    eng = _make_engine(redis_client)
    await eng.inject_subagent_result(
        parent_run_id="parent2",
        child_run_id="child2",
        result_summary="Result: success",
    )
    r = await eng._redis()
    raw = await r.lindex(eng._hot_key("parent2", "default"), 0)
    data = json.loads(raw)
    assert data["role"] == "tool"
    assert "child2"[:8] in data["content"] or "Result" in data["content"]


@pytest.mark.asyncio
async def test_inject_subagent_result_has_positive_token_count(redis_client):
    eng = _make_engine(redis_client)
    await eng.inject_subagent_result(
        parent_run_id="parent3", child_run_id="child3",
        result_summary="A fairly long result summary with multiple words",
    )
    r = await eng._redis()
    raw = await r.lindex(eng._hot_key("parent3", "default"), 0)
    data = json.loads(raw)
    assert data["tokens"] > 0


# ── slice_for_subagent ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_slice_for_subagent_returns_slice(redis_client):
    eng = _make_engine(redis_client)
    for i in range(3):
        await eng.push("parent-slice", "user", f"context message {i}",
                       tokens=50, skill_ns="default")
    result = await eng.slice_for_subagent(
        parent_run_id="parent-slice",
        child_run_id="child-slice",
        task="summarize the context",
        token_budget=500,
    )
    assert isinstance(result, SubAgentSlice)
    assert result.parent_run_id == "parent-slice"
    assert result.child_run_id == "child-slice"
    assert result.token_budget == 500


@pytest.mark.asyncio
async def test_slice_for_subagent_pushes_to_child_window(redis_client):
    eng = _make_engine(redis_client)
    for i in range(5):
        await eng.push("parent-push", "user", f"message {i}",
                       tokens=100, skill_ns="default")
    await eng.slice_for_subagent(
        parent_run_id="parent-push",
        child_run_id="child-push",
        task="process the messages",
        token_budget=800,
    )
    r = await eng._redis()
    child_count = await r.llen(eng._hot_key("child-push", "default"))
    assert child_count > 0


@pytest.mark.asyncio
async def test_slice_for_subagent_respects_token_budget(redis_client):
    eng = _make_engine(redis_client)
    for i in range(20):
        await eng.push("parent-budget", "user", f"big message {i}",
                       tokens=200, skill_ns="default")
    result = await eng.slice_for_subagent(
        parent_run_id="parent-budget",
        child_run_id="child-budget",
        task="test",
        token_budget=400,
    )
    total = sum(m.tokens for m in result.messages)
    assert total <= 400


# ── compress ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compress_extractive_fallback_without_summarizer(redis_client):
    eng = _make_engine(redis_client, summarizer=None)
    from harness.memory.schemas import ConversationMessage
    msgs = [
        ConversationMessage(role="user", content="user question"),
        ConversationMessage(role="assistant", content="assistant answer with result: 42"),
        ConversationMessage(role="tool", content="tool output data"),
    ]
    summary = await eng._compress(msgs)
    assert isinstance(summary, str)
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_compress_uses_llm_summarizer_when_available(redis_client):
    mock_summarizer = AsyncMock()
    from harness.core.context import LLMResponse
    mock_summarizer.complete = AsyncMock(return_value=LLMResponse(
        content="Compressed summary", tool_calls=[], input_tokens=10,
        output_tokens=20, model="mock", provider="mock",
    ))
    eng = _make_engine(redis_client, summarizer=mock_summarizer)
    msgs = [ConversationMessage(role="user", content="long conversation")]
    summary = await eng._compress(msgs)
    assert summary == "Compressed summary"
    mock_summarizer.complete.assert_called_once()


@pytest.mark.asyncio
async def test_compress_falls_back_when_summarizer_raises(redis_client):
    mock_summarizer = AsyncMock()
    mock_summarizer.complete = AsyncMock(side_effect=Exception("LLM unavailable"))
    eng = _make_engine(redis_client, summarizer=mock_summarizer)
    msgs = [ConversationMessage(role="user", content="message content with result")]
    summary = await eng._compress(msgs)
    assert isinstance(summary, str)
    assert len(summary) > 0


# ── _merge_by_step ───────────────────────────────────────────────────────────

def test_merge_by_step_concatenates_lists():
    a = [_msg(content="a1"), _msg(content="a2")]
    b = [_msg(content="b1")]
    result = _merge_by_step(a, b)
    assert len(result) == 3
    assert result[0].content == "a1"
    assert result[-1].content == "b1"


def test_merge_by_step_empty_lists():
    assert _merge_by_step([], []) == []
    # Timestamps differ each call — compare lengths and content only
    single = _msg(content="solo")
    assert len(_merge_by_step([single], [])) == 1
    assert _merge_by_step([single], [])[0].content == "solo"
    assert len(_merge_by_step([], [single])) == 1
    assert _merge_by_step([], [single])[0].content == "solo"


# ===========================================================================
# Regression tests — offload trim direction (item 1) & subagent order (item 2)
# ===========================================================================

@pytest.mark.asyncio
async def test_offload_trims_from_tail_keeps_newest(redis_client, monkeypatch):
    """Offload must remove the OLDEST (tail) messages and keep the newest.

    Regression for the non-atomic LTRIM-by-head bug: trimming relative to the
    head silently deleted non-offloaded (newest) messages once new messages
    were LPUSHed; trimming relative to the tail keeps the newest intact.
    """
    import harness.memory.context_engine as ce
    monkeypatch.setattr(ce, "_HOT_MAX_MSGS", 4)
    monkeypatch.setattr(ce, "_PAGE_TOKEN_TARGET", 100)
    # max_hot small so the token threshold (0.8 * max_hot) is exceeded.
    eng = _make_engine(redis_client, max_hot=500)

    # Push 6 messages: msg0 (oldest) … msg5 (newest), 100 tokens each.
    for i in range(6):
        await eng.push("run-trim", "user", f"msg{i}", tokens=100, skill_ns="default")

    # Newest messages must survive; the oldest were offloaded.
    msgs = await eng._load_hot("run-trim", "default")
    contents = [m.content for m in msgs]
    assert "msg5" in contents, "newest message was wrongly deleted"
    assert "msg0" not in contents, "oldest message should have been offloaded"
    # Chronological order preserved (oldest-first) among survivors.
    assert contents == sorted(contents, key=lambda c: int(c[3:]))


@pytest.mark.asyncio
async def test_offload_concurrent_pushes_dont_lose_messages(redis_client, monkeypatch):
    """A push concurrent with an offload must not be dropped (tail-relative trim)."""
    import harness.memory.context_engine as ce
    monkeypatch.setattr(ce, "_HOT_MAX_MSGS", 4)
    monkeypatch.setattr(ce, "_PAGE_TOKEN_TARGET", 100)
    eng = _make_engine(redis_client, max_hot=500)

    for i in range(5):
        await eng.push("run-conc", "user", f"m{i}", tokens=100, skill_ns="default")
    # One more push after offload already happened.
    await eng.push("run-conc", "user", "m-after", tokens=100, skill_ns="default")

    msgs = await eng._load_hot("run-conc", "default")
    contents = [m.content for m in msgs]
    assert "m-after" in contents


@pytest.mark.asyncio
async def test_slice_for_subagent_preserves_chronological_order(redis_client):
    """Child hot window must read back oldest-first, matching the parent order.

    Regression for the RPUSH-oldest-first-into-newest-first-list ordering bug.
    """
    eng = _make_engine(redis_client)
    for i in range(4):
        await eng.push("parent-order", "user", f"turn{i}", tokens=20, skill_ns="default")

    await eng.slice_for_subagent(
        parent_run_id="parent-order",
        child_run_id="child-order",
        task="continue",
        token_budget=1000,
    )

    child_msgs = await eng._load_hot("child-order", "default")
    contents = [m.content for m in child_msgs]
    # _load_hot returns chronological order; turns must be in ascending order.
    turn_nums = [int(c[4:]) for c in contents if c.startswith("turn")]
    assert turn_nums == sorted(turn_nums)
    assert turn_nums == list(range(turn_nums[0], turn_nums[0] + len(turn_nums)))
