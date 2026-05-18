"""Unit tests for real-time feedback channel."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from harness.feedback.channel import (
    FeedbackChannel,
    FeedbackEvent,
    should_inject,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redis():
    return fakeredis.FakeRedis(decode_responses=True)


def _ev(type_="hint", content="test", score=None, source="human", priority=2):
    return FeedbackEvent(type=type_, content=content, score=score,
                         source=source, priority=priority)


# ===========================================================================
# FeedbackEvent
# ===========================================================================

def test_feedback_event_defaults():
    ev = FeedbackEvent()
    assert ev.type == "hint"
    assert ev.source == "human"
    assert ev.priority == 2
    assert ev.applied is False
    assert ev.score is None
    assert len(ev.feedback_id) == 32  # uuid4 hex


def test_feedback_event_round_trip():
    ev = FeedbackEvent(type="correction", content="fix this",
                       score=None, source="evaluator", priority=3)
    ev2 = FeedbackEvent.from_dict(ev.to_dict())
    assert ev2.type == ev.type
    assert ev2.content == ev.content
    assert ev2.source == ev.source
    assert ev2.priority == ev.priority
    assert ev2.feedback_id == ev.feedback_id


def test_feedback_event_score_round_trip():
    ev = FeedbackEvent(type="score", content="low", score=0.25)
    ev2 = FeedbackEvent.from_dict(ev.to_dict())
    assert ev2.score == pytest.approx(0.25)


def test_feedback_event_applied_round_trip():
    now = datetime.now(timezone.utc)
    ev = FeedbackEvent(type="hint", content="x", applied=True, applied_at=now)
    ev2 = FeedbackEvent.from_dict(ev.to_dict())
    assert ev2.applied is True
    assert ev2.applied_at is not None


def test_feedback_event_to_context_message_correction():
    ev = FeedbackEvent(type="correction", content="Use LEFT JOIN not INNER JOIN")
    msg = ev.to_context_message()
    assert "[FEEDBACK:CORRECTION]" in msg
    assert "LEFT JOIN" in msg


def test_feedback_event_to_context_message_hint():
    ev = FeedbackEvent(type="hint", content="check the schema")
    assert "[FEEDBACK:HINT]" in ev.to_context_message()


def test_feedback_event_to_context_message_stop():
    ev = FeedbackEvent(type="stop", content="user cancelled")
    assert "[FEEDBACK:STOP]" in ev.to_context_message()


def test_feedback_event_to_context_message_redirect():
    ev = FeedbackEvent(type="redirect", content="now focus on orders table")
    assert "[FEEDBACK:REDIRECT]" in ev.to_context_message()


def test_feedback_event_to_context_message_score_includes_value():
    ev = FeedbackEvent(type="score", content="low quality", score=0.31)
    msg = ev.to_context_message()
    assert "0.31" in msg
    assert "[FEEDBACK:SCORE]" in msg


def test_feedback_event_to_context_message_no_score():
    ev = FeedbackEvent(type="hint", content="hint text", score=None)
    msg = ev.to_context_message()
    assert "None" not in msg


def test_feedback_event_empty_content():
    ev = FeedbackEvent(type="stop", content="")
    msg = ev.to_context_message()
    assert "[FEEDBACK:STOP]" in msg


def test_feedback_event_from_dict_missing_optional_fields():
    d = {"type": "hint", "content": "x"}
    ev = FeedbackEvent.from_dict(d)
    assert ev.type == "hint"
    assert ev.score is None
    assert ev.applied is False


# ===========================================================================
# should_inject
# ===========================================================================

def test_should_inject_correction():
    assert should_inject(FeedbackEvent(type="correction", content="fix")) is True


def test_should_inject_hint():
    assert should_inject(FeedbackEvent(type="hint", content="try")) is True


def test_should_inject_redirect():
    assert should_inject(FeedbackEvent(type="redirect", content="new goal")) is True


def test_should_inject_score_below_threshold():
    assert should_inject(FeedbackEvent(type="score", score=0.30, content="bad")) is True


def test_should_inject_score_at_threshold():
    # threshold is < 0.40, so score==0.40 does NOT inject (not strictly below threshold)
    assert should_inject(FeedbackEvent(type="score", score=0.40, content="ok")) is False


def test_should_inject_score_above_threshold():
    assert should_inject(FeedbackEvent(type="score", score=0.50, content="fine")) is False


def test_should_inject_score_high():
    assert should_inject(FeedbackEvent(type="score", score=0.95, content="great")) is False


def test_should_inject_stop():
    assert should_inject(FeedbackEvent(type="stop", content="")) is False


def test_should_inject_score_no_score_field():
    # score=None treated as no score → do not inject
    assert should_inject(FeedbackEvent(type="score", score=None, content="")) is False


# ===========================================================================
# FeedbackChannel — publish / poll
# ===========================================================================

@pytest.fixture
def channel():
    return FeedbackChannel(_redis())


@pytest.mark.asyncio
async def test_publish_returns_entry_id(channel):
    entry_id = await channel.publish("run1", _ev())
    assert entry_id  # non-empty string


@pytest.mark.asyncio
async def test_publish_sets_run_id(channel):
    ev = _ev()
    await channel.publish("run42", ev)
    assert ev.run_id == "run42"


@pytest.mark.asyncio
async def test_poll_returns_published_event(channel):
    ev = _ev(type_="correction", content="use LEFT JOIN")
    await channel.publish("run1", ev)
    events, _ = await channel.poll("run1", "0")
    assert len(events) == 1
    assert events[0].type == "correction"
    assert events[0].content == "use LEFT JOIN"


@pytest.mark.asyncio
async def test_poll_returns_all_types(channel):
    for t in ("correction", "hint", "score", "stop", "redirect"):
        await channel.publish("run2", _ev(type_=t))
    events, _ = await channel.poll("run2", "0")
    assert len(events) == 5
    types = {ev.type for ev in events}
    assert types == {"correction", "hint", "score", "stop", "redirect"}


@pytest.mark.asyncio
async def test_poll_advances_cursor(channel):
    await channel.publish("run3", _ev(content="first"))
    events1, last_id = await channel.poll("run3", "0")
    assert len(events1) == 1

    await channel.publish("run3", _ev(content="second"))
    events2, _ = await channel.poll("run3", last_id)
    assert len(events2) == 1
    assert events2[0].content == "second"


@pytest.mark.asyncio
async def test_poll_no_events_returns_empty(channel):
    events, last_id = await channel.poll("empty_run", "0")
    assert events == []
    assert last_id == "0"


@pytest.mark.asyncio
async def test_poll_respects_count_limit(channel):
    for i in range(10):
        await channel.publish("run4", _ev(content=f"msg{i}"))
    events, _ = await channel.poll("run4", "0", count=5)
    assert len(events) == 5


@pytest.mark.asyncio
async def test_poll_preserves_score(channel):
    ev = _ev(type_="score", score=0.37)
    await channel.publish("run5", ev)
    events, _ = await channel.poll("run5", "0")
    assert events[0].score == pytest.approx(0.37)


@pytest.mark.asyncio
async def test_poll_preserves_priority(channel):
    ev = _ev(priority=3)
    await channel.publish("run6", ev)
    events, _ = await channel.poll("run6", "0")
    assert events[0].priority == 3


@pytest.mark.asyncio
async def test_poll_preserves_source(channel):
    ev = _ev(source="evaluator")
    await channel.publish("run7", ev)
    events, _ = await channel.poll("run7", "0")
    assert events[0].source == "evaluator"


# ===========================================================================
# mark_applied / is_applied / history
# ===========================================================================

@pytest.mark.asyncio
async def test_mark_applied_sets_flag(channel):
    ev = _ev()
    await channel.publish("run8", ev)
    await channel.mark_applied("run8", [ev.feedback_id])
    assert await channel.is_applied("run8", ev.feedback_id) is True


@pytest.mark.asyncio
async def test_is_applied_false_before_mark(channel):
    ev = _ev()
    await channel.publish("run9", ev)
    assert await channel.is_applied("run9", ev.feedback_id) is False


@pytest.mark.asyncio
async def test_mark_applied_empty_list_no_error(channel):
    await channel.mark_applied("run10", [])  # must not raise


@pytest.mark.asyncio
async def test_history_returns_all_events(channel):
    for i in range(5):
        await channel.publish("run11", _ev(content=f"ev{i}"))
    history = await channel.history("run11")
    assert len(history) == 5


@pytest.mark.asyncio
async def test_history_marks_applied_events(channel):
    ev1 = _ev(content="first")
    ev2 = _ev(content="second")
    await channel.publish("run12", ev1)
    await channel.publish("run12", ev2)
    await channel.mark_applied("run12", [ev1.feedback_id])
    history = await channel.history("run12")
    applied = {ev.feedback_id: ev.applied for ev in history}
    assert applied[ev1.feedback_id] is True
    assert applied[ev2.feedback_id] is False


@pytest.mark.asyncio
async def test_history_empty_run(channel):
    history = await channel.history("unknown_run")
    assert history == []


@pytest.mark.asyncio
async def test_history_count_limit(channel):
    for i in range(20):
        await channel.publish("run13", _ev(content=f"m{i}"))
    history = await channel.history("run13", count=10)
    assert len(history) == 10


# ===========================================================================
# clear
# ===========================================================================

@pytest.mark.asyncio
async def test_clear_removes_events(channel):
    await channel.publish("run14", _ev())
    await channel.clear("run14")
    history = await channel.history("run14")
    assert history == []


@pytest.mark.asyncio
async def test_clear_removes_applied_set(channel):
    ev = _ev()
    await channel.publish("run15", ev)
    await channel.mark_applied("run15", [ev.feedback_id])
    await channel.clear("run15")
    assert await channel.is_applied("run15", ev.feedback_id) is False


@pytest.mark.asyncio
async def test_clear_nonexistent_run_no_error(channel):
    await channel.clear("does_not_exist")  # must not raise


# ===========================================================================
# Redis failure resilience
# ===========================================================================

@pytest.mark.asyncio
async def test_publish_redis_failure_returns_empty():
    bad_redis = MagicMock()
    bad_redis.xadd = AsyncMock(side_effect=RuntimeError("connection refused"))
    bad_redis.expire = AsyncMock()
    ch = FeedbackChannel(bad_redis)
    entry_id = await ch.publish("run", _ev())
    assert entry_id == ""


@pytest.mark.asyncio
async def test_poll_redis_failure_returns_empty():
    bad_redis = MagicMock()
    bad_redis.xread = AsyncMock(side_effect=RuntimeError("connection refused"))
    ch = FeedbackChannel(bad_redis)
    events, last_id = await ch.poll("run", "0")
    assert events == []
    assert last_id == "0"


@pytest.mark.asyncio
async def test_is_applied_redis_failure_returns_false():
    bad_redis = MagicMock()
    bad_redis.sismember = AsyncMock(side_effect=RuntimeError("redis down"))
    ch = FeedbackChannel(bad_redis)
    result = await ch.is_applied("run", "some_id")
    assert result is False


# ===========================================================================
# Integration: publish → poll → apply → history
# ===========================================================================

@pytest.mark.asyncio
async def test_full_feedback_lifecycle():
    ch = FeedbackChannel(_redis())
    run_id = "lifecycle_run"

    # Publish several events
    events_sent = [
        _ev("correction", "Use users table not usr", priority=3),
        _ev("score",      "partial result",          score=0.35, priority=2),
        _ev("hint",       "add WHERE clause",        priority=1),
    ]
    for ev in events_sent:
        await ch.publish(run_id, ev)

    # Poll — agent reads them
    polled, last_id = await ch.poll(run_id, "0")
    assert len(polled) == 3

    # Mark correction and score as applied (hint stays pending)
    to_apply = [ev for ev in polled if ev.type in ("correction", "score")]
    await ch.mark_applied(run_id, [ev.feedback_id for ev in to_apply])

    # Second poll gets nothing new
    polled2, _ = await ch.poll(run_id, last_id)
    assert polled2 == []

    # History shows correct applied flags
    history = await ch.history(run_id)
    assert len(history) == 3
    applied_types = {ev.type for ev in history if ev.applied}
    pending_types = {ev.type for ev in history if not ev.applied}
    assert "correction" in applied_types
    assert "score" in applied_types
    assert "hint" in pending_types
