"""Tests for background workers: publish_run_completion, worker event structure."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from harness.workers.rlvr_worker import publish_run_completion, _decode


# ===========================================================================
# publish_run_completion
# ===========================================================================

@pytest.mark.asyncio
async def test_publish_run_completion_writes_to_stream():
    redis = fakeredis.FakeRedis(decode_responses=True)
    await publish_run_completion(redis, "run123", "sql", "completed")

    entries = await redis.xrange("harness:rlvr:completions")
    assert len(entries) == 1
    _entry_id, fields = entries[0]
    assert fields["run_id"] == "run123"
    assert fields["agent_type"] == "sql"
    assert fields["status"] == "completed"


@pytest.mark.asyncio
async def test_publish_run_completion_sets_ttl():
    redis = fakeredis.FakeRedis(decode_responses=True)
    await publish_run_completion(redis, "run1", "code", "failed")
    ttl = await redis.ttl("harness:rlvr:completions")
    assert ttl > 0


@pytest.mark.asyncio
async def test_publish_run_completion_multiple_runs():
    redis = fakeredis.FakeRedis(decode_responses=True)
    for i in range(5):
        await publish_run_completion(redis, f"run_{i}", "sql", "completed")

    entries = await redis.xrange("harness:rlvr:completions")
    assert len(entries) == 5


@pytest.mark.asyncio
async def test_publish_run_completion_default_status():
    redis = fakeredis.FakeRedis(decode_responses=True)
    await publish_run_completion(redis, "run1", "base")
    entries = await redis.xrange("harness:rlvr:completions")
    _eid, fields = entries[0]
    assert fields["status"] == "completed"


@pytest.mark.asyncio
async def test_publish_run_completion_redis_failure_no_raise():
    bad_redis = MagicMock()
    bad_redis.xadd = AsyncMock(side_effect=RuntimeError("redis down"))
    bad_redis.expire = AsyncMock()
    await publish_run_completion(bad_redis, "run1", "sql")  # must not raise


# ===========================================================================
# _decode helper
# ===========================================================================

def test_decode_string():
    assert _decode("hello") == "hello"


def test_decode_bytes():
    assert _decode(b"hello") == "hello"


def test_decode_none():
    assert _decode(None) == ""


def test_decode_empty_string():
    assert _decode("") == ""


# ===========================================================================
# run_rlvr_cycle — duplicate detection
# ===========================================================================

@pytest.mark.asyncio
async def test_run_rlvr_cycle_skips_already_processed():
    from harness.workers.rlvr_worker import run_rlvr_cycle

    redis = fakeredis.FakeRedis(decode_responses=True)
    # Mark run as already processed
    await redis.sadd("harness:rlvr:processed", "run_done")

    mock_loop = MagicMock()
    mock_loop.process_episode = AsyncMock()

    await run_rlvr_cycle("run_done", "sql", mock_loop, redis)
    mock_loop.process_episode.assert_not_called()


@pytest.mark.asyncio
async def test_run_rlvr_cycle_processes_new_run():
    from harness.workers.rlvr_worker import run_rlvr_cycle

    redis = fakeredis.FakeRedis(decode_responses=True)
    mock_loop = MagicMock()
    mock_loop.process_episode = AsyncMock(return_value=None)

    await run_rlvr_cycle("new_run", "sql", mock_loop, redis)
    mock_loop.process_episode.assert_called_once_with("new_run", "sql")


@pytest.mark.asyncio
async def test_run_rlvr_cycle_marks_processed():
    from harness.workers.rlvr_worker import run_rlvr_cycle

    redis = fakeredis.FakeRedis(decode_responses=True)
    mock_loop = MagicMock()
    mock_loop.process_episode = AsyncMock(return_value=None)

    await run_rlvr_cycle("run_new", "sql", mock_loop, redis)

    is_member = await redis.sismember("harness:rlvr:processed", "run_new")
    assert bool(is_member) is True


@pytest.mark.asyncio
async def test_run_rlvr_cycle_handles_loop_exception():
    from harness.workers.rlvr_worker import run_rlvr_cycle

    redis = fakeredis.FakeRedis(decode_responses=True)
    mock_loop = MagicMock()
    mock_loop.process_episode = AsyncMock(side_effect=RuntimeError("loop failed"))

    # Should not raise — errors are caught and logged
    await run_rlvr_cycle("run_err", "sql", mock_loop, redis)


# ===========================================================================
# Hermes worker — run_hermes_cycle callable
# ===========================================================================

def test_hermes_worker_module_imports():
    from harness.workers.hermes_worker import run_hermes_cycle
    assert callable(run_hermes_cycle)


def test_agent_worker_module_imports():
    from harness.workers.agent_worker import process_run_job_async
    assert callable(process_run_job_async)


def test_rlvr_worker_module_imports():
    from harness.workers.rlvr_worker import run_event_driven, run_poll_mode, main
    assert callable(run_event_driven)
    assert callable(run_poll_mode)
    assert callable(main)
