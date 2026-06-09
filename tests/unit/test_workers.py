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


# ===========================================================================
# Agent worker — _build_agent / build_agent_factory wiring
# ===========================================================================

def test_build_agent_is_module_level():
    # Regression: _build_agent used to live inside process_run_job_async,
    # so build_agent_factory()'s closure raised NameError when invoked.
    from harness.workers import agent_worker
    assert callable(getattr(agent_worker, "_build_agent", None))


def test_build_agent_factory_passes_redis_client(monkeypatch):
    from harness.workers import agent_worker

    sentinel_redis = object()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **kw: sentinel_redis)

    calls = {}

    def fake_build_agent(agent_type, cfg, config_dict=None, redis_client=None):
        calls["agent_type"] = agent_type
        calls["cfg"] = cfg
        calls["redis_client"] = redis_client
        return "agent_instance"

    monkeypatch.setattr(agent_worker, "_build_agent", fake_build_agent)

    cfg = MagicMock(redis_url="redis://localhost:6379/0")
    factory = agent_worker.build_agent_factory(cfg)
    assert factory("code") == "agent_instance"
    assert calls["agent_type"] == "code"
    assert calls["cfg"] is cfg
    assert calls["redis_client"] is sentinel_redis


@pytest.mark.asyncio
async def test_worker_agent_factory_passes_live_redis(monkeypatch):
    # Regression: the worker's agent_factory closure called _build_agent
    # without the connected redis client, so event streaming and cost
    # tracking silently got redis_client=None.
    from harness.workers import agent_worker

    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **kw: fake_redis)

    built = {}

    def fake_build_agent(agent_type, cfg, config_dict=None, redis_client=None):
        built["agent_type"] = agent_type
        built["redis_client"] = redis_client
        return MagicMock()

    monkeypatch.setattr(agent_worker, "_build_agent", fake_build_agent)

    captured = {}

    class _FakeRunner:
        def __init__(self, *, agent_factory, **kwargs):
            captured["agent_factory"] = agent_factory

        async def execute_run(self, run_id):
            return MagicMock(status="completed")

    monkeypatch.setattr("harness.orchestrator.runner.AgentRunner", _FakeRunner)

    await agent_worker.process_run_job_async("run_factory_test", {})

    factory = captured["agent_factory"]
    factory("sql")
    assert built["agent_type"] == "sql"
    assert built["redis_client"] is fake_redis
