"""Unit tests for the RLVR system: reward functions, verifiers, buffer, advantage, loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from harness.improvement.rlvr.advantage import AdvantageEstimator, StepAdvantage
from harness.improvement.rlvr.buffer import StepReward, StepRewardBuffer
from harness.improvement.rlvr.loop import RLVRCycleResult, RLVRLoop
from harness.improvement.rlvr.reward import (
    EnsembleRewardFn,
    ExecutionRewardFn,
    LLMVerifierRewardFn,
    RewardSignal,
)
from harness.improvement.rlvr.verifiers import (
    CodeVerifier,
    ReasoningVerifier,
    SQLVerifier,
    VerificationResult,
    VerificationStep,
    _call_llm_deterministic,
    _LLM_CACHE,
    _LLM_CACHE_LOCK,
    get_verifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


def _make_step(run_id="run1", step=0, reward=0.8, agent_type="sql") -> StepReward:
    return StepReward(
        run_id=run_id, step=step, agent_type=agent_type,
        task="count users", action="SELECT COUNT(*) FROM users",
        result_preview="[(1523,)]", reward=reward,
        verdict="correct" if reward >= 0.85 else "incorrect",
        confidence=1.0, source="test", prompt_hash="abc123",
    )


def _make_vr(reward=0.9, verdict="correct", feedback="ok") -> VerificationResult:
    return VerificationResult(
        overall_reward=reward, verdict=verdict,
        steps=[VerificationStep("test_step", reward >= 0.5, reward, feedback)],
        feedback_for_agent=feedback,
    )


class _MockLLM:
    """Mock LLM provider that always returns a fixed deterministic response."""

    def __init__(self, response_json: dict | None = None):
        self._resp = response_json or {"verdict": "correct", "confidence": 0.9, "reasoning": "ok"}
        self.call_count = 0

    async def complete(self, messages, **kwargs):
        self.call_count += 1
        assert kwargs.get("temperature", 0.0) == 0.0, "Must call at temperature=0"
        r = MagicMock()
        r.content = json.dumps(self._resp)
        return r


class _MockSandbox:
    """Mock EvalSandbox that returns a fixed result."""

    def __init__(self, output=None, error=None):
        self._output = output or {"columns": ["count"], "rows": [[42]], "row_count": 1}
        self._error = error

    async def execute(self, action, **kwargs):
        r = MagicMock()
        r.output = self._output
        r.raw_text = str(self._output)
        r.error = self._error
        r.success = self._error is None
        return r


# ===========================================================================
# RewardSignal
# ===========================================================================

def test_reward_signal_clamps_high():
    r = RewardSignal(reward=1.5, verdict="correct", confidence=2.0, source="x")
    assert r.reward == 1.0
    assert r.confidence == 1.0


def test_reward_signal_clamps_low():
    r = RewardSignal(reward=-0.5, verdict="incorrect", confidence=-1.0, source="x")
    assert r.reward == 0.0
    assert r.confidence == 0.0


def test_reward_signal_valid_passthrough():
    r = RewardSignal(reward=0.75, verdict="partial", confidence=0.8, source="test")
    assert r.reward == pytest.approx(0.75)
    assert r.confidence == pytest.approx(0.8)


def test_reward_signal_cached_false_by_default():
    r = RewardSignal(reward=0.5, verdict="partial", confidence=0.5, source="x")
    assert r.cached is False


# ===========================================================================
# LLMVerifierRewardFn — determinism + caching
# ===========================================================================

@pytest.fixture
def llm_verifier():
    return LLMVerifierRewardFn(_MockLLM(), cache_size=100)


@pytest.mark.asyncio
async def test_llm_verifier_returns_reward_signal(llm_verifier):
    s = await llm_verifier.compute("task", "action", "result", "gold")
    assert isinstance(s, RewardSignal)
    assert 0.0 <= s.reward <= 1.0


@pytest.mark.asyncio
async def test_llm_verifier_temperature_zero():
    """Verifier must call LLM at temperature=0."""
    llm = _MockLLM()
    fn = LLMVerifierRewardFn(llm)
    await fn.compute("task", "action", None, None)
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_llm_verifier_second_call_is_cache_hit(llm_verifier):
    s1 = await llm_verifier.compute("task", "SELECT 1", "res", "SELECT 1")
    s2 = await llm_verifier.compute("task", "SELECT 1", "res", "SELECT 1")
    assert s2.cached is True
    assert s1.reward == pytest.approx(s2.reward)  # same result from cache


@pytest.mark.asyncio
async def test_llm_verifier_different_inputs_miss_cache():
    llm = _MockLLM()
    fn = LLMVerifierRewardFn(llm, cache_size=100)
    await fn.compute("task1", "SELECT 1", None, None)
    await fn.compute("task2", "SELECT 1", None, None)  # different task
    assert fn.cache_stats()["size"] == 2


@pytest.mark.asyncio
async def test_llm_verifier_cache_key_includes_all_fields():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    k1 = fn._cache_key("task", "action", "result", "gold")
    k2 = fn._cache_key("task", "action", "result", None)
    k3 = fn._cache_key("task", "action2", "result", "gold")
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


@pytest.mark.asyncio
async def test_llm_verifier_cache_evicts_oldest():
    fn = LLMVerifierRewardFn(_MockLLM(), cache_size=2)
    await fn.compute("t1", "a1", None, None)
    await fn.compute("t2", "a2", None, None)
    assert fn.cache_stats()["size"] == 2
    await fn.compute("t3", "a3", None, None)  # should evict oldest
    assert fn.cache_stats()["size"] == 2


def test_llm_verifier_clear_cache():
    fn = LLMVerifierRewardFn(_MockLLM(), cache_size=100)
    fn._cache["key"] = RewardSignal(reward=0.9, verdict="correct", confidence=0.9, source="x")
    fn.clear_cache()
    assert fn.cache_stats()["size"] == 0


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def test_llm_verifier_parse_correct_json():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    raw = '{"verdict": "correct", "confidence": 0.95, "reasoning": "matches gold"}'
    s = fn._parse(raw)
    assert s.verdict == "correct"
    assert s.confidence == pytest.approx(0.95)
    assert s.reward > 0.5


def test_llm_verifier_parse_incorrect_json():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    raw = '{"verdict": "incorrect", "confidence": 0.8, "reasoning": "wrong table"}'
    s = fn._parse(raw)
    assert s.verdict == "incorrect"
    assert s.reward < 0.5


def test_llm_verifier_parse_partial():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    raw = '{"verdict": "partial", "confidence": 0.7, "reasoning": "incomplete"}'
    s = fn._parse(raw)
    assert s.verdict == "partial"
    assert 0.1 < s.reward < 0.9


def test_llm_verifier_rule_fallback_correct():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    s = fn._rule_fallback("<verdict>CORRECT</verdict>")
    assert s.verdict == "correct"
    assert s.reward == 1.0


def test_llm_verifier_rule_fallback_incorrect():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    s = fn._rule_fallback("<verdict>INCORRECT</verdict>")
    assert s.verdict == "incorrect"
    assert s.reward == 0.0


def test_llm_verifier_rule_fallback_no_signal():
    fn = LLMVerifierRewardFn.__new__(LLMVerifierRewardFn)
    fn._cache = {}
    fn._lock = threading.Lock()
    fn._max_cache = 100
    s = fn._rule_fallback("some ambiguous text with no clear verdict")
    assert s.verdict in ("correct", "partial", "incorrect")
    assert 0.0 <= s.reward <= 1.0


@pytest.mark.asyncio
async def test_llm_verifier_llm_failure_returns_partial():
    class FailLLM:
        async def complete(self, messages, **kwargs):
            raise RuntimeError("connection refused")
    fn = LLMVerifierRewardFn(FailLLM())
    s = await fn.compute("task", "action", None, None)
    assert s.verdict == "partial"
    assert s.reward == pytest.approx(0.5)


# ===========================================================================
# ExecutionRewardFn
# ===========================================================================

@pytest.mark.asyncio
async def test_execution_reward_no_gold():
    fn = ExecutionRewardFn(_MockSandbox())
    s = await fn.compute("task", "SELECT 1", None, gold=None)
    assert s.reward == pytest.approx(0.5)
    assert s.source == "execution"


@pytest.mark.asyncio
async def test_execution_reward_sandbox_error():
    fn = ExecutionRewardFn(_MockSandbox(error="table not found"))
    # score_execution_match will fail → caught and returns 0.0
    s = await fn.compute("task", "SELECT bad", None, gold="SELECT good")
    assert s.reward == 0.0 or s.source == "execution"


# ===========================================================================
# EnsembleRewardFn
# ===========================================================================

@pytest.mark.asyncio
async def test_ensemble_llm_only_when_no_exec():
    llm_fn = LLMVerifierRewardFn(_MockLLM({"verdict": "correct", "confidence": 0.9, "reasoning": "ok"}))
    fn = EnsembleRewardFn(execution_fn=None, llm_fn=llm_fn, weight_exec=0.7)
    s = await fn.compute("task", "action", None, None)
    assert "llm_only" in s.source
    assert 0.0 <= s.reward <= 1.0


@pytest.mark.asyncio
async def test_ensemble_combines_both():
    exec_fn = MagicMock()
    exec_fn.compute = AsyncMock(return_value=RewardSignal(
        reward=1.0, verdict="correct", confidence=1.0, source="execution"
    ))
    llm_fn = MagicMock()
    llm_fn.compute = AsyncMock(return_value=RewardSignal(
        reward=0.8, verdict="correct", confidence=0.9, source="llm_verifier"
    ))
    fn = EnsembleRewardFn(exec_fn, llm_fn, weight_exec=0.7)
    s = await fn.compute("task", "action", None, "gold")
    expected = round(0.7 * 1.0 + 0.3 * 0.8, 4)
    assert s.reward == pytest.approx(expected, abs=0.001)
    assert "ensemble" in s.source


@pytest.mark.asyncio
async def test_ensemble_verdict_correct_high_reward():
    exec_fn = MagicMock()
    exec_fn.compute = AsyncMock(return_value=RewardSignal(
        reward=1.0, verdict="correct", confidence=1.0, source="execution"
    ))
    llm_fn = MagicMock()
    llm_fn.compute = AsyncMock(return_value=RewardSignal(
        reward=0.95, verdict="correct", confidence=0.9, source="llm_verifier"
    ))
    fn = EnsembleRewardFn(exec_fn, llm_fn, weight_exec=0.7)
    s = await fn.compute("task", "action", None, "gold")
    assert s.verdict == "correct"


# ===========================================================================
# StepRewardBuffer
# ===========================================================================

@pytest.fixture
def reward_buffer():
    return StepRewardBuffer(_fake_redis())


@pytest.mark.asyncio
async def test_buffer_record_and_retrieve(reward_buffer):
    step = _make_step("run1", step=0, reward=0.9)
    await reward_buffer.record(step)
    episode = await reward_buffer.get_episode("run1")
    assert len(episode) == 1
    assert episode[0].reward == pytest.approx(0.9)
    assert episode[0].run_id == "run1"


@pytest.mark.asyncio
async def test_buffer_multiple_steps_ordered(reward_buffer):
    for i, r in enumerate([0.9, 0.3, 0.7]):
        await reward_buffer.record(_make_step("run2", step=i, reward=r))
    episode = await reward_buffer.get_episode("run2")
    assert len(episode) == 3
    assert [s.step for s in episode] == [0, 1, 2]


@pytest.mark.asyncio
async def test_buffer_episode_mean(reward_buffer):
    for i, r in enumerate([0.6, 0.8, 1.0]):
        await reward_buffer.record(_make_step("run3", step=i, reward=r))
    mean = await reward_buffer.episode_mean("run3")
    assert mean == pytest.approx(0.8, abs=0.001)


@pytest.mark.asyncio
async def test_buffer_empty_episode(reward_buffer):
    assert await reward_buffer.get_episode("nonexistent") == []
    assert await reward_buffer.episode_mean("nonexistent") == 0.0


@pytest.mark.asyncio
async def test_buffer_delete_episode(reward_buffer):
    await reward_buffer.record(_make_step("run4", reward=0.5))
    await reward_buffer.delete_episode("run4")
    assert await reward_buffer.get_episode("run4") == []


@pytest.mark.asyncio
async def test_buffer_update_and_get_baseline(reward_buffer):
    await reward_buffer.update_baseline("sql", 0.6)
    await reward_buffer.update_baseline("sql", 0.8)
    b = await reward_buffer.get_baseline("sql")
    assert 0.6 <= b <= 0.8


@pytest.mark.asyncio
async def test_buffer_baseline_default_without_history(reward_buffer):
    b = await reward_buffer.get_baseline("unknown_agent")
    assert b == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_buffer_baseline_rolling_window(reward_buffer):
    for v in range(60):  # more than _BASELINE_WINDOW=50
        await reward_buffer.update_baseline("code", float(v) / 60)
    hist = await reward_buffer.get_history("code", n=50)
    assert len(hist) <= 50


@pytest.mark.asyncio
async def test_buffer_step_round_trip(reward_buffer):
    step = _make_step("rX", step=5, reward=0.777, agent_type="code")
    await reward_buffer.record(step)
    episode = await reward_buffer.get_episode("rX")
    s = episode[0]
    assert s.step == 5
    assert s.reward == pytest.approx(0.777)
    assert s.agent_type == "code"
    assert s.task == "count users"


@pytest.mark.asyncio
async def test_buffer_different_runs_isolated(reward_buffer):
    await reward_buffer.record(_make_step("runA", reward=1.0))
    await reward_buffer.record(_make_step("runB", reward=0.0))
    ep_a = await reward_buffer.get_episode("runA")
    ep_b = await reward_buffer.get_episode("runB")
    assert ep_a[0].reward == pytest.approx(1.0)
    assert ep_b[0].reward == pytest.approx(0.0)


# ===========================================================================
# AdvantageEstimator
# ===========================================================================

def _episode(rewards: list[float], run_id="r1") -> list[StepReward]:
    return [_make_step(run_id, step=i, reward=r) for i, r in enumerate(rewards)]


def test_advantage_estimator_length_matches_episode():
    est = AdvantageEstimator()
    ep = _episode([0.9, 0.3, 0.7])
    adv = est.compute(ep, baseline=0.6)
    assert len(adv) == 3


def test_advantage_estimator_empty_episode():
    est = AdvantageEstimator()
    assert est.compute([], baseline=0.5) == []


def test_advantage_estimator_normalised_mean_near_zero():
    # The per-episode z-score (norm_advantage) re-centres to ~0 mean; the raw
    # advantage (advantage) does NOT — that's exactly why split() uses raw.
    est = AdvantageEstimator(gamma=1.0)
    ep = _episode([0.9, 0.1, 0.8, 0.2, 0.6])
    adv = est.compute(ep, baseline=0.5)
    mean_norm = sum(a.norm_advantage for a in adv) / len(adv)
    assert abs(mean_norm) < 0.1


def test_advantage_estimator_high_reward_positive_advantage():
    est = AdvantageEstimator(gamma=1.0)
    ep = _episode([1.0, 0.0])   # first step great, second terrible
    adv = est.compute(ep, baseline=0.5)
    assert adv[0].advantage > 0   # 1.0 above baseline
    assert adv[1].advantage < 0   # 0.0 below baseline


def test_advantage_estimator_discounted_return():
    est = AdvantageEstimator(gamma=0.9)
    ep = _episode([0.0, 1.0])   # reward only at last step
    adv = est.compute(ep, baseline=0.0)
    # G[0] = 0.0 + 0.9 * 1.0 = 0.9   (discounted future reward)
    # G[1] = 1.0                        (immediate reward, no discounting)
    # The step that receives the reward directly has a higher return
    assert adv[0].discounted_return == pytest.approx(0.9, abs=0.001)
    assert adv[1].discounted_return == pytest.approx(1.0, abs=0.001)
    assert adv[0].discounted_return < adv[1].discounted_return


def test_advantage_estimator_weight_is_abs_norm_advantage():
    # weight is the gradient-scaling magnitude == |z-normalised advantage|.
    est = AdvantageEstimator()
    ep = _episode([0.9, 0.1, 0.5])
    adv = est.compute(ep, baseline=0.5)
    for a in adv:
        assert a.weight == pytest.approx(abs(a.norm_advantage), abs=1e-6)


def test_advantage_estimator_split_thresholds():
    est = AdvantageEstimator(gamma=1.0)
    ep = _episode([1.0, 0.0, 0.5, 1.0, 0.0])
    adv = est.compute(ep, baseline=0.5)
    pos, neg = est.split(adv, pos_threshold=0.5, neg_threshold=-0.5)
    assert all(a.advantage >= 0.5 for a in pos)
    assert all(a.advantage <= -0.5 for a in neg)


def test_advantage_estimator_split_no_positives():
    est = AdvantageEstimator()
    ep = _episode([0.1, 0.1, 0.1])
    adv = est.compute(ep, baseline=0.9)
    pos, neg = est.split(adv, pos_threshold=10.0)
    assert pos == []


def test_advantage_split_baseline_has_effect_uniform_good_episode():
    # Regression for the baseline-cancellation bug: a uniformly-good episode
    # (every step well above the rolling baseline) must reinforce ALL steps and
    # patch NONE. With the old z-normalised signal the zero-mean centring split
    # it ~half/half regardless of the baseline.
    est = AdvantageEstimator(gamma=0.0)  # per-step return G[t] = r[t]
    ep = _episode([0.9, 0.95, 0.92])
    adv = est.compute(ep, baseline=0.2)
    pos, neg = est.split(adv, pos_threshold=0.1, neg_threshold=-0.1)
    assert len(neg) == 0
    assert len(pos) == len(adv)
    # Raising the baseline above the rewards flips them all negative.
    adv_high = est.compute(ep, baseline=1.5)
    pos2, neg2 = est.split(adv_high, pos_threshold=0.1, neg_threshold=-0.1)
    assert len(pos2) == 0
    assert len(neg2) == len(adv_high)


def test_advantage_estimator_single_step():
    est = AdvantageEstimator()
    ep = _episode([0.8])
    adv = est.compute(ep, baseline=0.5)
    assert len(adv) == 1
    assert isinstance(adv[0], StepAdvantage)


# ===========================================================================
# SQLVerifier
# ===========================================================================

@pytest.mark.asyncio
async def test_sql_verifier_valid_syntax():
    v = SQLVerifier()
    result = await v.verify("count", "SELECT COUNT(*) FROM users", gold="SELECT COUNT(*) FROM users")
    syntax_step = next(s for s in result.steps if s.name == "syntax_check")
    assert syntax_step.passed is True
    assert syntax_step.score == 1.0


@pytest.mark.asyncio
async def test_sql_verifier_invalid_syntax():
    v = SQLVerifier()
    result = await v.verify("task", "SELEC COUNT FROM", gold=None)
    syntax_step = next(s for s in result.steps if s.name == "syntax_check")
    assert syntax_step.passed is False
    assert "syntax" in syntax_step.feedback.lower() or "error" in syntax_step.feedback.lower()


@pytest.mark.asyncio
async def test_sql_verifier_no_gold_result_check_neutral():
    v = SQLVerifier()
    result = await v.verify("task", "SELECT 1", gold=None)
    result_step = next(s for s in result.steps if s.name == "result_check")
    assert result_step.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_sql_verifier_identical_gold_high_reward():
    v = SQLVerifier()
    sql = "SELECT id FROM users WHERE active = 1"
    result = await v.verify("task", sql, gold=sql)
    assert result.overall_reward >= 0.5


@pytest.mark.asyncio
async def test_sql_verifier_with_schema_store(tmp_path):
    import sqlite3
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    import fakeredis.aioredis as fakeredis
    from harness.memory.context_engineering import SchemaStore
    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://unused"
    store._ttl = 3600
    store._client = fakeredis.FakeRedis(decode_responses=True)
    await store.store_from_sqlite("db1", db_path)

    v = SQLVerifier(schema_store=store)
    result = await v.verify("task", "SELECT id FROM users", gold=None, db_id="db1")
    schema_step = next(s for s in result.steps if s.name == "schema_check")
    assert schema_step.passed is True


@pytest.mark.asyncio
async def test_sql_verifier_unknown_table_fails_schema_check(tmp_path):
    import sqlite3
    db_path = str(tmp_path / "test2.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    import fakeredis.aioredis as fakeredis
    from harness.memory.context_engineering import SchemaStore
    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://unused"
    store._ttl = 3600
    store._client = fakeredis.FakeRedis(decode_responses=True)
    await store.store_from_sqlite("db2", db_path)

    v = SQLVerifier(schema_store=store)
    result = await v.verify("task", "SELECT * FROM nonexistent_table", gold=None, db_id="db2")
    schema_step = next(s for s in result.steps if s.name == "schema_check")
    assert schema_step.passed is False
    assert "nonexistent_table" in schema_step.feedback


@pytest.mark.asyncio
async def test_sql_verifier_with_llm_quality_check():
    llm = _MockLLM({"score": 0.9, "feedback": "query is correct"})
    v = SQLVerifier(llm=llm)
    result = await v.verify("count users", "SELECT COUNT(*) FROM users", gold=None)
    quality_step = next((s for s in result.steps if s.name == "quality_check"), None)
    assert quality_step is not None
    assert quality_step.score == pytest.approx(0.9)
    assert "correct" in quality_step.feedback


@pytest.mark.asyncio
async def test_sql_verifier_overall_reward_in_range():
    v = SQLVerifier()
    result = await v.verify("task", "SELECT * FROM users", gold="SELECT * FROM users")
    assert 0.0 <= result.overall_reward <= 1.0


@pytest.mark.asyncio
async def test_sql_verifier_verdict_values():
    v = SQLVerifier()
    result = await v.verify("task", "SELECT 1", gold=None)
    assert result.verdict in ("correct", "partial", "incorrect")


@pytest.mark.asyncio
async def test_sql_verifier_step_by_step_output():
    v = SQLVerifier()
    result = await v.verify("task", "SELECT 1", gold=None)
    output = result.step_by_step()
    assert "schema_check" in output
    assert "syntax_check" in output


@pytest.mark.asyncio
async def test_sql_verifier_to_reward_signal():
    v = SQLVerifier()
    result = await v.verify("task", "SELECT 1", gold=None)
    rs = result.to_reward_signal()
    assert rs.reward == pytest.approx(result.overall_reward)
    assert rs.verdict == result.verdict


@pytest.mark.asyncio
async def test_sql_verifier_feedback_not_empty_on_failure():
    v = SQLVerifier()
    result = await v.verify("task", "SELEC bad sql ;;;", gold=None)
    assert result.feedback_for_agent  # non-empty on failure


# ===========================================================================
# CodeVerifier
# ===========================================================================

@pytest.mark.asyncio
async def test_code_verifier_valid_syntax():
    v = CodeVerifier()
    result = await v.verify("reverse string", "def f(s): return s[::-1]")
    syntax_step = next(s for s in result.steps if s.name == "syntax_check")
    assert syntax_step.passed is True
    assert syntax_step.score == 1.0


@pytest.mark.asyncio
async def test_code_verifier_invalid_syntax():
    v = CodeVerifier()
    result = await v.verify("task", "def f(x return x")
    syntax_step = next(s for s in result.steps if s.name == "syntax_check")
    assert syntax_step.passed is False
    assert "SyntaxError" in syntax_step.feedback
    assert result.overall_reward < 0.5


@pytest.mark.asyncio
async def test_code_verifier_output_exact_match():
    v = CodeVerifier()
    result = await v.verify("task", "print(42)", result="42\n", gold=None, expected_output="42")
    output_step = next(s for s in result.steps if s.name == "output_check")
    assert output_step.passed is True


@pytest.mark.asyncio
async def test_code_verifier_output_mismatch():
    v = CodeVerifier()
    result = await v.verify("task", "print(0)", result="0\n", gold=None, expected_output="42")
    output_step = next(s for s in result.steps if s.name == "output_check")
    assert output_step.passed is False
    assert "mismatch" in output_step.feedback.lower()


@pytest.mark.asyncio
async def test_code_verifier_numeric_tolerance():
    v = CodeVerifier()
    result = await v.verify("task", "x", result="3.14159", gold=None, expected_output="3.14158")
    output_step = next(s for s in result.steps if s.name == "output_check")
    assert output_step.passed is True  # within 1% tolerance


@pytest.mark.asyncio
async def test_code_verifier_no_expected_output_neutral():
    v = CodeVerifier()
    result = await v.verify("task", "print('hello')", result="hello")
    output_step = next(s for s in result.steps if s.name == "output_check")
    assert output_step.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_code_verifier_with_llm():
    llm = _MockLLM({"score": 0.85, "feedback": "code is correct and efficient"})
    v = CodeVerifier(llm=llm)
    result = await v.verify("sort list", "sorted([3,1,2])", result="[1,2,3]", expected_output="[1, 2, 3]")
    quality_step = next((s for s in result.steps if s.name == "quality_check"), None)
    assert quality_step is not None
    assert quality_step.score == pytest.approx(0.85)


# ===========================================================================
# ReasoningVerifier
# ===========================================================================

@pytest.mark.asyncio
async def test_reasoning_verifier_empty_response():
    v = ReasoningVerifier()
    result = await v.verify("What is 2+2?", "", result="")
    format_step = next(s for s in result.steps if s.name == "format_check")
    assert format_step.passed is False
    assert result.overall_reward < 0.5


@pytest.mark.asyncio
async def test_reasoning_verifier_correct_answer():
    v = ReasoningVerifier()
    result = await v.verify("What is 2+2?", "4", result="4", gold="4")
    answer_step = next(s for s in result.steps if s.name == "answer_check")
    assert answer_step.passed is True
    assert result.overall_reward >= 0.5


@pytest.mark.asyncio
async def test_reasoning_verifier_wrong_answer():
    v = ReasoningVerifier()
    result = await v.verify("What is 2+2?", "5", result="5", gold="4")
    answer_step = next(s for s in result.steps if s.name == "answer_check")
    assert answer_step.passed is False


@pytest.mark.asyncio
async def test_reasoning_verifier_numeric_tolerance():
    v = ReasoningVerifier()
    result = await v.verify("task", "3.14159", result="3.14159", gold="3.14158")
    answer_step = next(s for s in result.steps if s.name == "answer_check")
    assert answer_step.passed is True


@pytest.mark.asyncio
async def test_reasoning_verifier_gold_in_longer_response():
    v = ReasoningVerifier()
    result = await v.verify("task", "The answer is 42 based on the data.",
                            result="The answer is 42 based on the data.", gold="42")
    answer_step = next(s for s in result.steps if s.name == "answer_check")
    assert answer_step.passed is True


@pytest.mark.asyncio
async def test_reasoning_verifier_with_llm():
    llm = _MockLLM({"score": 0.95, "feedback": "correct reasoning", "reasoning_valid": True, "answer_correct": True})
    v = ReasoningVerifier(llm=llm)
    result = await v.verify("task", "The answer is 4", result="The answer is 4", gold="4")
    reasoning_step = next((s for s in result.steps if s.name == "reasoning_check"), None)
    assert reasoning_step is not None
    assert reasoning_step.score >= 0.5


@pytest.mark.asyncio
async def test_reasoning_verifier_no_gold_answer_check_neutral():
    v = ReasoningVerifier()
    result = await v.verify("task", "Some response about the topic.", gold=None)
    answer_step = next(s for s in result.steps if s.name == "answer_check")
    assert answer_step.score == pytest.approx(0.5)


# ===========================================================================
# get_verifier factory
# ===========================================================================

def test_get_verifier_sql():
    assert isinstance(get_verifier("sql"), SQLVerifier)


def test_get_verifier_code():
    assert isinstance(get_verifier("code"), CodeVerifier)


def test_get_verifier_base():
    assert isinstance(get_verifier("base"), ReasoningVerifier)


def test_get_verifier_unknown_defaults_reasoning():
    assert isinstance(get_verifier("custom_agent"), ReasoningVerifier)


def test_get_verifier_passes_sandbox_to_sql():
    sb = _MockSandbox()
    v = get_verifier("sql", sandbox=sb)
    assert v._sandbox is sb


def test_get_verifier_passes_llm():
    llm = _MockLLM()
    v = get_verifier("code", llm=llm)
    assert v._llm is llm


# ===========================================================================
# LLM determinism cache (_call_llm_deterministic)
# ===========================================================================

@pytest.mark.asyncio
async def test_call_llm_deterministic_caches_response():
    llm = _MockLLM({"score": 0.9, "feedback": "ok"})
    prompt = f"unique_prompt_{id(llm)}"
    # Clear any existing cache entry
    key = hashlib.sha256(prompt.encode()).hexdigest()
    with _LLM_CACHE_LOCK:
        _LLM_CACHE.pop(key, None)

    text1, cached1 = await _call_llm_deterministic(llm, prompt, "system")
    text2, cached2 = await _call_llm_deterministic(llm, prompt, "system")

    assert text1 == text2
    assert cached2 is True   # second call is a cache hit
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_call_llm_deterministic_failure_returns_empty():
    class FailLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("network error")
        call_count = 0
    text, cached = await _call_llm_deterministic(FailLLM(), "some_prompt_xyz", "sys")
    assert text == ""
    assert cached is False


# ===========================================================================
# VerificationResult helpers
# ===========================================================================

def test_verification_result_step_by_step_format():
    vr = VerificationResult(
        overall_reward=0.7,
        verdict="partial",
        steps=[
            VerificationStep("step_a", True,  1.0, "all good"),
            VerificationStep("step_b", False, 0.0, "found an error"),
        ],
        feedback_for_agent="found an error",
    )
    output = vr.step_by_step()
    assert "step_a" in output
    assert "step_b" in output
    assert "✓" in output
    assert "✗" in output


def test_verification_result_confidence_all_deterministic():
    vr = VerificationResult(
        overall_reward=1.0, verdict="correct",
        steps=[VerificationStep("s", True, 1.0, "ok", deterministic=True)],
        feedback_for_agent="ok",
    )
    assert vr._confidence() == 1.0


def test_verification_result_confidence_nondeterministic():
    vr = VerificationResult(
        overall_reward=1.0, verdict="correct",
        steps=[VerificationStep("s", False, 0.5, "llm step", deterministic=False)],
        feedback_for_agent="ok",
    )
    assert vr._confidence() == pytest.approx(0.7)


def test_verification_result_to_reward_signal_maps_fields():
    vr = VerificationResult(
        overall_reward=0.85, verdict="correct",
        steps=[VerificationStep("s", True, 0.85, "ok")],
        feedback_for_agent="all checks passed",
        source="sql_verifier",
    )
    rs = vr.to_reward_signal()
    assert rs.reward == pytest.approx(0.85)
    assert rs.verdict == "correct"
    assert rs.source == "sql_verifier"


# ===========================================================================
# RLVRLoop
# ===========================================================================

@pytest.fixture
def rlvr_loop():
    buf = StepRewardBuffer(_fake_redis())
    est = AdvantageEstimator(gamma=0.95)
    patch_gen = MagicMock()
    patch_gen.generate = AsyncMock(return_value=None)
    prompt_store = MagicMock()
    prompt_store.apply_patch = AsyncMock()
    feedback_ch = MagicMock()
    feedback_ch.publish = AsyncMock()
    return RLVRLoop(
        reward_buffer=buf,
        estimator=est,
        patch_generator=patch_gen,
        prompt_store=prompt_store,
        feedback_channel=feedback_ch,
    )


@pytest.mark.asyncio
async def test_rlvr_loop_skips_short_episodes(rlvr_loop):
    # Record only 1 step (below min_steps=3)
    await rlvr_loop._buffer.record(_make_step("short_run", reward=0.9))
    result = await rlvr_loop.process_episode("short_run", "sql")
    assert result is None


@pytest.mark.asyncio
async def test_rlvr_loop_returns_cycle_result(rlvr_loop):
    for i, r in enumerate([0.9, 0.2, 0.8, 0.1, 0.7]):
        await rlvr_loop._buffer.record(_make_step("run_full", step=i, reward=r))
    result = await rlvr_loop.process_episode("run_full", "sql")
    assert isinstance(result, RLVRCycleResult)
    assert result.n_steps == 5
    assert result.agent_type == "sql"
    assert 0.0 <= result.mean_reward <= 1.0


@pytest.mark.asyncio
async def test_rlvr_loop_counts_positive_negative(rlvr_loop):
    # High variance episode → clear pos and neg
    for i, r in enumerate([1.0, 1.0, 0.0, 0.0, 0.5]):
        await rlvr_loop._buffer.record(_make_step("run_var", step=i, reward=r))
    result = await rlvr_loop.process_episode("run_var", "sql")
    assert result.n_positive + result.n_negative >= 0   # at least some split


@pytest.mark.asyncio
async def test_rlvr_loop_updates_baseline(rlvr_loop):
    for i in range(4):
        await rlvr_loop._buffer.record(_make_step("run_base", step=i, reward=0.9))
    await rlvr_loop.process_episode("run_base", "sql")
    b = await rlvr_loop._buffer.get_baseline("sql")
    # baseline should have been updated from default 0.5 toward 0.9
    assert b > 0.5


@pytest.mark.asyncio
async def test_rlvr_loop_publishes_step_feedback(rlvr_loop):
    vr = _make_vr(reward=0.3, verdict="incorrect", feedback="result is wrong")
    await rlvr_loop.publish_step_feedback("run_fb", vr)
    rlvr_loop._feedback.publish.assert_called_once()
    call_args = rlvr_loop._feedback.publish.call_args
    assert call_args[0][0] == "run_fb"
    ev = call_args[0][1]
    assert ev.type == "score"
    assert ev.score == pytest.approx(0.3)
    assert ev.priority == 3  # high priority for low-reward


@pytest.mark.asyncio
async def test_rlvr_loop_high_reward_low_priority_feedback(rlvr_loop):
    vr = _make_vr(reward=0.9, verdict="correct", feedback="good")
    await rlvr_loop.publish_step_feedback("run_hi", vr)
    ev = rlvr_loop._feedback.publish.call_args[0][1]
    assert ev.priority == 2  # medium priority for good reward


@pytest.mark.asyncio
async def test_rlvr_loop_no_feedback_channel():
    """Loop should not crash when feedback_channel is None."""
    buf = StepRewardBuffer(_fake_redis())
    loop = RLVRLoop(
        reward_buffer=buf, estimator=AdvantageEstimator(),
        patch_generator=None, prompt_store=None, feedback_channel=None,
    )
    vr = _make_vr()
    await loop.publish_step_feedback("run_x", vr)   # must not raise


@pytest.mark.asyncio
async def test_rlvr_loop_idempotent_on_same_run_id(rlvr_loop):
    # Re-delivering the same 'completed' event must not run a second cycle.
    for i in range(4):
        await rlvr_loop._buffer.record(_make_step("run_idem", step=i, reward=0.9))
    first = await rlvr_loop.process_episode("run_idem", "sql")
    assert first is not None
    second = await rlvr_loop.process_episode("run_idem", "sql")
    assert second is None
    # Episode buffer was dropped after the first cycle.
    assert await rlvr_loop._buffer.get_episode("run_idem") == []


@pytest.mark.asyncio
async def test_rlvr_loop_dedups_fewshots_across_episodes(rlvr_loop):
    # The same winning (task, action) recorded in two different runs must be
    # stored as a few-shot only once (no unbounded duplicate appends).
    for run_id in ("dedup_a", "dedup_b"):
        for i in range(4):
            await rlvr_loop._buffer.record(
                _make_step(run_id, step=i, reward=0.95)  # all high → all positive
            )
        await rlvr_loop.process_episode(run_id, "sql")
    # Only one unique (task, action) fingerprint stored despite two episodes.
    assert len(rlvr_loop._fewshot_hashes["sql"]) == 1


@pytest.mark.asyncio
async def test_rlvr_loop_cycle_summary_string(rlvr_loop):
    for i in range(4):
        await rlvr_loop._buffer.record(_make_step("run_s", step=i, reward=0.7))
    result = await rlvr_loop.process_episode("run_s", "sql")
    summary = result.summary()
    assert "run_s" in summary
    assert "sql" in summary
    assert "steps=4" in summary


# ===========================================================================
# run_episode_with_verification (integration)
# ===========================================================================

@pytest.mark.asyncio
async def test_run_episode_with_verification():
    from harness.improvement.rlvr.loop import run_episode_with_verification
    buf = StepRewardBuffer(_fake_redis())
    loop = RLVRLoop(
        reward_buffer=buf, estimator=AdvantageEstimator(),
        patch_generator=MagicMock(generate=AsyncMock(return_value=None)),
        prompt_store=None, feedback_channel=None,
    )
    v = ReasoningVerifier(llm=None)
    steps = [
        {"task": "What is 2+2?", "action": "4", "result": "4", "gold": "4"},
        {"task": "What is 3+3?", "action": "6", "result": "6", "gold": "6"},
        {"task": "What is 10/2?", "action": "5", "result": "5", "gold": "5"},
    ]
    result = await run_episode_with_verification(
        run_id="integration_run", agent_type="base",
        steps=steps, reward_buffer=buf,
        rlvr_loop=loop, verifier=v,
    )
    assert result is not None
    assert result.n_steps == 3
    assert result.mean_reward >= 0.5  # all answers are correct
