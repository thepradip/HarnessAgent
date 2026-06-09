"""Unit tests for the heuristic complexity scorer."""

from __future__ import annotations

from harness.llm.complexity import (
    TIER_CHEAP,
    TIER_PREMIUM,
    TIER_STANDARD,
    HeuristicComplexityScorer,
    score_complexity,
)


def test_short_single_turn_is_cheap():
    assert score_complexity([{"role": "user", "content": "hi there"}]) == TIER_CHEAP


def test_tools_force_premium():
    out = score_complexity([{"role": "user", "content": "x"}], tools=[{"name": "t"}])
    assert out == TIER_PREMIUM


def test_code_fence_is_premium():
    out = score_complexity([{"role": "user", "content": "```py\nprint(1)\n```"}])
    assert out == TIER_PREMIUM


def test_hard_keyword_is_premium():
    out = score_complexity([{"role": "user", "content": "please optimize this SQL join"}])
    assert out == TIER_PREMIUM


def test_large_context_is_premium():
    out = score_complexity(
        [{"role": "user", "content": "summarize"}], required_context=50_000
    )
    assert out == TIER_PREMIUM


def test_long_input_is_premium():
    long_text = "word " * 2000  # ~2500 tokens at chars/4
    assert score_complexity([{"role": "user", "content": long_text}]) == TIER_PREMIUM


def test_many_turns_is_premium():
    msgs = [{"role": "user", "content": "ok"} for _ in range(15)]
    assert score_complexity(msgs) == TIER_PREMIUM


def test_medium_multiturn_is_standard():
    # 4 turns, moderate length, no hard signals -> standard
    msgs = [
        {"role": "user", "content": "Tell me about your day in a sentence."},
        {"role": "assistant", "content": "It was a fairly ordinary and calm sort of day overall."},
        {"role": "user", "content": "What did you have for lunch around noon today, roughly?"},
        {"role": "assistant", "content": "A simple sandwich with some salad on the side, nothing fancy."},
    ]
    assert score_complexity(msgs) == TIER_STANDARD


def test_thresholds_are_tunable():
    strict = HeuristicComplexityScorer(premium_min_tokens=5)
    # ~12 tokens of plain text (48 chars / 4) now trips premium under the tuned threshold
    text = "alpha beta gamma delta epsilon zeta eta theta"
    assert strict.score([{"role": "user", "content": text}]) == TIER_PREMIUM


def test_system_text_counts_toward_complexity():
    # A long system prompt should push a short user turn up from cheap.
    out = score_complexity(
        [{"role": "user", "content": "ok"}], system="optimize the recursive algorithm"
    )
    assert out == TIER_PREMIUM
