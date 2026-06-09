"""
RLVR reward functions.

Three implementations:
  ExecutionRewardFn  — ground truth from sandbox execution (SQL, code)
  LLMVerifierRewardFn — deterministic LLM verifier (temperature=0, SHA-256 cache)
  EnsembleRewardFn   — weighted combination of both

Determinism guarantee for LLMVerifierRewardFn:
  - temperature = 0.0 (no sampling)
  - Identical (task, action, result, gold) → SHA-256 key → cache hit
  - Structured JSON output parsed to a fixed schema
  - Fallback rule-based checks when parsing fails
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

Verdict = Literal["correct", "incorrect", "partial"]

_REWARD_MAP: dict[str, float] = {
    "correct":   1.0,
    "partial":   0.5,
    "incorrect": 0.0,
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    reward: float                  # 0.0 – 1.0
    verdict: Verdict
    confidence: float              # verifier's self-reported confidence
    source: str                    # "execution" | "llm_verifier" | "ensemble" | "rule"
    reasoning: str = ""
    cached: bool = False

    def __post_init__(self) -> None:
        self.reward     = max(0.0, min(1.0, self.reward))
        self.confidence = max(0.0, min(1.0, self.confidence))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RewardFunction(Protocol):
    async def compute(
        self,
        task: str,
        action: str,
        result: Any,
        gold: str | None = None,
        **kwargs: Any,
    ) -> RewardSignal: ...


# ---------------------------------------------------------------------------
# Execution reward — ground truth from EvalSandbox
# ---------------------------------------------------------------------------

class ExecutionRewardFn:
    """
    Runs action and gold in a sandbox; reward = execution accuracy.
    Returns RewardSignal(source="execution").
    """

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    async def compute(
        self,
        task: str,
        action: str,
        result: Any = None,
        gold: str | None = None,
        **kwargs: Any,
    ) -> RewardSignal:
        if not gold:
            return RewardSignal(reward=0.5, verdict="partial",
                                confidence=0.5, source="execution",
                                reasoning="no gold provided")
        try:
            from harness.eval.scorers import score_execution_match
            sr = await score_execution_match(action, gold, self._sandbox, **kwargs)
            verdict: Verdict = (
                "correct" if sr.score >= 0.99 else
                "partial" if sr.score >= 0.5 else
                "incorrect"
            )
            return RewardSignal(
                reward=sr.score,
                verdict=verdict,
                confidence=1.0,  # execution is ground truth
                source="execution",
                reasoning=sr.details,
            )
        except Exception as exc:
            logger.debug("ExecutionRewardFn failed: %s", exc)
            return RewardSignal(reward=0.0, verdict="incorrect",
                                confidence=0.5, source="execution",
                                reasoning=str(exc))


# ---------------------------------------------------------------------------
# LLM verifier — deterministic (temperature=0, SHA-256 cache)
# ---------------------------------------------------------------------------

_VERIFIER_PROMPT = """\
You are a strict correctness verifier for an AI agent. Your verdict must be \
deterministic — given the same inputs you always return the same answer.

## Task
{task}

## Agent Action
{action}

## Action Result
{result}

## Gold Answer (reference)
{gold}

## Instructions
Reason step-by-step, then give a verdict.

Rules:
- CORRECT   : action fully and accurately addresses the task; result matches gold
- PARTIAL   : action is on the right track but incomplete or has minor errors
- INCORRECT : action is wrong, irrelevant, or the result clearly does not match gold

Think step by step inside <reasoning>...</reasoning> tags, then output exactly:

<reasoning>
1. Does the action address the task?
2. Does the result match the gold answer?
3. Are there any errors or omissions?
</reasoning>

<verdict>CORRECT|PARTIAL|INCORRECT</verdict>
<confidence>0.0-1.0</confidence>

After the tags, output ONLY valid JSON:
{{"verdict": "correct|partial|incorrect", "confidence": <float 0-1>, "reasoning": "<one sentence>"}}
"""


class LLMVerifierRewardFn:
    """
    Deterministic LLM-based reward.

    Determinism achieved by:
      1. temperature=0.0  — no sampling
      2. SHA-256 cache    — same inputs → same reward without a second LLM call
      3. Structured output parsing with rule-based fallback
    """

    def __init__(
        self,
        llm_provider: Any,
        cache_size: int = 4096,
    ) -> None:
        self._llm = llm_provider
        self._cache: dict[str, RewardSignal] = {}
        self._lock = threading.Lock()
        self._max_cache = cache_size

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(task: str, action: str, result: Any, gold: str | None) -> str:
        payload = json.dumps(
            {"task": task, "action": action,
             "result": str(result)[:2000], "gold": gold or ""},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get(self, key: str) -> RewardSignal | None:
        with self._lock:
            return self._cache.get(key)

    def _set(self, key: str, signal: RewardSignal) -> None:
        with self._lock:
            if len(self._cache) >= self._max_cache:
                # Evict oldest entry (insertion-ordered dict)
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = signal

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._cache), "max": self._max_cache}

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    async def compute(
        self,
        task: str,
        action: str,
        result: Any = None,
        gold: str | None = None,
        **kwargs: Any,
    ) -> RewardSignal:
        key = self._cache_key(task, action, result, gold)

        # Cache hit — fully deterministic
        cached = self._get(key)
        if cached is not None:
            return RewardSignal(
                reward=cached.reward, verdict=cached.verdict,
                confidence=cached.confidence, source=cached.source,
                reasoning=cached.reasoning, cached=True,
            )

        prompt = _VERIFIER_PROMPT.format(
            task=task,
            action=str(action)[:1000],
            result=str(result)[:2000] if result is not None else "(no result)",
            gold=str(gold)[:1000] if gold else "(no gold provided)",
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                system=(
                    "You are a deterministic correctness verifier. "
                    "Always respond with the same verdict for the same inputs. "
                    "Use temperature=0 reasoning — no creativity, only facts."
                ),
                temperature=0.0,
                skip_cache=False,
            )
            raw = response.content.strip()
        except Exception as exc:
            logger.warning("LLMVerifierRewardFn LLM call failed: %s", exc)
            return RewardSignal(reward=0.5, verdict="partial",
                                confidence=0.0, source="llm_verifier",
                                reasoning=f"llm_error: {exc}")

        signal = self._parse(raw)
        self._set(key, signal)
        return signal

    # ------------------------------------------------------------------
    # Parsing (structured + rule fallback — always returns a valid signal)
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> RewardSignal:
        # Try JSON block
        json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if json_match:
            try:
                d = json.loads(json_match.group())
                verdict_raw = str(d.get("verdict", "incorrect")).lower().strip()
                verdict: Verdict = (
                    "correct"   if "correct"   in verdict_raw and "incorrect" not in verdict_raw else
                    "partial"   if "partial"   in verdict_raw else
                    "incorrect"
                )
                confidence = float(d.get("confidence", 0.7))
                confidence = max(0.0, min(1.0, confidence))
                reward = _REWARD_MAP[verdict] * confidence + _REWARD_MAP[verdict] * (1 - confidence) * 0.5
                return RewardSignal(
                    reward=round(reward, 4),
                    verdict=verdict,
                    confidence=confidence,
                    source="llm_verifier",
                    reasoning=str(d.get("reasoning", ""))[:300],
                )
            except Exception:
                pass

        # Rule-based fallback — still deterministic
        return self._rule_fallback(raw)

    @staticmethod
    def _rule_fallback(raw: str) -> RewardSignal:
        """Deterministic keyword scan when JSON parsing fails."""
        upper = raw.upper()
        if "<VERDICT>CORRECT</VERDICT>" in upper or '"VERDICT": "CORRECT"' in upper:
            return RewardSignal(reward=1.0, verdict="correct",
                                confidence=0.7, source="llm_verifier",
                                reasoning="rule: CORRECT tag found")
        if "<VERDICT>PARTIAL</VERDICT>" in upper or '"VERDICT": "PARTIAL"' in upper:
            return RewardSignal(reward=0.5, verdict="partial",
                                confidence=0.7, source="llm_verifier",
                                reasoning="rule: PARTIAL tag found")
        if "<VERDICT>INCORRECT</VERDICT>" in upper or '"VERDICT": "INCORRECT"' in upper:
            return RewardSignal(reward=0.0, verdict="incorrect",
                                confidence=0.7, source="llm_verifier",
                                reasoning="rule: INCORRECT tag found")
        # True fallback: count positive vs negative keywords. Use word-boundary
        # regexes so "incorrect" is not counted as "correct" and "know" is not
        # counted as "no". Negatives are checked first / on equal footing so an
        # ambiguous response never reads as positive by accident.
        lower = raw.lower()

        def _count(words: tuple[str, ...]) -> int:
            return sum(
                1 for w in words
                if re.search(r"\b" + re.escape(w) + r"\b", lower)
            )

        neg = _count(("incorrect", "wrong", "no", "error", "mismatch"))
        pos = _count(("correct", "yes", "right", "accurate", "match"))
        if pos > neg:
            return RewardSignal(reward=0.7, verdict="correct",
                                confidence=0.5, source="llm_verifier",
                                reasoning="rule: keyword majority positive")
        if neg > pos:
            return RewardSignal(reward=0.0, verdict="incorrect",
                                confidence=0.5, source="llm_verifier",
                                reasoning="rule: keyword majority negative")
        return RewardSignal(reward=0.5, verdict="partial",
                            confidence=0.3, source="llm_verifier",
                            reasoning="rule: no clear signal")


# ---------------------------------------------------------------------------
# Ensemble reward — combines execution (ground truth) + LLM verifier
# ---------------------------------------------------------------------------

class EnsembleRewardFn:
    """
    Weighted combination of execution reward and LLM verifier reward.

    When execution sandbox is available: trust execution more (weight_exec=0.7).
    When no sandbox: fall back to LLM verifier alone.
    """

    def __init__(
        self,
        execution_fn: ExecutionRewardFn | None,
        llm_fn: LLMVerifierRewardFn,
        weight_exec: float = 0.7,
    ) -> None:
        self._exec = execution_fn
        self._llm = llm_fn
        self._w_exec = weight_exec
        self._w_llm = 1.0 - weight_exec

    async def compute(
        self,
        task: str,
        action: str,
        result: Any = None,
        gold: str | None = None,
        **kwargs: Any,
    ) -> RewardSignal:
        # LLM verifier always runs
        llm_sig = await self._llm.compute(task, action, result, gold, **kwargs)

        if self._exec is None:
            return RewardSignal(
                reward=llm_sig.reward,
                verdict=llm_sig.verdict,
                confidence=llm_sig.confidence,
                source="ensemble(llm_only)",
                reasoning=llm_sig.reasoning,
                cached=llm_sig.cached,
            )

        exec_sig = await self._exec.compute(task, action, result, gold, **kwargs)

        combined = self._w_exec * exec_sig.reward + self._w_llm * llm_sig.reward
        combined = round(combined, 4)
        verdict: Verdict = (
            "correct"   if combined >= 0.85 else
            "partial"   if combined >= 0.40 else
            "incorrect"
        )
        confidence = self._w_exec * exec_sig.confidence + self._w_llm * llm_sig.confidence

        return RewardSignal(
            reward=combined,
            verdict=verdict,
            confidence=round(confidence, 4),
            source=f"ensemble(exec={exec_sig.reward:.2f},llm={llm_sig.reward:.2f})",
            reasoning=f"exec: {exec_sig.reasoning} | llm: {llm_sig.reasoning}",
            cached=llm_sig.cached and exec_sig.confidence == 1.0,
        )
