"""
Domain-specific verifiers for RLVR.

Each verifier runs an ordered pipeline of sub-checks and returns a
VerificationResult with per-step scores, verdicts, and actionable feedback.

Determinism guarantee per step:
  rule-based steps  — pure logic, always deterministic
  execution steps   — sandbox.execute() is deterministic for same input
  llm steps         — temperature=0.0 + SHA-256 cache (same input → same output)

Three verifiers:
  SQLVerifier       — schema → syntax → execution → result → llm quality
  CodeVerifier      — syntax → execution → test pass → output match → llm quality
  ReasoningVerifier — format → answer match → llm chain-of-thought

Usage:
    verifier = SQLVerifier(sandbox, llm_provider, schema_store)
    result = await verifier.verify(
        task="How many active users?",
        action="SELECT COUNT(*) FROM users WHERE active = 1",
        result=sandbox_result,
        gold="SELECT COUNT(*) FROM users WHERE active = 1",
    )
    print(result.feedback_for_agent)   # inject into FeedbackChannel
    print(result.overall_reward)
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationStep:
    """Result of a single verification sub-check."""
    name: str
    passed: bool
    score: float                  # 0.0 – 1.0
    feedback: str                 # actionable, specific hint for the agent
    deterministic: bool = True    # False only for uncached LLM steps


@dataclass
class VerificationResult:
    """Full verification result with per-step breakdown."""
    overall_reward: float
    verdict: str                  # "correct" | "partial" | "incorrect"
    steps: list[VerificationStep]
    feedback_for_agent: str       # combined feedback string → inject via FeedbackChannel
    cached: bool = False
    source: str = "verifier"

    def to_reward_signal(self) -> "Any":
        from harness.improvement.rlvr.reward import RewardSignal
        return RewardSignal(
            reward=self.overall_reward,
            verdict=self.verdict,  # type: ignore[arg-type]
            confidence=self._confidence(),
            source=self.source,
            reasoning=self.feedback_for_agent[:300],
            cached=self.cached,
        )

    def _confidence(self) -> float:
        det = [s for s in self.steps if s.deterministic]
        return 1.0 if det and all(s.passed for s in det) else 0.7

    def step_by_step(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            status = "✓" if s.passed else "✗"
            lines.append(f"{i}. [{status}] {s.name} (score={s.score:.2f}): {s.feedback}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call helper (deterministic: temp=0, SHA-256 cache)
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[str, str] = {}
_LLM_CACHE_LOCK = threading.Lock()
_LLM_CACHE_MAX = 8192


def _llm_cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


async def _call_llm_deterministic(
    llm: Any,
    prompt: str,
    system: str,
    max_tokens: int = 512,
) -> tuple[str, bool]:
    """Call LLM at temperature=0. Returns (response_text, was_cached)."""
    key = _llm_cache_key(prompt)
    with _LLM_CACHE_LOCK:
        if key in _LLM_CACHE:
            return _LLM_CACHE[key], True

    try:
        resp = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            system=system,
            temperature=0.0,
            skip_cache=False,
        )
        text = resp.content.strip()
    except Exception as exc:
        logger.debug("_call_llm_deterministic failed: %s", exc)
        return "", False

    with _LLM_CACHE_LOCK:
        if len(_LLM_CACHE) >= _LLM_CACHE_MAX:
            oldest = next(iter(_LLM_CACHE))
            del _LLM_CACHE[oldest]
        _LLM_CACHE[key] = text

    return text, False


def _parse_llm_score(text: str) -> tuple[float, str]:
    """Extract score and feedback from LLM response. Deterministic parsing."""
    m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    score = float(m.group(1)) if m else 0.5
    m2 = re.search(r'"feedback"\s*:\s*"([^"]+)"', text)
    feedback = m2.group(1) if m2 else text[:200]
    return max(0.0, min(1.0, score)), feedback


# ---------------------------------------------------------------------------
# SQL Verifier
# ---------------------------------------------------------------------------

_SQL_LLM_SYSTEM = (
    "You are a deterministic SQL correctness verifier. "
    "temperature=0. Same input → same output. Always respond with JSON."
)

_SQL_LLM_PROMPT = """\
Evaluate this SQL query for correctness and quality.

Task: {task}
Generated SQL: {sql}
Gold SQL: {gold}
Execution result: {result}

Steps to check:
1. Does the SQL correctly answer the task?
2. Are there any logical errors (wrong aggregation, missing WHERE, wrong JOIN)?
3. Does the result match the gold answer semantically?

Respond ONLY with JSON (no markdown):
{{"score": <0.0-1.0>, "feedback": "<one specific actionable sentence>"}}
"""


class SQLVerifier:
    """
    Five-step SQL verification pipeline:
    1. schema_check   — table/column names exist (rule, deterministic)
    2. syntax_check   — valid SQL via sqlglot (rule, deterministic)
    3. execution_check — runs without error (sandbox, deterministic)
    4. result_check   — result matches gold (execution accuracy, deterministic)
    5. quality_check  — LLM at temp=0 (cached, deterministic)
    """

    def __init__(
        self,
        sandbox: Any | None = None,
        llm: Any | None = None,
        schema_store: Any | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._llm = llm
        self._schema = schema_store

    async def verify(
        self,
        task: str,
        action: str,          # the generated SQL
        result: Any = None,   # SandboxResult or dict with rows/columns
        gold: str | None = None,
        db_id: str | None = None,
    ) -> VerificationResult:
        steps: list[VerificationStep] = []

        # 1. Schema check (pure rule)
        steps.append(await self._check_schema(action, db_id))

        # 2. Syntax check (pure rule)
        steps.append(self._check_syntax(action))

        # 3. Execution check (sandbox)
        if self._sandbox is not None and result is None:
            exec_step, exec_result = await self._check_execution(action)
            steps.append(exec_step)
            result = exec_result
        else:
            error = getattr(result, "error", None) if result else "no result"
            steps.append(VerificationStep(
                name="execution_check",
                passed=(result is not None and not error),
                score=1.0 if (result and not error) else 0.0,
                feedback="Execution result provided externally." if result else f"Execution error: {error}",
            ))

        # 4. Result match (execution accuracy against gold)
        if gold and self._sandbox is not None:
            steps.append(await self._check_result(action, gold))
        elif gold:
            steps.append(self._check_result_text(action, gold))
        else:
            steps.append(VerificationStep(
                name="result_check",
                passed=True,
                score=0.5,
                feedback="No gold SQL provided — result check skipped.",
            ))

        # 5. LLM quality check (temp=0, cached)
        if self._llm is not None:
            steps.append(await self._check_quality(task, action, result, gold))

        return self._aggregate(steps, source="sql_verifier")

    # ------------------------------------------------------------------

    async def _check_schema(self, sql: str, db_id: str | None) -> VerificationStep:
        if self._schema is None or not db_id:
            return VerificationStep("schema_check", True, 1.0,
                                    "Schema store not configured — skipped.")
        try:
            known = set(await self._schema.table_names(db_id))
            if not known:
                return VerificationStep("schema_check", True, 1.0,
                                        "No tables stored for this db_id.")
            import sqlglot
            tables_used = {
                t.name.lower()
                for stmt in sqlglot.parse(sql)
                for t in (stmt.find_all(sqlglot.exp.Table) if stmt else [])
                if t.name
            }
            unknown = tables_used - {t.lower() for t in known}
            if unknown:
                return VerificationStep(
                    "schema_check", False, 0.0,
                    f"Unknown tables: {sorted(unknown)}. Available: {sorted(known)[:10]}.",
                )
            return VerificationStep("schema_check", True, 1.0,
                                    "All referenced tables exist in schema.")
        except Exception as exc:
            return VerificationStep("schema_check", True, 0.8,
                                    f"Schema check skipped (sqlglot error): {exc}")

    def _check_syntax(self, sql: str) -> VerificationStep:
        try:
            import sqlglot
            errors = sqlglot.parse(sql, error_level=sqlglot.ErrorLevel.RAISE)
            return VerificationStep("syntax_check", True, 1.0, "SQL syntax is valid.")
        except Exception as exc:
            return VerificationStep("syntax_check", False, 0.0,
                                    f"SQL syntax error: {str(exc)[:200]}")

    async def _check_execution(self, sql: str) -> tuple[VerificationStep, Any]:
        try:
            result = await self._sandbox.execute(sql)
            if result.error:
                return VerificationStep(
                    "execution_check", False, 0.0,
                    f"Execution error: {result.error[:200]}. Check table names and SQL syntax.",
                ), result
            return VerificationStep(
                "execution_check", True, 1.0,
                f"Executed successfully. Returned {getattr(result.output, '__len__', lambda: '?')() if hasattr(result.output, '__len__') else '?'} result(s).",
            ), result
        except Exception as exc:
            return VerificationStep("execution_check", False, 0.0,
                                    f"Sandbox error: {exc}"), None

    async def _check_result(self, sql: str, gold: str) -> VerificationStep:
        try:
            from harness.eval.scorers import score_execution_match
            sr = await score_execution_match(sql, gold, self._sandbox)
            if sr.score >= 0.99:
                return VerificationStep("result_check", True, 1.0,
                                        "Result set matches gold exactly.")
            if sr.score >= 0.5:
                return VerificationStep("result_check", False, sr.score,
                                        f"Partial match (score={sr.score:.2f}): {sr.details}")
            return VerificationStep("result_check", False, sr.score,
                                    f"Result does not match gold (score={sr.score:.2f}): {sr.details}")
        except Exception as exc:
            return VerificationStep("result_check", False, 0.0,
                                    f"Result comparison failed: {exc}")

    def _check_result_text(self, sql: str, gold: str) -> VerificationStep:
        from harness.eval.scorers import score_sql_equivalence
        score = score_sql_equivalence(sql, gold)
        if score >= 1.0:
            return VerificationStep("result_check", True, 1.0, "SQL is equivalent to gold.")
        if score >= 0.5:
            return VerificationStep("result_check", False, 0.5,
                                    "SQL is structurally similar to gold but not identical.")
        return VerificationStep("result_check", False, 0.0,
                                "SQL differs significantly from gold — check logic and table selection.")

    async def _check_quality(self, task: str, sql: str, result: Any, gold: str | None) -> VerificationStep:
        result_str = str(getattr(result, "raw_text", result))[:500] if result else "(no result)"
        prompt = _SQL_LLM_PROMPT.format(
            task=task, sql=sql, gold=gold or "(none)", result=result_str
        )
        text, cached = await _call_llm_deterministic(self._llm, prompt, _SQL_LLM_SYSTEM)
        score, feedback = _parse_llm_score(text) if text else (0.5, "LLM check unavailable.")
        return VerificationStep(
            "quality_check", score >= 0.6, score, feedback, deterministic=cached
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(steps: list[VerificationStep], source: str) -> VerificationResult:
        weights = {"schema_check": 0.15, "syntax_check": 0.15,
                   "execution_check": 0.20, "result_check": 0.35, "quality_check": 0.15}
        total_w = sum(weights.get(s.name, 0.1) for s in steps)
        reward = sum(s.score * weights.get(s.name, 0.1) for s in steps) / max(total_w, 1e-9)
        reward = round(min(1.0, max(0.0, reward)), 4)
        verdict = "correct" if reward >= 0.85 else "partial" if reward >= 0.40 else "incorrect"
        failed = [s for s in steps if not s.passed]
        feedback_parts = [s.feedback for s in failed if s.feedback]
        feedback = " | ".join(feedback_parts) if feedback_parts else "All checks passed."
        cached = all(s.deterministic for s in steps)
        return VerificationResult(
            overall_reward=reward,
            verdict=verdict,
            steps=steps,
            feedback_for_agent=feedback,
            cached=cached,
            source=source,
        )


# ---------------------------------------------------------------------------
# Code Verifier
# ---------------------------------------------------------------------------

_CODE_LLM_SYSTEM = (
    "You are a deterministic code correctness verifier. "
    "temperature=0. Always respond with JSON."
)

_CODE_LLM_PROMPT = """\
Evaluate this code for correctness and quality.

Task: {task}
Generated code:
```python
{code}
```
Expected output: {expected}
Actual output: {actual}

Check:
1. Does the code correctly solve the task?
2. Are there any bugs, edge cases missed, or logical errors?
3. Does the output match the expected output?

Respond ONLY with JSON:
{{"score": <0.0-1.0>, "feedback": "<one specific actionable sentence>"}}
"""


class CodeVerifier:
    """
    Four-step code verification pipeline:
    1. syntax_check    — ast.parse (rule, deterministic)
    2. execution_check — runs without error (sandbox, deterministic)
    3. output_check    — output matches expected (rule, deterministic)
    4. quality_check   — LLM at temp=0 (cached, deterministic)
    """

    def __init__(self, sandbox: Any | None = None, llm: Any | None = None) -> None:
        self._sandbox = sandbox
        self._llm = llm

    async def verify(
        self,
        task: str,
        action: str,          # the generated code
        result: Any = None,
        gold: str | None = None,
        expected_output: str | None = None,
    ) -> VerificationResult:
        steps: list[VerificationStep] = []

        # 1. Syntax check (pure rule)
        steps.append(self._check_syntax(action))

        # 2. Execution check
        actual_output = ""
        if self._sandbox is not None and result is None:
            exec_step, exec_result = await self._check_execution(action)
            steps.append(exec_step)
            if exec_result:
                actual_output = str(getattr(exec_result, "raw_text", exec_result))[:500]
        else:
            err = getattr(result, "error", None) if result else None
            steps.append(VerificationStep(
                "execution_check",
                passed=(result is not None and not err),
                score=1.0 if (result and not err) else 0.0,
                feedback="Provided externally." if result else f"Error: {err}",
            ))
            if result:
                actual_output = str(getattr(result, "raw_text", result))[:500]

        # 3. Output check (rule)
        steps.append(self._check_output(actual_output, expected_output or gold))

        # 4. LLM quality check (temp=0, cached)
        if self._llm is not None:
            steps.append(await self._check_quality(task, action, actual_output, expected_output or gold))

        return self._aggregate(steps)

    def _check_syntax(self, code: str) -> VerificationStep:
        try:
            ast.parse(code)
            return VerificationStep("syntax_check", True, 1.0, "Python syntax is valid.")
        except SyntaxError as exc:
            return VerificationStep("syntax_check", False, 0.0,
                                    f"SyntaxError at line {exc.lineno}: {exc.msg}. Fix the syntax before running.")

    async def _check_execution(self, code: str) -> tuple[VerificationStep, Any]:
        try:
            result = await self._sandbox.execute(code)
            if result.error:
                return VerificationStep(
                    "execution_check", False, 0.0,
                    f"Runtime error: {result.error[:200]}. Check variable names and logic.",
                ), result
            return VerificationStep("execution_check", True, 1.0,
                                    "Code ran successfully without errors."), result
        except Exception as exc:
            return VerificationStep("execution_check", False, 0.0,
                                    f"Sandbox unavailable: {exc}"), None

    def _check_output(self, actual: str, expected: str | None) -> VerificationStep:
        if not expected:
            return VerificationStep("output_check", True, 0.5,
                                    "No expected output provided — output check skipped.")
        actual_clean = actual.strip().lower()
        expected_clean = expected.strip().lower()
        if actual_clean == expected_clean:
            return VerificationStep("output_check", True, 1.0, "Output matches expected exactly.")
        if expected_clean in actual_clean:
            return VerificationStep("output_check", True, 0.8,
                                    "Expected output is present in actual output.")
        # Numeric tolerance check
        try:
            a_val, e_val = float(actual_clean), float(expected_clean)
            if abs(a_val - e_val) / max(abs(e_val), 1.0) < 0.01:
                return VerificationStep("output_check", True, 1.0,
                                        "Numeric output matches within 1% tolerance.")
        except ValueError:
            pass
        return VerificationStep("output_check", False, 0.0,
                                f"Output mismatch. Got: {actual[:100]!r}. Expected: {expected[:100]!r}.")

    async def _check_quality(self, task: str, code: str, actual: str, expected: str | None) -> VerificationStep:
        prompt = _CODE_LLM_PROMPT.format(
            task=task, code=code[:800], expected=expected or "(none)", actual=actual or "(no output)"
        )
        text, cached = await _call_llm_deterministic(self._llm, prompt, _CODE_LLM_SYSTEM)
        score, feedback = _parse_llm_score(text) if text else (0.5, "LLM check unavailable.")
        return VerificationStep("quality_check", score >= 0.6, score, feedback, deterministic=cached)

    @staticmethod
    def _aggregate(steps: list[VerificationStep]) -> VerificationResult:
        weights = {"syntax_check": 0.10, "execution_check": 0.30,
                   "output_check": 0.45, "quality_check": 0.15}
        total_w = sum(weights.get(s.name, 0.1) for s in steps)
        reward = sum(s.score * weights.get(s.name, 0.1) for s in steps) / max(total_w, 1e-9)
        reward = round(min(1.0, max(0.0, reward)), 4)
        verdict = "correct" if reward >= 0.85 else "partial" if reward >= 0.40 else "incorrect"
        failed = [s for s in steps if not s.passed]
        feedback = " | ".join(s.feedback for s in failed) if failed else "All checks passed."
        return VerificationResult(
            overall_reward=reward, verdict=verdict, steps=steps,
            feedback_for_agent=feedback, cached=all(s.deterministic for s in steps),
            source="code_verifier",
        )


# ---------------------------------------------------------------------------
# Reasoning Verifier (general tasks, math, multi-hop)
# ---------------------------------------------------------------------------

_REASON_LLM_SYSTEM = (
    "You are a deterministic reasoning verifier. "
    "temperature=0. Verify step-by-step. Always respond with JSON."
)

_REASON_LLM_PROMPT = """\
Verify this agent response step-by-step.

Task: {task}
Agent response: {response}
Gold answer: {gold}

Verification steps:
1. Does the response address the task directly?
2. Is the reasoning chain (if present) logically correct?
3. Does the final answer match the gold answer?
4. Are there any factual errors or unsupported claims?

Respond ONLY with JSON:
{{"score": <0.0-1.0>, "feedback": "<one specific actionable sentence>", \
"reasoning_valid": <true|false>, "answer_correct": <true|false>}}
"""


class ReasoningVerifier:
    """
    Three-step reasoning verification:
    1. format_check   — response has non-empty answer (rule, deterministic)
    2. answer_check   — answer matches gold (exact/numeric, deterministic)
    3. reasoning_check — LLM chain-of-thought verification (temp=0, cached)
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    async def verify(
        self,
        task: str,
        action: str,       # the agent's text response
        result: Any = None,
        gold: str | None = None,
        **_: Any,
    ) -> VerificationResult:
        response = str(result) if result else action
        steps: list[VerificationStep] = []

        # 1. Format check (rule)
        steps.append(self._check_format(response))

        # 2. Answer match (rule)
        steps.append(self._check_answer(response, gold))

        # 3. LLM reasoning check (temp=0, cached)
        if self._llm is not None:
            steps.append(await self._check_reasoning(task, response, gold))

        return self._aggregate(steps)

    def _check_format(self, response: str) -> VerificationStep:
        if not response or not response.strip():
            return VerificationStep("format_check", False, 0.0,
                                    "Response is empty. The agent must produce an answer.")
        if len(response.strip()) < 10:
            return VerificationStep("format_check", False, 0.3,
                                    f"Response is too short ({len(response)} chars). Provide a complete answer.")
        return VerificationStep("format_check", True, 1.0, "Response format is valid.")

    def _check_answer(self, response: str, gold: str | None) -> VerificationStep:
        if not gold:
            return VerificationStep("answer_check", True, 0.5, "No gold answer — skipped.")
        resp_lower = response.strip().lower()
        gold_lower = gold.strip().lower()
        if gold_lower in resp_lower:
            return VerificationStep("answer_check", True, 1.0, "Gold answer found in response.")
        # Numeric check
        try:
            r_nums = re.findall(r"-?\d+(?:\.\d+)?", resp_lower)
            g_nums = re.findall(r"-?\d+(?:\.\d+)?", gold_lower)
            if r_nums and g_nums:
                r_val = float(r_nums[-1])
                g_val = float(g_nums[-1])
                if abs(r_val - g_val) / max(abs(g_val), 1e-9) < 0.01:
                    return VerificationStep("answer_check", True, 1.0,
                                            "Numeric answer matches gold within tolerance.")
                return VerificationStep("answer_check", False, 0.0,
                                        f"Numeric mismatch: got {r_val}, expected {g_val}.")
        except ValueError:
            pass
        return VerificationStep("answer_check", False, 0.2,
                                f"Answer does not match gold. Expected: {gold[:100]!r}.")

    async def _check_reasoning(self, task: str, response: str, gold: str | None) -> VerificationStep:
        prompt = _REASON_LLM_PROMPT.format(
            task=task, response=response[:800], gold=gold or "(none)"
        )
        text, cached = await _call_llm_deterministic(self._llm, prompt, _REASON_LLM_SYSTEM)
        if not text:
            return VerificationStep("reasoning_check", True, 0.5,
                                    "LLM reasoning check unavailable.", deterministic=True)
        score, feedback = _parse_llm_score(text)
        # Also check reasoning_valid flag
        if '"reasoning_valid": false' in text.lower():
            feedback = f"Reasoning chain has errors. {feedback}"
            score = min(score, 0.5)
        return VerificationStep("reasoning_check", score >= 0.6, score, feedback, deterministic=cached)

    @staticmethod
    def _aggregate(steps: list[VerificationStep]) -> VerificationResult:
        weights = {"format_check": 0.10, "answer_check": 0.60, "reasoning_check": 0.30}
        total_w = sum(weights.get(s.name, 0.1) for s in steps)
        reward = sum(s.score * weights.get(s.name, 0.1) for s in steps) / max(total_w, 1e-9)
        reward = round(min(1.0, max(0.0, reward)), 4)
        verdict = "correct" if reward >= 0.85 else "partial" if reward >= 0.40 else "incorrect"
        failed = [s for s in steps if not s.passed]
        feedback = " | ".join(s.feedback for s in failed) if failed else "All checks passed."
        return VerificationResult(
            overall_reward=reward, verdict=verdict, steps=steps,
            feedback_for_agent=feedback, cached=all(s.deterministic for s in steps),
            source="reasoning_verifier",
        )


# ---------------------------------------------------------------------------
# Factory — pick verifier by agent_type
# ---------------------------------------------------------------------------

def get_verifier(
    agent_type: str,
    sandbox: Any | None = None,
    llm: Any | None = None,
    schema_store: Any | None = None,
) -> "SQLVerifier | CodeVerifier | ReasoningVerifier":
    if agent_type == "sql":
        return SQLVerifier(sandbox=sandbox, llm=llm, schema_store=schema_store)
    if agent_type == "code":
        return CodeVerifier(sandbox=sandbox, llm=llm)
    return ReasoningVerifier(llm=llm)
