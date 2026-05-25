"""PEV (Plan-Execute-Verify) verifier protocol and lightweight implementations.

The Verifier is the missing third leg of the agent control loop:

    Plan → Execute → **Verify** → (feedback back into loop) → done

Without verification, agents terminate on implicit convergence — when they
stop requesting tool calls — not when their output is objectively correct.
This module provides a Verifier protocol and concrete implementations that
check correctness using execution signals rather than LLM self-assessment.

Usage::

    from harness.verification.verifier import CodeExitCodeVerifier
    ctx.metadata["verifier"] = CodeExitCodeVerifier()

The BaseAgent checks for ``ctx.metadata["verifier"]`` after every no-tool-call
response. If verification fails, the feedback is injected as a user message
and the loop continues (up to ``max_verification_attempts``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Tool names treated as code-execution tools for result tracking
_CODE_TOOL_NAMES: frozenset[str] = frozenset({
    "run_python", "run_code", "execute_code", "exec_python",
})


@dataclass
class VerificationResult:
    """Result of a single verification check.

    Attributes:
        passed:   True → agent output is acceptable, stop the loop.
        verdict:  "correct" | "incorrect" | "partial" | "skipped"
        feedback: Injected as a user message when not passed.
        score:    0.0–1.0 quality estimate (used by RLVR if available).
    """

    passed: bool
    verdict: str
    feedback: str
    score: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.passed and self.score == 0.0:
            self.score = 1.0

    @classmethod
    def skipped(cls) -> "VerificationResult":
        return cls(passed=True, verdict="skipped", feedback="", score=1.0)

    @classmethod
    def correct(cls, feedback: str = "") -> "VerificationResult":
        return cls(passed=True, verdict="correct", feedback=feedback, score=1.0)

    @classmethod
    def incorrect(cls, feedback: str, score: float = 0.0) -> "VerificationResult":
        return cls(passed=False, verdict="incorrect", feedback=feedback, score=score)

    @classmethod
    def partial(cls, feedback: str, score: float = 0.5) -> "VerificationResult":
        return cls(passed=False, verdict="partial", feedback=feedback, score=score)


@runtime_checkable
class Verifier(Protocol):
    """Check whether an agent's output is acceptable.

    Called by ``BaseAgent._verify_output()`` after each no-tool-call response.
    Implementations should be lightweight — avoid LLM calls here; prefer
    execution signals (exit codes, test results, linter output).
    """

    async def verify(
        self,
        ctx: Any,
        output: str,
        history: list[dict],
    ) -> VerificationResult:
        """Return a VerificationResult for the agent's current output.

        Args:
            ctx:     AgentContext for the current run.
            output:  Extracted final answer text.
            history: Full conversation history up to this point.
        """
        ...


# ---------------------------------------------------------------------------
# NoopVerifier — always passes (use for agents with no verifiable oracle)
# ---------------------------------------------------------------------------

class NoopVerifier:
    """Verifier that always passes — equivalent to no verification.

    Use when you want the PEV slot populated but have no objective oracle.
    """

    async def verify(self, ctx: Any, output: str, history: list[dict]) -> VerificationResult:
        return VerificationResult.skipped()


# ---------------------------------------------------------------------------
# CodeExitCodeVerifier — checks the last code execution result
# ---------------------------------------------------------------------------

class CodeExitCodeVerifier:
    """Verifier for CodeAgent: checks the last run_python result.

    Uses ``ctx.metadata["last_code_result"]`` which BaseAgent populates after
    every code-execution tool call. Zero overhead — no extra tool calls needed.

    Failure triggers:
    - ``exit_code != 0``
    - ``timed_out == True``
    - ``oom_killed == True`` (exit code 137)
    - stderr contains an uncaught exception class name

    Passes if no code was run (skipped) or the last run succeeded.
    """

    # Exception patterns that indicate a real failure (not just warnings)
    _EXCEPTION_PATTERNS = (
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "AssertionError",
        "NameError",
        "TypeError",
        "ValueError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "SyntaxError",
        "IndentationError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "RuntimeError",
        "FileNotFoundError",
        "PermissionError",
    )

    async def verify(self, ctx: Any, output: str, history: list[dict]) -> VerificationResult:
        result = None
        if hasattr(ctx, "metadata"):
            result = ctx.metadata.get("last_code_result")

        if result is None:
            return VerificationResult.skipped()

        exit_code = result.get("exit_code")
        timed_out = result.get("timed_out", False)
        stderr = result.get("stderr", "") or ""
        stdout = result.get("stdout", "") or ""

        # OOM kill
        if exit_code == 137 and not timed_out:
            return VerificationResult.incorrect(
                feedback=(
                    "The code was killed by the OOM killer (exit 137). "
                    "Reduce memory usage or switch to a larger sandbox workload profile."
                )
            )

        # Timeout
        if timed_out:
            return VerificationResult.incorrect(
                feedback="The code timed out. Optimize the algorithm or reduce the dataset size."
            )

        # Non-zero exit with no stderr — unusual, flag it
        if exit_code != 0 and not stderr:
            return VerificationResult.incorrect(
                feedback=f"Code exited with code {exit_code} but produced no error output. "
                         "Check for silent failures or missing return statements."
            )

        # Non-zero exit with stderr
        if exit_code != 0:
            snippet = stderr[:500]
            return VerificationResult.incorrect(
                feedback=f"The code failed (exit {exit_code}):\n```\n{snippet}\n```\nFix the error above."
            )

        # Exit 0 but stderr contains an uncaught exception
        for pattern in self._EXCEPTION_PATTERNS:
            if pattern in stderr:
                snippet = stderr[:400]
                return VerificationResult.incorrect(
                    feedback=(
                        f"Code exited 0 but stderr contains an exception:\n```\n{snippet}\n```"
                    ),
                    score=0.2,
                )

        return VerificationResult.correct()


# ---------------------------------------------------------------------------
# ExpectedOutputVerifier — checks stdout against a pattern or string
# ---------------------------------------------------------------------------

class ExpectedOutputVerifier:
    """Verifier that checks whether stdout contains an expected string or matches a pattern.

    Useful for tasks where the correct answer is known in advance (evals,
    smoke tests, regression checks).
    """

    def __init__(
        self,
        expected: str,
        exact: bool = False,
        case_sensitive: bool = True,
    ) -> None:
        """
        Args:
            expected:       The expected string (or substring if exact=False).
            exact:          If True, stdout must equal expected exactly (stripped).
            case_sensitive: If False, comparison is case-insensitive.
        """
        self._expected = expected
        self._exact = exact
        self._case_sensitive = case_sensitive

    async def verify(self, ctx: Any, output: str, history: list[dict]) -> VerificationResult:
        result = None
        if hasattr(ctx, "metadata"):
            result = ctx.metadata.get("last_code_result")

        stdout = (result or {}).get("stdout", "") or output
        candidate = stdout.strip()
        expected = self._expected.strip()

        if not self._case_sensitive:
            candidate = candidate.lower()
            expected = expected.lower()

        if self._exact:
            passed = candidate == expected
        else:
            passed = expected in candidate

        if passed:
            return VerificationResult.correct()

        return VerificationResult.incorrect(
            feedback=(
                f"Expected output {'exactly ' if self._exact else 'containing '}"
                f"'{self._expected}' but got:\n```\n{stdout[:300]}\n```"
            )
        )
