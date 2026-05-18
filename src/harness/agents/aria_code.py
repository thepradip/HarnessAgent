"""
AriaCode — self-correcting code agent.

Mirrors AriaSql but for Python / code tasks:
  1. Generates code from the task description
  2. Runs CodeVerifier (syntax → execution → output match → LLM quality)
  3. If score < threshold, injects feedback and retries (up to max_retries)
  4. Returns the code with the highest verifier score

Works standalone (for HumanEval / SWE benchmark):
    agent = AriaCode(llm_provider)
    code = await agent.generate_code("Write a function to reverse a list")

Works inside the harness (extends CodeAgent):
    class AriaCodeAgent(CodeAgent): ...
"""

from __future__ import annotations

import logging
import re
from typing import Any

from harness.agents.code_agent import CodeAgent
from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

_CORRECTION_THRESHOLD = 0.60
_MAX_RETRIES = 2

_GENERATE_PROMPT = """\
You are AriaCode, a precise Python code generation engine.
Write clean, correct Python code that solves the task.
Return ONLY the code — no explanation, no markdown fences.

Task: {task}

Expected output (if provided): {expected}

Code:"""

_CORRECT_PROMPT = """\
You are AriaCode. Your previous code had issues. Fix it.

Task: {task}

Previous code:
{code}

Verification feedback:
{feedback}

Write the corrected code only — no explanation, no markdown.

Code:"""


def _extract_code(text: str) -> str:
    """Strip markdown fences and extract the first code block."""
    text = text.strip()
    text = re.sub(r"^```python\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class AriaCode:
    """
    Standalone self-correcting code agent.

    Parameters
    ----------
    llm_provider : LLMProvider or LLMRouter
    verifier     : CodeVerifier | None
    max_retries  : int
    correction_threshold : float
    """

    def __init__(
        self,
        llm_provider: Any,
        verifier: Any | None = None,
        max_retries: int = _MAX_RETRIES,
        correction_threshold: float = _CORRECTION_THRESHOLD,
    ) -> None:
        self._llm = llm_provider
        self._verifier = verifier
        self._max_retries = max_retries
        self._threshold = correction_threshold

    async def generate_code(
        self,
        task: str,
        expected_output: str | None = None,
        gold_code: str | None = None,
    ) -> str:
        """
        Generate Python code for task with self-correction.

        1. Call LLM → candidate code
        2. Verify with CodeVerifier (if configured)
        3. If score < threshold, inject feedback and retry
        4. Return best code
        """
        code, score, feedback = await self._generate_and_verify(
            task, expected_output, gold_code
        )
        best_code, best_score = code, score

        for attempt in range(self._max_retries):
            if best_score >= self._threshold:
                break
            logger.debug(
                "AriaCode: score %.2f < %.2f — correction %d/%d",
                best_score, self._threshold, attempt + 1, self._max_retries,
            )
            code, score, feedback = await self._correct_and_verify(
                task, best_code, feedback, expected_output, gold_code
            )
            if score > best_score:
                best_code, best_score = code, score

        logger.debug("AriaCode: final score=%.2f task=%s", best_score, task[:60])
        return best_code

    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                system=(
                    "You are AriaCode. Return only Python code. "
                    "No markdown. No explanation."
                ),
                temperature=0.0,
                skip_cache=False,
            )
            return _extract_code(response.content)
        except Exception as exc:
            logger.warning("AriaCode LLM call failed: %s", exc)
            return "# generation failed"

    async def _generate_and_verify(
        self,
        task: str,
        expected: str | None,
        gold: str | None,
    ) -> tuple[str, float, str]:
        code = await self._call_llm(
            _GENERATE_PROMPT.format(task=task, expected=expected or "(none)")
        )
        return await self._verify(code, task, expected, gold)

    async def _correct_and_verify(
        self,
        task: str,
        prev_code: str,
        feedback: str,
        expected: str | None,
        gold: str | None,
    ) -> tuple[str, float, str]:
        code = await self._call_llm(
            _CORRECT_PROMPT.format(
                task=task, code=prev_code, feedback=feedback[:500]
            )
        )
        return await self._verify(code, task, expected, gold)

    async def _verify(
        self,
        code: str,
        task: str,
        expected: str | None,
        gold: str | None,
    ) -> tuple[str, float, str]:
        if self._verifier is None:
            return code, 1.0, ""
        try:
            vr = await self._verifier.verify(
                task=task,
                action=code,
                result=None,
                gold=gold,
                expected_output=expected,
            )
            return code, vr.overall_reward, vr.feedback_for_agent
        except Exception as exc:
            logger.debug("AriaCode verify failed: %s", exc)
            return code, 0.5, str(exc)

    @classmethod
    def from_config(
        cls,
        with_verifier: bool = True,
    ) -> "AriaCode":
        """Build AriaCode using the harness config (.env keys)."""
        from harness.core.config import get_config
        from harness.llm.factory import build_router
        from harness.improvement.rlvr.verifiers import CodeVerifier

        cfg = get_config()
        llm = build_router(cfg)
        verifier = CodeVerifier(sandbox=None, llm=llm) if with_verifier else None
        return cls(llm_provider=llm, verifier=verifier)


# ---------------------------------------------------------------------------
# Harness agent variant
# ---------------------------------------------------------------------------

class AriaCodeAgent(CodeAgent):
    """
    CodeAgent + AriaCode self-correction loop.

    Before the base agent's full tool-calling loop runs, first attempts direct
    generation + verification. Falls back to the base loop if score < threshold.
    """

    agent_type: str = "ariacode"

    def build_system_prompt(self, ctx: AgentContext) -> str:
        return (
            "You are AriaCode, an expert Python engineer.\n\n"
            "Rules:\n"
            "1. Write clean, type-annotated Python.\n"
            "2. Run your code to verify it works before returning.\n"
            "3. Fix all lint errors and test failures.\n"
            "4. Return the final code and a brief explanation.\n"
        )
