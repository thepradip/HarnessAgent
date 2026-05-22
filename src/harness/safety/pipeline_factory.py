"""Safety pipeline factory for HarnessAgent.

Constructs Guardrail Pipeline instances with appropriate stages
based on agent type and configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for a safety guardrail pipeline."""

    max_steps: int = 50
    max_tokens: int = 100_000
    max_wall_seconds: float = 300.0
    allowed_tools: list[str] | None = None  # None means all tools are allowed
    blocked_tools: list[str] = field(default_factory=list)
    allow_destructive_commands: bool = False
    pii_redact_output: bool = True
    injection_detect_input: bool = True
    loop_detection: bool = True
    loop_window: int = 10


def build_pipeline(
    agent_type: str,
    config: SafetyConfig,
    budget: Any | None = None,  # guardrail.intermediate.budget.Budget
) -> Any:
    """Build a Guardrail Pipeline configured for the given agent type.

    Pipeline composition:
    - Input stage:  InjectionDetector (if config.injection_detect_input)
    - Intermediate: Budget (steps / tokens / time), LoopDetector (if loop_detection),
                    ToolPolicy (allowed_tools / blocked_tools)
    - Output stage: PIIRedactor (if pii_redact_output)

    Returns a configured Pipeline instance.
    """
    try:
        from guardrail.pipeline import Pipeline, Stage
    except ImportError:
        logger.warning(
            "guardrail package not installed — using HardConstraintPipeline for agent_type=%s",
            agent_type,
        )
        return _HardConstraintPipeline(blocked_tools=config.blocked_tools)

    input_stages: list[Any] = []
    intermediate_stages: list[Any] = []
    output_stages: list[Any] = []

    # ------------------------------------------------------------------
    # Input guards
    # ------------------------------------------------------------------
    if config.injection_detect_input:
        try:
            from guardrail.input.injection_detector import InjectionDetector
            input_stages.append(InjectionDetector())
        except ImportError:
            logger.debug("InjectionDetector not available — skipping")

    # ------------------------------------------------------------------
    # Intermediate guards
    # ------------------------------------------------------------------

    # Budget guard
    if budget is None:
        try:
            from guardrail.intermediate.budget import Budget
            budget = Budget(
                max_steps=config.max_steps,
                max_tokens=config.max_tokens,
                max_wall_seconds=config.max_wall_seconds,
            )
        except ImportError:
            logger.debug("Budget guard not available — skipping")

    if budget is not None:
        intermediate_stages.append(budget)

    # Loop detector
    if config.loop_detection:
        try:
            from guardrail.intermediate.loop_detector import LoopDetector
            intermediate_stages.append(LoopDetector(window=config.loop_window))
        except ImportError:
            logger.debug("LoopDetector not available — skipping")

    # Tool policy
    if config.allowed_tools is not None or config.blocked_tools:
        try:
            from guardrail.intermediate.tool_policy import ToolPolicy
            try:
                # Try keyword args (newer guardrail versions)
                intermediate_stages.append(
                    ToolPolicy(
                        allowed=config.allowed_tools,
                        blocked=config.blocked_tools,
                    )
                )
            except TypeError:
                # Older guardrail API — positional args
                intermediate_stages.append(
                    ToolPolicy(config.allowed_tools, config.blocked_tools)
                )
        except (ImportError, Exception) as exc:
            logger.debug("ToolPolicy not available — skipping (%s)", exc)

    # ------------------------------------------------------------------
    # Output guards
    # ------------------------------------------------------------------
    if config.pii_redact_output:
        try:
            from guardrail.output.pii_redactor import PIIRedactor
            output_stages.append(PIIRedactor())
        except ImportError:
            logger.debug("PIIRedactor not available — skipping")

    # ------------------------------------------------------------------
    # Assemble pipeline
    # ------------------------------------------------------------------
    stages: list[Any] = []
    if input_stages:
        try:
            stages.append(Stage(name="input", guards=input_stages))
        except Exception:
            stages.extend(input_stages)
    if intermediate_stages:
        try:
            stages.append(Stage(name="intermediate", guards=intermediate_stages))
        except Exception:
            stages.extend(intermediate_stages)
    if output_stages:
        try:
            stages.append(Stage(name="output", guards=output_stages))
        except Exception:
            stages.extend(output_stages)

    try:
        pipeline = Pipeline(stages=stages, name=f"harness_{agent_type}")
    except Exception:
        # Some versions of guardrail use positional args or different API
        pipeline = Pipeline(stages)  # type: ignore[call-arg]

    logger.info(
        "Built safety pipeline for agent_type=%s: %d stages, %d guards",
        agent_type,
        len(stages),
        len(input_stages) + len(intermediate_stages) + len(output_stages),
    )
    return pipeline


def get_default_config(agent_type: str) -> SafetyConfig:
    """Return a sensible default SafetyConfig for the given agent type."""
    match agent_type:
        case "sql":
            return SafetyConfig(
                allowed_tools=[
                    "execute_sql",
                    "list_tables",
                    "describe_table",
                    "sample_rows",
                ],
                allow_destructive_commands=False,
                pii_redact_output=True,
                injection_detect_input=True,
                loop_detection=True,
            )
        case "code":
            return SafetyConfig(
                allowed_tools=[
                    "run_python",
                    "lint_code",
                    "read_file",
                    "write_file",
                    "apply_patch",
                    "list_workspace",
                ],
                allow_destructive_commands=False,
                pii_redact_output=True,
                injection_detect_input=True,
                loop_detection=True,
            )
        case "research":
            return SafetyConfig(
                allowed_tools=["read_file", "write_file", "list_workspace"],
                pii_redact_output=True,
                injection_detect_input=True,
            )
        case _:
            return SafetyConfig()


# ---------------------------------------------------------------------------
# Hard-constraint fallback for when guardrail is unavailable
# ---------------------------------------------------------------------------

import re as _re

_INJECTION_PATTERNS = [
    _re.compile(r"ignore\s+(all\s+)?previous\s+instructions", _re.I),
    _re.compile(r"forget\s+your\s+(system\s+)?instructions", _re.I),
    _re.compile(r"you\s+are\s+now\s+(a\s+different|DAN|an?\s+unrestricted)", _re.I),
    _re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", _re.I),
    _re.compile(r"jailbreak", _re.I),
    _re.compile(r"prompt\s+injection", _re.I),
    _re.compile(r"</?(system|SYSTEM)>", _re.I),
]

_PII_PATTERNS = [
    (_re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (_re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL REDACTED]"),
    (_re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE REDACTED]"),
    (_re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"), "[CARD REDACTED]"),
]


class _GuardResult:
    """Minimal GuardResult returned by _HardConstraintPipeline."""

    def __init__(self, blocked: bool, reason: str = "") -> None:
        self.blocked = blocked
        self.reason = reason
        self.decision = "block" if blocked else "allow"


class _HardConstraintPipeline:
    """Minimal safety pipeline used when the guardrail package is not installed.

    Provides three layers of protection with zero external dependencies:
    - Input: regex-based prompt-injection detection
    - Step: blocked tool-name enforcement
    - Output: PII redaction + secret leak detection/redaction
    """

    def __init__(self, blocked_tools: list[str] | None = None) -> None:
        self._blocked_tools: set[str] = set(blocked_tools or [])
        try:
            from harness.security.scanner import SecretScanner
            self._secret_scanner: Any = SecretScanner()
        except Exception:
            self._secret_scanner = None

    async def check_input(self, payload: Any) -> _GuardResult:
        content = ""
        if isinstance(payload, dict):
            content = str(payload.get("content", payload.get("query", "")))
        elif isinstance(payload, str):
            content = payload

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(content):
                logger.warning(
                    "HardConstraintPipeline: prompt injection detected: %r",
                    content[:120],
                )
                return _GuardResult(
                    blocked=True,
                    reason=f"Prompt injection pattern detected: {pattern.pattern}",
                )
        return _GuardResult(blocked=False)

    async def check_step(self, payload: Any) -> _GuardResult:
        tool_name = ""
        if isinstance(payload, dict):
            tool_name = str(payload.get("tool_name", payload.get("tool", "")))
        elif hasattr(payload, "name"):
            tool_name = payload.name

        if tool_name and tool_name in self._blocked_tools:
            return _GuardResult(
                blocked=True,
                reason=f"Tool '{tool_name}' is in the blocked list",
            )
        return _GuardResult(blocked=False)

    async def check_output(self, payload: Any) -> _GuardResult:
        """Scan output for leaked secrets. Never blocks — redacts and warns."""
        content = ""
        if isinstance(payload, dict):
            content = str(payload.get("content", ""))
        elif isinstance(payload, str):
            content = payload

        if content and self._secret_scanner is not None:
            matches = self._secret_scanner.scan(content)
            if matches:
                names = ", ".join(m.pattern_name for m in matches)
                logger.warning(
                    "Secret leak detected in LLM output (%s) — redacting before use",
                    names,
                )
                return _GuardResult(
                    blocked=False,
                    reason=f"secret_detected:{names}",
                )
        return _GuardResult(blocked=False)

    def redact(self, text: str) -> str:
        """Redact PII and API keys/tokens from text."""
        for pattern, replacement in _PII_PATTERNS:
            text = pattern.sub(replacement, text)
        if self._secret_scanner is not None:
            text = self._secret_scanner.redact(text)
        return text
