"""Error hierarchy and FailureClass enum for HarnessAgent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    """Canonical failure classification for all harness errors."""

    # LLM failures
    LLM_ERROR = "LLM_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_CONTEXT_LIMIT = "LLM_CONTEXT_LIMIT"
    LLM_PARSE_ERROR = "LLM_PARSE_ERROR"

    # Tool failures
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_SCHEMA_ERROR = "TOOL_SCHEMA_ERROR"
    TOOL_EXEC_ERROR = "TOOL_EXEC_ERROR"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_OUTPUT_ERROR = "TOOL_OUTPUT_ERROR"

    # MCP failures
    MCP_CONNECT_ERROR = "MCP_CONNECT_ERROR"
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"

    # Safety failures
    SAFETY_INPUT = "SAFETY_INPUT"
    SAFETY_STEP = "SAFETY_STEP"
    SAFETY_OUTPUT = "SAFETY_OUTPUT"

    # Budget failures
    BUDGET_STEPS = "BUDGET_STEPS"
    BUDGET_TOKENS = "BUDGET_TOKENS"
    BUDGET_TIME = "BUDGET_TIME"
    BUDGET_COST = "BUDGET_COST"

    # Cancellation (operator-initiated, e.g. DELETE /runs/{id})
    CANCELLED = "CANCELLED"

    # Memory failures
    MEMORY_REDIS = "MEMORY_REDIS"
    MEMORY_VECTOR = "MEMORY_VECTOR"
    MEMORY_GRAPH = "MEMORY_GRAPH"

    # Inter-agent failures
    INTER_AGENT_TIMEOUT = "INTER_AGENT_TIMEOUT"
    INTER_AGENT_REJECT = "INTER_AGENT_REJECT"

    # Orchestration failures
    PLAN_ERROR = "PLAN_ERROR"
    SKILL_MISSING = "SKILL_MISSING"

    # Catch-all
    UNKNOWN = "UNKNOWN"


class HarnessError(Exception):
    """Base exception for all HarnessAgent errors."""

    def __init__(
        self,
        message: str,
        failure_class: FailureClass = FailureClass.UNKNOWN,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.failure_class = failure_class
        self.context: dict[str, Any] = context or {}
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.error_id: str = uuid.uuid4().hex

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"failure_class={self.failure_class.value}, "
            f"error_id={self.error_id})"
        )


class LLMError(HarnessError):
    """Raised when an LLM provider call fails."""

    def __init__(
        self,
        message: str,
        failure_class: FailureClass = FailureClass.LLM_ERROR,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, failure_class, context)


class ToolError(HarnessError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        failure_class: FailureClass = FailureClass.TOOL_EXEC_ERROR,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, failure_class, context)
        self.tool_name = tool_name


class SafetyViolation(HarnessError):
    """Raised when a safety guard rejects content."""

    def __init__(
        self,
        message: str,
        guard_source: str,
        failure_class: FailureClass = FailureClass.SAFETY_INPUT,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, failure_class, context)
        self.guard_source = guard_source


class BudgetExceeded(HarnessError):
    """Raised when a run exceeds its step, token, or time budget."""

    def __init__(
        self,
        message: str,
        failure_class: FailureClass = FailureClass.BUDGET_STEPS,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, failure_class, context)


class CircuitOpenError(HarnessError):
    """Raised when a circuit breaker rejects a call in OPEN state."""

    def __init__(
        self,
        message: str,
        service_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, FailureClass.LLM_ERROR, context)
        self.service_name = service_name


class InterAgentTimeout(HarnessError):
    """Raised when an inter-agent call times out."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, FailureClass.INTER_AGENT_TIMEOUT, context)


class HITLRejected(HarnessError):
    """Raised when a human-in-the-loop reviewer rejects a proposed action."""

    def __init__(
        self,
        message: str,
        request_id: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, FailureClass.INTER_AGENT_REJECT, context)
        self.request_id = request_id


class RunCancelled(HarnessError):
    """Raised inside the agent loop when a run is cancelled by an operator
    (e.g. DELETE /runs/{id} flipping the persisted status to 'cancelled')."""

    def __init__(
        self,
        message: str = "Run cancelled",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, FailureClass.CANCELLED, context)


class SandboxError(HarnessError):
    """Raised when sandbox execution fails."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, FailureClass.TOOL_EXEC_ERROR, context)


class IngestionError(HarnessError):
    """Raised when document or data ingestion fails."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, FailureClass.MEMORY_VECTOR, context)


class RateLimitError(HarnessError):
    """Raised when a tenant exceeds rate or cost limits."""

    def __init__(
        self,
        message: str,
        retry_after: float = 0.0,
        failure_class: FailureClass = FailureClass.LLM_RATE_LIMIT,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, failure_class, context)
        self.retry_after = retry_after
