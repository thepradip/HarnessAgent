"""Patch generation — creates prompt improvement patches from error analysis."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from harness.improvement.error_collector import ErrorRecord

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    """A proposed prompt improvement patch.

    Attributes:
        patch_id:       Unique identifier.
        agent_type:     Which agent's prompt to patch.
        target:         Target resource (e.g. "prompt" or "tool_config").
        op:             Operation: append | prepend | replace | remove | set.
        path:           For replace/remove: the text to find in the current prompt.
        value:          The new text to insert / replace with.
        rationale:      LLM-generated explanation of why this patch helps.
        proposed_by:    Who proposed this patch ("hermes" or user ID).
        proposed_at:    UTC timestamp.
        score:          Eval score after testing (None until evaluated).
        status:         pending | approved | rejected | applied.
        based_on_errors: List of ErrorRecord IDs this patch addresses.
    """

    patch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_type: str = ""
    target: str = "prompt"
    op: str = "append"
    path: str = ""
    value: str = ""
    rationale: str = ""
    proposed_by: str = "hermes"
    proposed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: Optional[float] = None
    status: str = "pending"  # pending | approved | rejected | applied
    based_on_errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "agent_type": self.agent_type,
            "target": self.target,
            "op": self.op,
            "path": self.path,
            "value": self.value,
            "rationale": self.rationale,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at.isoformat(),
            "score": self.score,
            "status": self.status,
            "based_on_errors": self.based_on_errors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Patch":
        proposed_at = data.get("proposed_at")
        if isinstance(proposed_at, str):
            try:
                proposed_at = datetime.fromisoformat(proposed_at)
            except ValueError:
                proposed_at = datetime.now(timezone.utc)
        elif not isinstance(proposed_at, datetime):
            proposed_at = datetime.now(timezone.utc)

        return cls(
            patch_id=data.get("patch_id", uuid.uuid4().hex),
            agent_type=data.get("agent_type", ""),
            target=data.get("target", "prompt"),
            op=data.get("op", "append"),
            path=data.get("path", ""),
            value=data.get("value", ""),
            rationale=data.get("rationale", ""),
            proposed_by=data.get("proposed_by", "hermes"),
            proposed_at=proposed_at,
            score=data.get("score"),
            status=data.get("status", "pending"),
            based_on_errors=list(data.get("based_on_errors", [])),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Patch":
        return cls.from_dict(json.loads(raw))


@dataclass
class PatchOutcome:
    """Result of applying a patch and re-evaluating.

    Attributes:
        patch:          The Patch that was applied.
        baseline_score: Score before applying the patch.
        patched_score:  Score after applying the patch.
        improvement:    patched_score - baseline_score.
        accepted:       Whether the patch was accepted (improvement >= threshold).
        eval_summary:   Human-readable evaluation summary.
    """

    patch: Patch
    baseline_score: float = 0.0
    patched_score: float = 0.0
    improvement: float = 0.0
    accepted: bool = False
    eval_summary: str = ""

    def __post_init__(self) -> None:
        self.improvement = self.patched_score - self.baseline_score


# ---------------------------------------------------------------------------
# Patch generation prompt
# ---------------------------------------------------------------------------

_GEN_SYSTEM = (
    "You are Hermes, an AI self-improvement engine for multi-agent systems. "
    "You analyze error patterns and generate targeted improvements to agent prompts. "
    "Always output valid JSON."
)

_GEN_PROMPT_TEMPLATE = """\
You are analyzing failures in an AI agent system to improve its prompt.

## Agent Type
{agent_type}

## Current Prompt
```
{current_prompt}
```

## Recent Errors ({error_count} failures)
{error_summary}

## Most Common Failure Classes
{failure_classes}

## Task
Generate a single targeted patch to improve the agent's prompt based on these errors.

The patch should:
1. Address the root cause of the most common failures
2. Be specific and actionable (not generic advice)
3. Add guidance that was missing or clarify confusing instructions

Respond with a JSON object in exactly this format:
{{
  "op": "<append|prepend|replace|remove|set>",
  "path": "<exact text to find in prompt for replace/remove, empty for append/prepend/set>",
  "value": "<the new text to insert or replace with>",
  "rationale": "<2-3 sentence explanation of why this patch addresses the failures>"
}}

JSON:"""


class PatchGenerator:
    """Generates prompt patches by analyzing error records with an LLM.

    Args:
        llm_provider:   LLMProvider for generating patches.
        prompt_manager: PromptManager for reading current active prompts.
        patch_store:    Optional Redis-backed store for persisting patches.
    """

    def __init__(
        self,
        llm_provider: Any,
        prompt_manager: Any,
        patch_store: Optional[Any] = None,
    ) -> None:
        self._llm = llm_provider
        self._prompt_manager = prompt_manager
        self._patch_store = patch_store

    async def generate(
        self,
        agent_type: str,
        errors: list[ErrorRecord],
        max_errors_in_prompt: int = 10,
    ) -> Optional[Patch]:
        """Generate a patch based on recent error records.

        Args:
            agent_type:            The agent type to patch.
            errors:                List of recent ErrorRecord objects.
            max_errors_in_prompt:  Max error entries to include in the LLM prompt.

        Returns:
            A Patch object if generation succeeded, else None.
        """
        if not errors:
            logger.info("No errors to generate patch from for agent_type=%s", agent_type)
            return None

        # Get current prompt
        current_prompt = await self._prompt_manager.get_prompt(agent_type)

        # Summarise errors
        sample_errors = errors[:max_errors_in_prompt]
        error_lines = []
        for i, err in enumerate(sample_errors, 1):
            error_lines.append(
                f"{i}. [{err.failure_class}] {err.error_message[:200]}"
                + (f"\n   Task: {err.task[:100]}" if err.task else "")
            )
        error_summary = "\n".join(error_lines)

        # Count failure classes
        fc_counts: dict[str, int] = {}
        for err in errors:
            fc_counts[err.failure_class] = fc_counts.get(err.failure_class, 0) + 1
        failure_classes = ", ".join(
            f"{fc}={count}"
            for fc, count in sorted(fc_counts.items(), key=lambda x: -x[1])[:5]
        )

        prompt = _GEN_PROMPT_TEMPLATE.format(
            agent_type=agent_type,
            current_prompt=current_prompt[:3000],
            error_count=len(errors),
            error_summary=error_summary,
            failure_classes=failure_classes,
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                system=_GEN_SYSTEM,
            )
            raw = response.content.strip()

            # Strip markdown
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()

            data = json.loads(raw)

            patch = Patch(
                agent_type=agent_type,
                target="prompt",
                op=str(data.get("op", "append")).lower(),
                path=str(data.get("path", "")),
                value=str(data.get("value", "")),
                rationale=str(data.get("rationale", "")),
                proposed_by="hermes",
                based_on_errors=[err.record_id for err in sample_errors],
            )

            if self._patch_store is not None:
                await self._patch_store.save(patch)

            logger.info(
                "Generated patch %s for agent_type=%s (op=%s)",
                patch.patch_id[:8],
                agent_type,
                patch.op,
            )
            return patch

        except json.JSONDecodeError as exc:
            logger.warning("Patch generation JSON parse error: %s", exc)
            return None
        except Exception as exc:
            logger.error("Patch generation failed for agent_type=%s: %s", agent_type, exc)
            return None

    async def generate_retry_patch(
        self,
        agent_type: str,
        errors: list[ErrorRecord],
        tool_registry: Any | None = None,
    ) -> Optional[Patch]:
        """Propose a timeout increase for tools that consistently time out.

        Analyses TOOL_TIMEOUT failures, identifies the most-affected tool,
        and proposes doubling its ``timeout_seconds`` (capped at 120 s).

        Returns None if there are no timeout errors or the tool cannot be
        identified from error messages.
        """
        timeout_errors = [e for e in errors if "TOOL_TIMEOUT" in e.failure_class]
        if not timeout_errors:
            return None

        # Count which tool times out most often
        tool_counts: dict[str, int] = {}
        for err in timeout_errors:
            m = re.search(r"Tool '([^']+)' timed out", err.error_message or "")
            if m:
                tool_counts[m.group(1)] = tool_counts.get(m.group(1), 0) + 1

        if not tool_counts:
            return None

        worst_tool = max(tool_counts, key=lambda k: tool_counts[k])
        current_timeout = 30.0
        if tool_registry is not None:
            try:
                tool = tool_registry.get(worst_tool)
                if tool is not None:
                    current_timeout = float(getattr(tool, "timeout_seconds", 30.0))
            except Exception:
                pass

        suggested_timeout = min(current_timeout * 2, 120.0)

        patch = Patch(
            agent_type=agent_type,
            target="retry_config",
            op="set",
            path=worst_tool,
            value=str(suggested_timeout),
            rationale=(
                f"'{worst_tool}' timed out {tool_counts[worst_tool]} time(s) "
                f"across {len(timeout_errors)} failures. "
                f"Proposal: increase timeout_seconds from {current_timeout}s "
                f"to {suggested_timeout}s."
            ),
            proposed_by="hermes",
            based_on_errors=[e.record_id for e in timeout_errors[:5]],
        )
        if self._patch_store is not None:
            await self._patch_store.save(patch)
        logger.info(
            "Generated retry patch %s for agent_type=%s tool=%s (%.0f→%.0f s)",
            patch.patch_id[:8], agent_type, worst_tool,
            current_timeout, suggested_timeout,
        )
        return patch

    async def generate_permission_patch(
        self,
        agent_type: str,
        errors: list[ErrorRecord],
    ) -> Optional[Patch]:
        """Propose a policy update based on recurring safety violations.

        Analyses SAFETY_STEP / SAFETY_OUTPUT failures, identifies the most
        frequently blocked tool, and proposes adding it to ``blocked_tools``
        so future HITL gates fire immediately instead of executing and failing.

        Returns None if there are no safety errors.
        """
        safety_errors = [
            e for e in errors
            if any(fc in e.failure_class for fc in ("SAFETY_", "GUARDRAIL"))
        ]
        if not safety_errors:
            return None

        # Extract which tool was blocked most often
        tool_counts: dict[str, int] = {}
        for err in safety_errors:
            m = re.search(
                r"Tool '([^']+)' (?:is blocked|blocked by|blocked:)",
                err.error_message or "",
            )
            if m:
                tool_counts[m.group(1)] = tool_counts.get(m.group(1), 0) + 1

        if tool_counts:
            most_blocked = max(tool_counts, key=lambda k: tool_counts[k])
            value = json.dumps({
                "add_to_blocked_tools": most_blocked,
                "violation_count": tool_counts[most_blocked],
                "recommendation": (
                    f"Add '{most_blocked}' to HarnessPolicy.blocked_tools "
                    "to prevent repeated safety violations."
                ),
            })
            rationale = (
                f"'{most_blocked}' was blocked by the safety pipeline "
                f"{tool_counts[most_blocked]} time(s). "
                "Adding it to blocked_tools enforces the boundary at the "
                "policy gate before HITL is triggered."
            )
        else:
            value = json.dumps({
                "recommendation": "review_tool_permissions",
                "safety_failure_count": len(safety_errors),
            })
            rationale = (
                f"{len(safety_errors)} safety violation(s) detected. "
                "Review tool permissions and tighten policy boundaries."
            )

        patch = Patch(
            agent_type=agent_type,
            target="permission",
            op="set",
            path="blocked_tools",
            value=value,
            rationale=rationale,
            proposed_by="hermes",
            based_on_errors=[e.record_id for e in safety_errors[:5]],
        )
        if self._patch_store is not None:
            await self._patch_store.save(patch)
        logger.info(
            "Generated permission patch %s for agent_type=%s (%d safety errors)",
            patch.patch_id[:8], agent_type, len(safety_errors),
        )
        return patch

    async def generate_tool_patch(
        self,
        agent_type: str,
        errors: list[ErrorRecord],
        tool_registry: Any | None = None,
    ) -> Optional[Patch]:
        """
        Generate a tool description / schema patch from tool-related errors.

        Analyses TOOL_NOT_FOUND, TOOL_SCHEMA_ERROR, and TOOL_EXEC_ERROR failures
        and proposes updated tool descriptions or argument schema fixes.

        Args:
            agent_type:    The agent type whose tools to patch.
            errors:        Recent tool-related ErrorRecord objects.
            tool_registry: Optional ToolRegistry to read current tool definitions.

        Returns:
            A Patch with target="tool_config" if generation succeeded, else None.
        """
        tool_errors = [
            e for e in errors
            if any(fc in e.failure_class for fc in ("TOOL_", "MCP_"))
        ]
        if not tool_errors:
            return None

        # Build current tool definitions summary
        tools_summary = ""
        if tool_registry is not None:
            try:
                tools = tool_registry.list_tools()
                tools_summary = "\n".join(
                    f"- {t.name}: {t.description} | schema: {json.dumps(t.input_schema)[:200]}"
                    for t in tools[:10]
                )
            except Exception:
                pass

        sample = tool_errors[:8]
        error_lines = "\n".join(
            f"{i}. [{e.failure_class}] {e.error_message[:200]}"
            + (f"\n   Task: {e.task[:80]}" if e.task else "")
            for i, e in enumerate(sample, 1)
        )

        prompt = (
            f"Agent type: {agent_type}\n\n"
            f"Current tools:\n{tools_summary or 'Unknown'}\n\n"
            f"Recent tool failures ({len(tool_errors)} total):\n{error_lines}\n\n"
            "Propose a tool configuration fix as JSON:\n"
            '{"tool_name": str, "op": "update_description"|"update_schema"|"add_example", '
            '"value": str_or_dict, "rationale": str}'
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                system=(
                    "You are an agent tool configuration expert. "
                    "Analyse tool failures and propose precise tool definition fixes. "
                    "Return only valid JSON."
                ),
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()

            data = json.loads(raw)
            patch = Patch(
                agent_type=agent_type,
                target="tool_config",
                op=str(data.get("op", "update_description")).lower(),
                path=str(data.get("tool_name", "")),
                value=json.dumps(data.get("value", "")),
                rationale=str(data.get("rationale", "")),
                proposed_by="hermes",
                based_on_errors=[e.record_id for e in sample],
            )
            if self._patch_store is not None:
                await self._patch_store.save(patch)
            logger.info(
                "Generated tool patch %s for agent_type=%s tool=%s",
                patch.patch_id[:8], agent_type, patch.path,
            )
            return patch
        except Exception as exc:
            logger.warning("Tool patch generation failed: %s", exc)
            return None
