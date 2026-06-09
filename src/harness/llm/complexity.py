"""Query-complexity scoring for cost-aware LLM routing.

The router uses a *tier* string (``cheap`` / ``standard`` / ``premium``) to pick
which model serves a request. The most efficient way to choose that tier is a
heuristic — no extra LLM call, no added latency or cost on the request path.

``HeuristicComplexityScorer`` inspects signals already present in the request
(token estimate, tool use, code/SQL/reasoning keywords, turn count, required
context) and returns a tier. It implements the :class:`ComplexityScorer`
protocol, so a learned classifier (embedding-based via fastembed, or a small
LLM such as Haiku / gpt-4o-mini) can be dropped in later without touching the
router — see :class:`ComplexityScorer`.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

# Canonical tier order, cheapest/least-capable first. The router treats the
# returned string as the *preferred* tier and falls back through the remaining
# providers (by priority) if no provider in that tier can serve the request.
TIER_CHEAP = "cheap"
TIER_STANDARD = "standard"
TIER_PREMIUM = "premium"
TIER_ORDER: tuple[str, ...] = (TIER_CHEAP, TIER_STANDARD, TIER_PREMIUM)


@runtime_checkable
class ComplexityScorer(Protocol):
    """Returns a tier name for a request. Implement to swap the routing policy.

    The default implementation is heuristic (free, deterministic). A learned
    scorer — embedding classifier or a small LLM — can implement this same
    protocol; the router only depends on ``score`` returning a tier string.
    """

    def score(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        required_context: int = 0,
        max_tokens: int = 0,
    ) -> str:
        ...


# Keywords that signal a request likely needs a stronger model. Kept small and
# word-boundary matched to avoid false positives (e.g. "prove" in "approve").
_HARD_KEYWORDS = re.compile(
    r"\b("
    r"prove|proof|derive|theorem|optimi[sz]e|refactor|architect|"
    r"debug|stack ?trace|traceback|exception|"
    r"step[- ]by[- ]step|reason through|chain[- ]of[- ]thought|think carefully|"
    r"sql|select\s|join\b|subquery|cte\b|"
    r"regex|recursion|recursive|concurren|deadlock|race condition|"
    r"analy[sz]e|compare and contrast|trade[- ]?off"
    r")\b",
    re.IGNORECASE,
)

# Fenced code blocks are a strong "this is real work" signal.
_CODE_FENCE = re.compile(r"```")


def _text_of(messages: list[dict[str, Any]], system: str | None) -> str:
    parts: list[str] = [system or ""]
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


class HeuristicComplexityScorer:
    """Free, deterministic complexity scorer — the default routing policy.

    Thresholds are constructor args so a deployment can tune them without
    code changes. ``estimate_tokens`` is a coarse chars/4 heuristic; it does not
    call a tokenizer (that would add latency the routing decision is meant to
    save).
    """

    def __init__(
        self,
        *,
        cheap_max_tokens: int = 200,
        premium_min_tokens: int = 1500,
        premium_min_turns: int = 12,
        premium_context: int = 32_000,
        chars_per_token: int = 4,
    ) -> None:
        self._cheap_max_tokens = cheap_max_tokens
        self._premium_min_tokens = premium_min_tokens
        self._premium_min_turns = premium_min_turns
        self._premium_context = premium_context
        self._chars_per_token = max(1, chars_per_token)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // self._chars_per_token

    def score(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        required_context: int = 0,
        max_tokens: int = 0,
    ) -> str:
        text = _text_of(messages, system)
        est_tokens = self.estimate_tokens(text)
        turns = len(messages)

        # --- premium: anything that smells like real, hard, or long work ---
        if tools:
            return TIER_PREMIUM
        if required_context and required_context >= self._premium_context:
            return TIER_PREMIUM
        if est_tokens >= self._premium_min_tokens:
            return TIER_PREMIUM
        if turns >= self._premium_min_turns:
            return TIER_PREMIUM
        if _CODE_FENCE.search(text) or _HARD_KEYWORDS.search(text):
            return TIER_PREMIUM

        # --- cheap: short, single-shot, no complexity signals ---
        if turns <= 2 and est_tokens <= self._cheap_max_tokens:
            return TIER_CHEAP

        # --- everything else ---
        return TIER_STANDARD


# Module-level convenience scorer for callers that don't need to tune thresholds.
DEFAULT_SCORER = HeuristicComplexityScorer()


def score_complexity(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    required_context: int = 0,
    max_tokens: int = 0,
) -> str:
    """Score a request with the default heuristic scorer."""
    return DEFAULT_SCORER.score(
        messages,
        system=system,
        tools=tools,
        required_context=required_context,
        max_tokens=max_tokens,
    )
