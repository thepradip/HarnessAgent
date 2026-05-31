"""Runtime prompt-component overrides for offline optimization (e.g. GEPA).

An optimizer evaluates a candidate by injecting component texts into
``AgentContext.metadata['gepa_overrides']`` — a dict of component-name -> text.
Prompt-construction sites call :func:`gepa_override` to honor an override when
present, falling back to their normal value otherwise.

This keeps the optimizer decoupled from the runtime: a component (the agent
system prompt, an inter-agent communication/handoff prompt, a context-summary
prompt, ...) becomes optimizable simply by reading its override at the point
where its text is built — no optimizer-specific plumbing in the agent.
"""

from __future__ import annotations

from typing import Any

# Key under AgentContext.metadata holding {component_name: candidate_text}.
OVERRIDES_KEY = "gepa_overrides"


def gepa_override(ctx: Any, name: str, fallback: str) -> str:
    """Return an injected override for component ``name`` if present, else ``fallback``.

    Args:
        ctx:      An object with a ``.metadata`` dict (e.g. AgentContext).
        name:     The component name (e.g. ``"system_prompt"``).
        fallback: The value to use when no override is injected.
    """
    meta = getattr(ctx, "metadata", None)
    if isinstance(meta, dict):
        overrides = meta.get(OVERRIDES_KEY)
        if isinstance(overrides, dict):
            value = overrides.get(name)
            if value:
                return str(value)
    return fallback
