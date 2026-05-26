"""HarnessAgent — production harness for any AI agent framework."""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"
version = __version__


def wrap(
    obj: Any,
    safety_pipeline: Any | None = None,
    cost_tracker: Any | None = None,
    audit_logger: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Auto-detect the framework type and return a production-ready adapter.

    Wraps any LangGraph graph, CrewAI crew, or AutoGen agent with the full
    harness stack: safety guardrails, cost tracking, budget enforcement,
    and observability.

    Usage::

        from harness import wrap

        # LangGraph
        adapter = wrap(my_graph)

        # CrewAI
        adapter = wrap(my_crew, safety_pipeline=pipeline)

        # AutoGen
        adapter = wrap(initiator_agent, recipient=recipient_agent)

    Args:
        obj:             The framework object to wrap (StateGraph, Crew,
                         ConversableAgent, or compiled LangGraph).
        safety_pipeline: Optional guardrail pipeline to inject.
        cost_tracker:    Optional CostTracker to inject.
        audit_logger:    Optional AuditLogger to inject.
        **kwargs:        Extra keyword arguments forwarded to the adapter.

    Returns:
        A FrameworkAdapter subclass ready to call ``.run_with_harness()``.

    Raises:
        TypeError: If the object type is not recognised.
        ImportError: If the required framework is not installed.
    """
    adapter = _detect_and_build(obj, **kwargs)
    adapter.attach_harness(
        safety_pipeline=safety_pipeline,
        cost_tracker=cost_tracker,
        audit_logger=audit_logger,
    )
    return adapter


def _detect_and_build(obj: Any, **kwargs: Any) -> Any:
    """Detect framework type and instantiate the correct adapter."""

    # --- LangGraph ---
    # Compiled graphs have .invoke + .astream; raw StateGraph has .compile()
    _is_langgraph = False
    try:
        from langgraph.graph import StateGraph as _StateGraph  # type: ignore
        if isinstance(obj, _StateGraph):
            _is_langgraph = True
    except ImportError:
        pass

    if not _is_langgraph and hasattr(obj, "astream") and hasattr(obj, "get_graph"):
        _is_langgraph = True

    if _is_langgraph:
        from harness.adapters.langgraph import LangGraphAdapter
        return LangGraphAdapter(graph=obj, **kwargs)

    # --- CrewAI ---
    # Crew has .kickoff() and an .agents list
    if hasattr(obj, "kickoff") and hasattr(obj, "agents"):
        from harness.adapters.crewai import CrewAIAdapter
        return CrewAIAdapter(crew=obj, **kwargs)

    # --- AutoGen ConversableAgent ---
    # ConversableAgent and its subclasses (AssistantAgent, UserProxyAgent)
    # all have .initiate_chat()
    if hasattr(obj, "initiate_chat"):
        from harness.adapters.autogen import AutoGenAdapter
        recipient = kwargs.pop("recipient", None)
        if recipient is None:
            raise TypeError(
                "AutoGen wrap() requires a 'recipient' keyword argument: "
                "wrap(initiator, recipient=other_agent)"
            )
        return AutoGenAdapter(
            initiator_agent=obj,
            recipient_agent_or_groupchat=recipient,
            **kwargs,
        )

    raise TypeError(
        f"Cannot wrap {type(obj).__name__!r}. "
        "Supported types: LangGraph StateGraph / compiled graph, "
        "CrewAI Crew, AutoGen ConversableAgent. "
        "Pass the framework object directly to wrap()."
    )
