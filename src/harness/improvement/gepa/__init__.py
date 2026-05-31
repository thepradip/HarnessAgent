"""GEPA reflective prompt evolution for the Hermes self-improvement loop.

Public surface:

- :class:`GepaPatchGenerator` — a drop-in for ``PatchGenerator`` that evolves the
  agent system prompt with GEPA instead of a single heuristic edit.
- :func:`build_patch_generator` — selects the generator from a strategy string,
  falling back to the heuristic generator when GEPA is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from harness.improvement.gepa.generator import GepaPatchGenerator
from harness.improvement.patch_generator import PatchGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "EvalDatasetGepaAdapter",
    "EvalOptimizationResult",
    "GepaPatchGenerator",
    "build_patch_generator",
    "optimize_prompts_on_dataset",
]


def __getattr__(name: str) -> Any:
    # Lazy re-export of the eval-optimizer surface to avoid importing the eval
    # stack (datasets/scorers) unless it's actually used.
    if name in ("optimize_prompts_on_dataset", "EvalOptimizationResult"):
        from harness.improvement.gepa import eval_optimizer

        return getattr(eval_optimizer, name)
    if name == "EvalDatasetGepaAdapter":
        from harness.improvement.gepa.eval_adapter import EvalDatasetGepaAdapter

        return EvalDatasetGepaAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_patch_generator(
    strategy: str,
    *,
    llm_provider: Any,
    prompt_manager: Any,
    evaluator: Any | None = None,
    config: Any = None,
    patch_store: Any | None = None,
) -> Any:
    """Return the patch generator for ``strategy``.

    Args:
        strategy:       ``"gepa"`` for reflective prompt evolution, anything else
            (e.g. ``"heuristic"``) for the default :class:`PatchGenerator`.
        llm_provider:   LLM provider (reflection LM for GEPA / patch LM for heuristic).
        prompt_manager: PromptManager for reading the current prompt.
        evaluator:      Evaluator metric — *required* for GEPA. When missing, GEPA
            cannot score candidates, so the heuristic generator is used instead.
        config:         Settings carrying ``hermes_gepa_*`` keys.
        patch_store:    Optional store for persisting proposed patches.

    Returns:
        A generator exposing ``async generate(agent_type, errors, max_errors_in_prompt)``.
    """
    if strategy == "gepa":
        if evaluator is None:
            logger.warning(
                "Hermes strategy 'gepa' requested but no evaluator was provided — "
                "GEPA needs a metric to optimize against. Falling back to the "
                "heuristic PatchGenerator."
            )
        else:
            try:
                import gepa  # noqa: F401  (availability probe)
            except ImportError:
                logger.warning(
                    "Hermes strategy 'gepa' requested but the 'gepa' package is not "
                    "installed (pip install agent-haas[improvement]). Falling back to "
                    "the heuristic PatchGenerator."
                )
            else:
                logger.info("Hermes: using GEPA reflective prompt-evolution generator.")
                return GepaPatchGenerator(
                    llm_provider=llm_provider,
                    prompt_manager=prompt_manager,
                    evaluator=evaluator,
                    config=config,
                    patch_store=patch_store,
                )

    return PatchGenerator(
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        patch_store=patch_store,
    )
