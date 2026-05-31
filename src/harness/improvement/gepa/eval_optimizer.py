"""Optimize agent prompt components against a gold-labeled EvalDataset with GEPA.

This is the correctness-driven, multi-component optimizer. Give it a set of
seed components (system prompt, inter-agent communication/handoff prompt,
context-summary prompt, ...) and a labeled :class:`EvalDataset`; GEPA evolves the
component texts to maximize the gold scorers, reflecting on per-case diagnostics.

Offline only — intended for a CI/optimization job, not the request path. The
evolved texts are returned for review/promotion into the prompt store.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from harness.improvement.gepa.eval_adapter import EvalDatasetGepaAdapter
from harness.improvement.gepa.generator import _best_score
from harness.improvement.gepa.reflection import ProviderReflectionLM, make_coro_runner

logger = logging.getLogger(__name__)


@dataclass
class EvalOptimizationResult:
    """Outcome of a GEPA-over-EvalDataset run.

    Attributes:
        components:    Evolved component name -> text (the best candidate).
        seed:          The seed component texts the run started from.
        best_score:    Aggregate validation score of the best candidate, if known.
        improved:      Whether any component changed vs the seed.
        total_metric_calls: Number of candidate evaluations GEPA spent.
    """

    components: dict[str, str]
    seed: dict[str, str]
    best_score: float | None
    improved: bool
    total_metric_calls: int | None


async def optimize_prompts_on_dataset(
    *,
    eval_runner: Any,
    dataset: Any,
    llm_provider: Any,
    seed_prompts: dict[str, str],
    valset: Any = None,
    budget: int = 30,
    reflection_max_tokens: int = 4096,
    tenant_id: str = "gepa-eval",
    scorer: Any = None,
    pass_threshold: float = 0.5,
    concurrency: int = 3,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> EvalOptimizationResult:
    """Evolve ``seed_prompts`` to maximize gold scores on ``dataset``.

    Args:
        eval_runner:    EvalRunner instance (its scorers define correctness).
        dataset:        Labeled EvalDataset used as the GEPA trainset.
        llm_provider:   Reflection (teacher) LM provider.
        seed_prompts:   Component name -> current text. Multiple components are
            optimized jointly (GEPA's compound-system mode). The keys must match
            the ``gepa_override`` names read at runtime (e.g. ``"system_prompt"``).
        valset:         Optional EvalDataset for Pareto tracking (defaults to ``dataset``).
        budget:         Max candidate evaluations (``max_metric_calls``).
        Other args:     Reflection size, eval tenant/scorer/threshold/concurrency,
            and optional MLflow tracking.

    Returns:
        EvalOptimizationResult with the evolved components and metadata.
    """
    if not seed_prompts:
        raise ValueError("seed_prompts must contain at least one component")

    try:
        import gepa
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "GEPA optimization requires the 'gepa' package "
            "(pip install agent-haas[improvement])."
        ) from exc

    loop = asyncio.get_running_loop()
    run_coro = make_coro_runner(loop)

    adapter = EvalDatasetGepaAdapter(
        eval_runner,
        run_coro,
        tenant_id=tenant_id,
        scorer=scorer,
        pass_threshold=pass_threshold,
        concurrency=concurrency,
    )
    reflection_lm = ProviderReflectionLM(
        llm_provider=llm_provider, run_coro=run_coro, max_tokens=reflection_max_tokens
    )

    trainset = list(dataset.cases)
    valset_cases = list((valset or dataset).cases)

    gepa_optimize: Any = gepa.optimize  # type: ignore[attr-defined]

    def _optimize() -> Any:
        return gepa_optimize(
            seed_candidate=dict(seed_prompts),
            trainset=trainset,
            valset=valset_cases,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            max_metric_calls=budget,
            display_progress_bar=False,
            raise_on_exception=False,
            seed=0,
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri if use_mlflow else None,
            mlflow_experiment_name=mlflow_experiment_name if use_mlflow else None,
        )

    result = await asyncio.to_thread(_optimize)

    best = dict(getattr(result, "best_candidate", None) or {})
    if not best:
        best = dict(seed_prompts)

    improved = any(
        (best.get(k, "") or "").strip() != (seed_prompts.get(k, "") or "").strip()
        for k in seed_prompts
    )

    logger.info(
        "GEPA/eval: optimized %d component(s) over %d case(s); improved=%s score=%s",
        len(seed_prompts),
        len(trainset),
        improved,
        _best_score(result),
    )

    return EvalOptimizationResult(
        components=best,
        seed=dict(seed_prompts),
        best_score=_best_score(result),
        improved=improved,
        total_metric_calls=getattr(result, "total_metric_calls", None),
    )
