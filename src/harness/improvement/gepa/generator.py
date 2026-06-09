"""GepaPatchGenerator — GEPA reflective prompt evolution as a Hermes generator.

A drop-in replacement for :class:`PatchGenerator`: it exposes the same
``async generate(agent_type, errors, max_errors_in_prompt) -> Patch | None``
surface the :class:`HermesLoop` already calls, so the loop, its holdout
evaluation, the auto-apply threshold, and rollback all stay unchanged.

Where the heuristic generator asks an LLM for one targeted edit, this one runs
GEPA: it evolves a *population* of full-prompt candidates, scores each against
the injected :class:`Evaluator`, reflects on the failures, and keeps a Pareto
front across the sampled tasks. The winning prompt is returned as a single
``op="set"`` patch.

GEPA's engine is synchronous, so ``optimize()`` runs in a worker thread while the
adapter and reflection LM bounce their coroutines back to this loop (see
``reflection.make_coro_runner``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from harness.improvement.gepa.adapter import COMPONENT, HarnessGepaAdapter
from harness.improvement.gepa.reflection import ProviderReflectionLM, make_coro_runner
from harness.improvement.patch_generator import Patch, PatchGenerator

logger = logging.getLogger(__name__)


class GepaPatchGenerator:
    """Generate prompt patches via GEPA reflective prompt evolution.

    GEPA owns only the *prompt* patch path (``generate``). The HermesLoop also
    routes timeout/safety/tool-dominant failures to specialized non-prompt
    generators (``generate_retry_patch`` / ``generate_permission_patch`` /
    ``generate_tool_patch``); those are delegated to an inner heuristic
    :class:`PatchGenerator` so a tool timeout still gets a one-line config bump
    instead of an expensive prompt evolution.

    Args:
        llm_provider:   Provider used as GEPA's reflection (teacher) LM. Must expose
            ``async complete(messages, *, max_tokens, system=None)``.
        prompt_manager: PromptManager (``async get_prompt(agent_type) -> str``) — the
            current active prompt seeds the optimization.
        evaluator:      Evaluator (``async score(patch, test_cases, agent_type)``) —
            GEPA's metric. This is the same scorer the Hermes loop gates on.
        config:         Settings carrying ``hermes_gepa_*`` keys (optional; sane
            defaults are used when absent).
        patch_store:    Optional store with ``async save(patch)`` for persistence.
    """

    def __init__(
        self,
        llm_provider: Any,
        prompt_manager: Any,
        evaluator: Any,
        config: Any = None,
        patch_store: Any | None = None,
    ) -> None:
        self._llm = llm_provider
        self._prompt_manager = prompt_manager
        self._evaluator = evaluator
        self._config = config
        self._patch_store = patch_store
        # Specialized non-prompt patches (timeout/permission/tool) stay heuristic.
        self._heuristic = PatchGenerator(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            patch_store=patch_store,
        )

        self._budget: int = _cfg(config, "hermes_gepa_budget", 30)
        self._reflection_max_tokens: int = _cfg(
            config, "hermes_gepa_reflection_max_tokens", 4096
        )
        self._min_train: int = _cfg(config, "hermes_gepa_min_train", 3)
        self._use_mlflow: bool = bool(getattr(config, "hermes_gepa_use_mlflow", False))
        self._mlflow_tracking_uri: str | None = getattr(config, "mlflow_tracking_uri", None)
        self._mlflow_experiment_name: str | None = getattr(
            config, "mlflow_experiment_name", None
        )

    async def generate(
        self,
        agent_type: str,
        errors: list[Any],
        max_errors_in_prompt: int = 10,
    ) -> Patch | None:
        """Evolve a new system prompt for ``agent_type`` from recent failures.

        Returns a ``op="set"`` Patch with the evolved prompt, or ``None`` when
        there is nothing to do (too few errors, GEPA unavailable, no improvement,
        or an unrecoverable optimization error — all non-fatal to the loop).
        """
        if not errors:
            logger.info("GEPA: no errors to optimize from for agent_type=%s", agent_type)
            return None

        if len(errors) < self._min_train:
            logger.info(
                "GEPA: too few errors for %s (%d < %d) — skipping optimization",
                agent_type,
                len(errors),
                self._min_train,
            )
            return None

        try:
            import gepa  # local import: optional dependency
        except ImportError:
            logger.warning(
                "GEPA: the 'gepa' package is not installed "
                "(pip install agent-haas[improvement]) — cannot optimize."
            )
            return None

        seed_prompt = await self._prompt_manager.get_prompt(agent_type)
        if not seed_prompt:
            logger.warning("GEPA: empty seed prompt for %s — skipping", agent_type)
            return None

        # DataInst = ErrorRecord. Cap the working set to bound rollout cost.
        dataset = list(errors[: max(self._min_train, max_errors_in_prompt)])

        # Temporal train/val split (mirrors hermes.py): errors are newest-first,
        # so the newer half is held out for validation and the older half drives
        # optimization. Validating on the training set (the old trainset==valset)
        # makes the reported "validation score" in-sample and uninformative.
        split = max(1, len(dataset) // 2)
        valset = dataset[:split]        # newest — held-out validation only
        trainset = dataset[split:]      # older  — used to evolve the prompt
        if not trainset:
            # Sample too small to split — fall back to shared set but flag it.
            trainset = dataset
            valset = dataset
            in_sample = True
        else:
            in_sample = False

        loop = asyncio.get_running_loop()
        run_coro = make_coro_runner(loop)
        adapter = HarnessGepaAdapter(
            evaluator=self._evaluator, agent_type=agent_type, run_coro=run_coro
        )
        reflection_lm = ProviderReflectionLM(
            llm_provider=self._llm,
            run_coro=run_coro,
            max_tokens=self._reflection_max_tokens,
        )

        # gepa ships without type stubs; treat optimize as untyped here.
        gepa_optimize: Any = gepa.optimize  # type: ignore[attr-defined]

        def _optimize() -> Any:
            return gepa_optimize(
                seed_candidate={COMPONENT: seed_prompt},
                trainset=trainset,
                valset=valset,
                adapter=adapter,
                reflection_lm=reflection_lm,
                candidate_selection_strategy="pareto",
                max_metric_calls=self._budget,
                display_progress_bar=False,
                raise_on_exception=False,
                seed=0,
                use_mlflow=self._use_mlflow,
                mlflow_tracking_uri=self._mlflow_tracking_uri if self._use_mlflow else None,
                mlflow_experiment_name=(
                    self._mlflow_experiment_name if self._use_mlflow else None
                ),
            )

        try:
            result = await asyncio.to_thread(_optimize)
        except Exception as exc:  # never break the Hermes cycle
            logger.error("GEPA: optimization failed for %s: %s", agent_type, exc)
            return None

        best_candidate = getattr(result, "best_candidate", None) or {}
        best_prompt = best_candidate.get(COMPONENT, "")

        if not best_prompt or best_prompt.strip() == seed_prompt.strip():
            logger.info(
                "GEPA: no improved prompt found for %s — returning no patch", agent_type
            )
            return None

        best_score = _best_score(result)
        rationale = (
            f"GEPA reflective prompt evolution over {len(trainset)} training "
            f"task(s) (val={len(valset)}) "
            f"using {getattr(result, 'total_metric_calls', '?')} metric call(s)"
        )
        if best_score is not None:
            score_label = "in-sample score" if in_sample else "validation score"
            rationale += f"; best {score_label} {best_score:.3f}"
        rationale += "."

        patch = Patch(
            agent_type=agent_type,
            target="prompt",
            op="set",
            path=COMPONENT,
            value=best_prompt,
            rationale=rationale,
            proposed_by="hermes-gepa",
            based_on_errors=[
                getattr(e, "record_id", "") for e in dataset if getattr(e, "record_id", "")
            ],
        )
        if best_score is not None:
            patch.score = best_score

        if self._patch_store is not None:
            try:
                await self._patch_store.save(patch)
            except Exception as exc:
                logger.debug("GEPA: could not persist patch %s: %s", patch.patch_id[:8], exc)

        logger.info(
            "GEPA: generated patch %s for %s (op=set, %d chars, score=%s)",
            patch.patch_id[:8],
            agent_type,
            len(best_prompt),
            f"{best_score:.3f}" if best_score is not None else "n/a",
        )
        return patch

    # ------------------------------------------------------------------
    # Specialized non-prompt patches — delegated to the heuristic generator.
    # The HermesLoop routes timeout/safety/tool-dominant batches here (probed
    # via hasattr); keeping these means GEPA only handles the prompt path.
    # ------------------------------------------------------------------

    async def generate_retry_patch(
        self, agent_type: str, errors: list[Any], tool_registry: Any | None = None
    ) -> Patch | None:
        return await self._heuristic.generate_retry_patch(
            agent_type=agent_type, errors=errors, tool_registry=tool_registry
        )

    async def generate_permission_patch(
        self, agent_type: str, errors: list[Any]
    ) -> Patch | None:
        return await self._heuristic.generate_permission_patch(
            agent_type=agent_type, errors=errors
        )

    async def generate_tool_patch(
        self, agent_type: str, errors: list[Any], tool_registry: Any | None = None
    ) -> Patch | None:
        return await self._heuristic.generate_tool_patch(
            agent_type=agent_type, errors=errors, tool_registry=tool_registry
        )


def _cfg(config: Any, name: str, default: int) -> int:
    """Read an int config attribute with a fallback."""
    try:
        value = getattr(config, name, default)
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _best_score(result: Any) -> float | None:
    """Extract the best candidate's aggregate validation score, if available.

    GEPA exposes per-candidate aggregate scores as ``val_aggregate_scores`` in
    ``to_dict()``; the ``val_aggregate_subscores`` attribute is only populated
    under some eval policies, so we prefer the former and fall back to the latter.
    """
    best_idx = getattr(result, "best_idx", None)

    scores: Any = None
    try:
        scores = result.to_dict().get("val_aggregate_scores")
    except Exception:
        scores = None
    if not scores:
        scores = getattr(result, "val_aggregate_subscores", None)

    try:
        if scores and best_idx is not None and 0 <= best_idx < len(scores):
            return float(scores[best_idx])
    except (TypeError, ValueError, IndexError):
        pass
    return None
