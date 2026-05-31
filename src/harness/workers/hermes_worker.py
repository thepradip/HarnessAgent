"""Hermes background worker — runs the self-improvement cycle on a schedule."""

from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hermes cycle task
# ---------------------------------------------------------------------------


async def run_hermes_cycle(config_override: dict | None = None) -> None:
    """Execute one Hermes self-improvement cycle for all registered agent types.

    Initialises all required dependencies (Redis, LLM, prompt manager,
    error collector), then runs the full Hermes cycle for each known
    agent type (sql, code, research, orchestrator).

    Args:
        config_override: Optional dict to override config values for testing.
    """
    config_override = config_override or {}

    import redis.asyncio as aioredis

    from harness.core.config import get_config
    from harness.improvement.error_collector import ErrorCollector
    from harness.improvement.gepa import build_patch_generator
    from harness.improvement.patch_generator import Patch
    from harness.prompts.manager import PromptManager
    from harness.prompts.store import PromptStore

    cfg = get_config()
    redis_url = config_override.get("redis_url", cfg.redis_url)

    # Initialise Redis
    redis_client = aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=10,
    )

    try:
        await redis_client.ping()
    except Exception as exc:
        logger.error("Hermes worker: Redis connection failed: %s", exc)
        return

    # Initialise components
    error_collector = ErrorCollector(redis=redis_client)
    prompt_store = PromptStore(redis=redis_client)
    prompt_manager = PromptManager(store=prompt_store)

    # Initialise LLM provider
    try:
        from harness.llm.anthropic import AnthropicProvider  # type: ignore
        llm = AnthropicProvider(
            api_key=config_override.get("anthropic_api_key", cfg.anthropic_api_key),
            model=config_override.get("default_model", cfg.default_model),
        )
    except Exception as exc:
        logger.warning("Hermes: LLM provider not available: %s", exc)
        await redis_client.aclose()
        return

    # Patch store backed by Redis
    class _RedisPatchStore:
        _PREFIX = "harness:patch"

        def __init__(self, r):
            self._r = r

        async def save(self, patch: Patch) -> None:
            await self._r.set(f"{self._PREFIX}:{patch.patch_id}", patch.to_json())

    patch_store = _RedisPatchStore(redis_client)
    strategy = config_override.get("hermes_strategy", getattr(cfg, "hermes_strategy", "heuristic"))

    agent_types = config_override.get(
        "agent_types", ["sql", "code", "research", "orchestrator"]
    )
    min_errors = config_override.get(
        "hermes_min_errors_to_trigger", cfg.hermes_min_errors_to_trigger
    )
    auto_apply = config_override.get("hermes_auto_apply", cfg.hermes_auto_apply)

    # Evaluator-backed path: run the full HermesLoop, which scores patches by
    # replaying failing tasks (AgentRunner + Evaluator) before the apply/rollback
    # gate. This is what makes GEPA usable in production. On any setup failure we
    # fall back to the lightweight generate-and-queue path below.
    use_evaluator = config_override.get(
        "hermes_use_evaluator", getattr(cfg, "hermes_use_evaluator", False)
    )
    if use_evaluator:
        try:
            hermes = build_hermes_loop(
                redis_client=redis_client,
                cfg=cfg,
                llm_provider=llm,
                error_collector=error_collector,
                prompt_manager=prompt_manager,
                patch_store=patch_store,
                strategy=strategy,
                config_override=config_override,
            )
            logger.info(
                "Hermes (evaluator-backed) cycle starting: strategy=%s agent_types=%s",
                strategy,
                agent_types,
            )
            loop_outcomes = await hermes.run_all_agents(agent_types)
            logger.info(
                "Hermes (evaluator-backed) cycle complete: %d outcome(s)",
                len(loop_outcomes),
            )
            for loop_outcome in loop_outcomes:
                logger.info(
                    "  %s: applied=%s — %s",
                    getattr(getattr(loop_outcome, "patch", None), "agent_type", "?"),
                    getattr(loop_outcome, "applied", False),
                    getattr(loop_outcome, "reason", ""),
                )
            await redis_client.aclose()
            return
        except Exception as exc:
            logger.exception(
                "Hermes: evaluator-backed loop failed (%s) — falling back to the "
                "lightweight generate-and-queue path.",
                exc,
            )

    # GEPA needs an Evaluator metric to score evolved prompts. This lightweight
    # path does not construct one (it applies/queues without replaying tasks),
    # so build_patch_generator transparently falls back to the heuristic generator
    # when strategy="gepa" and evaluator is None.
    patch_generator = build_patch_generator(
        strategy,
        llm_provider=llm,
        prompt_manager=prompt_manager,
        evaluator=None,
        config=cfg,
        patch_store=patch_store,
    )

    logger.info(
        "Hermes cycle starting: agent_types=%s, min_errors=%d, auto_apply=%s",
        agent_types,
        min_errors,
        auto_apply,
    )

    outcomes: list[dict] = []

    for agent_type in agent_types:
        try:
            # Get recent errors
            errors = await error_collector.get_recent(agent_type, limit=100)
            error_count = len(errors)

            if error_count < min_errors:
                logger.info(
                    "Hermes: Skipping %s (only %d errors, need %d)",
                    agent_type,
                    error_count,
                    min_errors,
                )
                outcomes.append({
                    "agent_type": agent_type,
                    "action": "skipped",
                    "reason": f"insufficient_errors ({error_count}/{min_errors})",
                })
                continue

            logger.info(
                "Hermes: Generating patch for %s from %d errors",
                agent_type,
                error_count,
            )

            # Generate patch
            patch = await patch_generator.generate(
                agent_type=agent_type,
                errors=errors,
            )

            if patch is None:
                logger.warning("Hermes: Patch generation returned None for %s", agent_type)
                outcomes.append({
                    "agent_type": agent_type,
                    "action": "generation_failed",
                    "patch_id": None,
                })
                continue

            logger.info(
                "Hermes: Generated patch %s for %s (op=%s)",
                patch.patch_id[:8],
                agent_type,
                patch.op,
            )

            if auto_apply:
                # Apply directly without evaluation
                await prompt_manager.apply_patch(patch)
                patch.status = "applied"
                await patch_store.save(patch)
                outcomes.append({
                    "agent_type": agent_type,
                    "action": "applied",
                    "patch_id": patch.patch_id,
                })
                logger.info(
                    "Hermes: Auto-applied patch %s for %s",
                    patch.patch_id[:8],
                    agent_type,
                )
            else:
                # Queue for human review
                patch.status = "pending"
                await patch_store.save(patch)
                outcomes.append({
                    "agent_type": agent_type,
                    "action": "queued_for_review",
                    "patch_id": patch.patch_id,
                })
                logger.info(
                    "Hermes: Queued patch %s for review (agent_type=%s)",
                    patch.patch_id[:8],
                    agent_type,
                )

        except Exception as exc:
            logger.exception("Hermes: Error in cycle for agent_type=%s: %s", agent_type, exc)
            outcomes.append({
                "agent_type": agent_type,
                "action": "error",
                "error": str(exc),
            })

    logger.info("Hermes cycle complete: %d outcomes", len(outcomes))
    for outcome in outcomes:
        logger.info("  %s: %s", outcome.get("agent_type"), outcome.get("action"))

    await redis_client.aclose()


# ---------------------------------------------------------------------------
# Evaluator-backed HermesLoop assembly
# ---------------------------------------------------------------------------


def build_hermes_loop(
    *,
    redis_client: Any,
    cfg: Any,
    llm_provider: Any,
    error_collector: Any,
    prompt_manager: Any,
    patch_store: Any,
    strategy: str,
    config_override: dict[str, Any] | None = None,
    agent_factory: Any = None,
) -> Any:
    """Assemble a fully-wired, evaluator-backed :class:`HermesLoop`.

    Builds the AgentRunner + Evaluator (which replay failing tasks to score
    patches), the strategy-selected patch generator *with* that evaluator (so
    GEPA can optimize against it), an OnlineLearningMonitor for rollback, and an
    optional MLflow tracer. The agent factory is reused from the same builder the
    API server uses, so replayed runs match production agent construction.

    Args:
        redis_client:    Async Redis client.
        cfg:             Settings.
        llm_provider:    LLM provider (reflection LM for GEPA / patch LM otherwise).
        error_collector: Shared ErrorCollector.
        prompt_manager:  PromptManager (also serves as the prompt store for the loop).
        patch_store:     Store with ``async save(patch)``.
        strategy:        ``"gepa"`` or ``"heuristic"``.
        config_override: Optional overrides (e.g. ``workspace_base_path``).
        agent_factory:   Optional ``Callable(agent_type) -> agent``. When omitted,
            ``build_agent_factory(cfg)`` is used (the production builder).

    Returns:
        A ready-to-run :class:`HermesLoop`.
    """
    from harness.improvement.evaluator import Evaluator
    from harness.improvement.gepa import build_patch_generator
    from harness.improvement.hermes import HermesLoop
    from harness.improvement.online_monitor import OnlineLearningMonitor
    from harness.observability.metrics import get_prometheus_metrics
    from harness.orchestrator.runner import AgentRunner

    config_override = config_override or {}

    if agent_factory is None:
        from harness.workers.agent_worker import build_agent_factory
        agent_factory = build_agent_factory(cfg)

    agent_runner = AgentRunner(
        redis=redis_client,
        agent_factory=agent_factory,
        workspace_base=config_override.get(
            "workspace_base_path", getattr(cfg, "workspace_base_path", "/workspaces")
        ),
        error_collector=error_collector,
    )
    evaluator = Evaluator(agent_runner=agent_runner, error_collector=error_collector)

    generator = build_patch_generator(
        strategy,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        evaluator=evaluator,
        config=cfg,
        patch_store=patch_store,
    )

    online_monitor = OnlineLearningMonitor(redis=redis_client)

    mlflow_tracer = None
    if getattr(cfg, "hermes_gepa_use_mlflow", False):
        try:
            from harness.observability.mlflow_tracer import MLflowAgentTracer
            mlflow_tracer = MLflowAgentTracer(
                tracking_uri=cfg.mlflow_tracking_uri,
                experiment_name=cfg.mlflow_experiment_name,
            )
        except Exception as exc:
            logger.debug("Hermes: MLflow tracer unavailable: %s", exc)

    # HermesLoop reads counters off the metrics object via getattr; expose the
    # Prometheus registry's named metrics as attributes.
    metrics = SimpleNamespace(**get_prometheus_metrics())

    return HermesLoop(
        collector=error_collector,
        generator=generator,
        evaluator=evaluator,
        prompt_store=prompt_manager,
        metrics=metrics,
        config=cfg,
        online_monitor=online_monitor,
        mlflow_tracer=mlflow_tracer,
    )


# ---------------------------------------------------------------------------
# APScheduler-based scheduler
# ---------------------------------------------------------------------------


def start_hermes_worker(
    interval_seconds: float | None = None,
    run_once: bool = False,
) -> None:
    """Start the Hermes background scheduler.

    Uses APScheduler AsyncIOScheduler to run ``run_hermes_cycle()`` at the
    configured interval.

    Args:
        interval_seconds: Override for the cycle interval (for testing).
        run_once:         If True, run once and exit (useful for testing).
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
        from apscheduler.triggers.interval import IntervalTrigger  # type: ignore
    except ImportError:
        logger.error(
            "APScheduler not installed. Install with: pip install apscheduler>=3.10"
        )
        sys.exit(1)

    from harness.core.config import get_config

    cfg = get_config()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    effective_interval = interval_seconds or cfg.hermes_interval_seconds

    async def _run() -> None:
        if run_once:
            await run_hermes_cycle()
            return

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            run_hermes_cycle,
            trigger=IntervalTrigger(seconds=effective_interval),
            id="hermes_cycle",
            name="Hermes Self-Improvement Cycle",
            replace_existing=True,
            max_instances=1,  # prevent overlapping runs
        )
        scheduler.start()
        logger.info(
            "Hermes worker started (interval=%.0fs)",
            effective_interval,
        )

        try:
            # Run once immediately at startup
            await run_hermes_cycle()
            # Keep running until interrupted
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Hermes worker shutting down")
            scheduler.shutdown(wait=False)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Hermes worker stopped")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    start_hermes_worker()
