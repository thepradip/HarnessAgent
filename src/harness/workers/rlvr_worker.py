"""
RLVR background worker — processes completed episodes and runs the advantage-
weighted prompt improvement cycle.

Two modes:
  event-driven  Subscribe to Redis Streams for run completion events; process
                each episode as it completes. Zero-latency, preferred for
                production.
  scheduled     Poll for unprocessed episodes on a fixed interval. Fallback
                when the event stream is unavailable.

Usage:
    python -m harness.workers.rlvr_worker            # event-driven
    python -m harness.workers.rlvr_worker --poll 30  # poll every 30 s
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_EVENTS_STREAM = "harness:events:{run_id}"
_PROCESSED_KEY = "harness:rlvr:processed"   # SET of already-processed run_ids
_PROCESSED_TTL = 86_400 * 3                  # 3 days
_AGENT_TYPES = ("sql", "code", "base")


# ---------------------------------------------------------------------------
# Core cycle (shared by both modes)
# ---------------------------------------------------------------------------

async def run_rlvr_cycle(
    run_id: str,
    agent_type: str,
    rlvr_loop: Any,
    redis: Any,
) -> None:
    """
    Process one completed episode.

    Guards against double-processing with a Redis SET.
    """
    already = await redis.sismember(_PROCESSED_KEY, run_id)
    if already:
        logger.debug("RLVR worker: run %s already processed — skipping", run_id[:8])
        return

    logger.info("RLVR worker: processing episode run=%s agent=%s", run_id[:8], agent_type)
    try:
        result = await rlvr_loop.process_episode(run_id, agent_type)
        if result:
            logger.info("RLVR worker: %s", result.summary())
        await redis.sadd(_PROCESSED_KEY, run_id)
        await redis.expire(_PROCESSED_KEY, _PROCESSED_TTL)
    except Exception as exc:
        logger.warning("RLVR worker: episode %s failed: %s", run_id[:8], exc)


# ---------------------------------------------------------------------------
# Event-driven mode
# ---------------------------------------------------------------------------

async def run_event_driven(
    redis_url: str,
    agent_types: tuple[str, ...] = _AGENT_TYPES,
    idle_sleep: float = 1.0,
) -> None:
    """
    Subscribe to all per-run event streams via XREAD and process
    'completed' / 'failed' events as they arrive.
    """
    import redis.asyncio as aioredis
    from harness.core.config import get_config

    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    rlvr_loop, _ = await _build_loop(redis_client)

    # Track last-read position per stream
    stream_cursors: dict[str, str] = {}
    # We subscribe to a global run-completion channel
    completion_key = "harness:rlvr:completions"
    last_comp_id = "0"

    logger.info("RLVR worker: event-driven mode started")

    try:
        while True:
            # Read from the completions stream (published by AgentRunner on finish)
            try:
                entries = await redis_client.xread(
                    {completion_key: last_comp_id}, count=20, block=1000
                )
            except Exception as exc:
                logger.debug("RLVR xread error: %s", exc)
                await asyncio.sleep(idle_sleep)
                continue

            if not entries:
                continue

            for _stream, messages in entries:
                for entry_id, fields in messages:
                    last_comp_id = entry_id if isinstance(entry_id, str) else entry_id.decode()
                    run_id = _decode(fields.get("run_id", ""))
                    agent_type = _decode(fields.get("agent_type", "base"))
                    if run_id and agent_type in agent_types:
                        await run_rlvr_cycle(run_id, agent_type, rlvr_loop, redis_client)

    except asyncio.CancelledError:
        logger.info("RLVR worker: event-driven mode stopped")
    finally:
        await redis_client.aclose()


# ---------------------------------------------------------------------------
# Scheduled (poll) mode
# ---------------------------------------------------------------------------

async def run_poll_mode(
    redis_url: str,
    interval: float = 30.0,
    agent_types: tuple[str, ...] = _AGENT_TYPES,
) -> None:
    """
    Poll for unprocessed completed runs on a fixed interval.
    Reads from harness:rlvr:completions stream using a checkpoint cursor.
    """
    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    rlvr_loop, _ = await _build_loop(redis_client)

    completion_key = "harness:rlvr:completions"
    cursor_key = "harness:rlvr:poll_cursor"

    logger.info("RLVR worker: poll mode started (interval=%.0fs)", interval)

    try:
        while True:
            last_id = await redis_client.get(cursor_key) or "0"
            try:
                # Non-blocking read; poll interval handles the wait
                entries = await redis_client.xread(
                    {completion_key: last_id}, count=50
                )
            except Exception as exc:
                logger.debug("RLVR poll read error: %s", exc)
                await asyncio.sleep(interval)
                continue

            if entries:
                for _stream, messages in entries:
                    for entry_id, fields in messages:
                        new_cursor = entry_id if isinstance(entry_id, str) else entry_id.decode()
                        run_id = _decode(fields.get("run_id", ""))
                        agent_type = _decode(fields.get("agent_type", "base"))
                        if run_id and agent_type in agent_types:
                            await run_rlvr_cycle(run_id, agent_type, rlvr_loop, redis_client)
                        await redis_client.set(cursor_key, new_cursor)

            await asyncio.sleep(interval)

    except asyncio.CancelledError:
        logger.info("RLVR worker: poll mode stopped")
    finally:
        await redis_client.aclose()


# ---------------------------------------------------------------------------
# Dependency builder
# ---------------------------------------------------------------------------

async def _build_loop(redis_client: Any) -> tuple[Any, Any]:
    """Construct RLVRLoop from harness config."""
    from harness.core.config import get_config
    from harness.improvement.rlvr.advantage import AdvantageEstimator
    from harness.improvement.rlvr.buffer import StepRewardBuffer
    from harness.improvement.rlvr.loop import RLVRLoop
    from harness.improvement.patch_generator import PatchGenerator
    from harness.prompts.manager import PromptManager
    from harness.prompts.store import PromptStore
    from harness.feedback.channel import FeedbackChannel

    cfg = get_config()

    buf = StepRewardBuffer(redis_client)
    est = AdvantageEstimator(
        gamma=getattr(cfg, "rlvr_gamma", 0.95),
    )
    prompt_store = PromptStore(redis=redis_client)
    prompt_manager = PromptManager(store=prompt_store)
    feedback_ch = FeedbackChannel(redis_client)

    # PatchGenerator requires an LLM — load lazily
    patch_gen = None
    try:
        from harness.llm.factory import build_llm_provider
        llm = build_llm_provider(cfg)
        patch_gen = PatchGenerator(llm=llm)
    except Exception as exc:
        logger.warning("RLVR worker: PatchGenerator unavailable (%s) — patches disabled", exc)

    loop = RLVRLoop(
        reward_buffer=buf,
        estimator=est,
        patch_generator=patch_gen,
        prompt_store=prompt_manager,
        feedback_channel=feedback_ch,
        min_steps=getattr(cfg, "rlvr_min_steps", 3),
        pos_threshold=getattr(cfg, "rlvr_pos_threshold", 0.5),
        neg_threshold=getattr(cfg, "rlvr_neg_threshold", -0.5),
    )
    return loop, buf


# ---------------------------------------------------------------------------
# AgentRunner hook — publish completion event
# ---------------------------------------------------------------------------

async def publish_run_completion(
    redis: Any,
    run_id: str,
    agent_type: str,
    status: str = "completed",
) -> None:
    """
    Called by AgentRunner when a run finishes.
    Publishes a completion event that the RLVR worker picks up.
    """
    completion_key = "harness:rlvr:completions"
    try:
        await redis.xadd(
            completion_key,
            {"run_id": run_id, "agent_type": agent_type, "status": status},
        )
        await redis.expire(completion_key, 86_400 * 7)
    except Exception as exc:
        logger.debug("publish_run_completion failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v else ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="HarnessAgent RLVR worker")
    parser.add_argument("--poll", type=float, default=0,
                        help="Poll interval in seconds. 0 = event-driven (default).")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                        help="Redis URL.")
    args = parser.parse_args()

    if args.poll > 0:
        asyncio.run(run_poll_mode(args.redis_url, interval=args.poll))
    else:
        asyncio.run(run_event_driven(args.redis_url))


if __name__ == "__main__":
    main()
