"""High-level inter-agent communication patterns built on AgentMessageBus."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from harness.core.errors import InterAgentTimeout
from harness.messaging.bus import AgentMessageBus
from harness.messaging.schema import AgentMessage

logger = logging.getLogger(__name__)


class MessagePatterns:
    """
    High-level messaging patterns:
    - pipeline    : linear chain of agents
    - scatter_gather: fan-out + selective collect
    - heartbeat_monitor: periodic liveness checking
    """

    def __init__(self, bus: AgentMessageBus) -> None:
        self._bus = bus

    async def pipeline(
        self,
        agents: list[str],
        initial_payload: dict[str, Any],
        timeout: float = 60.0,
        sender_id: str = "orchestrator",
    ) -> AgentMessage:
        """
        Chain agents in sequence: output of agent[i] becomes input of agent[i+1].

        Returns the final AgentMessage from the last agent.
        Raises InterAgentTimeout if any agent in the chain doesn't respond.
        """
        if not agents:
            raise ValueError("agents list must not be empty")

        current_payload = initial_payload

        for i, agent_id in enumerate(agents):
            logger.debug(
                "Pipeline step %d/%d: sending to %s", i + 1, len(agents), agent_id
            )
            try:
                reply = await self._bus.request(
                    sender_id=sender_id,
                    recipient_id=agent_id,
                    payload=current_payload,
                    timeout=timeout,
                )
                if reply.message_type == "error":
                    logger.warning(
                        "Pipeline error at step %d (agent=%s): %s",
                        i + 1,
                        agent_id,
                        reply.payload,
                    )
                    return reply
                current_payload = reply.payload
            except InterAgentTimeout:
                logger.error(
                    "Pipeline timeout at step %d (agent=%s)", i + 1, agent_id
                )
                raise

        # Build synthetic final reply wrapping last payload
        return AgentMessage(
            sender_id=agents[-1],
            recipient_id=sender_id,
            message_type="result",
            payload=current_payload,
        )

    async def scatter_gather(
        self,
        sender_id: str,
        recipient_ids: list[str],
        payload: dict[str, Any],
        timeout: float = 60.0,
        min_responses: int = 1,
    ) -> list[AgentMessage]:
        """
        Fan-out ``payload`` to all recipients and gather replies.

        Returns as soon as ``min_responses`` have been received OR timeout elapses.
        Partial results (fewer than len(recipient_ids)) are acceptable.
        """
        import uuid as _uuid

        correlation_id = _uuid.uuid4().hex
        pending: set[str] = set(recipient_ids)
        replies: list[AgentMessage] = []

        # Snapshot the reply-stream position BEFORE sending so replies that
        # arrive before the subscriber's first XREAD are not lost.
        start_ids = await self._bus.snapshot_stream_ids(sender_id)

        # Send to all in parallel
        send_tasks = [
            self._bus.send(
                AgentMessage(
                    sender_id=sender_id,
                    recipient_id=rid,
                    message_type="task",
                    payload=payload,
                    correlation_id=correlation_id,
                )
            )
            for rid in recipient_ids
        ]
        await asyncio.gather(*send_tasks, return_exceptions=True)

        async def _collect() -> None:
            async for reply in self._bus.subscribe(
                sender_id, message_types=["result", "error"], last_ids=start_ids
            ):
                if (
                    reply.correlation_id == correlation_id
                    and reply.sender_id in pending
                ):
                    replies.append(reply)
                    pending.discard(reply.sender_id)

                    if len(replies) >= min_responses:
                        return
                    if not pending:
                        return

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # partial results are acceptable

        if len(replies) < min_responses:
            logger.warning(
                "scatter_gather timeout: got %d/%d responses",
                len(replies),
                len(recipient_ids),
            )

        return replies

    async def heartbeat_monitor(
        self,
        agent_ids: list[str],
        interval: float = 30.0,
        sender_id: str = "monitor",
    ) -> AsyncIterator[dict[str, str]]:
        """
        Async generator that periodically pings each agent and yields a
        health-status dict: {agent_id: "alive" | "dead" | "unknown"}.

        Uses a fire-and-forget heartbeat pattern: sends a ``heartbeat`` message
        and waits up to ``interval/2`` seconds for a reply.
        """
        while True:
            status: dict[str, str] = {aid: "unknown" for aid in agent_ids}

            check_tasks = [
                self._ping_agent(sender_id, agent_id, interval / 2)
                for agent_id in agent_ids
            ]
            results = await asyncio.gather(*check_tasks, return_exceptions=True)

            for agent_id, result in zip(agent_ids, results):
                if isinstance(result, Exception):
                    status[agent_id] = "dead"
                elif result:
                    status[agent_id] = "alive"
                else:
                    status[agent_id] = "dead"

            yield status
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ping_agent(
        self, sender_id: str, agent_id: str, timeout: float
    ) -> bool:
        """Send a heartbeat and return True if the agent replies in time."""
        try:
            reply = await self._bus.request(
                sender_id=sender_id,
                recipient_id=agent_id,
                payload={"type": "ping"},
                timeout=timeout,
                message_types=["result", "error", "status", "heartbeat"],
            )
            return reply.message_type in ("result", "status", "heartbeat")
        except InterAgentTimeout:
            return False
        except Exception as exc:
            logger.debug("Heartbeat ping to %s failed: %s", agent_id, exc)
            return False


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


async def pipeline(
    bus: AgentMessageBus,
    agents: list[str],
    initial_payload: dict[str, Any],
    timeout: float = 60.0,
) -> AgentMessage:
    """Chain agents sequentially using the provided bus."""
    patterns = MessagePatterns(bus)
    return await patterns.pipeline(
        agents=agents,
        initial_payload=initial_payload,
        timeout=timeout,
    )


async def scatter_gather(
    bus: AgentMessageBus,
    sender_id: str,
    recipient_ids: list[str],
    payload: dict[str, Any],
    timeout: float = 60.0,
    min_responses: int = 1,
) -> list[AgentMessage]:
    """Fan-out to all recipients and collect at least ``min_responses`` replies."""
    patterns = MessagePatterns(bus)
    return await patterns.scatter_gather(
        sender_id=sender_id,
        recipient_ids=recipient_ids,
        payload=payload,
        timeout=timeout,
        min_responses=min_responses,
    )


async def heartbeat_monitor(
    bus: AgentMessageBus,
    agent_ids: list[str],
    interval: float = 30.0,
) -> AsyncIterator[dict[str, str]]:
    """Yield periodic health status dicts for the given agent IDs."""
    patterns = MessagePatterns(bus)
    async for status in patterns.heartbeat_monitor(
        agent_ids=agent_ids,
        interval=interval,
    ):
        yield status
