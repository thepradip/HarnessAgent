"""PromptStore — versioned prompt persistence backed by Redis."""

from __future__ import annotations

import json
import logging
from typing import Optional

from harness.prompts.schemas import PromptVersion

logger = logging.getLogger(__name__)

_KEY_PREFIX = "harness:prompt"
_INDEX_PREFIX = "harness:prompt_index"
_ACTIVE_PREFIX = "harness:prompt_active"
_COUNTER_PREFIX = "harness:prompt_counter"


def _version_key(agent_type: str, version_id: str) -> str:
    return f"{_KEY_PREFIX}:{agent_type}:{version_id}"


def _index_key(agent_type: str) -> str:
    return f"{_INDEX_PREFIX}:{agent_type}"


def _active_key(agent_type: str) -> str:
    return f"{_ACTIVE_PREFIX}:{agent_type}"


def _counter_key(agent_type: str) -> str:
    return f"{_COUNTER_PREFIX}:{agent_type}"


class PromptStore:
    """Versioned prompt storage in Redis.

    Each prompt version is stored as a JSON hash at:
        harness:prompt:{agent_type}:{version_id}

    An ordered index per agent_type is maintained as a sorted set at:
        harness:prompt_index:{agent_type}
    scored by version_number, so ZREVRANGE retrieves newest first.

    The active version ID per agent_type is stored as a plain string at:
        harness:prompt_active:{agent_type}

    Args:
        redis: An aioredis-compatible async Redis client.
    """

    def __init__(self, redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save(self, version: PromptVersion) -> PromptVersion:
        """Persist a PromptVersion to Redis.

        Stores the JSON blob and updates the sorted index for the agent_type.

        Args:
            version: The PromptVersion to save.

        Returns:
            The saved PromptVersion (possibly with updated version_id).
        """
        key = _version_key(version.agent_type, version.version_id)
        index_key = _index_key(version.agent_type)

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(key, version.to_json())
            pipe.zadd(index_key, {version.version_id: float(version.version_number)})
            if version.active:
                pipe.set(_active_key(version.agent_type), version.version_id)
            await pipe.execute()

        logger.debug(
            "Saved prompt version %s for agent_type=%s (v%d, active=%s)",
            version.version_id[:8],
            version.agent_type,
            version.version_number,
            version.active,
        )
        return version

    async def create_version(
        self,
        agent_type: str,
        content: str,
        **kwargs,
    ) -> PromptVersion:
        """Create a new PromptVersion with auto-incremented version_number.

        Args:
            agent_type: The agent type the prompt belongs to.
            content:    The prompt text.
            **kwargs:   Extra keyword arguments forwarded to PromptVersion().

        Returns:
            The newly created PromptVersion (saved to Redis).
        """
        # Determine next version number via a monotonic per-agent counter.
        # zcard+1 collides after delete() (count shrinks) and under concurrent
        # creates (two callers read the same count), corrupting the
        # version_number-scored index that rollback() depends on. INCR is atomic
        # and never reuses a number.
        next_version = int(await self._redis.incr(_counter_key(agent_type)))

        version = PromptVersion(
            agent_type=agent_type,
            content=content,
            version_number=next_version,
            **kwargs,
        )
        await self.save(version)
        return version

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get(self, version_id: str, agent_type: str = "") -> Optional[PromptVersion]:
        """Retrieve a PromptVersion by its version_id.

        If agent_type is not provided, scans known agent types.

        Args:
            version_id: The UUID of the version to retrieve.
            agent_type: Optional hint to narrow the key space.

        Returns:
            PromptVersion if found, else None.
        """
        if agent_type:
            raw = await self._redis.get(_version_key(agent_type, version_id))
            if raw:
                return PromptVersion.from_json(raw if isinstance(raw, str) else raw.decode())
            return None

        # Scan for the version across all known agent types
        pattern = f"{_KEY_PREFIX}:*:{version_id}"
        keys = []
        async for key in self._redis.scan_iter(match=pattern, count=100):
            keys.append(key)

        for key in keys:
            raw = await self._redis.get(key)
            if raw:
                return PromptVersion.from_json(raw if isinstance(raw, str) else raw.decode())
        return None

    async def get_active(self, agent_type: str) -> Optional[PromptVersion]:
        """Return the currently active PromptVersion for agent_type, or None.

        Args:
            agent_type: The agent type to look up.

        Returns:
            The active PromptVersion, or None if none is set.
        """
        active_id_raw = await self._redis.get(_active_key(agent_type))
        if not active_id_raw:
            return None
        active_id = active_id_raw if isinstance(active_id_raw, str) else active_id_raw.decode()
        return await self.get(version_id=active_id, agent_type=agent_type)

    async def list_versions(
        self, agent_type: str, limit: int = 20
    ) -> list[PromptVersion]:
        """Return up to *limit* versions for *agent_type*, newest first.

        Args:
            agent_type: The agent type to list versions for.
            limit:      Maximum number of versions to return.

        Returns:
            List of PromptVersion objects ordered by version_number descending.
        """
        index_key = _index_key(agent_type)
        # ZREVRANGE returns members with highest scores first (highest version_number)
        version_ids: list = await self._redis.zrevrange(
            index_key, 0, limit - 1
        )

        versions: list[PromptVersion] = []
        for vid_raw in version_ids:
            vid = vid_raw if isinstance(vid_raw, str) else vid_raw.decode()
            v = await self.get(version_id=vid, agent_type=agent_type)
            if v is not None:
                versions.append(v)

        return versions

    # ------------------------------------------------------------------
    # Promotion and rollback
    # ------------------------------------------------------------------

    async def promote(self, version_id: str) -> PromptVersion:
        """Make the specified version the active one for its agent_type.

        Deactivates all other versions for the same agent_type.

        Args:
            version_id: The version to promote.

        Returns:
            The promoted PromptVersion.

        Raises:
            KeyError: If version_id is not found.
        """
        # Fetch the target version (scan since we don't always know agent_type)
        target = await self.get(version_id=version_id)
        if target is None:
            raise KeyError(f"PromptVersion not found: {version_id}")

        agent_type = target.agent_type

        # Deactivate all existing versions for this agent_type
        all_versions = await self.list_versions(agent_type, limit=1000)
        async with self._redis.pipeline(transaction=True) as pipe:
            for v in all_versions:
                if v.version_id != version_id and v.active:
                    v.active = False
                    pipe.set(_version_key(agent_type, v.version_id), v.to_json())
            # Activate target
            target.active = True
            pipe.set(_version_key(agent_type, version_id), target.to_json())
            pipe.set(_active_key(agent_type), version_id)
            await pipe.execute()

        logger.info(
            "Promoted prompt version %s (v%d) for agent_type=%s",
            version_id[:8],
            target.version_number,
            agent_type,
        )
        return target

    async def rollback(self, agent_type: str, steps: int = 1) -> PromptVersion:
        """Roll back to the version N steps before the current active version.

        Args:
            agent_type: The agent type to roll back.
            steps:      Number of versions to step back (default 1).

        Returns:
            The newly promoted (rolled-back) PromptVersion.

        Raises:
            ValueError: If there are not enough versions to roll back.
        """
        versions = await self.list_versions(agent_type, limit=1000)
        if len(versions) < steps + 1:
            raise ValueError(
                f"Cannot roll back {steps} step(s): only {len(versions)} version(s) exist "
                f"for agent_type={agent_type!r}"
            )

        # Find the current active version index
        active_idx = next(
            (i for i, v in enumerate(versions) if v.active), 0
        )
        target_idx = active_idx + steps
        if target_idx >= len(versions):
            raise ValueError(
                f"Rollback by {steps} step(s) would go past the oldest version."
            )

        target = versions[target_idx]
        return await self.promote(target.version_id)

    # ------------------------------------------------------------------
    # Score tracking
    # ------------------------------------------------------------------

    async def update_score(
        self, agent_type: str, version_id: str, score: float
    ) -> None:
        """Update the evaluation score on an existing prompt version.

        Called by the Hermes evaluator and online monitor after measuring
        live performance. Does nothing if the version is not found.

        Args:
            agent_type: The agent type the version belongs to.
            version_id: The version to update.
            score:      New score in [0, 1].
        """
        version = await self.get(version_id=version_id, agent_type=agent_type)
        if version is None:
            logger.debug("update_score: version %s not found for %s", version_id[:8], agent_type)
            return
        version.score = max(0.0, min(1.0, score))
        await self.save(version)
        logger.debug(
            "Updated score for version %s (agent_type=%s): %.3f",
            version_id[:8],
            agent_type,
            score,
        )

    async def get_performance_history(
        self, agent_type: str, limit: int = 20
    ) -> list[dict]:
        """Return version performance history in chronological order.

        Useful for charting score progression across prompt versions.

        Args:
            agent_type: The agent type to query.
            limit:      Maximum number of versions to include.

        Returns:
            List of dicts with version_number, score, active, created_at,
            created_by, patch_id — oldest first.
        """
        versions = await self.list_versions(agent_type, limit=limit)
        # list_versions returns newest-first; reverse for chronological order
        return [
            {
                "version_number": v.version_number,
                "version_id": v.version_id,
                "score": v.score,
                "active": v.active,
                "created_at": v.created_at.isoformat(),
                "created_by": v.created_by,
                "patch_id": v.patch_id,
                "tags": v.tags,
            }
            for v in reversed(versions)
        ]

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete(self, agent_type: str, version_id: str) -> None:
        """Remove a version from storage and the sorted index.

        Args:
            agent_type: The agent type the version belongs to.
            version_id: The version to delete.
        """
        key = _version_key(agent_type, version_id)
        index_key = _index_key(agent_type)
        active_key = _active_key(agent_type)

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.zrem(index_key, version_id)
            # Clear active pointer if this was the active version
            active_raw = await self._redis.get(active_key)
            if active_raw:
                active_id = active_raw if isinstance(active_raw, str) else active_raw.decode()
                if active_id == version_id:
                    pipe.delete(active_key)
            await pipe.execute()
