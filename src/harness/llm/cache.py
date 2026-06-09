"""Semantic LLM response cache using Redis and dense vector embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_CACHE_NS = "harness:llm_cache"
_SCAN_CAP = 200  # max entries for vector scan; oldest beyond this are skipped


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _messages_hash(messages: list[dict[str, Any]], tenant_id: str) -> str:
    """Return a stable SHA-256 hex digest of the messages + tenant."""
    payload = json.dumps({"tenant": tenant_id, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _hget(raw: dict[Any, Any], field: str, default: str = "") -> str:
    """Read a hash field that may be keyed/valued as bytes or str.

    Redis clients differ: ``decode_responses=True`` yields str keys/values,
    otherwise bytes. Look the field up under both forms and normalise the value
    to ``str`` so the cache hits in both client modes.
    """
    val = raw.get(field)
    if val is None:
        val = raw.get(field.encode())
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def _is_single_turn(messages: list[dict[str, Any]]) -> bool:
    """True if the request has no prior assistant turns.

    Semantic (fuzzy) matching is only safe for single-turn requests: two
    unrelated multi-turn conversations whose final user message is identical
    (e.g. "yes") must not cross-hit, since their meaning depends on the prior
    turns the embedding ignores.
    """
    return not any(m.get("role") == "assistant" for m in messages)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
    return ""


class SemanticLLMCache:
    """Cache LLM responses using semantic similarity matching via Redis hashes."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        embedding_provider: Any,  # EmbeddingProvider protocol
        tenant_id: str = "default",
        namespace: str = _CACHE_NS,
    ) -> None:
        self._redis = redis_client
        self._embedder = embedding_provider
        self._tenant_id = tenant_id
        self._namespace = namespace

    def _key(self, msg_hash: str) -> str:
        """Build the Redis hash key for a cached entry."""
        return f"{self._namespace}:{self._tenant_id}:{msg_hash}"

    def _index_key(self) -> str:
        """Build the Redis sorted-set key that indexes all cache entries."""
        return f"{self._namespace}:{self._tenant_id}:index"

    async def get(
        self,
        messages: list[dict[str, Any]],
        threshold: float = 0.97,
    ) -> str | None:
        """Return a cached response whose embedding is above threshold, or None."""
        query_text = _last_user_text(messages)
        if not query_text:
            return None

        # Fast path: exact SHA-256 match — no embedding needed. Safe for any
        # number of turns because the hash covers the full message list.
        msg_hash = _messages_hash(messages, self._tenant_id)
        exact_key = self._key(msg_hash)
        try:
            raw_exact = await self._redis.hgetall(exact_key)
            if raw_exact:
                response = _hget(raw_exact, "response")
                if response:
                    logger.debug("Cache exact HIT tenant=%s", self._tenant_id)
                    return response
        except Exception as exc:
            logger.debug("Cache exact lookup failed: %s", exc)

        # Slow (semantic) path is only safe for single-turn requests: with prior
        # assistant turns the final user message ("yes") is ambiguous and would
        # cross-hit unrelated conversations. Skip the vector scan otherwise.
        if not _is_single_turn(messages):
            logger.debug(
                "Cache: skipping semantic scan for multi-turn request tenant=%s",
                self._tenant_id,
            )
            return None

        # Slow path: vector scan — cap at _SCAN_CAP most-recent entries
        try:
            query_embedding = (await self._embedder.embed([query_text]))[0]
        except Exception as exc:
            logger.warning("Cache get: embedding failed: %s", exc)
            return None

        index_key = self._index_key()
        members: list[bytes] = await self._redis.zrevrange(index_key, 0, _SCAN_CAP - 1)
        if not members:
            return None

        best_score = -1.0
        best_response: str | None = None

        for member in members:
            entry_key = member.decode() if isinstance(member, bytes) else member
            raw = await self._redis.hgetall(entry_key)
            if not raw:
                continue
            try:
                stored_embedding: list[float] = json.loads(
                    _hget(raw, "embedding", "[]")
                )
                response: str = _hget(raw, "response")
                score = _cosine_similarity(query_embedding, stored_embedding)
                if score > best_score:
                    best_score = score
                    best_response = response
            except Exception as exc:
                logger.debug("Cache entry parse error: %s", exc)
                continue

        if best_score >= threshold and best_response is not None:
            logger.debug(
                "Cache HIT (score=%.4f, threshold=%.4f, tenant=%s)",
                best_score,
                threshold,
                self._tenant_id,
            )
            return best_response

        logger.debug(
            "Cache MISS (best_score=%.4f, threshold=%.4f, tenant=%s)",
            best_score,
            threshold,
            self._tenant_id,
        )
        return None

    async def set(
        self,
        messages: list[dict[str, Any]],
        response: str,
        ttl: int = 3600,
    ) -> None:
        """Store the response embedding and text in Redis."""
        query_text = _last_user_text(messages)
        if not query_text:
            return

        try:
            embedding = (await self._embedder.embed([query_text]))[0]
        except Exception as exc:
            logger.warning("Cache set: embedding failed: %s", exc)
            return

        msg_hash = _messages_hash(messages, self._tenant_id)
        entry_key = self._key(msg_hash)
        now = time.time()

        pipe = self._redis.pipeline(transaction=False)
        pipe.hset(
            entry_key,
            mapping={
                "embedding": json.dumps(embedding),
                "response": response,
                "created_at": str(now),
                "query_text": query_text[:500],
            },
        )
        pipe.expire(entry_key, ttl)
        # Track entry in the index sorted set by timestamp
        pipe.zadd(self._index_key(), {entry_key: now})
        pipe.expire(self._index_key(), ttl + 60)
        await pipe.execute()
        logger.debug("Cache SET: tenant=%s hash=%s", self._tenant_id, msg_hash[:12])

    async def invalidate(self, pattern: str = "*") -> int:
        """Delete cache entries whose keys match the given glob pattern."""
        full_pattern = f"{self._namespace}:{self._tenant_id}:{pattern}"
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=full_pattern, count=100)
            if keys:
                pipe = self._redis.pipeline(transaction=False)
                for key in keys:
                    pipe.delete(key)
                    # Remove from index
                    k = key.decode() if isinstance(key, bytes) else key
                    pipe.zrem(self._index_key(), k)
                results = await pipe.execute()
                deleted += sum(1 for r in results if r == 1)
            if cursor == 0:
                break
        logger.info(
            "Cache invalidate: pattern=%s deleted=%d tenant=%s",
            pattern,
            deleted,
            self._tenant_id,
        )
        return deleted
