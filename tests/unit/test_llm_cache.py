"""Regression tests for SemanticLLMCache.

Covers two verified correctness bugs:
- Cross-conversation false hits: two unrelated multi-turn conversations whose
  final user message is identical (e.g. "yes") must NOT cross-hit via the
  semantic (embedding) path.
- Redis key-mode mismatch: the cache must hit with both a bytes client (default)
  and a decode_responses=True client.
"""

from __future__ import annotations

import pytest
import fakeredis.aioredis as fakeredis

from harness.llm.cache import SemanticLLMCache


class _ConstEmbedder:
    """Embedding provider that maps every text to the same vector.

    Forces a perfect semantic match (cosine = 1.0) so the only thing that can
    prevent a hit is the cache's own single-turn / multi-turn guard.
    """

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_multi_turn_does_not_cross_hit_semantically():
    redis = fakeredis.FakeRedis()
    cache = SemanticLLMCache(redis, _ConstEmbedder(), tenant_id="t1")

    # Conversation A: store a response keyed on its full multi-turn history.
    conv_a = [
        {"role": "user", "content": "Should I deploy on Friday?"},
        {"role": "assistant", "content": "That is risky."},
        {"role": "user", "content": "yes"},
    ]
    await cache.set(conv_a, "Deploying on Friday — proceeding.")

    # Conversation B: unrelated multi-turn ending in the same final "yes".
    conv_b = [
        {"role": "user", "content": "Should I delete the production database?"},
        {"role": "assistant", "content": "That would be destructive."},
        {"role": "user", "content": "yes"},
    ]
    # Despite identical embeddings, the semantic path must be skipped for
    # multi-turn requests — no cross-hit.
    assert await cache.get(conv_b) is None


@pytest.mark.asyncio
async def test_single_turn_semantic_hit_still_works():
    redis = fakeredis.FakeRedis()
    cache = SemanticLLMCache(redis, _ConstEmbedder(), tenant_id="t1")

    await cache.set([{"role": "user", "content": "What is 2+2?"}], "4")
    # A different single-turn query with identical embedding should hit.
    hit = await cache.get([{"role": "user", "content": "Compute two plus two"}])
    assert hit == "4"


@pytest.mark.asyncio
async def test_exact_hit_works_for_multi_turn():
    """The exact (full-history hash) path is still safe for multi-turn."""
    redis = fakeredis.FakeRedis()
    cache = SemanticLLMCache(redis, _ConstEmbedder(), tenant_id="t1")
    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    await cache.set(conv, "goodbye")
    assert await cache.get(conv) == "goodbye"


@pytest.mark.asyncio
async def test_cache_hits_with_decode_responses_client():
    """With decode_responses=True the hash fields are str, not bytes — the
    cache must still hit (previously embedding defaulted to [] and never hit)."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    cache = SemanticLLMCache(redis, _ConstEmbedder(), tenant_id="t1")

    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    await cache.set(msgs, "Paris")

    # Exact hit
    assert await cache.get(msgs) == "Paris"
    # Semantic hit (different single-turn text, identical embedding)
    assert await cache.get([{"role": "user", "content": "France capital city?"}]) == "Paris"
