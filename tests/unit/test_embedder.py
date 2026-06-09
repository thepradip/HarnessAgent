"""Unit tests for SentenceTransformerEmbedder (mocked model — no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from harness.memory.embedder import SentenceTransformerEmbedder, estimate_tokens

_DIMS = 3


class _FakeSentenceTransformer:
    """Mimics sentence_transformers.SentenceTransformer.encode() semantics.

    Per the library API:
    - ``convert_to_numpy=True``  → 2D numpy ndarray
    - ``convert_to_numpy=False`` → plain Python list of tensors
      (the list itself has NO ``.tolist()`` — the original bug).
    """

    def __init__(self) -> None:
        self.encode_calls = 0
        self.last_kwargs: dict = {}

    def encode(self, texts, convert_to_numpy=False, show_progress_bar=False):
        self.encode_calls += 1
        self.last_kwargs = {
            "convert_to_numpy": convert_to_numpy,
            "show_progress_bar": show_progress_bar,
        }
        vectors = np.array(
            [[float(len(t)), 1.0, 2.0] for t in texts], dtype=np.float32
        )
        if convert_to_numpy:
            return vectors
        return list(vectors)  # list of tensor-like rows — no .tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return _DIMS


@pytest.fixture
def embedder():
    emb = SentenceTransformerEmbedder(model_name="fake-model")
    emb._model = _FakeSentenceTransformer()  # skip lazy load (no torch in CI)
    return emb


@pytest.mark.asyncio
async def test_embed_returns_plain_float_lists(embedder):
    """Regression: convert_to_numpy=False returned a list with no .tolist()
    → AttributeError on every uncached call."""
    result = await embedder.embed(["ab", "abcd"])
    assert len(result) == 2
    for vec in result:
        assert isinstance(vec, list)
        assert len(vec) == _DIMS
        assert all(isinstance(v, float) for v in vec)
    assert result[0][0] == 2.0  # len("ab")
    assert result[1][0] == 4.0  # len("abcd")


@pytest.mark.asyncio
async def test_embed_requests_numpy_output(embedder):
    await embedder.embed(["hello"])
    assert embedder._model.last_kwargs["convert_to_numpy"] is True
    assert embedder._model.last_kwargs["show_progress_bar"] is False


@pytest.mark.asyncio
async def test_embed_uses_lru_cache_for_repeated_texts(embedder):
    first = await embedder.embed(["same text"])
    second = await embedder.embed(["same text"])
    assert first == second
    assert embedder._model.encode_calls == 1  # second call served from cache


@pytest.mark.asyncio
async def test_embed_mixed_cached_and_uncached_preserves_order(embedder):
    await embedder.embed(["aa"])  # prime the cache
    result = await embedder.embed(["bbbb", "aa", "cccccc"])
    assert [vec[0] for vec in result] == [4.0, 2.0, 6.0]
    assert embedder._model.encode_calls == 2  # only uncached texts re-encoded


@pytest.mark.asyncio
async def test_dimensions_after_model_loaded(embedder):
    await embedder.embed(["probe"])
    assert embedder.dimensions == _DIMS


def test_dimensions_raises_before_load():
    emb = SentenceTransformerEmbedder(model_name="fake-model")
    with pytest.raises(RuntimeError):
        _ = emb.dimensions


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("one two three") == int(3 * 1.3)
