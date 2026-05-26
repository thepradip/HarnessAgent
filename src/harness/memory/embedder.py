"""SentenceTransformer embedding provider with LRU caching."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_LRU_MAX = 1_000


class _LRUCache:
    """Simple thread-safe LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> list[float] | None:
        async with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    async def set(self, key: str, value: list[float]) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value


def _text_cache_key(text: str) -> str:
    """Deterministic cache key for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: word_count * 1.3, minimum 1."""
    words = len(text.split())
    return max(1, int(words * 1.3))


class SentenceTransformerEmbedder:
    """
    Async-compatible sentence-transformer embedder.

    The underlying SentenceTransformer model is loaded lazily on the first
    ``embed()`` call to avoid blocking the event loop at import time.
    Embeddings are cached in an in-process LRU cache (max 1 000 entries).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None  # SentenceTransformer instance (loaded lazily)
        self._load_lock = asyncio.Lock()
        self._cache = _LRUCache(_LRU_MAX)

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return embedding dimension; raises if model not yet loaded."""
        if self._model is None:
            raise RuntimeError(
                "Embedder model not loaded yet — call embed() first."
            )
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Return dense embeddings for the provided texts.

        Uses the in-process LRU cache; only uncached texts are forwarded to
        the underlying SentenceTransformer model.
        """
        await self._ensure_model_loaded()

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(texts):
            key = _text_cache_key(text)
            cached = await self._cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            loop = asyncio.get_running_loop()
            # Run CPU-bound encoding in thread pool to avoid blocking the event loop
            raw_embeddings: list[list[float]] = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    uncached_texts, convert_to_numpy=False, show_progress_bar=False
                ).tolist(),
            )
            for idx, text, embedding in zip(
                uncached_indices, uncached_texts, raw_embeddings
            ):
                key = _text_cache_key(text)
                await self._cache.set(key, embedding)
                results[idx] = embedding

        # At this point all slots must be filled
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_model_loaded(self) -> None:
        """Lazily load the SentenceTransformer model exactly once."""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            logger.info("Loading SentenceTransformer model: %s", self._model_name)
            loop = asyncio.get_running_loop()
            try:
                from sentence_transformers import SentenceTransformer

                self._model = await loop.run_in_executor(
                    None, lambda: SentenceTransformer(self._model_name)
                )
                logger.info(
                    "SentenceTransformer model loaded — dimensions=%d",
                    self._model.get_sentence_embedding_dimension(),
                )
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers (PyTorch ~1.5 GB) is required for this embedder.\n"
                    "Install: pip install agent-haas[embed-full]\n"
                    "For a lightweight alternative (~100 MB, no PyTorch): "
                    "use FastEmbedEmbedder — pip install agent-haas[vector]"
                ) from exc


class FastEmbedEmbedder:
    """Lightweight ONNX embedder using fastembed — no PyTorch required.

    Install size: ~100 MB vs ~1.5 GB for sentence-transformers + torch.
    Default model: ``BAAI/bge-small-en-v1.5`` (384 dims, quality on par
    with ``all-MiniLM-L6-v2``).

    This is the default embedder when ``pip install agent-haas[vector]``
    is used. Switch to ``SentenceTransformerEmbedder`` only when you need
    a specific model that fastembed does not support.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dims: int = 384          # default for bge-small; updated after load
        self._load_lock = asyncio.Lock()
        self._cache = _LRUCache(_LRU_MAX)

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return ONNX embeddings for ``texts`` with LRU caching."""
        await self._ensure_loaded()

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(texts):
            key = _text_cache_key(text)
            cached = await self._cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            loop = asyncio.get_running_loop()
            raw: list[list[float]] = await loop.run_in_executor(
                None,
                lambda: [emb.tolist() for emb in self._model.embed(uncached_texts)],
            )
            for idx, text, embedding in zip(uncached_indices, uncached_texts, raw):
                key = _text_cache_key(text)
                await self._cache.set(key, embedding)
                results[idx] = embedding

        return [r for r in results if r is not None]

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            logger.info("Loading FastEmbed model: %s", self._model_name)
            loop = asyncio.get_running_loop()
            try:
                from fastembed import TextEmbedding  # type: ignore[import]

                self._model = await loop.run_in_executor(
                    None, lambda: TextEmbedding(self._model_name)
                )
                # Probe dimensions on a dummy input
                probe = list(self._model.embed(["probe"]))
                if probe:
                    self._dims = len(probe[0])
                logger.info("FastEmbed model loaded — dimensions=%d", self._dims)
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed package is required for FastEmbedEmbedder.\n"
                    "Install: pip install agent-haas[vector]\n"
                    "For PyTorch-based embeddings: pip install agent-haas[embed-full]"
                ) from exc
