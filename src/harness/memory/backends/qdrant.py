"""Qdrant vector store backend."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from harness.core.errors import FailureClass, HarnessError
from harness.core.protocols import VectorHit

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "harness_memories"


class QdrantVectorStore:
    """
    VectorStore implementation backed by Qdrant.

    One collection (``harness_memories``) is used for all tenants; tenant_id
    is stored in the point payload for filtering.  The collection is created
    automatically with COSINE distance if it does not exist.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        embedder: Any = None,
        vector_size: int = 384,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        self._url = url
        self._embedder = embedder
        self._vector_size = vector_size
        self._collection_name = collection_name
        self._client: Any = None
        self._collection_ready = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client package is required. Install with: pip install qdrant-client"
                ) from exc
            try:
                self._client = AsyncQdrantClient(url=self._url)
            except Exception as exc:
                raise HarnessError(
                    f"Qdrant client creation failed: {exc}",
                    failure_class=FailureClass.MEMORY_VECTOR,
                ) from exc
        return self._client

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        client = await self._get_client()
        try:
            from qdrant_client.models import (  # type: ignore[import]
                Distance,
                VectorParams,
            )

            existing = await client.get_collections()
            names = [c.name for c in existing.collections]
            if self._collection_name not in names:
                vector_size = await self._embedding_dimensions()
                await client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' with vector_size=%d",
                    self._collection_name,
                    vector_size,
                )
            self._collection_ready = True
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Qdrant collection init failed: %s", exc)
            raise HarnessError(
                f"Qdrant collection error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            raise HarnessError(
                "QdrantVectorStore requires an embedder for vector generation.",
                failure_class=FailureClass.MEMORY_VECTOR,
            )
        embeddings = await self._embedder.embed([text])
        return embeddings[0]

    async def _embedding_dimensions(self) -> int:
        """Resolve the embedding dimension when creating the collection.

        Lazy embedders raise on ``.dimensions`` until they have embedded at
        least once, so a query/upsert that has to create the collection before
        any embed call would crash. Probe with a real embed first, then read
        ``.dimensions``; fall back to the measured vector length or the
        configured default.
        """
        if self._embedder is None:
            return self._vector_size
        try:
            probe = await self._embedder.embed(["probe"])
            dim = getattr(self._embedder, "dimensions", None)
            if dim:
                return int(dim)
            if probe and probe[0]:
                return len(probe[0])
        except Exception as exc:
            logger.debug("Embedder dimension probe failed: %s", exc)
        return self._vector_size

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def upsert(
        self,
        id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        """Insert or update a point in Qdrant."""
        await self._ensure_collection()
        client = await self._get_client()

        vector = embedding if embedding is not None else await self._embed(text)

        try:
            from qdrant_client.models import PointStruct  # type: ignore[import]

            payload: dict[str, Any] = {
                "text": text,
                "metadata": metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            point = PointStruct(id=_str_to_uuid(id), vector=vector, payload=payload)
            await client.upsert(
                collection_name=self._collection_name,
                points=[point],
            )
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Qdrant upsert failed: %s", exc)
            raise HarnessError(
                f"Qdrant upsert error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def query(
        self,
        text: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        hybrid_alpha: float | None = None,
    ) -> list[VectorHit]:
        """Return top-k nearest neighbours with optional metadata filtering."""
        await self._ensure_collection()
        client = await self._get_client()
        vector = await self._embed(text)

        try:
            qdrant_filter = _build_qdrant_filter(filter) if filter else None

            if hybrid_alpha is not None:
                # Qdrant 1.7+ supports sparse+dense hybrid; use sparse vectors if available.
                # Fall back to dense-only if sparse vectors not configured.
                results = await client.search(
                    collection_name=self._collection_name,
                    query_vector=vector,
                    query_filter=qdrant_filter,
                    limit=k,
                    with_payload=True,
                )
            else:
                results = await client.search(
                    collection_name=self._collection_name,
                    query_vector=vector,
                    query_filter=qdrant_filter,
                    limit=k,
                    with_payload=True,
                )
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Qdrant query failed: %s", exc)
            raise HarnessError(
                f"Qdrant query error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

        hits: list[VectorHit] = []
        for r in results:
            payload = r.payload or {}
            meta = payload.get("metadata", {})
            hits.append(
                VectorHit(
                    id=str(r.id),
                    text=payload.get("text", ""),
                    score=float(r.score),
                    metadata=meta,
                )
            )
        return hits

    async def delete(self, id: str) -> None:
        """Remove a point by ID."""
        await self._ensure_collection()
        client = await self._get_client()
        try:
            from qdrant_client.models import PointIdsList  # type: ignore[import]

            await client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=[_str_to_uuid(id)]),
            )
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Qdrant delete failed: %s", exc)
            raise HarnessError(
                f"Qdrant delete error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Return document count, optionally filtered."""
        await self._ensure_collection()
        client = await self._get_client()
        try:
            qdrant_filter = _build_qdrant_filter(filter) if filter else None
            result = await client.count(
                collection_name=self._collection_name,
                count_filter=qdrant_filter,
                exact=True,
            )
            return result.count
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Qdrant count failed: %s", exc)
            raise HarnessError(
                f"Qdrant count error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def close(self) -> None:
        """Close the underlying Qdrant async client."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.debug("Qdrant client close failed: %s", exc)
            self._client = None
            self._collection_ready = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_to_uuid(id_str: str) -> str:
    """Convert an arbitrary string ID to a UUID-compatible string for Qdrant."""
    try:
        # If it already IS a UUID, use it directly.
        uuid.UUID(id_str)
        return id_str
    except ValueError:
        # Otherwise, derive a deterministic UUID from the string.
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


def _build_qdrant_filter(filter: dict[str, Any]) -> Any:
    """Convert a flat {field: value} dict to a Qdrant Filter object."""
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import]

        conditions = []
        for field_name, value in filter.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{field_name}",
                    match=MatchValue(value=value),
                )
            )
        return Filter(must=conditions)
    except ImportError:
        return None
