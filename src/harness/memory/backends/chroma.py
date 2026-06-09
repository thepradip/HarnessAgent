"""ChromaDB vector store backend."""

from __future__ import annotations

import logging
from typing import Any

from harness.core.errors import FailureClass, HarnessError
from harness.core.protocols import VectorHit

logger = logging.getLogger(__name__)

_COLLECTION_PREFIX = "harness"


class ChromaVectorStore:
    """
    VectorStore implementation backed by ChromaDB.

    - path == ":memory:" → EphemeralClient (in-process, no persistence).
    - Any other path    → PersistentClient at that directory.

    Collections are named "harness_{tenant_id}" when a tenant_id is provided,
    otherwise a single "harness" collection is used with tenant_id stored in
    the document metadata.
    """

    def __init__(
        self,
        path: str = ":memory:",
        tenant_id: str | None = None,
        embedder: Any = None,
    ) -> None:
        self._path = path
        self._tenant_id = tenant_id
        self._embedder = embedder
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "chromadb package is required. Install with: pip install chromadb"
            ) from exc

        try:
            if self._path == ":memory:":
                self._client = chromadb.EphemeralClient()
            else:
                self._client = chromadb.PersistentClient(path=self._path)
        except Exception as exc:
            logger.error("Failed to create ChromaDB client: %s", exc)
            raise HarnessError(
                f"ChromaDB client creation failed: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def _ensure_collection(self) -> Any:
        await self._ensure_client()
        if self._collection is not None:
            return self._collection

        collection_name = (
            f"{_COLLECTION_PREFIX}_{self._tenant_id}"
            if self._tenant_id
            else _COLLECTION_PREFIX
        )
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.error("Failed to get/create ChromaDB collection: %s", exc)
            raise HarnessError(
                f"ChromaDB collection error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc
        return self._collection

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
        """Insert or update a document.

        If embedding is not provided and an embedder is configured, the text
        is embedded automatically. Otherwise ChromaDB handles embedding via its
        default embedding function.
        """
        collection = await self._ensure_collection()
        safe_metadata = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in metadata.items()
        }
        if self._tenant_id:
            safe_metadata.setdefault("tenant_id", self._tenant_id)

        try:
            kwargs: dict[str, Any] = {
                "ids": [id],
                "documents": [text],
                "metadatas": [safe_metadata],
            }
            if embedding is not None:
                kwargs["embeddings"] = [embedding]
            elif self._embedder is not None:
                emb_list = await self._embedder.embed([text])
                kwargs["embeddings"] = emb_list

            collection.upsert(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB upsert failed: %s", exc)
            raise HarnessError(
                f"ChromaDB upsert error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def query(
        self,
        text: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        hybrid_alpha: float | None = None,  # noqa: ARG002 — ChromaDB does not support hybrid
    ) -> list[VectorHit]:
        """Return top-k nearest neighbours.

        ``hybrid_alpha`` is accepted for protocol compatibility but ignored —
        ChromaDB does not support sparse/dense hybrid search.
        """
        collection = await self._ensure_collection()

        where: dict[str, Any] | None = None
        if filter:
            where = self._build_where(filter)

        try:
            query_kwargs: dict[str, Any] = {
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }
            if self._embedder is not None:
                emb_list = await self._embedder.embed([text])
                query_kwargs["query_embeddings"] = emb_list
            else:
                query_kwargs["query_texts"] = [text]

            if where:
                query_kwargs["where"] = where

            results = collection.query(**query_kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            raise HarnessError(
                f"ChromaDB query error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

        hits: list[VectorHit] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma cosine distance → similarity score: 1 - distance
            score = float(1.0 - dist)
            hits.append(
                VectorHit(
                    id=doc_id,
                    text=doc or "",
                    score=score,
                    metadata=meta or {},
                )
            )
        return hits

    async def delete(self, id: str) -> None:
        """Remove a document by ID."""
        collection = await self._ensure_collection()
        try:
            collection.delete(ids=[id])
        except Exception as exc:
            logger.error("ChromaDB delete failed: %s", exc)
            raise HarnessError(
                f"ChromaDB delete error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Return document count, optionally filtered."""
        collection = await self._ensure_collection()
        try:
            if filter:
                where = self._build_where(filter)
                results = collection.get(where=where, include=[])
                return len(results.get("ids", []))
            return collection.count()
        except Exception as exc:
            logger.error("ChromaDB count failed: %s", exc)
            raise HarnessError(
                f"ChromaDB count error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def close(self) -> None:
        """Release the ChromaDB client.

        Ephemeral/persistent Chroma clients hold no network sockets, but we
        drop references so a fresh client is created on next use and so callers
        can treat all backends uniformly.
        """
        self._collection = None
        self._client = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_where(self, filter: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a flat {field: value} dict to Chroma ``where`` syntax.

        Multiple conditions are combined with ``$and``.
        """
        if len(filter) == 1:
            field, value = next(iter(filter.items()))
            return {field: {"$eq": value}}

        conditions = [{field: {"$eq": value}} for field, value in filter.items()]
        return {"$and": conditions}
