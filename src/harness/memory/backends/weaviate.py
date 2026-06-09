"""Weaviate vector store backend."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from harness.core.errors import FailureClass, HarnessError
from harness.core.protocols import VectorHit

logger = logging.getLogger(__name__)

_CLASS_NAME = "HarnessMemory"

# Metadata keys that are promoted to first-class, filterable Weaviate properties
# (in addition to being kept inside metadata_json for round-trip fidelity).
# _build_weaviate_filter targets these via Filter.by_property(...).
_FILTERABLE_FIELDS = ("run_id", "tenant_id", "type")

_CLASS_SCHEMA: dict[str, Any] = {
    "class": _CLASS_NAME,
    "description": "Harness long-term memory store",
    "vectorizer": "none",  # we supply our own vectors
    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
            "description": "Raw memory text",
        },
        {
            "name": "metadata_json",
            "dataType": ["text"],
            "description": "JSON-serialised metadata dict",
        },
        {
            "name": "created_at",
            "dataType": ["text"],
            "description": "ISO-8601 creation timestamp",
        },
        # Promoted filterable fields (kept as TEXT for exact-match equals).
        {"name": "run_id", "dataType": ["text"], "description": "run_id (filterable)"},
        {"name": "tenant_id", "dataType": ["text"], "description": "tenant_id (filterable)"},
        {"name": "type", "dataType": ["text"], "description": "memory type (filterable)"},
    ],
}


class WeaviateVectorStore:
    """
    VectorStore implementation backed by Weaviate.

    Connects using ``weaviate.use_async_with_local()`` for self-hosted instances.
    Class schema is auto-created on first use.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        embedder: Any = None,
        api_key: str | None = None,
    ) -> None:
        self._url = url
        self._embedder = embedder
        self._api_key = api_key
        self._client: Any = None
        self._schema_ready = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import weaviate  # type: ignore[import]
                import weaviate.classes as wvc  # noqa: F401 — ensures v4 SDK
            except ImportError as exc:
                raise ImportError(
                    "weaviate-client >=4.x is required. "
                    "Install with: pip install weaviate-client"
                ) from exc

            try:
                if self._api_key:
                    self._client = await weaviate.use_async_with_weaviate_cloud(
                        cluster_url=self._url,
                        auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                    ).__aenter__()
                else:
                    self._client = await weaviate.use_async_with_local(
                        host=self._url.replace("http://", "").replace("https://", "").split(":")[0],
                        port=int(self._url.split(":")[-1]) if ":" in self._url else 8080,
                    ).__aenter__()
            except Exception as exc:
                logger.error("Weaviate connection failed: %s", exc)
                raise HarnessError(
                    f"Weaviate connection error: {exc}",
                    failure_class=FailureClass.MEMORY_VECTOR,
                ) from exc
        return self._client

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        client = await self._get_client()
        try:
            if not await client.collections.exists(_CLASS_NAME):
                import weaviate.classes.config as wcc  # type: ignore[import]

                await client.collections.create(
                    name=_CLASS_NAME,
                    description="Harness long-term memory store",
                    vectorizer_config=wcc.Configure.Vectorizer.none(),
                    properties=[
                        wcc.Property(name="text", data_type=wcc.DataType.TEXT),
                        wcc.Property(name="metadata_json", data_type=wcc.DataType.TEXT),
                        wcc.Property(name="created_at", data_type=wcc.DataType.TEXT),
                        # Promoted filterable fields — see _FILTERABLE_FIELDS.
                        wcc.Property(name="run_id", data_type=wcc.DataType.TEXT),
                        wcc.Property(name="tenant_id", data_type=wcc.DataType.TEXT),
                        wcc.Property(name="type", data_type=wcc.DataType.TEXT),
                    ],
                )
                logger.info("Created Weaviate class '%s'", _CLASS_NAME)
            self._schema_ready = True
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Weaviate schema init failed: %s", exc)
            raise HarnessError(
                f"Weaviate schema error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            raise HarnessError(
                "WeaviateVectorStore requires an embedder.",
                failure_class=FailureClass.MEMORY_VECTOR,
            )
        embeddings = await self._embedder.embed([text])
        return embeddings[0]

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
        """Insert or update a Weaviate object."""
        await self._ensure_schema()
        client = await self._get_client()
        vector = embedding if embedding is not None else await self._embed(text)

        properties = {
            "text": text,
            "metadata_json": json.dumps(metadata, default=str),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Promote filterable fields to top-level properties so
        # _build_weaviate_filter (Filter.by_property) can match them.
        for fld in _FILTERABLE_FIELDS:
            if fld in metadata and metadata[fld] is not None:
                properties[fld] = str(metadata[fld])
        obj_uuid = _to_uuid(id)

        try:
            collection = client.collections.get(_CLASS_NAME)
            # Check if exists then update; otherwise insert
            try:
                await collection.data.update(
                    uuid=obj_uuid,
                    properties=properties,
                    vector=vector,
                )
            except Exception:
                await collection.data.insert(
                    properties=properties,
                    vector=vector,
                    uuid=obj_uuid,
                )
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Weaviate upsert failed: %s", exc)
            raise HarnessError(
                f"Weaviate upsert error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def query(
        self,
        text: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        hybrid_alpha: float | None = None,
    ) -> list[VectorHit]:
        """
        Near-vector search; uses hybrid search when ``hybrid_alpha`` is given.

        ``hybrid_alpha=0`` → pure BM25, ``hybrid_alpha=1`` → pure vector,
        ``hybrid_alpha=0.5`` → balanced.
        """
        await self._ensure_schema()
        client = await self._get_client()
        vector = await self._embed(text)

        try:
            import weaviate.classes.query as wcq  # type: ignore[import]

            collection = client.collections.get(_CLASS_NAME)

            where_filter = _build_weaviate_filter(filter) if filter else None

            if hybrid_alpha is not None:
                response = await collection.query.hybrid(
                    query=text,
                    vector=vector,
                    alpha=hybrid_alpha,
                    limit=k,
                    filters=where_filter,
                    return_metadata=wcq.MetadataQuery(score=True, distance=True),
                )
            else:
                response = await collection.query.near_vector(
                    near_vector=vector,
                    limit=k,
                    filters=where_filter,
                    return_metadata=wcq.MetadataQuery(distance=True),
                )

            hits: list[VectorHit] = []
            for obj in response.objects:
                props = obj.properties or {}
                meta_json = props.get("metadata_json", "{}")
                try:
                    meta = json.loads(meta_json)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

                # Hybrid search ranks by a fused relevance score (higher is
                # better) and does NOT populate distance. Pure near-vector search
                # populates distance (lower is better). Read the right field for
                # each path, otherwise every hybrid hit scores 0.0 and gets
                # discarded by downstream threshold filters.
                md = obj.metadata
                if hybrid_alpha is not None and md and md.score is not None:
                    score = float(md.score)
                else:
                    distance = (
                        md.distance if md and md.distance is not None else 1.0
                    )
                    score = float(1.0 - distance)

                hits.append(
                    VectorHit(
                        id=str(obj.uuid),
                        text=props.get("text", ""),
                        score=score,
                        metadata=meta,
                    )
                )
            return hits

        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Weaviate query failed: %s", exc)
            raise HarnessError(
                f"Weaviate query error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def delete(self, id: str) -> None:
        """Remove an object by UUID."""
        await self._ensure_schema()
        client = await self._get_client()
        try:
            collection = client.collections.get(_CLASS_NAME)
            await collection.data.delete_by_id(_to_uuid(id))
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Weaviate delete failed: %s", exc)
            raise HarnessError(
                f"Weaviate delete error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Return object count, optionally filtered."""
        await self._ensure_schema()
        client = await self._get_client()
        try:
            collection = client.collections.get(_CLASS_NAME)
            where_filter = _build_weaviate_filter(filter) if filter else None
            result = await collection.aggregate.over_all(
                total_count=True,
                filters=where_filter,
            )
            return result.total_count or 0
        except HarnessError:
            raise
        except Exception as exc:
            logger.error("Weaviate count failed: %s", exc)
            raise HarnessError(
                f"Weaviate count error: {exc}",
                failure_class=FailureClass.MEMORY_VECTOR,
            ) from exc

    async def close(self) -> None:
        """Close Weaviate async client."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uuid(id_str: str) -> str:
    try:
        uuid.UUID(id_str)
        return id_str
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


def _build_weaviate_filter(filter: dict[str, Any]) -> Any:
    """Convert a flat {field: value} dict to Weaviate Filter objects."""
    try:
        import weaviate.classes.query as wcq  # type: ignore[import]

        filters = []
        for field_name, value in filter.items():
            # Promoted fields are stored as TEXT, so equals must compare strings.
            if field_name in _FILTERABLE_FIELDS and value is not None:
                value = str(value)
            filters.append(
                wcq.Filter.by_property(field_name).equal(value)
            )
        if len(filters) == 1:
            return filters[0]
        result = filters[0]
        for f in filters[1:]:
            result = result & f
        return result
    except ImportError:
        return None
