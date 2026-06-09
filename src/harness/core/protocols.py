"""Structural Protocol ABCs and supporting dataclasses for HarnessAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from harness.core.context import LLMResponse, ToolResult


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VectorHit:
    """A single search result from the vector store."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    type: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    type: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPath:
    """A traversal path through the knowledge graph."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_weight: float = 0.0


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Contract that all LLM provider adapters must satisfy."""

    provider_name: str
    model: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request and return the normalised response."""
        ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text deltas from the LLM."""
        ...

    # Optional capability: streaming WITH tool-call accumulation and exact
    # provider usage. Providers that cannot stream-with-tools simply do not
    # implement this — callers must feature-detect with ``hasattr`` and fall
    # back to ``complete()``. Declared here (not enforced at runtime) so the
    # contract is discoverable and typed.
    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_text: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Stream text deltas (via ``on_text``) and return a fully-populated
        LLMResponse with real usage + accumulated tool calls."""
        ...

    async def health_check(self) -> bool:
        """Return True if the provider is reachable and healthy."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Contract for vector database adapters."""

    async def upsert(
        self,
        id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        """Insert or update a document in the vector store."""
        ...

    async def query(
        self,
        text: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        hybrid_alpha: float | None = None,
    ) -> list[VectorHit]:
        """Return the top-k nearest neighbours for the query text."""
        ...

    async def delete(self, id: str) -> None:
        """Remove a document by its ID."""
        ...

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Return the number of documents matching the optional filter."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Contract for knowledge-graph adapters."""

    async def add_node(self, id: str, type: str, props: dict[str, Any]) -> None:
        """Add or update a node in the graph."""
        ...

    async def add_edge(
        self,
        src: str,
        tgt: str,
        type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        """Add or update a directed edge between two nodes."""
        ...

    async def traverse(
        self,
        start_ids: list[str],
        max_hops: int = 2,
    ) -> list[GraphPath]:
        """Perform a BFS/DFS traversal and return all discovered paths."""
        ...

    async def find_nodes(
        self,
        names: list[str],
        fuzzy: bool = True,
    ) -> list[GraphNode]:
        """Find nodes by name, optionally using fuzzy matching."""
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Contract for tool implementations callable by agents."""

    name: str
    description: str
    input_schema: dict[str, Any]
    timeout_seconds: float

    async def execute(
        self,
        ctx: Any,  # AgentContext — avoid circular import
        args: dict[str, Any],
    ) -> ToolResult:
        """Execute the tool with the given arguments and return a result."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for text embedding adapters."""

    model: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return dense vector embeddings for the provided texts."""
        ...
