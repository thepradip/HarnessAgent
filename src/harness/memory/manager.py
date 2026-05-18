"""MemoryManager: unified interface to all memory tiers."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from harness.core.protocols import EmbeddingProvider, GraphPath, VectorStore
from harness.memory.context_engine import (
    ActionRecord,
    ActionSummary,
    BuiltContext,
    ContextEngine,
    SubAgentSlice,
)
from harness.memory.context_engineering import (
    AssembledContext,
    ContextPipeline,
    SchemaStore,
    TableSchema,
)
from harness.memory.context_manager import ContextWindowManager
from harness.memory.graph import get_graph_memory
from harness.memory.graph_rag import GraphRAGEngine
from harness.memory.schemas import (
    ConversationMessage,
    ContextWindow,
    MemoryEntry,
    RetrievalResult,
)
from harness.memory.session_memory import SessionMemory, SessionMemoryRegistry
from harness.memory.short_term import ShortTermMemory
from harness.memory.vector_factory import build_embedding_provider, build_vector_store

if TYPE_CHECKING:
    from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

# PII redaction patterns
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),                    # SSN
    (re.compile(r"\b\d{3}[.\-\s]?\d{3}[.\-\s]?\d{4}\b"), "[PHONE]"),   # US phone
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),  # email
    (re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"), "[CC]"),               # Visa card
    (re.compile(r"\b5[1-5][0-9]{14}\b"), "[CC]"),                       # Mastercard
]


def _redact_pii(text: str) -> str:
    """Apply basic PII masking to text before long-term storage."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class MemoryManager:
    """
    Unified memory interface providing access to:
    - Short-term conversation history (Redis)
    - Long-term vector memory (Chroma / Qdrant / Weaviate)
    - Knowledge graph (NetworkX / Neo4j)
    - GraphRAG smart retrieval
    - Context window management
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        vector_store: VectorStore,
        graph: Any,
        embedder: EmbeddingProvider,
        context_manager: ContextWindowManager,
        context_engine: ContextEngine | None = None,
        session_registry: SessionMemoryRegistry | None = None,
        context_pipeline: ContextPipeline | None = None,
    ) -> None:
        self._short_term = short_term
        self._vector_store = vector_store
        self._graph = graph
        self._embedder = embedder
        self._context_manager = context_manager
        self._context_engine = context_engine
        self._graph_rag = GraphRAGEngine(graph, vector_store, embedder)
        self._sessions = session_registry
        self._pipeline = context_pipeline

    # ------------------------------------------------------------------
    # Context pipeline (L1 / L2 / L3)
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> ContextPipeline | None:
        """The L1/L2/L3 ContextPipeline, if configured."""
        return self._pipeline

    @property
    def schema(self) -> SchemaStore | None:
        """Direct access to the SQL SchemaStore (L3 tier)."""
        if self._pipeline is not None:
            return self._pipeline.l3.schema
        return None

    async def assemble_context(
        self,
        run_id: str,
        query: str,
        token_budget: int = 60_000,
        skill_ns: str = "default",
        db_id: str | None = None,
        relevant_tables: list[str] | None = None,
        include_schema: bool = True,
        include_kg: bool = True,
        vector_filter: dict | None = None,
    ) -> AssembledContext | None:
        """
        Assemble a full L1/L2/L3 context window.

        Returns None if no ContextPipeline is configured — callers
        should fall back to build_context() in that case.
        """
        if self._pipeline is None:
            return None
        return await self._pipeline.assemble(
            run_id=run_id,
            query=query,
            token_budget=token_budget,
            skill_ns=skill_ns,
            db_id=db_id,
            relevant_tables=relevant_tables,
            include_schema=include_schema,
            include_kg=include_kg,
            vector_filter=vector_filter,
        )

    # ------------------------------------------------------------------
    # Conversation (short-term + context engine)
    # ------------------------------------------------------------------

    async def push_message(
        self,
        run_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        skill_ns: str = "default",
        step: int = 0,
    ) -> None:
        """Append a message; feeds both the legacy short-term store and ContextEngine."""
        await self._short_term.push_message(run_id, role, content, tokens)
        if self._context_engine is not None:
            await self._context_engine.push(
                run_id, role, content, tokens=tokens, skill_ns=skill_ns, step=step
            )

    async def get_history(
        self,
        run_id: str,
        last_n: int = 20,
    ) -> list[ConversationMessage]:
        """Return the most recent ``last_n`` messages (chronological order)."""
        return await self._short_term.get_history(run_id, last_n=last_n)

    async def fit_history(
        self,
        run_id: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> ContextWindow:
        """Retrieve history and fit it into the given token budget."""
        messages = await self._short_term.get_history(run_id)
        return await self._context_manager.fit(
            messages=messages,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Long-term (vector)
    # ------------------------------------------------------------------

    async def remember(
        self,
        text: str,
        metadata: dict[str, Any],
        tenant_id: str | None = None,
    ) -> str:
        """
        Store text in the long-term vector store after PII redaction.

        Returns the generated document ID.
        """
        clean_text = _redact_pii(text)
        doc_id = uuid.uuid4().hex

        if tenant_id:
            metadata = {**metadata, "tenant_id": tenant_id}

        embeddings = await self._embedder.embed([clean_text])
        await self._vector_store.upsert(
            id=doc_id,
            text=clean_text,
            metadata=metadata,
            embedding=embeddings[0],
        )
        return doc_id

    async def recall(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve the top-k most semantically similar memories."""
        hits = await self._vector_store.query(text=query, k=k, filter=filter)
        return [
            MemoryEntry(
                id=h.id,
                text=h.text,
                metadata=h.metadata,
                score=h.score,
                created_at=datetime.now(timezone.utc),
                source="long",
            )
            for h in hits
        ]

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a subject-predicate-object triple to the knowledge graph.

        Creates nodes for subject and object if they don't exist, then adds
        a directed edge with the predicate as the edge type.
        """
        props = metadata or {}
        await self._graph.add_node(id=subject, type="Entity", props={"name": subject})
        await self._graph.add_node(id=object_, type="Entity", props={"name": object_})
        await self._graph.add_edge(
            src=subject,
            tgt=object_,
            type=predicate.replace(" ", "_").upper(),
            props={**props, "weight": weight},
        )

    async def graph_query(
        self,
        start_ids: list[str],
        max_hops: int = 2,
    ) -> list[GraphPath]:
        """Traverse the knowledge graph from the given start node IDs."""
        return await self._graph.traverse(start_ids=start_ids, max_hops=max_hops)

    # ------------------------------------------------------------------
    # Smart retrieval
    # ------------------------------------------------------------------

    async def smart_retrieve(
        self,
        query: str,
        ctx: "AgentContext",
    ) -> RetrievalResult:
        """
        Graph-first retrieval with vector fallback.

        Delegates to GraphRAGEngine which handles entity extraction,
        graph traversal, vector supplementation, and strategy annotation.
        """
        return await self._graph_rag.retrieve(query=query, ctx=ctx)

    # ------------------------------------------------------------------
    # ContextEngine — paged, skill-isolated, action-scored
    # ------------------------------------------------------------------

    async def build_context(
        self,
        run_id: str,
        query: str,
        skill_ns: str = "default",
        token_budget: int | None = None,
    ) -> BuiltContext | ContextWindow:
        """
        Build an LLM-ready context window.

        Uses ContextEngine (with offload + cold retrieval) when available;
        falls back to legacy ContextWindowManager otherwise.
        """
        if self._context_engine is not None:
            return await self._context_engine.build_context(
                run_id=run_id,
                query=query,
                skill_ns=skill_ns,
                token_budget=token_budget,
            )
        # Legacy path
        return await self.fit_history(run_id, max_tokens=token_budget or 100_000)

    async def evaluate_action(
        self,
        run_id: str,
        step: int,
        goal: str,
        llm_content: str,
        tool_name: str | None = None,
        tool_result: str | None = None,
        is_error: bool = False,
        skill_ns: str = "default",
    ) -> ActionRecord | None:
        """Score an agent action and persist the record. No-op without ContextEngine."""
        if self._context_engine is None:
            return None
        return await self._context_engine.evaluate_action(
            run_id=run_id,
            step=step,
            goal=goal,
            llm_content=llm_content,
            tool_name=tool_name,
            tool_result=tool_result,
            is_error=is_error,
            skill_ns=skill_ns,
        )

    async def get_action_summary(self, run_id: str) -> ActionSummary | None:
        """Return aggregated action metrics. None if ContextEngine not configured."""
        if self._context_engine is None:
            return None
        return await self._context_engine.get_action_summary(run_id)

    async def slice_for_subagent(
        self,
        parent_run_id: str,
        child_run_id: str,
        task: str,
        token_budget: int,
        skill_ns: str = "default",
    ) -> SubAgentSlice | None:
        """Slice parent context for a child agent. None if ContextEngine not configured."""
        if self._context_engine is None:
            return None
        return await self._context_engine.slice_for_subagent(
            parent_run_id=parent_run_id,
            child_run_id=child_run_id,
            task=task,
            token_budget=token_budget,
            skill_ns=skill_ns,
        )

    async def inject_subagent_result(
        self,
        parent_run_id: str,
        child_run_id: str,
        result_summary: str,
        skill_ns: str = "default",
    ) -> None:
        """Inject child result into parent hot window. No-op without ContextEngine."""
        if self._context_engine is not None:
            await self._context_engine.inject_subagent_result(
                parent_run_id=parent_run_id,
                child_run_id=child_run_id,
                result_summary=result_summary,
                skill_ns=skill_ns,
            )

    # ------------------------------------------------------------------
    # Session memory (cross-run, persistent)
    # ------------------------------------------------------------------

    def session(self, tenant_id: str, session_id: str = "default") -> SessionMemory:
        """Return the SessionMemory for (tenant_id, session_id).

        SessionMemory persists across runs — use it to store user preferences,
        accumulated decisions, and cross-session facts.

        Example
        -------
        sm = memory.session("acme", "user-42")
        await sm.remember("preferred_model", "claude-sonnet")
        model = await sm.recall("preferred_model")
        """
        if self._sessions is None:
            raise RuntimeError(
                "SessionMemoryRegistry not configured. "
                "Pass a redis client to MemoryManager.create() to enable session memory."
            )
        return self._sessions.get(tenant_id, session_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def clear_session(self, run_id: str) -> None:
        """Clear all short-term memory for the given run_id."""
        await self._short_term.clear(run_id)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: Any, llm_provider: Any = None) -> "MemoryManager":
        """
        Build a fully configured MemoryManager from harness config.

        Constructs:
        - ShortTermMemory (Redis)
        - EmbeddingProvider (SentenceTransformer)
        - VectorStore (Chroma / Qdrant / Weaviate based on config)
        - GraphMemory (NetworkX / Neo4j based on config)
        - ContextWindowManager (legacy 100k token budget)
        - ContextEngine (paged, skill-isolated, action-scored)
        """
        embedder = build_embedding_provider(config)
        vector_store = build_vector_store(config, embedder)
        short_term = ShortTermMemory(redis_url=config.redis_url)
        graph = get_graph_memory(config)
        context_manager = ContextWindowManager(max_tokens=100_000)

        context_engine = ContextEngine.create(
            redis_url=config.redis_url,
            vector_store=vector_store,
            embedder=embedder,
            summarizer=llm_provider,
            max_hot_tokens=getattr(config, "context_max_hot_tokens", 80_000),
            reserve_output=getattr(config, "context_reserve_output", 2_000),
            offload_threshold=getattr(config, "context_offload_threshold", 0.80),
            cold_pages_per_query=getattr(config, "context_cold_pages", 3),
        )

        # Session memory registry — shared Redis client, 7-day TTL
        import redis.asyncio as _aioredis
        _redis_for_sessions = _aioredis.from_url(
            config.redis_url, decode_responses=True, max_connections=5
        )
        session_registry = SessionMemoryRegistry(
            redis=_redis_for_sessions,
            default_ttl=getattr(config, "session_memory_ttl", 604_800),
        )

        # L1 / L2 / L3 context engineering pipeline
        context_pipeline = ContextPipeline.create(
            redis_url=config.redis_url,
            vector_store=vector_store,
            embedder=embedder,
            graph=graph,
            context_engine=context_engine,
            relevance_threshold=getattr(config, "vector_relevance_threshold", 0.70),
            top_k=getattr(config, "vector_top_k", 5),
            l1_fraction=getattr(config, "context_l1_fraction", 0.60),
            l2_fraction=getattr(config, "context_l2_fraction", 0.25),
            l3_fraction=getattr(config, "context_l3_fraction", 0.15),
            schema_ttl=getattr(config, "schema_ttl", 86_400 * 7),
        )

        return cls(
            short_term=short_term,
            vector_store=vector_store,
            graph=graph,
            embedder=embedder,
            context_manager=context_manager,
            context_engine=context_engine,
            session_registry=session_registry,
            context_pipeline=context_pipeline,
        )
