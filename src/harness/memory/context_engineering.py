"""
Context Engineering — L1 / L2 / L3 tiered context assembly for any agentic app.

Tiers
─────
L1  Chat history cache   Redis hot window    sub-ms     per-run conversation
L2  Vector cache         VectorStore         ms         semantic search over memory
L3  Knowledge store      Graph + SchemaStore ms–s       entities, relations, SQL schema

Token budget allocation (defaults, configurable)
────────────────────────────────────────────────
  L1  60 %  — most recent chat history
  L2  25 %  — semantic search hits
  L3  15 %  — knowledge graph facts + SQL schema

Usage
─────
    pipeline = ContextPipeline.create(
        redis_url="redis://localhost:6379",
        vector_store=my_vector_store,
        embedder=my_embedder,
        graph=my_graph,
    )

    # Store SQL schema once (e.g. on agent startup)
    await pipeline.l3.schema.store("mydb", tables)

    # Assemble context before every LLM call
    ctx = await pipeline.assemble(
        run_id=ctx.run_id,
        query=user_query,
        skill_ns="sql",
        token_budget=60_000,
        db_id="mydb",              # optional: include schema context
    )
    # ctx.messages → pass to LLM
    # ctx.stats    → log / trace
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Token budget splits ───────────────────────────────────────────────────────

_L1_FRACTION = 0.60
_L2_FRACTION = 0.25
_L3_FRACTION = 0.15

# ── Redis key prefixes ────────────────────────────────────────────────────────

_SCHEMA_PFX = "harness:schema:"        # HASH per db_id
_SCHEMA_TTL = 86_400 * 7               # 7 days
_KG_FACT_PFX = "harness:kg:facts:"     # ZSET per tenant_id
_KG_FACT_TTL = 86_400 * 30             # 30 days


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnDef:
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type,
                "nullable": self.nullable, "primary_key": self.primary_key}

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnDef":
        return cls(name=d["name"], type=d["type"],
                   nullable=d.get("nullable", True), primary_key=d.get("primary_key", False))


@dataclass
class ForeignKey:
    column: str
    references_table: str
    references_column: str

    def to_dict(self) -> dict:
        return {"column": self.column,
                "references_table": self.references_table,
                "references_column": self.references_column}

    @classmethod
    def from_dict(cls, d: dict) -> "ForeignKey":
        return cls(column=d["column"],
                   references_table=d["references_table"],
                   references_column=d["references_column"])


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: int | None = None
    sample_rows: list[dict] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "row_count": self.row_count,
            "sample_rows": self.sample_rows,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableSchema":
        return cls(
            name=d["name"],
            columns=[ColumnDef.from_dict(c) for c in d.get("columns", [])],
            foreign_keys=[ForeignKey.from_dict(fk) for fk in d.get("foreign_keys", [])],
            row_count=d.get("row_count"),
            sample_rows=d.get("sample_rows", []),
            description=d.get("description", ""),
        )

    def to_context_block(self, include_samples: bool = True) -> str:
        """Format as a compact context string for LLM injection."""
        pk_cols = [c.name for c in self.columns if c.primary_key]
        col_strs = []
        for c in self.columns:
            pk = " PK" if c.primary_key else ""
            null = "" if c.nullable else " NOT NULL"
            col_strs.append(f"  {c.name} {c.type}{pk}{null}")
        lines = [f"TABLE {self.name}"]
        if self.description:
            lines.append(f"  -- {self.description}")
        lines.extend(col_strs)
        if self.foreign_keys:
            for fk in self.foreign_keys:
                lines.append(f"  FK: {fk.column} → {fk.references_table}.{fk.references_column}")
        if self.row_count is not None:
            lines.append(f"  -- {self.row_count:,} rows")
        if include_samples and self.sample_rows:
            lines.append(f"  -- sample: {self.sample_rows[0]}")
        return "\n".join(lines)


@dataclass
class TierResult:
    tier: Literal["L1", "L2", "L3"]
    messages: list[dict]          # {"role": str, "content": str}
    tokens_used: int
    source: str                   # human-readable description
    latency_ms: float
    hit: bool = True              # False when tier returned nothing


@dataclass
class AssembledContext:
    messages: list[dict]          # final list ready to pass to LLM
    total_tokens: int
    budget: int
    tier_results: list[TierResult]
    query: str
    skill_ns: str
    assembled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "budget": self.budget,
            "utilisation": round(self.total_tokens / max(self.budget, 1), 3),
            "tiers": {
                r.tier: {
                    "tokens": r.tokens_used,
                    "source": r.source,
                    "latency_ms": round(r.latency_ms, 2),
                    "hit": r.hit,
                }
                for r in self.tier_results
            },
        }

    def l1_tokens(self) -> int:
        return next((r.tokens_used for r in self.tier_results if r.tier == "L1"), 0)

    def l2_tokens(self) -> int:
        return next((r.tokens_used for r in self.tier_results if r.tier == "L2"), 0)

    def l3_tokens(self) -> int:
        return next((r.tokens_used for r in self.tier_results if r.tier == "L3"), 0)


# ── L1: Chat history cache ────────────────────────────────────────────────────

class L1ChatCache:
    """
    Hot chat history per (run_id, skill_ns).
    Backed by Redis LIST; delegates to ContextEngine for offload logic.
    Exposes a clean get/stats interface over the existing hot window.
    """

    def __init__(self, context_engine: Any) -> None:
        self._engine = context_engine  # ContextEngine

    async def get(
        self,
        run_id: str,
        skill_ns: str = "default",
        token_budget: int = 48_000,
    ) -> TierResult:
        t0 = time.monotonic()
        try:
            msgs = await self._engine._load_hot(run_id, skill_ns)
            # Also pull shared namespace if skill-specific
            if skill_ns != "default":
                shared = await self._engine._load_hot(run_id, "default")
                msgs = shared + msgs  # chronological: shared first, then skill-specific
        except Exception as exc:
            logger.debug("L1 load failed: %s", exc)
            msgs = []

        # Fit to budget
        kept, tokens, _ = _fit_to_budget(
            [{"role": m.role, "content": m.content, "tokens": m.tokens} for m in msgs],
            token_budget,
        )
        # Strip the internal "tokens" bookkeeping key — these dicts end up in
        # AssembledContext.messages and are passed straight to the LLM, where
        # strict APIs reject unknown message fields.
        kept = [{"role": m["role"], "content": m["content"]} for m in kept]

        return TierResult(
            tier="L1",
            messages=kept,
            tokens_used=tokens,
            source=f"chat_history run={run_id[:8]} ns={skill_ns}",
            latency_ms=(time.monotonic() - t0) * 1000,
            hit=len(kept) > 0,
        )

    async def push(
        self,
        run_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        skill_ns: str = "default",
        step: int = 0,
    ) -> None:
        await self._engine.push(run_id, role, content,
                                tokens=tokens, skill_ns=skill_ns, step=step)


# ── L2: Vector cache ──────────────────────────────────────────────────────────

class L2VectorCache:
    """
    Semantic search over the vector store.
    Returns relevant memory passages within a token budget and relevance threshold.
    """

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        relevance_threshold: float = 0.70,
        top_k: int = 5,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._threshold = relevance_threshold
        self._top_k = top_k

    async def get(
        self,
        query: str,
        token_budget: int = 20_000,
        filter: dict | None = None,
    ) -> TierResult:
        t0 = time.monotonic()

        if self._vs is None:
            return TierResult("L2", [], 0, "vector_store=None", 0.0, hit=False)

        try:
            hits = await self._vs.query(text=query, k=self._top_k, filter=filter or {})
        except Exception as exc:
            logger.debug("L2 query failed: %s", exc)
            return TierResult("L2", [], 0, f"query_error: {exc}", 0.0, hit=False)

        msgs: list[dict] = []
        tokens_used = 0
        sources: list[str] = []

        for hit in hits:
            if getattr(hit, "score", 1.0) < self._threshold:
                continue
            t = _count_tokens(hit.text)
            if tokens_used + t > token_budget:
                continue
            msgs.append({"role": "system", "content": f"[Memory] {hit.text}"})
            tokens_used += t
            sources.append(hit.id)

        return TierResult(
            tier="L2",
            messages=msgs,
            tokens_used=tokens_used,
            source=f"vector_search k={self._top_k} hits={len(msgs)}",
            latency_ms=(time.monotonic() - t0) * 1000,
            hit=len(msgs) > 0,
        )

    async def store(
        self,
        text: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Embed and upsert text into the vector store."""
        import uuid as _uuid
        uid = doc_id or _uuid.uuid4().hex
        try:
            embedding = (await self._embedder.embed([text]))[0]
            await self._vs.upsert(id=uid, text=text,
                                  metadata=metadata or {}, embedding=embedding)
        except Exception as exc:
            logger.debug("L2 store failed: %s", exc)
        return uid


# ── L3a: SQL Schema Store ─────────────────────────────────────────────────────

class SchemaStore:
    """
    Redis-backed SQL schema registry.

    Stores table definitions (columns, types, PKs, FKs, row counts, sample rows)
    per db_id. Used by the SQL agent to inject compact schema context without
    re-fetching live database metadata on every query.
    """

    def __init__(self, redis_url: str, ttl: int = _SCHEMA_TTL) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Any | None = None

    async def _r(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._redis_url, decode_responses=True, max_connections=5
            )
        return self._client

    # ── Write ──────────────────────────────────────────────────────────────────

    async def store(self, db_id: str, tables: list[TableSchema]) -> None:
        """Persist all table schemas for db_id."""
        r = await self._r()
        key = f"{_SCHEMA_PFX}{db_id}"
        mapping: dict[str, str] = {}
        for tbl in tables:
            mapping[tbl.name] = json.dumps(tbl.to_dict())
        if mapping:
            await r.hset(key, mapping=mapping)
            await r.expire(key, self._ttl)
            logger.debug("SchemaStore: stored %d tables for db_id=%s", len(tables), db_id)

    async def store_from_sqlite(self, db_id: str, db_path: str) -> list[TableSchema]:
        """Introspect a SQLite database and store its schema automatically."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        tables: list[TableSchema] = []
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            table_names = [r[0] for r in cur.fetchall()]
            for tname in table_names:
                # Columns
                cur = conn.execute(f"PRAGMA table_info({tname})")
                cols = [
                    ColumnDef(
                        name=row[1], type=row[2],
                        nullable=(row[3] == 0),
                        primary_key=(row[5] > 0),
                    )
                    for row in cur.fetchall()
                ]
                # FKs
                cur = conn.execute(f"PRAGMA foreign_key_list({tname})")
                fks = [
                    ForeignKey(column=row[3],
                               references_table=row[2],
                               references_column=row[4])
                    for row in cur.fetchall()
                ]
                # Row count
                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM {tname}")
                    row_count = cur.fetchone()[0]
                except Exception:
                    row_count = None
                # Sample rows (up to 2)
                sample_rows: list[dict] = []
                try:
                    cur = conn.execute(f"SELECT * FROM {tname} LIMIT 2")
                    col_names = [d[0] for d in cur.description]
                    for row in cur.fetchall():
                        sample_rows.append(dict(zip(col_names, row)))
                except Exception:
                    pass

                tables.append(TableSchema(
                    name=tname, columns=cols, foreign_keys=fks,
                    row_count=row_count, sample_rows=sample_rows,
                ))
        finally:
            conn.close()
        await self.store(db_id, tables)
        return tables

    # ── Read ───────────────────────────────────────────────────────────────────

    async def get(
        self,
        db_id: str,
        table_names: list[str] | None = None,
    ) -> list[TableSchema]:
        """Retrieve all (or named) table schemas for db_id."""
        r = await self._r()
        key = f"{_SCHEMA_PFX}{db_id}"
        if table_names:
            raw_values = await r.hmget(key, table_names)
            items = zip(table_names, raw_values)
        else:
            raw_map = await r.hgetall(key)
            items = raw_map.items()  # type: ignore[assignment]

        result: list[TableSchema] = []
        for _, raw in items:
            if raw:
                try:
                    result.append(TableSchema.from_dict(json.loads(raw)))
                except Exception as exc:
                    logger.debug("SchemaStore.get parse error: %s", exc)
        return result

    async def table_names(self, db_id: str) -> list[str]:
        """Return all stored table names for db_id."""
        r = await self._r()
        return list(await r.hkeys(f"{_SCHEMA_PFX}{db_id}"))

    async def delete(self, db_id: str) -> None:
        r = await self._r()
        await r.delete(f"{_SCHEMA_PFX}{db_id}")

    async def get_context_block(
        self,
        db_id: str,
        relevant_tables: list[str] | None = None,
        include_samples: bool = True,
        max_tables: int = 15,
    ) -> str:
        """
        Format stored schema as a compact context string for LLM injection.

        If relevant_tables is provided, only those tables are included.
        Otherwise all stored tables are included (capped at max_tables).
        """
        tables = await self.get(db_id, relevant_tables)
        if not tables:
            return ""
        tables = tables[:max_tables]
        blocks = [tbl.to_context_block(include_samples=include_samples) for tbl in tables]
        header = f"-- Database: {db_id} ({len(tables)} tables) --"
        return header + "\n\n" + "\n\n".join(blocks)


# ── L3b: Knowledge Graph cache ────────────────────────────────────────────────

class KGCache:
    """
    Retrieves entity facts from the knowledge graph relevant to the current query.
    Uses entity extraction to find relevant nodes, then graph traversal.
    """

    def __init__(self, graph: Any, entity_extractor: Any | None = None) -> None:
        self._graph = graph
        self._extractor = entity_extractor

    async def get(
        self,
        query: str,
        token_budget: int = 6_000,
        max_hops: int = 2,
    ) -> TierResult:
        t0 = time.monotonic()

        if self._graph is None:
            return TierResult("L3", [], 0, "graph=None", 0.0, hit=False)

        # Extract named entities from query to use as graph start nodes
        entities: list[str] = []
        if self._extractor is not None:
            try:
                entities = await self._extractor.extract_entities(query)
            except Exception:
                pass
        if not entities:
            # Fallback: use noun-like words > 3 chars
            import re
            entities = [w for w in re.findall(r"\b[A-Za-z][a-z]{3,}\b", query)][:5]

        if not entities:
            return TierResult("L3", [], 0, "no_entities", 0.0, hit=False)

        try:
            paths = await self._graph.traverse(start_ids=entities, max_hops=max_hops)
        except Exception as exc:
            logger.debug("KG traverse failed: %s", exc)
            return TierResult("L3", [], 0, f"traverse_error: {exc}", 0.0, hit=False)

        facts: list[str] = []
        for path in paths:
            for edge in getattr(path, "edges", []):
                src = getattr(edge, "src", "")
                rel = getattr(edge, "type", "")
                tgt = getattr(edge, "tgt", "")
                if src and rel and tgt:
                    facts.append(f"{src} {rel} {tgt}")

        if not facts:
            return TierResult("L3", [], 0, "no_facts", 0.0, hit=False)

        # Deduplicate and build a single system message
        seen: set[str] = set()
        unique_facts = [f for f in facts if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

        content = "Knowledge graph facts:\n" + "\n".join(f"• {f}" for f in unique_facts[:50])
        tokens = _count_tokens(content)
        if tokens > token_budget:
            # Truncate facts list
            content = "Knowledge graph facts:\n" + "\n".join(
                f"• {f}" for f in unique_facts[:max(1, token_budget // 15)]
            )
            tokens = _count_tokens(content)

        return TierResult(
            tier="L3",
            messages=[{"role": "system", "content": content}],
            tokens_used=tokens,
            source=f"kg_traverse entities={len(entities)} facts={len(unique_facts)}",
            latency_ms=(time.monotonic() - t0) * 1000,
            hit=True,
        )


# ── L3: Combined knowledge store ─────────────────────────────────────────────

class L3KnowledgeStore:
    """
    Combines the SQL schema store and knowledge graph.
    Both contribute to the L3 context tier with a shared token budget.
    """

    def __init__(self, schema: SchemaStore, kg: KGCache) -> None:
        self.schema = schema
        self.kg = kg

    async def get(
        self,
        query: str,
        token_budget: int = 12_000,
        db_id: str | None = None,
        relevant_tables: list[str] | None = None,
        include_schema: bool = True,
        include_kg: bool = True,
        max_hops: int = 2,
    ) -> TierResult:
        t0 = time.monotonic()
        msgs: list[dict] = []
        tokens_used = 0
        sources: list[str] = []

        # Schema block — up to 2/3 of L3 budget
        if include_kg and include_schema:
            schema_budget = token_budget * 2 // 3
            kg_budget = token_budget - schema_budget
        elif include_schema:
            schema_budget = token_budget
            kg_budget = 0
        else:
            schema_budget = 0
            kg_budget = token_budget

        if include_schema and db_id:
            schema_block = await self.schema.get_context_block(
                db_id, relevant_tables, include_samples=True
            )
            if schema_block:
                t = _count_tokens(schema_block)
                if t <= schema_budget:
                    msgs.append({"role": "system", "content": schema_block})
                    tokens_used += t
                    sources.append(f"schema:{db_id}")
                else:
                    # Truncate: no samples, fewer tables
                    schema_block = await self.schema.get_context_block(
                        db_id, relevant_tables, include_samples=False, max_tables=8
                    )
                    t = _count_tokens(schema_block)
                    # Re-verify the truncated block actually fits — truncation is
                    # heuristic and may still exceed the schema budget. Only add
                    # it when it fits to keep the L3 tier within budget.
                    if schema_block and t <= schema_budget:
                        msgs.append({"role": "system", "content": schema_block})
                        tokens_used += t
                        sources.append(f"schema:{db_id}(truncated)")

        if include_kg:
            kg_result = await self.kg.get(
                query, token_budget=max(0, kg_budget), max_hops=max_hops
            )
            if kg_result.hit:
                msgs.extend(kg_result.messages)
                tokens_used += kg_result.tokens_used
                sources.append(kg_result.source)

        return TierResult(
            tier="L3",
            messages=msgs,
            tokens_used=tokens_used,
            source="; ".join(sources) if sources else "empty",
            latency_ms=(time.monotonic() - t0) * 1000,
            hit=len(msgs) > 0,
        )


# ── ContextPipeline ───────────────────────────────────────────────────────────

class ContextPipeline:
    """
    Assembles a complete LLM context window by drawing from L1, L2, and L3.

    Token budget allocation (configurable):
        L1 (chat history)  : l1_fraction × budget
        L2 (vector search) : l2_fraction × budget
        L3 (knowledge/schema): l3_fraction × budget

    Assembly order in the final message list:
        [L3 schema/facts] → [L2 semantic hits] → [L1 chat history]
    (Oldest/most-general context first so LLM sees it before the recent chat.)
    """

    def __init__(
        self,
        l1: L1ChatCache,
        l2: L2VectorCache,
        l3: L3KnowledgeStore,
        l1_fraction: float = _L1_FRACTION,
        l2_fraction: float = _L2_FRACTION,
        l3_fraction: float = _L3_FRACTION,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self._l1_frac = l1_fraction
        self._l2_frac = l2_fraction
        self._l3_frac = l3_fraction

    async def assemble(
        self,
        run_id: str,
        query: str,
        token_budget: int = 60_000,
        skill_ns: str = "default",
        # L2 options
        vector_filter: dict | None = None,
        # L3 options
        db_id: str | None = None,
        relevant_tables: list[str] | None = None,
        include_schema: bool = True,
        include_kg: bool = True,
        # Override fractions per-call
        l1_fraction: float | None = None,
        l2_fraction: float | None = None,
        l3_fraction: float | None = None,
    ) -> AssembledContext:
        """
        Build and return an AssembledContext ready for the LLM.

        The fractions are re-normalised if they don't sum to 1.0.
        """
        f1 = l1_fraction if l1_fraction is not None else self._l1_frac
        f2 = l2_fraction if l2_fraction is not None else self._l2_frac
        f3 = l3_fraction if l3_fraction is not None else self._l3_frac
        total = f1 + f2 + f3
        if total > 0:
            f1, f2, f3 = f1 / total, f2 / total, f3 / total

        b1 = int(token_budget * f1)
        b2 = int(token_budget * f2)
        b3 = token_budget - b1 - b2   # L3 gets remainder to avoid rounding loss

        # Fetch all tiers in parallel
        import asyncio
        l1_res, l2_res, l3_res = await asyncio.gather(
            self.l1.get(run_id, skill_ns, token_budget=b1),
            self.l2.get(query, token_budget=b2, filter=vector_filter),
            self.l3.get(
                query, token_budget=b3, db_id=db_id,
                relevant_tables=relevant_tables,
                include_schema=include_schema, include_kg=include_kg,
            ),
        )

        # Assemble: L3 first (schema/facts), L2 (semantic), L1 (chat history)
        all_messages: list[dict] = []
        for result in (l3_res, l2_res, l1_res):
            all_messages.extend(result.messages)

        total_tokens = l1_res.tokens_used + l2_res.tokens_used + l3_res.tokens_used

        return AssembledContext(
            messages=all_messages,
            total_tokens=total_tokens,
            budget=token_budget,
            tier_results=[l1_res, l2_res, l3_res],
            query=query,
            skill_ns=skill_ns,
        )

    @classmethod
    def create(
        cls,
        redis_url: str,
        vector_store: Any = None,
        embedder: Any = None,
        graph: Any = None,
        entity_extractor: Any = None,
        context_engine: Any = None,
        relevance_threshold: float = 0.70,
        top_k: int = 5,
        l1_fraction: float = _L1_FRACTION,
        l2_fraction: float = _L2_FRACTION,
        l3_fraction: float = _L3_FRACTION,
        schema_ttl: int = _SCHEMA_TTL,
    ) -> "ContextPipeline":
        """Build a ContextPipeline from plain config values."""
        # L1 requires a ContextEngine
        if context_engine is None:
            from harness.memory.context_engine import ContextEngine
            context_engine = ContextEngine.create(
                redis_url=redis_url,
                vector_store=vector_store,
                embedder=embedder,
            )

        l1 = L1ChatCache(context_engine)
        l2 = L2VectorCache(vector_store, embedder,
                           relevance_threshold=relevance_threshold, top_k=top_k)
        schema = SchemaStore(redis_url=redis_url, ttl=schema_ttl)
        kg = KGCache(graph, entity_extractor)
        l3 = L3KnowledgeStore(schema=schema, kg=kg)

        return cls(l1=l1, l2=l2, l3=l3,
                   l1_fraction=l1_fraction,
                   l2_fraction=l2_fraction,
                   l3_fraction=l3_fraction)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token)."""
    try:
        from harness.memory.embedder import estimate_tokens
        return estimate_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _fit_to_budget(
    messages: list[dict],
    budget: int,
) -> tuple[list[dict], int, bool]:
    """Keep the most-recent messages that fit within token budget."""
    kept: list[dict] = []
    used = 0
    for msg in reversed(messages):
        t = msg.get("tokens", _count_tokens(msg.get("content", "")))
        if used + t <= budget:
            kept.insert(0, msg)
            used += t
        else:
            break
    return kept, used, len(kept) < len(messages)
