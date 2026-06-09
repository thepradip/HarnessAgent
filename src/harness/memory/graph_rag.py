"""GraphRAG engine: weighted multi-hop graph retrieval with vector bridging.

Improvements over v1:
- Edge weights: joins (1.5) > used_by_query (1.2) > has_column (0.8)
- Path scoring: paths ranked by cumulative edge weight, not just count
- Query history nodes: past successful SQL queries surface as context
- Error pattern nodes: past failures surface as warnings
- Vector-to-graph bridging: when regex finds no anchors, vector hits seed the graph
- Frequency scoring: tables accessed often rank first in rendered output
- Bidirectional anchoring: traverse from all anchor nodes simultaneously
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from harness.core.protocols import EmbeddingProvider, GraphNode, GraphPath, VectorStore
from harness.memory.entity_extractor import extract_entities
from harness.memory.schemas import RetrievalResult

if TYPE_CHECKING:
    from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

# Edge type weights — higher means more relevant path
_EDGE_WEIGHTS: dict[str, float] = {
    "joins":          1.5,   # join relationships are most valuable
    "used_by_query":  1.2,   # past query usage is very informative
    "has_column":     0.8,   # schema structure
    "references":     1.0,   # FK reference
    "occurred_in":    0.6,   # error → query (lower priority)
}

_DEFAULT_EDGE_WEIGHT = 0.5


@dataclass
class ScoredPath:
    path: GraphPath
    score: float = 0.0


class GraphRAGEngine:
    """Multi-hop retrieval combining weighted graph traversal with vector bridging.

    Retrieval strategy (in priority order):
    1. Extract entities from query via regex
    2. Anchor to matching graph nodes
    3. If no anchors found, bridge from vector search results (new)
    4. BFS traversal scored by edge weights (new)
    5. Rank paths, render top-N as compact context
    6. Supplement with vector hits if graph coverage is thin
    """

    def __init__(
        self,
        graph: Any,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        llm_provider: Any | None = None,
        max_rendered_paths: int = 20,
    ) -> None:
        self._graph = graph
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm_provider   # optional: used for NL entity extraction
        self._max_paths = max_rendered_paths
        # In-memory frequency counter: node_id -> access count
        # Lock guards concurrent coroutine writes in the same event loop.
        self._node_freq: dict[str, int] = defaultdict(int)
        self._freq_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        ctx: "AgentContext",
        max_hops: int = 2,
        vector_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve context using weighted graph traversal with vector bridging."""
        entities = await extract_entities(query, llm_provider=self._llm)
        graph_paths: list[GraphPath] = []
        graph_context = ""
        anchor_source = "none"

        # Step 1: find anchor nodes from entity names
        anchor_ids: list[str] = []
        if entities:
            try:
                anchor_nodes = await self._graph.find_nodes(entities, fuzzy=True)
                anchor_ids = [n.id for n in anchor_nodes]
                anchor_source = "regex"
            except Exception as exc:
                logger.warning("Graph anchor lookup failed: %s", exc)

        # Step 2: vector-to-graph bridging if regex found nothing
        vector_hits = []
        try:
            tenant_filter = {"tenant_id": ctx.tenant_id} if ctx.tenant_id else None
            vector_hits = await self._vector_store.query(
                text=query, k=vector_k, filter=tenant_filter
            )
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)

        if not anchor_ids and vector_hits:
            # Pull entity-like words from the top vector results and try those as anchors
            bridge_entities: list[str] = []
            for hit in vector_hits[:3]:
                bridge_entities.extend(await extract_entities(hit.text))
            if bridge_entities:
                try:
                    bridge_nodes = await self._graph.find_nodes(bridge_entities, fuzzy=True)
                    anchor_ids = [n.id for n in bridge_nodes]
                    anchor_source = "vector_bridge"
                except Exception as exc:
                    logger.warning("Vector bridge anchor lookup failed: %s", exc)

        # Step 3: weighted BFS traversal from all anchors simultaneously
        if anchor_ids:
            try:
                raw_paths = await self._graph.traverse(anchor_ids, max_hops=max_hops)
                scored = self._score_paths(raw_paths)
                # Take top paths by score
                top_paths = [sp.path for sp in scored[:self._max_paths]]
                graph_paths = top_paths
                # Update frequency counters (lock guards concurrent coroutine writes)
                async with self._freq_lock:
                    for path in graph_paths:
                        for node in path.nodes:
                            self._node_freq[node.id] += 1
                graph_context = self._render_paths(graph_paths)
            except Exception as exc:
                logger.warning("Graph traversal failed: %s", exc)

        vector_context = [h.text for h in vector_hits]

        # Strategy annotation
        if len(graph_paths) >= 3 and not vector_hits:
            strategy = "graph_primary"
        elif graph_paths and vector_hits:
            strategy = "hybrid"
        elif anchor_source == "vector_bridge":
            strategy = "vector_bridge"
        elif not graph_paths and entities:
            strategy = "vector_fallback"
        else:
            strategy = "vector_primary"

        return RetrievalResult(
            graph_paths=graph_paths,
            graph_context=graph_context,
            vector_hits=vector_hits,
            vector_context=vector_context,
            total_tokens_estimate=self._estimate_tokens(
                graph_context, " ".join(vector_context)
            ),
            strategy=strategy,  # type: ignore[arg-type]
        )

    async def record_query(
        self,
        query_sql: str,
        tables_used: list[str],
        run_id: str,
        tenant_id: str,
        success: bool,
        error_message: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Add a query execution to the knowledge graph.

        Call this after every SQL execution (success or failure).
        Builds up a history of which queries touched which tables, making
        future retrievals smarter for similar questions.
        """
        now = datetime.now(timezone.utc).isoformat()
        # Stable across processes — builtin hash() is salted per-process
        # (PYTHONHASHSEED), which would create duplicate Query nodes for the
        # same SQL after a restart.
        query_hash = hashlib.sha1(query_sql.encode("utf-8")).hexdigest()[:8]
        query_id = f"query:{run_id}:{query_hash}"

        try:
            # Add Query node
            await self._graph.add_node(
                id=query_id,
                type="Query",
                props={
                    "sql": query_sql[:500],
                    "success": success,
                    "run_id": run_id,
                    "tenant_id": tenant_id,
                    "latency_ms": latency_ms,
                    "created_at": now,
                },
            )

            # Link query to tables it used
            for table in tables_used:
                weight = _EDGE_WEIGHTS["used_by_query"]
                await self._graph.add_edge(
                    src=query_id,
                    tgt=table,
                    type="used_by_query",
                    props={"weight": weight, "success": success},
                )

            # If it failed, add an Error node
            if not success and error_message:
                error_id = f"error:{query_id}"
                await self._graph.add_node(
                    id=error_id,
                    type="Error",
                    props={
                        "message": error_message[:300],
                        "query_id": query_id,
                        "created_at": now,
                    },
                )
                await self._graph.add_edge(
                    src=error_id,
                    tgt=query_id,
                    type="occurred_in",
                    props={"weight": _EDGE_WEIGHTS["occurred_in"]},
                )

        except Exception as exc:
            logger.warning("Failed to record query in graph: %s", exc)

    async def populate_schema(
        self,
        tables_info: list[dict[str, Any]],
        ctx: "AgentContext",
    ) -> None:
        """Populate graph with SQL schema. Called once per agent startup."""
        for table in tables_info:
            table_name = table["name"]
            await self._graph.add_node(
                id=table_name,
                type="Table",
                props={"name": table_name, "tenant_id": ctx.tenant_id},
            )
            for col in table.get("columns", []):
                col_id = f"{table_name}.{col['name']}"
                await self._graph.add_node(
                    id=col_id,
                    type="Column",
                    props={
                        "name": col["name"],
                        "col_type": col.get("type", "UNKNOWN"),
                        "nullable": col.get("nullable", True),
                        "table": table_name,
                    },
                )
                await self._graph.add_edge(
                    src=table_name,
                    tgt=col_id,
                    type="has_column",
                    props={"weight": _EDGE_WEIGHTS["has_column"]},
                )
            for fk in table.get("foreign_keys", []):
                await self._graph.add_edge(
                    src=table_name,
                    tgt=fk["ref_table"],
                    type="joins",
                    props={
                        "on": f"{fk['col']}={fk['ref_col']}",
                        "local_col": fk["col"],
                        "ref_col": fk["ref_col"],
                        "weight": _EDGE_WEIGHTS["joins"],
                    },
                )
        logger.info(
            "Populated schema graph: %d tables for tenant %s",
            len(tables_info),
            ctx.tenant_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_paths(self, paths: list[GraphPath]) -> list[ScoredPath]:
        """Score each path by cumulative edge weight + node frequency bonus."""
        scored: list[ScoredPath] = []
        for path in paths:
            edge_score = sum(
                p.props.get("weight", _DEFAULT_EDGE_WEIGHT)
                for p in path.edges
            ) if path.edges else _DEFAULT_EDGE_WEIGHT

            # Frequency bonus: nodes accessed more often rank higher
            freq_bonus = sum(
                min(self._node_freq.get(n.id, 0) * 0.1, 1.0)
                for n in path.nodes
            )
            scored.append(ScoredPath(path=path, score=edge_score + freq_bonus))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def _render_paths(self, paths: list[GraphPath]) -> str:
        """Render paths as compact, deduplicated context text.

        Sections:
        [SCHEMA]    - table/column structure
        [JOINS]     - join relationships
        [HISTORY]   - past queries that touched these tables
        [ERRORS]    - past failures (so agent avoids repeating them)
        """
        table_cols: dict[str, list[str]] = {}
        join_lines: list[str] = []
        query_lines: list[str] = []
        error_lines: list[str] = []
        seen: set[str] = set()

        # Sort tables by frequency so most-used come first
        for path in paths:
            for node in path.nodes:
                nid = node.id
                if node.type == "Table" and nid not in table_cols:
                    table_cols[nid] = []
                elif node.type == "Column":
                    table = node.props.get("table", "")
                    col_name = node.props.get("name", nid)
                    col_type = node.props.get("col_type", "?")
                    nullable = "" if node.props.get("nullable", True) else " NOT NULL"
                    key = f"{table}.{col_name}"
                    if table and key not in seen:
                        seen.add(key)
                        table_cols.setdefault(table, []).append(
                            f"{col_name}({col_type}{nullable})"
                        )
                elif node.type == "Query":
                    sql = node.props.get("sql", "")
                    success = node.props.get("success", True)
                    latency = node.props.get("latency_ms")
                    if sql and sql not in seen:
                        seen.add(sql)
                        lat_str = f", {latency:.0f}ms" if latency else ""
                        status = "ok" if success else "FAILED"
                        query_lines.append(f"  [{status}{lat_str}] {sql[:200]}")
                elif node.type == "Error":
                    msg = node.props.get("message", "")
                    if msg and msg not in seen:
                        seen.add(msg)
                        error_lines.append(f"  {msg[:200]}")

            for edge in path.edges:
                if edge.type == "joins":
                    on_clause = edge.props.get("on", "")
                    line = f"{edge.source_id} --joins--> {edge.target_id}"
                    if on_clause:
                        line += f" ON {on_clause}"
                    if line not in seen:
                        seen.add(line)
                        join_lines.append(line)

        # Render — sort tables by frequency (descending)
        sections: list[str] = []

        if table_cols:
            sections.append("[SCHEMA]")
            for table_name in sorted(
                table_cols,
                key=lambda t: self._node_freq.get(t, 0),
                reverse=True,
            ):
                cols = table_cols[table_name]
                col_str = ", ".join(cols) if cols else "(no columns)"
                freq = self._node_freq.get(table_name, 0)
                freq_note = f" (used {freq}x)" if freq > 0 else ""
                sections.append(f"{table_name}: Table | cols: {col_str}{freq_note}")

        if join_lines:
            sections.append("[JOINS]")
            sections.extend(join_lines)

        if query_lines:
            sections.append("[PAST QUERIES]")
            sections.extend(query_lines[:5])  # cap at 5 to keep context tight

        if error_lines:
            sections.append("[PAST ERRORS - avoid repeating]")
            sections.extend(error_lines[:3])

        return "\n".join(sections)

    def _estimate_tokens(self, *texts: str) -> int:
        return max(1, sum(len(t) for t in texts) // 4)
