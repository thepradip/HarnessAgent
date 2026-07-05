"""CodeGraphRAG: weighted multi-hop retrieval over the code knowledge graph.

The code twin of :class:`harness.memory.graph_rag.GraphRAGEngine`:

1. Extract code identifiers from the query (regex → LLM → fallback)
2. Anchor to matching code nodes (``code:`` namespace only)
3. If no anchors, bridge from vector hits over the symbol chunks
4. Weighted BFS traversal (calls 1.5 > inherits 1.4 > imports 1.0 > contains 0.8)
5. Render top paths **signatures-first** — the token saver: the agent sees
   compact structure (signatures, docstrings, relations) instead of file dumps,
   and expands full source only for the symbols it actually needs via
   :meth:`CodeGraphRAG.expand_symbol`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from harness.core.protocols import GraphPath, VectorStore
from harness.memory.code_entity_extractor import extract_code_entities
from harness.memory.code_graph import CODE_EDGE_WEIGHTS
from harness.memory.schemas import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_EDGE_WEIGHT = 0.5
_CODE_NODE_PREFIX = "code:"
_MAX_SYMBOL_LINES = 400  # expand_symbol safety cap


class CodeGraphRAG:
    """Multi-hop code retrieval combining the code graph with symbol vectors.

    Args:
        graph:        Any GraphStore populated by CodeGraphIndexer.
        vector_store: Optional VectorStore holding the per-symbol chunks.
        llm_provider: Optional — enables the LLM entity-extraction tier.
        repo_root:    Optional repository root; enables expand_symbol().
        max_rendered_paths: Cap on scored paths rendered into context.
    """

    def __init__(
        self,
        graph: Any,
        vector_store: VectorStore | None = None,
        llm_provider: Any | None = None,
        repo_root: Path | str | None = None,
        max_rendered_paths: int = 20,
    ) -> None:
        self._graph = graph
        self._vector_store = vector_store
        self._llm = llm_provider
        self._repo_root = Path(repo_root).resolve() if repo_root else None
        self._max_paths = max_rendered_paths
        self._node_freq: dict[str, int] = defaultdict(int)
        self._freq_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        tenant_id: str | None = None,
        max_hops: int = 2,
        vector_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve code context for *query* via graph traversal + vectors."""
        entities = await extract_code_entities(query, llm_provider=self._llm)
        graph_paths: list[GraphPath] = []
        graph_context = ""
        anchored_via_vector = False

        # Step 1: anchor nodes from extracted identifiers
        anchor_ids: list[str] = []
        if entities:
            try:
                anchor_nodes = await self._graph.find_nodes(entities, fuzzy=True)
                anchor_ids = [
                    n.id for n in anchor_nodes if n.id.startswith(_CODE_NODE_PREFIX)
                ]
            except Exception as exc:
                logger.warning("Code graph anchor lookup failed: %s", exc)

        # Step 2: vector search over symbol chunks (also used for bridging)
        vector_hits: list[Any] = []
        if self._vector_store is not None:
            try:
                vector_filter = {"tenant_id": tenant_id} if tenant_id else None
                vector_hits = await self._vector_store.query(
                    text=query, k=vector_k, filter=vector_filter
                )
            except Exception as exc:
                logger.warning("Code vector retrieval failed: %s", exc)

        if not anchor_ids and vector_hits:
            # Bridge: symbol chunks carry their graph node id in metadata
            bridge_ids = [
                hit.metadata.get("symbol_id", "")
                for hit in vector_hits[:3]
                if hit.metadata.get("symbol_id", "").startswith(_CODE_NODE_PREFIX)
            ]
            anchor_ids = [b for b in bridge_ids if b]
            anchored_via_vector = bool(anchor_ids)

        # Step 3: weighted BFS from all anchors
        if anchor_ids:
            try:
                raw_paths = await self._graph.traverse(anchor_ids, max_hops=max_hops)
                scored = self._score_paths(raw_paths)
                graph_paths = [sp[0] for sp in scored[: self._max_paths]]
                async with self._freq_lock:
                    for path in graph_paths:
                        for node in path.nodes:
                            self._node_freq[node.id] += 1
                graph_context = self._render_paths(graph_paths)
            except Exception as exc:
                logger.warning("Code graph traversal failed: %s", exc)

        vector_context = [h.text for h in vector_hits]

        if graph_paths and vector_hits:
            strategy = "hybrid"
        elif graph_paths:
            strategy = "graph_primary"
        elif vector_hits:
            strategy = "vector_fallback" if anchored_via_vector else "vector_primary"
        else:
            strategy = "vector_fallback"

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

    async def expand_symbol(self, symbol_id: str) -> str | None:
        """Return the full source of one symbol (expand-on-demand token saver).

        The retrieval context shows signatures only; agents call this for the
        two or three symbols they actually need to read. Requires ``repo_root``.
        The symbol id encodes the file (``code:sym:<rel_path>::<qualname>``);
        the line range comes from the graph node, which the indexer keeps
        fresh. Returns None for malformed ids, path escapes, or missing files.
        """
        if self._repo_root is None:
            logger.debug("expand_symbol called without repo_root configured")
            return None
        if not symbol_id.startswith("code:sym:"):
            return None
        remainder = symbol_id[len("code:sym:") :]
        rel_path, sep, _qualname = remainder.partition("::")
        if not sep:
            return None

        target = (self._repo_root / rel_path).resolve()
        try:
            target.relative_to(self._repo_root)
        except ValueError:
            logger.warning("expand_symbol path escape blocked: %s", symbol_id)
            return None
        if not target.is_file():
            return None
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("expand_symbol read failed for %s: %s", rel_path, exc)
            return None

        try:
            nodes = await self._graph.find_nodes([symbol_id], fuzzy=False)
        except Exception as exc:
            logger.warning("expand_symbol node lookup failed: %s", exc)
            return None
        if not nodes:
            return None  # unknown symbol — never dump the whole file

        start = max(int(nodes[0].props.get("line", 1)) - 1, 0)
        end = min(
            int(nodes[0].props.get("end_line", start + 1)),
            start + _MAX_SYMBOL_LINES,
        )
        return "\n".join(lines[start:end])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_paths(self, paths: list[GraphPath]) -> list[tuple[GraphPath, float]]:
        """Score paths by cumulative edge weight + node-frequency bonus."""
        scored: list[tuple[GraphPath, float]] = []
        for path in paths:
            edge_score = (
                sum(
                    edge.props.get(
                        "weight", CODE_EDGE_WEIGHTS.get(edge.type, _DEFAULT_EDGE_WEIGHT)
                    )
                    for edge in path.edges
                )
                if path.edges
                else _DEFAULT_EDGE_WEIGHT
            )
            freq_bonus = sum(
                min(self._node_freq.get(n.id, 0) * 0.1, 1.0) for n in path.nodes
            )
            scored.append((path, edge_score + freq_bonus))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _render_paths(self, paths: list[GraphPath]) -> str:
        """Render paths as compact, deduplicated, signatures-first context.

        Sections:
        [FILES]       - files touched, with module + docstring
        [SYMBOLS]     - signature + docstring + location (never full bodies)
        [CALL GRAPH]  - caller → callee edges
        [INHERITANCE] - subclass → base edges
        [IMPORTS]     - file → file dependencies
        """
        file_lines: dict[str, str] = {}
        symbol_lines: dict[str, str] = {}
        call_lines: list[str] = []
        inherit_lines: list[str] = []
        import_lines: list[str] = []
        seen: set[str] = set()
        display: dict[str, str] = {}  # node id → short display name

        for path in paths:
            for node in path.nodes:
                props = node.props
                if node.type == "CodeFile":
                    display[node.id] = props.get("file", node.id)
                    if node.id not in file_lines:
                        doc = props.get("docstring", "")
                        module = props.get("module", "")
                        line = f"{props.get('file', node.id)}"
                        if module:
                            line += f"  ({module})"
                        if doc:
                            line += f" — {doc}"
                        file_lines[node.id] = line
                elif node.type in ("CodeClass", "CodeFunction", "CodeMethod"):
                    display[node.id] = props.get(
                        "qualified_name", props.get("name", node.id)
                    )
                    if node.id not in symbol_lines:
                        signature = props.get("signature", node.id)
                        doc = props.get("docstring", "")
                        loc = f"{props.get('file', '?')}:{props.get('line', '?')}"
                        line = f"{signature}   [{loc}]"
                        if doc:
                            line += f"\n    {doc}"
                        symbol_lines[node.id] = line
                elif node.type == "CodeModule":
                    display[node.id] = props.get("name", node.id)

            for edge in path.edges:
                src = display.get(edge.source_id, edge.source_id)
                tgt = display.get(edge.target_id, edge.target_id)
                if edge.type == "calls":
                    line = f"{src} --calls--> {tgt}"
                    if line not in seen:
                        seen.add(line)
                        call_lines.append(line)
                elif edge.type == "inherits":
                    line = f"{src} --inherits--> {tgt}"
                    if line not in seen:
                        seen.add(line)
                        inherit_lines.append(line)
                elif edge.type == "imports":
                    line = f"{src} --imports--> {tgt}"
                    if line not in seen:
                        seen.add(line)
                        import_lines.append(line)

        sections: list[str] = []
        if file_lines:
            sections.append("[FILES]")
            sections.extend(file_lines.values())
        if symbol_lines:
            sections.append("[SYMBOLS]")
            # Most-frequently retrieved symbols first
            for node_id in sorted(
                symbol_lines, key=lambda n: self._node_freq.get(n, 0), reverse=True
            ):
                sections.append(symbol_lines[node_id])
        if call_lines:
            sections.append("[CALL GRAPH]")
            sections.extend(call_lines[:30])
        if inherit_lines:
            sections.append("[INHERITANCE]")
            sections.extend(inherit_lines[:15])
        if import_lines:
            sections.append("[IMPORTS]")
            sections.extend(import_lines[:15])

        return "\n".join(sections)

    def _estimate_tokens(self, *texts: str) -> int:
        return max(1, sum(len(t) for t in texts) // 4)
