"""Graph memory implementations: NetworkX (dev) and Neo4j (production)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from harness.core.errors import FailureClass, HarnessError
from harness.core.protocols import GraphEdge, GraphNode, GraphPath

logger = logging.getLogger(__name__)

# Cypher labels and relationship types are interpolated directly into the query
# string (they cannot be parameterised), so they MUST be restricted to a safe
# identifier charset to prevent Cypher injection.
_SAFE_IDENT = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_label(value: str, kind: str) -> str:
    """Validate a Cypher label / relationship type before interpolation.

    Raises HarnessError if ``value`` contains anything outside [A-Za-z0-9_],
    which prevents injection via crafted node types or relationship predicates.
    """
    if not value or not _SAFE_IDENT.match(value):
        raise HarnessError(
            f"Invalid Cypher {kind} {value!r}: only [A-Za-z0-9_] are allowed.",
            failure_class=FailureClass.MEMORY_GRAPH,
        )
    return value


# ---------------------------------------------------------------------------
# NetworkX implementation (default, dev/testing)
# ---------------------------------------------------------------------------


class NetworkXGraphMemory:
    """
    In-process graph memory backed by networkx.DiGraph.

    All mutations are protected by an asyncio.Lock.
    The graph can be persisted to / loaded from a JSON file in the workspace.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        try:
            import networkx as nx  # type: ignore[import]

            self._nx = nx
        except ImportError as exc:
            raise ImportError(
                "networkx is required. Install with: pip install networkx"
            ) from exc

        self._G: Any = self._nx.DiGraph()
        self._lock = asyncio.Lock()
        self._persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load_from_file(persist_path)

    # ------------------------------------------------------------------
    # GraphStore protocol
    # ------------------------------------------------------------------

    async def add_node(self, id: str, type: str, props: dict[str, Any]) -> None:
        async with self._lock:
            self._G.add_node(id, node_type=type, **props)
            await self._persist()

    async def add_edge(
        self,
        src: str,
        tgt: str,
        type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self._G.add_edge(src, tgt, edge_type=type, **(props or {}))
            await self._persist()

    async def traverse(
        self,
        start_ids: list[str],
        max_hops: int = 2,
    ) -> list[GraphPath]:
        """BFS from each start_id; return list of GraphPath objects."""
        async with self._lock:
            paths: list[GraphPath] = []
            visited_edges: set[tuple[str, str]] = set()

            for start_id in start_ids:
                if start_id not in self._G:
                    continue

                # BFS layer by layer
                current_layer = [start_id]
                path_nodes = [self._make_node(start_id)]
                path_edges: list[GraphEdge] = []

                for _ in range(max_hops):
                    next_layer: list[str] = []
                    for node_id in current_layer:
                        for neighbour in list(self._G.successors(node_id)) + list(
                            self._G.predecessors(node_id)
                        ):
                            edge_key = (min(node_id, neighbour), max(node_id, neighbour))
                            if edge_key in visited_edges:
                                continue
                            visited_edges.add(edge_key)
                            next_layer.append(neighbour)

                            # Build GraphEdge (prefer forward direction)
                            if self._G.has_edge(node_id, neighbour):
                                edge_data = self._G.edges[node_id, neighbour]
                                path_edges.append(
                                    GraphEdge(
                                        source_id=node_id,
                                        target_id=neighbour,
                                        type=edge_data.get("edge_type", "related"),
                                        props={
                                            k: v
                                            for k, v in edge_data.items()
                                            if k != "edge_type"
                                        },
                                    )
                                )
                            else:
                                edge_data = self._G.edges[neighbour, node_id]
                                path_edges.append(
                                    GraphEdge(
                                        source_id=neighbour,
                                        target_id=node_id,
                                        type=edge_data.get("edge_type", "related"),
                                        props={
                                            k: v
                                            for k, v in edge_data.items()
                                            if k != "edge_type"
                                        },
                                    )
                                )
                            path_nodes.append(self._make_node(neighbour))

                    current_layer = next_layer
                    if not current_layer:
                        break

                if len(path_nodes) > 1:
                    paths.append(GraphPath(nodes=path_nodes, edges=path_edges))

            return paths

    async def find_nodes(
        self,
        names: list[str],
        fuzzy: bool = True,
    ) -> list[GraphNode]:
        """Find nodes by exact ID match, then case-insensitive contains match."""
        async with self._lock:
            found: list[GraphNode] = []
            found_ids: set[str] = set()

            # Exact match first
            for name in names:
                if name in self._G and name not in found_ids:
                    found.append(self._make_node(name))
                    found_ids.add(name)

            if fuzzy:
                name_lower = [n.lower() for n in names]
                for node_id in self._G.nodes:
                    if node_id in found_ids:
                        continue
                    node_data = self._G.nodes[node_id]
                    node_name = str(node_data.get("name", node_id)).lower()
                    if any(q in node_name or q in node_id.lower() for q in name_lower):
                        found.append(self._make_node(node_id))
                        found_ids.add(node_id)

            return found

    async def remove_nodes_by_prop(self, key: str, value: Any) -> int:
        """Remove every node whose property *key* equals *value*.

        Incident edges are removed with the nodes (networkx semantics).
        Used by the code-graph indexer to drop a file's stale subgraph before
        re-indexing. Returns the number of nodes removed.
        """
        async with self._lock:
            doomed = [
                node_id
                for node_id, attrs in self._G.nodes(data=True)
                if attrs.get(key) == value
            ]
            self._G.remove_nodes_from(doomed)
            if doomed:
                await self._persist()
            return len(doomed)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self) -> None:
        if self._persist_path is None:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_to_file, self._persist_path)
        except Exception as exc:
            logger.warning("Graph persist failed: %s", exc)

    def _save_to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {"id": nid, **attrs}
                for nid, attrs in self._G.nodes(data=True)
            ],
            "edges": [
                {"src": src, "tgt": tgt, **attrs}
                for src, tgt, attrs in self._G.edges(data=True)
            ],
        }
        # Crash-safe atomic write: write to a unique temp file in the same dir,
        # flush + fsync, then os.replace(). A unique name (not a fixed *.tmp)
        # avoids racing concurrent saves and avoids matching the workspace
        # cleanup *.tmp pattern that could delete an in-flight file.
        payload = json.dumps(data, indent=2, default=str)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f"{path.name}.", suffix=".swap"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _load_from_file(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for node in data.get("nodes", []):
                nid = node.pop("id")
                self._G.add_node(nid, **node)
            for edge in data.get("edges", []):
                src = edge.pop("src")
                tgt = edge.pop("tgt")
                self._G.add_edge(src, tgt, **edge)
            logger.info("Loaded graph from %s (%d nodes)", path, len(self._G.nodes))
        except Exception as exc:
            logger.warning("Failed to load graph from %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_node(self, node_id: str) -> GraphNode:
        data = dict(self._G.nodes[node_id])
        node_type = data.pop("node_type", "unknown")
        return GraphNode(id=node_id, type=node_type, props=data)


# ---------------------------------------------------------------------------
# Neo4j implementation (production)
# ---------------------------------------------------------------------------


class Neo4jGraphMemory:
    """
    Graph memory backed by a Neo4j database (async driver).

    All operations use the async Neo4j Python driver with automatic retry on
    ServiceUnavailable.
    """

    _MAX_RETRIES = 3

    def __init__(
        self,
        url: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "harnesspassword",
    ) -> None:
        self._url = url
        self._user = user
        self._password = password
        self._driver: Any = None

    async def _get_driver(self) -> Any:
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "neo4j package is required. Install with: pip install neo4j"
                ) from exc
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self._url,
                    auth=(self._user, self._password),
                    max_connection_pool_size=50,
                )
            except Exception as exc:
                raise HarnessError(
                    f"Neo4j driver creation failed: {exc}",
                    failure_class=FailureClass.MEMORY_GRAPH,
                ) from exc
        return self._driver

    async def _run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[Any]:
        """Execute a Cypher query and return rows as dicts (``result.data()``).

        Note: ``data()`` serialises graph entities (nodes, relationships,
        paths) into plain dicts/lists — labels and Path structure are lost.
        Use :meth:`_run_query_values` when the caller needs real graph objects.
        """
        return await self._run(cypher, params, values=False)

    async def _run_query_values(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """Execute a Cypher query and return raw value rows (``result.values()``).

        Preserves neo4j graph objects (Node, Relationship, Path) so callers
        like :meth:`traverse` can walk real paths — ``result.data()`` would
        flatten them into dicts and drop labels/structure.
        """
        return await self._run(cypher, params, values=True)

    async def _run(
        self, cypher: str, params: dict[str, Any] | None, values: bool
    ) -> list[Any]:
        """Shared retry loop for Cypher execution."""
        driver = await self._get_driver()
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                async with driver.session() as session:
                    result = await session.run(cypher, parameters=params or {})
                    if values:
                        return list(await result.values())
                    return list(await result.data())
            except Exception as exc:
                if "ServiceUnavailable" in type(exc).__name__ and attempt < self._MAX_RETRIES - 1:
                    wait = 0.5 * (attempt + 1)
                    logger.warning(
                        "Neo4j ServiceUnavailable (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        self._MAX_RETRIES,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    last_exc = exc
                else:
                    raise HarnessError(
                        f"Neo4j query failed: {exc}",
                        failure_class=FailureClass.MEMORY_GRAPH,
                    ) from exc
        raise HarnessError(
            f"Neo4j query failed after {self._MAX_RETRIES} retries: {last_exc}",
            failure_class=FailureClass.MEMORY_GRAPH,
        )

    # ------------------------------------------------------------------
    # GraphStore protocol
    # ------------------------------------------------------------------

    async def add_node(self, id: str, type: str, props: dict[str, Any]) -> None:
        label = _safe_label(type, "label")
        cypher = (
            f"MERGE (n:{label} {{id: $id}}) "
            "SET n += $properties"
        )
        await self._run_query(cypher, {"id": id, "properties": {**props, "id": id}})

    async def add_edge(
        self,
        src: str,
        tgt: str,
        type: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        rel_type = _safe_label(type, "relationship type")
        cypher = (
            "MATCH (a {id: $src_id}) "
            "MATCH (b {id: $tgt_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += $properties"
        )
        await self._run_query(
            cypher,
            {
                "src_id": src,
                "tgt_id": tgt,
                "properties": props or {},
            },
        )

    async def traverse(
        self,
        start_ids: list[str],
        max_hops: int = 2,
    ) -> list[GraphPath]:
        """Multi-hop traversal via Cypher variable-length path query."""
        cypher = (
            f"MATCH path = (start)-[*1..{max_hops}]-(end) "
            "WHERE start.id IN $ids "
            "RETURN path "
            "LIMIT 200"
        )
        # values() keeps real neo4j Path objects; data() would flatten them
        # into dicts and _convert_neo4j_path would silently drop every path.
        rows = await self._run_query_values(cypher, {"ids": start_ids})

        paths: list[GraphPath] = []
        for row in rows:
            path_data = row[0] if row else None
            if path_data is None:
                continue
            try:
                gp = self._convert_neo4j_path(path_data)
                if gp:
                    paths.append(gp)
            except Exception as exc:
                logger.debug("Path conversion error: %s", exc)

        return paths

    async def find_nodes(
        self,
        names: list[str],
        fuzzy: bool = True,
    ) -> list[GraphNode]:
        # Return labels explicitly — result.data() serialises nodes to plain
        # property dicts, so labels would otherwise be lost ("unknown").
        if fuzzy and names:
            q = names[0]
            cypher = (
                "MATCH (n) WHERE n.id IN $ids "
                "OR toLower(n.name) CONTAINS toLower($q) "
                "RETURN n, labels(n) AS labels LIMIT 50"
            )
            rows = await self._run_query(cypher, {"ids": names, "q": q})
        else:
            cypher = "MATCH (n) WHERE n.id IN $ids RETURN n, labels(n) AS labels"
            rows = await self._run_query(cypher, {"ids": names})

        nodes: list[GraphNode] = []
        for row in rows:
            props = dict(row.get("n", {}))
            node_id = props.pop("id", "")
            labels = row.get("labels") or []
            node_type = labels[0] if labels else "unknown"
            nodes.append(GraphNode(id=node_id, type=node_type, props=props))
        return nodes

    async def remove_nodes_by_prop(self, key: str, value: Any) -> int:
        """Remove every node whose property *key* equals *value* (DETACH DELETE).

        Property access is dynamic (``n[$key]``) so both key and value stay
        parameterised — no Cypher injection surface. Returns nodes removed.
        """
        cypher = (
            "MATCH (n) WHERE n[$key] = $value "
            "DETACH DELETE n "
            "RETURN count(n) AS removed"
        )
        rows = await self._run_query(cypher, {"key": key, "value": value})
        return int(rows[0].get("removed", 0)) if rows else 0

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_neo4j_path(self, path: Any) -> GraphPath | None:
        """Convert a Neo4j Path object to our GraphPath dataclass."""
        try:
            nodes: list[GraphNode] = []
            edges: list[GraphEdge] = []

            for node in path.nodes:
                props = dict(node)
                node_id = props.pop("id", str(node.id))
                node_type = list(node.labels)[0] if node.labels else "unknown"
                nodes.append(GraphNode(id=node_id, type=node_type, props=props))

            for rel in path.relationships:
                src_props = dict(rel.start_node)
                tgt_props = dict(rel.end_node)
                src_id = src_props.get("id", str(rel.start_node.id))
                tgt_id = tgt_props.get("id", str(rel.end_node.id))
                edges.append(
                    GraphEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        type=rel.type,
                        props=dict(rel),
                    )
                )

            return GraphPath(nodes=nodes, edges=edges) if nodes else None
        except Exception as exc:
            logger.debug("Neo4j path conversion failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_graph_memory(config: Any) -> Any:
    """
    Return the appropriate GraphMemory instance based on config.graph_backend.

    Returns NetworkXGraphMemory for "networkx" (default/dev),
    Neo4jGraphMemory for "neo4j" (production).
    """
    backend = getattr(config, "graph_backend", "networkx")

    if backend == "neo4j":
        return Neo4jGraphMemory(
            url=config.neo4j_url,
            user=config.neo4j_user,
            password=config.neo4j_password,
        )

    # Default: networkx
    workspace_base = getattr(config, "workspace_base_path", "/workspaces")
    persist_path = Path(workspace_base) / "graph_state.json"
    return NetworkXGraphMemory(persist_path=persist_path)
