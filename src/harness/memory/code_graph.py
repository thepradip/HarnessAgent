"""Code knowledge graph: index source-code structure into graph memory.

Populates any :class:`harness.core.protocols.GraphStore` backend (NetworkX or
Neo4j) with code structure — files, classes, functions, methods — and the
relations between them (contains, defines, imports, calls, inherits), mirroring
how :meth:`GraphRAGEngine.populate_schema` loads SQL schemas.

Node types:   CodeFile, CodeClass, CodeFunction, CodeMethod, CodeModule
Edge types:   contains, imports, calls, inherits

Node IDs are deterministic so re-indexing is idempotent:
    code:file:<rel_path>
    code:sym:<rel_path>::<qualified_name>
    code:mod:<module_name>            (external modules, optional)

Incremental indexing: every CodeFile node stores the file's SHA-256 content
hash. On re-index, unchanged files are skipped; changed files have their old
subgraph removed (when the backend supports ``remove_nodes_by_prop``) before
fresh facts are written, so deleted symbols do not linger.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.core.protocols import GraphStore
from harness.ingestion.chunker import Chunk
from harness.ingestion.code_extractor import (
    CodeFileFacts,
    extract_code_facts,
)
from harness.ingestion.code_loader import SourceFile, load_source_files

logger = logging.getLogger(__name__)

# Edge type weights — mirrors the SQL graph weights in graph_rag.py.
CODE_EDGE_WEIGHTS: dict[str, float] = {
    "calls":    1.5,   # call relationships are the most valuable signal
    "inherits": 1.4,   # subclassing is nearly as informative
    "imports":  1.0,   # module dependency
    "contains": 0.8,   # structural containment (file→symbol, class→method)
}

_SYMBOL_NODE_TYPES: dict[str, str] = {
    "class": "CodeClass",
    "function": "CodeFunction",
    "method": "CodeMethod",
}

# Cap per-symbol chunk content so a giant class cannot blow the embed budget.
_CHUNK_MAX_CHARS = 4_000


def file_node_id(rel_path: str) -> str:
    return f"code:file:{rel_path}"


def symbol_node_id(rel_path: str, qualified_name: str) -> str:
    return f"code:sym:{rel_path}::{qualified_name}"


def module_node_id(module: str) -> str:
    return f"code:mod:{module}"


def _parse_symbol_id(node_id: str) -> tuple[str, str] | None:
    """Split ``code:sym:<rel_path>::<qualname>`` into (rel_path, qualname)."""
    if not node_id.startswith("code:sym:"):
        return None
    remainder = node_id[len("code:sym:") :]
    rel_path, sep, qualname = remainder.partition("::")
    if not sep:
        return None
    return rel_path, qualname


@dataclass
class CodeIndexStats:
    """Result of a CodeGraphIndexer.index_repo run."""

    files_seen: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_failed_parse: int = 0
    symbols_indexed: int = 0
    edges_added: int = 0
    calls_resolved: int = 0
    calls_unresolved: int = 0
    chunks_embedded: int = 0
    duration_ms: float = 0.0
    parse_errors: dict[str, str] = field(default_factory=dict)


class _SymbolResolver:
    """Resolves raw callee / base-class names to symbol node IDs.

    Built once per index run from ALL parsed files (including unchanged ones)
    so cross-file references resolve regardless of which files changed.
    """

    def __init__(self, all_facts: list[CodeFileFacts]) -> None:
        # qualified name within its file → node id
        self._by_file_qual: dict[tuple[str, str], str] = {}
        # simple name → node ids (for unique-match fallback)
        self._by_simple: dict[str, list[str]] = {}
        # project module name → file path (e.g. "src.harness.memory.graph")
        self._module_to_file: dict[str, str] = {}
        # symbol kind lookup for preferring classes on inheritance edges
        self._kind: dict[str, str] = {}
        # per-file: parent map (qualified name → parent qualified name)
        self._parent: dict[tuple[str, str], str] = {}
        # per-file: class → base names as written (for inherited-method lookup)
        self._bases: dict[tuple[str, str], list[str]] = {}
        # per-file import maps
        self._from_imports: dict[str, dict[str, str]] = {}   # file → {name: module}
        self._module_aliases: dict[str, dict[str, str]] = {} # file → {alias: module}

        for facts in all_facts:
            self._module_to_file[facts.module_name] = facts.file_path
            for symbol in facts.symbols:
                node_id = symbol_node_id(facts.file_path, symbol.qualified_name)
                self._by_file_qual[(facts.file_path, symbol.qualified_name)] = node_id
                self._by_simple.setdefault(symbol.name, []).append(node_id)
                self._kind[node_id] = symbol.kind
                self._parent[(facts.file_path, symbol.qualified_name)] = symbol.parent
            for inh in facts.inherits:
                self._bases.setdefault(
                    (facts.file_path, inh.class_name), []
                ).append(inh.base_name)

            from_map: dict[str, str] = {}
            alias_map: dict[str, str] = {}
            for imp in facts.imports:
                module = self._absolute_module(imp.module, imp.level, facts.module_name)
                if imp.names:
                    for name in imp.names:
                        from_map[name] = module
                elif imp.alias:
                    alias_map[imp.alias] = module
                else:
                    # "import a.b.c" binds the top-level name "a"
                    alias_map[imp.module.split(".")[0]] = imp.module.split(".")[0]
            self._from_imports[facts.file_path] = from_map
            self._module_aliases[facts.file_path] = alias_map

    # ------------------------------------------------------------------

    @staticmethod
    def _absolute_module(module: str, level: int, current_module: str) -> str:
        """Resolve a possibly-relative import to a dotted module path."""
        if level == 0:
            return module
        parts = current_module.split(".")
        # level 1 = current package, 2 = parent, … (current_module includes
        # the file's own module segment, hence the extra -1).
        keep = len(parts) - level
        if keep < 0:
            keep = 0
        base = parts[:keep]
        if module:
            base.append(module)
        return ".".join(base)

    def resolve_module_file(self, module: str) -> str | None:
        """Map an imported module name to an indexed file path (suffix-aware).

        Handles src-layout repos where the import path ("harness.memory.graph")
        is a suffix of the repo module path ("src.harness.memory.graph").
        """
        if not module:
            return None
        if module in self._module_to_file:
            return self._module_to_file[module]
        suffix = f".{module}"
        matches = [
            path
            for mod, path in self._module_to_file.items()
            if mod.endswith(suffix)
        ]
        return matches[0] if len(matches) == 1 else None

    def resolve_symbol(
        self,
        raw_name: str,
        facts: CodeFileFacts,
        caller: str = "",
        prefer_class: bool = False,
    ) -> str | None:
        """Resolve *raw_name* (as written in code) to a symbol node id."""
        return self._resolve(raw_name, facts.file_path, caller, prefer_class)

    def _resolve(
        self,
        raw_name: str,
        file_path: str,
        caller: str = "",
        prefer_class: bool = False,
    ) -> str | None:
        # self.foo / cls.foo → method on the caller's class or its bases
        if raw_name.startswith(("self.", "cls.")) and caller:
            attr = raw_name.split(".", 1)[1]
            klass = self._enclosing_class(file_path, caller)
            if klass:
                return self._resolve_on_class_chain(file_path, klass, attr)
            return None

        head, _, _rest = raw_name.partition(".")

        # Plain or dotted name defined in this file (e.g. "helper",
        # "ClassName.method")
        node = self._by_file_qual.get((file_path, raw_name))
        if node:
            return node

        from_map = self._from_imports.get(file_path, {})
        if head in from_map:
            # "from x import foo" then foo(...)
            target_file = self.resolve_module_file(from_map[head])
            if target_file:
                node = self._by_file_qual.get((target_file, raw_name))
                if node:
                    return node
            # "from pkg import submodule" then submodule.func(...)
            if "." in raw_name:
                parent_module = from_map[head]
                sub_module = f"{parent_module}.{head}" if parent_module else head
                sub_file = self.resolve_module_file(sub_module)
                if sub_file:
                    node = self._by_file_qual.get(
                        (sub_file, raw_name.split(".", 1)[1])
                    )
                    if node:
                        return node

        # Module alias: "import harness.memory.graph as g" then g.foo(...)
        alias_map = self._module_aliases.get(file_path, {})
        if head in alias_map and "." in raw_name:
            target_file = self.resolve_module_file(alias_map[head])
            if target_file:
                remainder = raw_name.split(".", 1)[1]
                node = self._by_file_qual.get((target_file, remainder))
                if node:
                    return node

        # Unique global simple-name match (last resort — skip ambiguous)
        simple = raw_name.rsplit(".", 1)[-1]
        candidates = self._by_simple.get(simple, [])
        if prefer_class:
            class_candidates = [
                c for c in candidates if self._kind.get(c) == "class"
            ]
            if class_candidates:
                candidates = class_candidates
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _resolve_on_class_chain(
        self, file_path: str, klass: str, attr: str
    ) -> str | None:
        """Look up *attr* on *klass*, walking base classes (MRO-lite)."""
        current_file, current_class = file_path, klass
        for _ in range(8):  # depth guard
            node = self._by_file_qual.get((current_file, f"{current_class}.{attr}"))
            if node:
                return node
            next_hop: tuple[str, str] | None = None
            for base_name in self._bases.get((current_file, current_class), []):
                base_id = self._resolve(base_name, current_file, prefer_class=True)
                if base_id is None:
                    continue
                parsed = _parse_symbol_id(base_id)
                if parsed is not None:
                    next_hop = parsed
                    break
            if next_hop is None:
                return None
            current_file, current_class = next_hop
        return None

    def _enclosing_class(self, file_path: str, qualified: str) -> str | None:
        """Walk up the parent chain from *qualified* to the nearest class."""
        current = qualified
        for _ in range(16):  # cycle guard
            parent = self._parent.get((file_path, current), "")
            if not parent:
                return None
            parent_id = self._by_file_qual.get((file_path, parent))
            if parent_id and self._kind.get(parent_id) == "class":
                return parent
            current = parent
        return None


class CodeGraphIndexer:
    """Indexes a repository's code structure into graph (and vector) memory.

    Usage::

        graph = NetworkXGraphMemory()
        indexer = CodeGraphIndexer(graph)
        stats = await indexer.index_repo("/path/to/repo")

    Pass a ``vector_store`` to also embed one chunk per top-level symbol,
    with metadata linking each chunk back to its graph node — this is what
    lets vector hits anchor graph traversal in CodeGraphRAG.
    """

    def __init__(
        self,
        graph: GraphStore,
        vector_store: Any | None = None,
        parser: str = "auto",
        track_external_modules: bool = False,
    ) -> None:
        self._graph = graph
        self._vector_store = vector_store
        self._parser = parser
        self._track_external = track_external_modules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_repo(
        self,
        root: Path | str,
        languages: set[str] | None = None,
        extra_exclude_dirs: set[str] | None = None,
        exclude_globs: list[str] | None = None,
        tenant_id: str = "",
        force: bool = False,
    ) -> CodeIndexStats:
        """Index (or incrementally re-index) the repository at *root*.

        Args:
            root:               Repository root directory.
            languages:          Restrict to these languages (default: all known).
            extra_exclude_dirs: Extra directory names to skip.
            exclude_globs:      fnmatch patterns to skip (relative paths).
            tenant_id:          Stamped on every node for tenant filtering.
            force:              Re-index every file even if its hash is unchanged.

        Returns:
            CodeIndexStats with real counts from this run.
        """
        started = time.monotonic()
        stats = CodeIndexStats()

        sources = load_source_files(
            root,
            languages=languages,
            extra_exclude_dirs=extra_exclude_dirs,
            exclude_globs=exclude_globs,
        )
        stats.files_seen = len(sources)

        # Parse everything up front (cheap, in-memory) so the resolver sees
        # the whole project even when only a few files changed.
        all_facts: list[CodeFileFacts] = []
        facts_by_path: dict[str, CodeFileFacts] = {}
        for source in sources:
            facts = extract_code_facts(source, parser=self._parser)
            if facts.parse_error and not facts.symbols:
                stats.files_failed_parse += 1
                stats.parse_errors[source.path] = facts.parse_error
                continue
            all_facts.append(facts)
            facts_by_path[source.path] = facts

        resolver = _SymbolResolver(all_facts)
        sources_by_path = {s.path: s for s in sources}

        for facts in all_facts:
            source = sources_by_path[facts.file_path]
            if not force and await self._is_unchanged(source):
                stats.files_skipped_unchanged += 1
                continue
            await self._remove_stale_subgraph(source.path)
            await self._index_file(source, facts, resolver, tenant_id, stats)
            stats.files_indexed += 1

        stats.duration_ms = (time.monotonic() - started) * 1000
        logger.info(
            "Code graph index: %d/%d files indexed (%d unchanged, %d parse "
            "failures), %d symbols, %d edges, %d/%d calls resolved in %.0f ms",
            stats.files_indexed,
            stats.files_seen,
            stats.files_skipped_unchanged,
            stats.files_failed_parse,
            stats.symbols_indexed,
            stats.edges_added,
            stats.calls_resolved,
            stats.calls_resolved + stats.calls_unresolved,
            stats.duration_ms,
        )
        return stats

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    async def _is_unchanged(self, source: SourceFile) -> bool:
        """True when the file node already exists with the same content hash."""
        try:
            existing = await self._graph.find_nodes(
                [file_node_id(source.path)], fuzzy=False
            )
        except Exception as exc:
            logger.warning("Code graph hash lookup failed for %s: %s", source.path, exc)
            return False
        return bool(existing) and existing[0].props.get("content_hash") == source.content_hash

    async def _remove_stale_subgraph(self, rel_path: str) -> None:
        """Remove all nodes previously indexed for *rel_path*.

        Uses the optional ``remove_nodes_by_prop`` backend capability so
        deleted/renamed symbols do not linger. Backends without it fall back
        to MERGE-overwrite semantics (stale symbols persist until a full
        rebuild — logged once at debug level).
        """
        remover = getattr(self._graph, "remove_nodes_by_prop", None)
        if remover is None:
            logger.debug(
                "Graph backend %s lacks remove_nodes_by_prop; stale symbols in "
                "%s will persist until a forced rebuild",
                type(self._graph).__name__,
                rel_path,
            )
            return
        try:
            await remover("file", rel_path)
        except Exception as exc:
            logger.warning("Stale subgraph removal failed for %s: %s", rel_path, exc)

    async def _index_file(
        self,
        source: SourceFile,
        facts: CodeFileFacts,
        resolver: _SymbolResolver,
        tenant_id: str,
        stats: CodeIndexStats,
    ) -> None:
        fid = file_node_id(source.path)
        await self._graph.add_node(
            id=fid,
            type="CodeFile",
            props={
                "name": source.path,
                "file": source.path,
                "module": facts.module_name,
                "language": source.language,
                "docstring": facts.docstring,
                "content_hash": source.content_hash,
                "line_count": source.line_count,
                "tenant_id": tenant_id,
            },
        )

        # Symbol nodes + containment edges
        for symbol in facts.symbols:
            sid = symbol_node_id(source.path, symbol.qualified_name)
            await self._graph.add_node(
                id=sid,
                type=_SYMBOL_NODE_TYPES.get(symbol.kind, "CodeFunction"),
                props={
                    "name": symbol.name,
                    "qualified_name": symbol.qualified_name,
                    "kind": symbol.kind,
                    "file": source.path,
                    "line": symbol.line,
                    "end_line": symbol.end_line,
                    "signature": symbol.signature,
                    "docstring": symbol.docstring,
                    "tenant_id": tenant_id,
                },
            )
            stats.symbols_indexed += 1

            parent_id = (
                symbol_node_id(source.path, symbol.parent) if symbol.parent else fid
            )
            await self._graph.add_edge(
                src=parent_id,
                tgt=sid,
                type="contains",
                props={"weight": CODE_EDGE_WEIGHTS["contains"], "file": source.path},
            )
            stats.edges_added += 1

        # Import edges (file → file, or file → external module)
        for imp in facts.imports:
            module = resolver._absolute_module(imp.module, imp.level, facts.module_name)
            target_file = resolver.resolve_module_file(module)
            if target_file:
                target_id = file_node_id(target_file)
            elif self._track_external and module:
                target_id = module_node_id(module)
                await self._graph.add_node(
                    id=target_id,
                    type="CodeModule",
                    props={"name": module, "external": True, "tenant_id": tenant_id},
                )
            else:
                continue
            await self._graph.add_edge(
                src=fid,
                tgt=target_id,
                type="imports",
                props={
                    "weight": CODE_EDGE_WEIGHTS["imports"],
                    "file": source.path,
                    "names": ", ".join(imp.names[:10]),
                },
            )
            stats.edges_added += 1

        # Inheritance edges (class → base class)
        for inh in facts.inherits:
            src_id = symbol_node_id(source.path, inh.class_name)
            base_id = resolver.resolve_symbol(
                inh.base_name, facts, prefer_class=True
            )
            if base_id is None:
                continue
            await self._graph.add_edge(
                src=src_id,
                tgt=base_id,
                type="inherits",
                props={
                    "weight": CODE_EDGE_WEIGHTS["inherits"],
                    "file": source.path,
                    "base": inh.base_name,
                },
            )
            stats.edges_added += 1

        # Call edges (symbol → symbol)
        for call in facts.calls:
            callee_id = resolver.resolve_symbol(
                call.callee_name, facts, caller=call.caller
            )
            if callee_id is None:
                stats.calls_unresolved += 1
                continue
            caller_id = (
                symbol_node_id(source.path, call.caller) if call.caller else fid
            )
            if caller_id == callee_id:
                continue  # direct recursion adds noise, not signal
            await self._graph.add_edge(
                src=caller_id,
                tgt=callee_id,
                type="calls",
                props={
                    "weight": CODE_EDGE_WEIGHTS["calls"],
                    "file": source.path,
                    "line": call.line,
                },
            )
            stats.calls_resolved += 1
            stats.edges_added += 1

        # Vector chunks — one per top-level symbol, linked to its graph node
        if self._vector_store is not None:
            for chunk in build_symbol_chunks(source, facts, tenant_id=tenant_id):
                try:
                    await self._vector_store.upsert(
                        id=chunk.chunk_id,
                        text=chunk.content,
                        metadata=chunk.metadata,
                    )
                    stats.chunks_embedded += 1
                except Exception as exc:
                    logger.warning(
                        "Vector upsert failed for %s: %s", chunk.chunk_id, exc
                    )


# ---------------------------------------------------------------------------
# Code-aware chunking
# ---------------------------------------------------------------------------


def build_symbol_chunks(
    source: SourceFile,
    facts: CodeFileFacts,
    tenant_id: str = "",
) -> list[Chunk]:
    """Build one embeddable chunk per *top-level* class or function.

    Chunks never split a symbol mid-body. Each chunk carries a header with the
    file path and signature (so the embedding captures identity, not just the
    body) and metadata linking back to the symbol's graph node id.

    Chunk IDs are deterministic (``code:sym:<path>::<qualname>``) so re-indexing
    upserts in place instead of duplicating.
    """
    lines = source.content.splitlines()
    chunks: list[Chunk] = []

    for symbol in facts.symbols:
        if symbol.parent:
            continue  # methods/nested defs ride along inside their parent
        body = "\n".join(lines[symbol.line - 1 : symbol.end_line])
        if len(body) > _CHUNK_MAX_CHARS:
            body = body[:_CHUNK_MAX_CHARS] + "\n# …[truncated]"
        header = f"# {source.path}:{symbol.line}\n# {symbol.signature}\n"
        if symbol.docstring:
            header += f"# {symbol.docstring}\n"
        content = header + body
        sid = symbol_node_id(source.path, symbol.qualified_name)
        chunks.append(
            Chunk(
                chunk_id=sid,
                doc_id=file_node_id(source.path),
                content=content,
                start_char=0,
                end_char=len(content),
                metadata={
                    "symbol_id": sid,
                    "file": source.path,
                    "qualified_name": symbol.qualified_name,
                    "kind": symbol.kind,
                    "language": source.language,
                    "start_line": symbol.line,
                    "end_line": symbol.end_line,
                    "signature": symbol.signature,
                    "tenant_id": tenant_id,
                    "source": "code_graph",
                },
            )
        )
    return chunks
