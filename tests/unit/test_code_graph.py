"""Tests for the code knowledge graph: indexer, incremental re-index,
symbol resolution, chunking, CodeGraphRAG retrieval, and agent tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from harness.core.protocols import VectorHit
from harness.memory.code_graph import (
    CodeGraphIndexer,
    build_symbol_chunks,
    file_node_id,
    symbol_node_id,
)
from harness.memory.code_graph_rag import CodeGraphRAG
from harness.memory.graph import NetworkXGraphMemory
from harness.tools.code_graph_tools import (
    ExpandCodeSymbolTool,
    SearchCodeGraphTool,
    build_code_graph_tools,
)

# ===========================================================================
# Sample repository fixture
# ===========================================================================

_UTILS = '''"""Utility helpers."""


def validate(item: dict, strict: bool = False) -> bool:
    """Check an item for validity."""
    return bool(item) or strict


def helper() -> bool:
    """Validate an empty item."""
    return validate({})
'''

_MODELS = '''"""Data models."""

from app.utils import validate


class Base:
    """Base model."""

    def save(self) -> None:
        """Persist the model."""
        validate({})


class User(Base):
    """A user model."""

    async def rename(self, new_name: str) -> str:
        """Rename the user."""
        self.save()
        return new_name
'''

_MAIN = '''"""Entry point."""

from app import utils
from app.models import User


def run(count: int = 1) -> None:
    """Run the application."""
    user = User()
    utils.helper()
'''


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "utils.py").write_text(_UTILS)
    (tmp_path / "app" / "models.py").write_text(_MODELS)
    (tmp_path / "app" / "main.py").write_text(_MAIN)
    return tmp_path


@pytest.fixture
def graph():
    return NetworkXGraphMemory(persist_path=None)


class FakeVectorStore:
    """Minimal VectorStore double: naive word-overlap ranking."""

    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, dict]] = {}

    async def upsert(self, id, text, metadata, embedding=None):
        self.docs[id] = (text, metadata)

    async def query(self, text, k=5, filter=None, hybrid_alpha=None):
        words = set(text.lower().split())
        scored = []
        for doc_id, (doc_text, metadata) in self.docs.items():
            overlap = len(words & set(doc_text.lower().split()))
            scored.append((overlap, doc_id, doc_text, metadata))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            VectorHit(id=doc_id, text=doc_text, score=float(overlap), metadata=meta)
            for overlap, doc_id, doc_text, meta in scored[:k]
        ]

    async def delete(self, id):
        self.docs.pop(id, None)

    async def count(self, filter=None):
        return len(self.docs)


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.tenant_id = ""
    ctx.run_id = "run-1"
    return ctx


# ===========================================================================
# Indexing
# ===========================================================================

async def test_index_repo_counts(sample_repo, graph):
    stats = await CodeGraphIndexer(graph).index_repo(sample_repo)
    assert stats.files_seen == 4
    assert stats.files_indexed == 4
    assert stats.files_failed_parse == 0
    # utils: validate, helper | models: Base, save, User, rename | main: run
    assert stats.symbols_indexed == 7
    assert stats.calls_unresolved == 1  # bool() builtin
    assert stats.calls_resolved == 5
    assert stats.duration_ms > 0


async def test_index_creates_expected_nodes(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    nodes = await graph.find_nodes(
        [
            file_node_id("app/models.py"),
            symbol_node_id("app/models.py", "User"),
            symbol_node_id("app/models.py", "User.rename"),
            symbol_node_id("app/utils.py", "validate"),
        ],
        fuzzy=False,
    )
    assert len(nodes) == 4
    by_id = {n.id: n for n in nodes}
    user = by_id[symbol_node_id("app/models.py", "User")]
    assert user.type == "CodeClass"
    assert user.props["signature"] == "class User(Base)"
    assert user.props["docstring"] == "A user model."
    rename = by_id[symbol_node_id("app/models.py", "User.rename")]
    assert rename.type == "CodeMethod"
    assert rename.props["signature"] == "async def rename(self, new_name: str) -> str"


async def test_index_resolves_cross_file_and_inherited_calls(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    edges = [
        (src, tgt, data["edge_type"])
        for src, tgt, data in graph._G.edges(data=True)
    ]
    # from-import call: Base.save → utils.validate
    assert (
        symbol_node_id("app/models.py", "Base.save"),
        symbol_node_id("app/utils.py", "validate"),
        "calls",
    ) in edges
    # inherited method: User.rename → self.save() resolves to Base.save
    assert (
        symbol_node_id("app/models.py", "User.rename"),
        symbol_node_id("app/models.py", "Base.save"),
        "calls",
    ) in edges
    # submodule from-import: run → utils.helper()
    assert (
        symbol_node_id("app/main.py", "run"),
        symbol_node_id("app/utils.py", "helper"),
        "calls",
    ) in edges
    # inheritance edge
    assert (
        symbol_node_id("app/models.py", "User"),
        symbol_node_id("app/models.py", "Base"),
        "inherits",
    ) in edges
    # import edge between files
    assert (
        file_node_id("app/models.py"),
        file_node_id("app/utils.py"),
        "imports",
    ) in edges


async def test_traverse_from_symbol_reaches_related(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    paths = await graph.traverse(
        [symbol_node_id("app/models.py", "User")], max_hops=2
    )
    assert paths
    node_ids = {n.id for p in paths for n in p.nodes}
    assert symbol_node_id("app/models.py", "Base") in node_ids


# ===========================================================================
# Incremental re-index
# ===========================================================================

async def test_reindex_skips_unchanged(sample_repo, graph):
    indexer = CodeGraphIndexer(graph)
    await indexer.index_repo(sample_repo)
    stats = await indexer.index_repo(sample_repo)
    assert stats.files_indexed == 0
    assert stats.files_skipped_unchanged == 4


async def test_reindex_force_reindexes_all(sample_repo, graph):
    indexer = CodeGraphIndexer(graph)
    await indexer.index_repo(sample_repo)
    stats = await indexer.index_repo(sample_repo, force=True)
    assert stats.files_indexed == 4


async def test_reindex_changed_file_removes_stale_symbols(sample_repo, graph):
    indexer = CodeGraphIndexer(graph)
    await indexer.index_repo(sample_repo)

    # Rename helper → helper_two
    (sample_repo / "app" / "utils.py").write_text(
        _UTILS.replace("def helper()", "def helper_two()")
    )
    stats = await indexer.index_repo(sample_repo)
    assert stats.files_indexed == 1
    assert stats.files_skipped_unchanged == 3

    stale = await graph.find_nodes(
        [symbol_node_id("app/utils.py", "helper")], fuzzy=False
    )
    fresh = await graph.find_nodes(
        [symbol_node_id("app/utils.py", "helper_two")], fuzzy=False
    )
    assert stale == []
    assert len(fresh) == 1


async def test_parse_error_recorded_not_fatal(sample_repo, graph):
    (sample_repo / "app" / "broken.py").write_text("def broken(:\n")
    stats = await CodeGraphIndexer(graph).index_repo(sample_repo)
    assert stats.files_failed_parse == 1
    assert "app/broken.py" in stats.parse_errors
    assert stats.files_indexed == 4  # the rest still indexed


# ===========================================================================
# remove_nodes_by_prop (NetworkX backend)
# ===========================================================================

async def test_remove_nodes_by_prop(graph):
    await graph.add_node("a", "T", {"file": "x.py"})
    await graph.add_node("b", "T", {"file": "x.py"})
    await graph.add_node("c", "T", {"file": "y.py"})
    await graph.add_edge("a", "c", "rel")
    removed = await graph.remove_nodes_by_prop("file", "x.py")
    assert removed == 2
    remaining = await graph.find_nodes(["a", "b", "c"], fuzzy=False)
    assert [n.id for n in remaining] == ["c"]


# ===========================================================================
# Code-aware chunking + vector integration
# ===========================================================================

async def test_symbol_chunks_top_level_only(sample_repo, graph):
    from harness.ingestion.code_extractor import extract_code_facts
    from harness.ingestion.code_loader import load_source_files

    (models,) = [
        s for s in load_source_files(sample_repo) if s.path == "app/models.py"
    ]
    chunks = build_symbol_chunks(models, extract_code_facts(models))
    names = {c.metadata["qualified_name"] for c in chunks}
    assert names == {"Base", "User"}  # methods ride inside their class chunk
    user_chunk = next(c for c in chunks if c.metadata["qualified_name"] == "User")
    assert "async def rename" in user_chunk.content       # body included
    assert "# class User(Base)" in user_chunk.content      # signature header
    assert user_chunk.chunk_id == symbol_node_id("app/models.py", "User")
    assert user_chunk.metadata["symbol_id"] == user_chunk.chunk_id


async def test_index_embeds_chunks_into_vector_store(sample_repo, graph):
    store = FakeVectorStore()
    stats = await CodeGraphIndexer(graph, vector_store=store).index_repo(sample_repo)
    # top-level: validate, helper, Base, User, run
    assert stats.chunks_embedded == 5
    assert await store.count() == 5
    assert symbol_node_id("app/models.py", "User") in store.docs


# ===========================================================================
# CodeGraphRAG retrieval
# ===========================================================================

async def test_retrieve_signatures_first(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    rag = CodeGraphRAG(graph, repo_root=sample_repo)
    result = await rag.retrieve("How does User.rename work?")

    assert result.strategy == "graph_primary"
    assert result.graph_paths
    context = result.graph_context
    assert "[SYMBOLS]" in context
    assert "async def rename(self, new_name: str) -> str" in context
    assert "[CALL GRAPH]" in context
    assert "User.rename --calls--> Base.save" in context
    assert "[INHERITANCE]" in context
    assert "User --inherits--> Base" in context
    # Signatures-first: bodies never rendered
    assert "return new_name" not in context


async def test_retrieve_vector_bridge_when_no_anchors(sample_repo, graph):
    store = FakeVectorStore()
    await CodeGraphIndexer(graph, vector_store=store).index_repo(sample_repo)
    rag = CodeGraphRAG(graph, vector_store=store, repo_root=sample_repo)
    # No code-shaped identifiers and no graph name matches → vector bridge
    result = await rag.retrieve("zzz qqq nonexistent words")
    assert result.vector_hits
    assert result.graph_context  # bridged into the graph via symbol_id metadata
    assert result.strategy in ("hybrid", "vector_fallback")


async def test_retrieve_unknown_symbol_empty(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    rag = CodeGraphRAG(graph)
    result = await rag.retrieve("`TotallyMissingClass.method`")
    assert result.graph_paths == []
    assert result.graph_context == ""


async def test_expand_symbol_returns_source(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    rag = CodeGraphRAG(graph, repo_root=sample_repo)
    source = await rag.expand_symbol(symbol_node_id("app/models.py", "User"))
    assert source is not None
    assert source.startswith("class User(Base):")
    assert "async def rename" in source


async def test_expand_symbol_rejects_bad_ids(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    rag = CodeGraphRAG(graph, repo_root=sample_repo)
    assert await rag.expand_symbol("not-a-symbol-id") is None
    assert await rag.expand_symbol("code:sym:../../etc/passwd::x") is None
    assert await rag.expand_symbol("code:sym:app/missing.py::x") is None


async def test_expand_symbol_requires_repo_root(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    rag = CodeGraphRAG(graph)  # no repo_root
    assert await rag.expand_symbol(symbol_node_id("app/models.py", "User")) is None


# ===========================================================================
# Agent tools
# ===========================================================================

async def test_search_code_graph_tool(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    tool = SearchCodeGraphTool(CodeGraphRAG(graph, repo_root=sample_repo))
    result = await tool.execute(_ctx(), {"query": "who calls `validate`"})
    assert not result.is_error
    assert "[SYMBOLS]" in result.data
    assert "def validate" in result.data
    assert result.metadata["graph_paths"] >= 1
    assert result.metadata["strategy"] == "graph_primary"


async def test_search_tool_no_match_message(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    tool = SearchCodeGraphTool(CodeGraphRAG(graph))
    result = await tool.execute(_ctx(), {"query": "`NoSuchThing.at_all`"})
    assert not result.is_error
    assert "No indexed code matched" in result.data


async def test_expand_code_symbol_tool(sample_repo, graph):
    await CodeGraphIndexer(graph).index_repo(sample_repo)
    tool = ExpandCodeSymbolTool(CodeGraphRAG(graph, repo_root=sample_repo))
    ok = await tool.execute(
        _ctx(), {"symbol_id": symbol_node_id("app/utils.py", "validate")}
    )
    assert not ok.is_error
    assert ok.data.startswith("def validate")

    bad = await tool.execute(_ctx(), {"symbol_id": "code:sym:app/utils.py::gone"})
    assert bad.is_error


def test_build_code_graph_tools_names():
    tools = build_code_graph_tools(MagicMock())
    assert [t.name for t in tools] == ["search_code_graph", "expand_code_symbol"]
    for tool in tools:
        assert tool.description
        assert tool.input_schema["type"] == "object"
        assert tool.timeout_seconds > 0
