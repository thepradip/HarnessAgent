"""Integration test: index the harness repo's own source into a code graph.

No external services required — NetworkX backend, stdlib ast parser, real
files from src/. Verifies the full pipeline end-to-end: load → extract →
graph → retrieve → expand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.memory.code_graph import CodeGraphIndexer, symbol_node_id
from harness.memory.code_graph_rag import CodeGraphRAG
from harness.memory.graph import NetworkXGraphMemory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"


@pytest.fixture
async def indexed():
    """Index src/harness for real (~100 files, stdlib ast — a few seconds)."""
    graph = NetworkXGraphMemory(persist_path=None)
    indexer = CodeGraphIndexer(graph)
    stats = await indexer.index_repo(_SRC)
    return graph, indexer, stats


async def test_indexes_real_repo(indexed):
    _graph, _indexer, stats = indexed
    assert stats.files_seen > 80
    assert stats.files_indexed == stats.files_seen - stats.files_failed_parse
    assert stats.files_failed_parse == 0, stats.parse_errors
    assert stats.symbols_indexed > 500
    assert stats.calls_resolved > 200
    assert stats.edges_added > 1000


async def test_real_symbols_present(indexed):
    graph, _indexer, _stats = indexed
    nodes = await graph.find_nodes(
        [
            symbol_node_id("harness/memory/graph_rag.py", "GraphRAGEngine"),
            symbol_node_id("harness/memory/graph.py", "NetworkXGraphMemory.add_node"),
            symbol_node_id("harness/memory/code_graph.py", "CodeGraphIndexer"),
        ],
        fuzzy=False,
    )
    assert len(nodes) == 3


async def test_real_call_edge_resolved(indexed):
    graph, _indexer, _stats = indexed
    # GraphRAGEngine.retrieve calls extract_entities (from-import, cross-file)
    edges = [
        (src, tgt)
        for src, tgt, data in graph._G.edges(data=True)
        if data.get("edge_type") == "calls"
    ]
    assert (
        symbol_node_id("harness/memory/graph_rag.py", "GraphRAGEngine.retrieve"),
        symbol_node_id("harness/memory/entity_extractor.py", "extract_entities"),
    ) in edges


async def test_retrieval_end_to_end(indexed):
    graph, _indexer, _stats = indexed
    rag = CodeGraphRAG(graph, repo_root=_SRC)
    result = await rag.retrieve("who calls extract_entities?")
    assert result.graph_paths
    assert "def extract_entities" in result.graph_context
    assert "[CALL GRAPH]" in result.graph_context
    assert "extract_entities" in result.graph_context

    # Expand-on-demand returns the real source
    source = await rag.expand_symbol(
        symbol_node_id("harness/memory/entity_extractor.py", "extract_entities")
    )
    assert source is not None
    assert source.lstrip().startswith("async def extract_entities")


async def test_incremental_reindex_all_unchanged(indexed):
    _graph, indexer, stats = indexed
    stats2 = await indexer.index_repo(_SRC)
    assert stats2.files_indexed == 0
    assert stats2.files_skipped_unchanged == stats.files_indexed
