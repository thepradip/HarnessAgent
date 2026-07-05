"""End-to-end code knowledge graph demo — no API keys, no services.

Builds a tiny sample project in a temp directory, indexes it into an
in-process NetworkX graph, then walks through the full agent workflow:

  1. index_repo         — structural extraction into the graph
  2. search_code_graph  — signatures-first retrieval (the token saver)
  3. expand_code_symbol — full source for exactly one symbol
  4. incremental re-index after editing a file

Run:  python examples/code_graph_demo.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from harness.memory.code_graph import CodeGraphIndexer, symbol_node_id
from harness.memory.code_graph_rag import CodeGraphRAG
from harness.memory.graph import NetworkXGraphMemory
from harness.tools.code_graph_tools import ExpandCodeSymbolTool, SearchCodeGraphTool

_UTILS = '''"""Validation helpers."""


def validate(item: dict, strict: bool = False) -> bool:
    """Check an item for validity."""
    return bool(item) or strict
'''

_MODELS = '''"""Data models."""

from app.utils import validate


class Base:
    """Base model with persistence."""

    def save(self) -> None:
        """Validate then persist the model."""
        validate({})


class User(Base):
    """A user account."""

    async def rename(self, new_name: str) -> str:
        """Rename the user and persist."""
        self.save()
        return new_name
'''


def make_sample_repo(root: Path) -> None:
    app = root / "app"
    app.mkdir(parents=True)
    (app / "__init__.py").write_text("")
    (app / "utils.py").write_text(_UTILS)
    (app / "models.py").write_text(_MODELS)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        make_sample_repo(repo)

        # ------------------------------------------------------------------
        # 1. Index the repository
        # ------------------------------------------------------------------
        graph = NetworkXGraphMemory()  # swap for Neo4jGraphMemory in production
        indexer = CodeGraphIndexer(graph)
        stats = await indexer.index_repo(repo)
        print("=== 1. index_repo ===")
        print(
            f"files={stats.files_indexed}  symbols={stats.symbols_indexed}  "
            f"edges={stats.edges_added}  calls resolved={stats.calls_resolved}  "
            f"({stats.duration_ms:.0f} ms)\n"
        )

        # ------------------------------------------------------------------
        # 2. Search — what an agent sees instead of file dumps
        # ------------------------------------------------------------------
        rag = CodeGraphRAG(graph, repo_root=repo)
        search = SearchCodeGraphTool(rag)
        ctx = SimpleNamespace(tenant_id="", run_id="demo")  # stands in for AgentContext

        result = await search.execute(ctx, {"query": "How does User.rename work?"})
        print("=== 2. search_code_graph('How does User.rename work?') ===")
        print(result.data)
        print(f"(strategy={result.metadata['strategy']}, "
              f"~{result.metadata['tokens_estimate']} tokens)\n")

        # ------------------------------------------------------------------
        # 3. Expand exactly one symbol (token saver: only what's needed)
        # ------------------------------------------------------------------
        expand = ExpandCodeSymbolTool(rag)
        symbol_id = symbol_node_id("app/models.py", "User")
        result = await expand.execute(ctx, {"symbol_id": symbol_id})
        print(f"=== 3. expand_code_symbol('{symbol_id}') ===")
        print(result.data)
        print()

        # ------------------------------------------------------------------
        # 4. Incremental re-index — edit one file, only it re-indexes
        # ------------------------------------------------------------------
        (repo / "app" / "utils.py").write_text(
            _UTILS + '\n\ndef sanitize(text: str) -> str:\n    """Strip text."""\n    return text.strip()\n'
        )
        stats = await indexer.index_repo(repo)
        print("=== 4. incremental re-index after editing app/utils.py ===")
        print(
            f"re-indexed={stats.files_indexed}  "
            f"skipped unchanged={stats.files_skipped_unchanged}"
        )
        (fresh,) = await graph.find_nodes(
            [symbol_node_id("app/utils.py", "sanitize")], fuzzy=False
        )
        print(f"new symbol visible: {fresh.props['signature']}")


if __name__ == "__main__":
    asyncio.run(main())
