"""Code-graph retrieval tools for HarnessAgent agents.

Exposes the code knowledge graph (see ``harness.memory.code_graph``) to
agents as two tools:

- ``search_code_graph``  — signatures-first structural retrieval
- ``expand_code_symbol`` — fetch one symbol's full source on demand

Together they implement the token-saving contract: search returns compact
structure (signatures, docstrings, call/inheritance/import edges); the agent
expands only the symbols it actually needs to read.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from harness.core.context import AgentContext, ToolResult

logger = logging.getLogger(__name__)


class SearchCodeGraphTool:
    """Search the code knowledge graph for symbols related to a question."""

    name = "search_code_graph"
    description = (
        "Search the indexed code knowledge graph. Returns a compact structural "
        "view — file paths, symbol signatures with docstrings, call graph, "
        "inheritance, and import edges — for the parts of the codebase related "
        "to the query. Use expand_code_symbol to read a symbol's full source. "
        "Prefer this over reading whole files: it is dramatically cheaper."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language question or symbol name, e.g. "
                    "'who calls LLMRouter.complete' or 'retry logic for providers'."
                ),
            },
            "max_hops": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "maximum": 4,
                "description": "Graph traversal depth from anchor symbols (default 2).",
            },
        },
        "required": ["query"],
    }
    timeout_seconds: float = 30.0

    def __init__(self, code_rag: Any) -> None:
        """*code_rag* is a harness.memory.code_graph_rag.CodeGraphRAG instance."""
        self._code_rag = code_rag

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        query: str = args["query"]
        max_hops: int = int(args.get("max_hops", 2))
        try:
            result = await self._code_rag.retrieve(
                query,
                tenant_id=ctx.tenant_id or None,
                max_hops=max_hops,
            )
        except Exception as exc:
            logger.warning("search_code_graph failed: %s", exc)
            return ToolResult(data=None, error=f"Code graph search failed: {exc}")

        if not result.graph_context and not result.vector_context:
            return ToolResult(
                data="No indexed code matched the query. The repository may not "
                "be indexed yet, or the symbol name may be misspelled.",
                metadata={"strategy": result.strategy},
            )

        sections: list[str] = []
        if result.graph_context:
            sections.append(result.graph_context)
        if result.vector_context:
            sections.append("[RELATED CODE — vector matches]")
            sections.extend(text[:600] for text in result.vector_context[:3])

        return ToolResult(
            data="\n".join(sections),
            metadata={
                "strategy": result.strategy,
                "graph_paths": len(result.graph_paths),
                "vector_hits": len(result.vector_hits),
                "tokens_estimate": result.total_tokens_estimate,
            },
        )


class ExpandCodeSymbolTool:
    """Fetch the full source of one symbol from the code graph."""

    name = "expand_code_symbol"
    description = (
        "Return the full source code of a single symbol previously seen in "
        "search_code_graph results. Pass the symbol id exactly as shown "
        "(format: code:sym:<file_path>::<qualified_name>)."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "symbol_id": {
                "type": "string",
                "description": "Symbol node id, e.g. 'code:sym:src/app/llm.py::HaasLLM.chat'.",
            },
        },
        "required": ["symbol_id"],
    }
    timeout_seconds: float = 10.0

    def __init__(self, code_rag: Any) -> None:
        self._code_rag = code_rag

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        symbol_id: str = args["symbol_id"]
        try:
            source = await self._code_rag.expand_symbol(symbol_id)
        except Exception as exc:
            logger.warning("expand_code_symbol failed: %s", exc)
            return ToolResult(data=None, error=f"Symbol expansion failed: {exc}")

        if source is None:
            return ToolResult(
                data=None,
                error=(
                    f"Symbol '{symbol_id}' could not be expanded — check the id "
                    "against search_code_graph output (and that the code graph "
                    "was built with repo_root set)."
                ),
            )
        return ToolResult(data=source, metadata={"symbol_id": symbol_id})


def build_code_graph_tools(code_rag: Any) -> list[Any]:
    """Return the code-graph tool set for registration.

    Usage::

        rag = CodeGraphRAG(graph, vector_store, repo_root=repo)
        for tool in build_code_graph_tools(rag):
            registry.register(tool)
    """
    return [SearchCodeGraphTool(code_rag), ExpandCodeSymbolTool(code_rag)]
