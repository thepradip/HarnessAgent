# Code Knowledge Graph (CodeGraphRAG)

Index a repository's *structure* — files, classes, functions, methods, and the
relationships between them (imports, calls, inheritance) — into harness graph
memory, and retrieve it with the same weighted GraphRAG machinery the SQL
agents use for schemas.

**Why:** agents that read whole files burn tokens and lose focus. The code
graph gives them a compact structural view first — signatures, docstrings,
call graph — and full source only for the symbols they actually need
(*expand-on-demand*). Typical savings are 10–20× versus file dumps, and
answers are more consistent because the same question anchors to the same
graph neighborhood every time.

```
load_source_files ─→ extract_code_facts ─→ CodeGraphIndexer ─→ GraphStore (NetworkX / Neo4j)
      (walk repo)      (ast / tree-sitter)        │
                                                  └─→ VectorStore (Chroma / Qdrant / Weaviate)
                                                        one chunk per top-level symbol

query ─→ extract_code_entities ─→ CodeGraphRAG.retrieve ─→ signatures-first context
              (regex → LLM)              │
                                         └─→ expand_symbol(id) → full source of ONE symbol
```

## Installation

The core works with **zero extra dependencies** — Python files are parsed with
the stdlib `ast` module. Optional extras:

```bash
pip install agent-haas                # Python indexing works out of the box
pip install agent-haas[code-graph]   # + tree-sitter parser tier
pip install agent-haas[vector]       # + Chroma/Qdrant/Weaviate symbol embeddings
pip install agent-haas[graph]        # + Neo4j production graph backend
```

## Quickstart

```python
import asyncio
from harness.memory import CodeGraphIndexer, CodeGraphRAG
from harness.memory.graph import NetworkXGraphMemory

async def main():
    graph = NetworkXGraphMemory()          # or get_graph_memory(config) / Neo4j
    indexer = CodeGraphIndexer(graph)

    # 1. Index the repository
    stats = await indexer.index_repo("/path/to/repo")
    print(f"{stats.symbols_indexed} symbols, {stats.edges_added} edges, "
          f"{stats.calls_resolved} calls resolved in {stats.duration_ms:.0f} ms")

    # 2. Ask structural questions
    rag = CodeGraphRAG(graph, repo_root="/path/to/repo")
    result = await rag.retrieve("who calls LLMRouter.complete?")
    print(result.graph_context)
    # [FILES]
    # app/llm.py  (app.llm) — LLM routing layer.
    # [SYMBOLS]
    # async def complete(self, messages, *, max_tokens) -> LLMResponse   [app/llm.py:42]
    #     Route a completion to the healthiest provider.
    # [CALL GRAPH]
    # HaasLLM.chat --calls--> LLMRouter.complete
    # ...

    # 3. Expand only what you need (the token saver)
    source = await rag.expand_symbol("code:sym:app/llm.py::LLMRouter.complete")
    print(source)

asyncio.run(main())
```

Runnable end-to-end demo (no API keys, no services):

```bash
python examples/code_graph_demo.py
```

## Graph schema

| Node type      | ID format                              | Key props |
|----------------|----------------------------------------|-----------|
| `CodeFile`     | `code:file:<rel_path>`                 | `module`, `language`, `docstring`, `content_hash`, `line_count` |
| `CodeClass`    | `code:sym:<rel_path>::<qualname>`      | `signature`, `docstring`, `line`, `end_line`, `file` |
| `CodeFunction` | `code:sym:<rel_path>::<qualname>`      | same as class |
| `CodeMethod`   | `code:sym:<rel_path>::<qualname>`      | same as class |
| `CodeModule`   | `code:mod:<module>` (external, opt-in) | `external: true` |

| Edge type  | Weight | Meaning |
|------------|--------|---------|
| `calls`    | 1.5    | symbol → symbol call (resolved cross-file, through imports, aliases, `self.x` including base classes) |
| `inherits` | 1.4    | class → base class |
| `imports`  | 1.0    | file → file (or external module) |
| `contains` | 0.8    | file → symbol, class → method |

Weights drive path scoring in retrieval, exactly like `joins > used_by_query >
has_column` in the SQL GraphRAG.

## Incremental re-indexing

Every `CodeFile` node stores a SHA-256 content hash. On re-index:

- **unchanged files are skipped** (`stats.files_skipped_unchanged`),
- **changed files** first have their old subgraph removed
  (`remove_nodes_by_prop("file", path)` — implemented on both NetworkX and
  Neo4j backends), so renamed/deleted symbols do not linger,
- `force=True` rebuilds everything.

```python
stats = await indexer.index_repo(repo)         # first run: everything
stats = await indexer.index_repo(repo)         # second run: 0 indexed, all skipped
stats = await indexer.index_repo(repo, force=True)
```

Files that fail to parse are recorded in `stats.parse_errors` and never abort
the run.

## Vector integration (semantic bridge)

Pass any harness `VectorStore` and the indexer embeds **one chunk per
top-level symbol** — signature + docstring header + body, chunked on symbol
boundaries (never mid-function). Chunk IDs equal the symbol's graph node id,
so vector hits *anchor graph traversal* when the query has no recognizable
identifiers:

```python
from harness.memory.vector_factory import VectorStoreFactory, build_embedding_provider
from harness.core.config import get_config

cfg = get_config()
embedder = build_embedding_provider(cfg)
store = VectorStoreFactory.build(cfg, embedder)       # chroma / qdrant / weaviate

indexer = CodeGraphIndexer(graph, vector_store=store)
await indexer.index_repo(repo, tenant_id="acme")

rag = CodeGraphRAG(graph, vector_store=store, repo_root=repo)
result = await rag.retrieve("where do we deduplicate cached responses?")
# no identifier in the query → vector hit on the symbol chunk → its symbol_id
# metadata seeds the graph traversal (strategy = "vector_fallback" / "hybrid")
```

## Agent tools

Two tools expose the graph to any harness agent:

```python
from harness.memory import CodeGraphRAG
from harness.tools import ToolRegistry, build_code_graph_tools

rag = CodeGraphRAG(graph, vector_store=store, repo_root=repo)
registry = ToolRegistry(safety_pipeline=..., audit_logger=...)
for tool in build_code_graph_tools(rag):
    registry.register(tool)
```

| Tool | Input | Output |
|------|-------|--------|
| `search_code_graph` | `query` (NL or symbol name), `max_hops` (1–4, default 2) | signatures-first structural context + strategy metadata |
| `expand_code_symbol` | `symbol_id` from search results | full source of that one symbol |

The `CodeAgent` system prompt already instructs the model to search the graph
before reading files and to expand only the symbols it must modify.

## Query-side entity extraction

`extract_code_entities` mirrors the SQL three-tier extractor:

1. **Precise regex** — backticked names, file paths, dotted paths
   (`HaasLLM.chat`), call syntax, CamelCase, snake_case. Substring fragments
   are suppressed (`HaasLLM.chat` wins over `HaasLLM`).
2. **LLM** (optional `llm_provider`) — for fuzzy questions with no identifiers.
3. **Generic identifier regex** — stopword-filtered fallback.

## Parsers

| Parser | When | Notes |
|--------|------|-------|
| stdlib `ast` (default) | Python files | exact signatures, docstrings, line ranges; zero deps |
| tree-sitter (`parser="tree-sitter"`) | opt-in; extension point for JS/TS/Go/… | requires `agent-haas[code-graph]`; behaviourally aligned with the ast tier for Python |

```python
indexer = CodeGraphIndexer(graph, parser="tree-sitter")   # force tree-sitter
```

Non-Python languages are detected by the loader (`.js .ts .tsx .go .java .rs`)
but skipped until their grammars are wired in — `stats.parse_errors` tells you
what was skipped.

## Call resolution — what resolves, what doesn't

Resolved to graph edges:

- same-file calls (`helper()`), including nested/qualified (`Class.method`)
- from-imports (`from app.utils import validate` → `validate()`)
- submodule imports (`from app import utils` → `utils.helper()`)
- module aliases (`import app.utils as u` → `u.validate()`)
- `self.x()` / `cls.x()` — walks base classes within the project (MRO-lite)
- unique global simple-name matches (skipped when ambiguous)

Counted as `calls_unresolved` (deliberately, to keep the graph noise-free):

- builtins (`bool()`, `len()`), stdlib/third-party calls (unless
  `track_external_modules=True` for import edges)
- attribute calls on local variables (`user.rename()` where `user = User()`
  — no type inference)
- dynamic dispatch (`getattr`, decorators that rebind)

## Production notes

- **Backends:** `NetworkXGraphMemory` (in-process, JSON persistence) for dev;
  `Neo4jGraphMemory` for shared/production deployments — both implement
  `remove_nodes_by_prop` for incremental re-index.
- **Tenancy:** pass `tenant_id` to `index_repo`; it is stamped on every node
  and used to filter vector queries.
- **Safety:** `expand_symbol` refuses path escapes outside `repo_root` and
  unknown symbol ids (it never dumps whole files); tool results are capped by
  the registry's standard 8k truncation.
- **Sizing:** files > 1 MB and non-UTF-8 files are skipped; per-symbol chunks
  are capped at 4k chars; vendored/generated dirs (`.venv`, `node_modules`,
  `__pycache__`, …) are excluded by default — extend with
  `extra_exclude_dirs` / `exclude_globs`.

## API reference (quick)

```python
# Indexing
CodeGraphIndexer(graph, vector_store=None, parser="auto", track_external_modules=False)
await indexer.index_repo(root, languages=None, extra_exclude_dirs=None,
                         exclude_globs=None, tenant_id="", force=False) -> CodeIndexStats

# Retrieval
CodeGraphRAG(graph, vector_store=None, llm_provider=None, repo_root=None,
             max_rendered_paths=20)
await rag.retrieve(query, tenant_id=None, max_hops=2, vector_k=5) -> RetrievalResult
await rag.expand_symbol(symbol_id) -> str | None

# Tools
build_code_graph_tools(rag) -> [SearchCodeGraphTool, ExpandCodeSymbolTool]

# Lower level
load_source_files(root, ...) -> list[SourceFile]
extract_code_facts(source, parser="auto") -> CodeFileFacts
build_symbol_chunks(source, facts, tenant_id="") -> list[Chunk]
```
