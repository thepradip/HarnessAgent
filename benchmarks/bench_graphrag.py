"""Benchmark 1: GraphRAG vs naive vector search — token efficiency.

Measures how many context tokens each approach delivers to the LLM for 20
representative SQL-agent queries on a 10-table e-commerce schema.

Run:
    PYTHONPATH=src python benchmarks/bench_graphrag.py

Output:
    benchmarks/results/graphrag_token_efficiency.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema: 10-table e-commerce system
# ---------------------------------------------------------------------------

SCHEMA: list[dict[str, Any]] = [
    {
        "name": "users",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "email", "type": "VARCHAR", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
            {"name": "plan", "type": "VARCHAR", "nullable": True},
            {"name": "created_at", "type": "TIMESTAMP", "nullable": False},
        ],
        "foreign_keys": [],
    },
    {
        "name": "products",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": False},
            {"name": "category_id", "type": "INT", "nullable": True},
            {"name": "price", "type": "DECIMAL", "nullable": False},
            {"name": "inventory_count", "type": "INT", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "category_id", "ref_table": "categories", "ref_col": "id"}
        ],
    },
    {
        "name": "orders",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "user_id", "type": "INT", "nullable": False},
            {"name": "total_usd", "type": "DECIMAL", "nullable": False},
            {"name": "status", "type": "VARCHAR", "nullable": False},
            {"name": "created_at", "type": "TIMESTAMP", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "user_id", "ref_table": "users", "ref_col": "id"}
        ],
    },
    {
        "name": "order_items",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "order_id", "type": "INT", "nullable": False},
            {"name": "product_id", "type": "INT", "nullable": False},
            {"name": "qty", "type": "INT", "nullable": False},
            {"name": "unit_price", "type": "DECIMAL", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "order_id", "ref_table": "orders", "ref_col": "id"},
            {"col": "product_id", "ref_table": "products", "ref_col": "id"},
        ],
    },
    {
        "name": "payments",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "order_id", "type": "INT", "nullable": False},
            {"name": "method", "type": "VARCHAR", "nullable": False},
            {"name": "amount_usd", "type": "DECIMAL", "nullable": False},
            {"name": "status", "type": "VARCHAR", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "order_id", "ref_table": "orders", "ref_col": "id"}
        ],
    },
    {
        "name": "addresses",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "user_id", "type": "INT", "nullable": False},
            {"name": "street", "type": "VARCHAR", "nullable": False},
            {"name": "city", "type": "VARCHAR", "nullable": False},
            {"name": "state", "type": "VARCHAR", "nullable": False},
            {"name": "zip", "type": "VARCHAR", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "user_id", "ref_table": "users", "ref_col": "id"}
        ],
    },
    {
        "name": "categories",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": False},
            {"name": "parent_id", "type": "INT", "nullable": True},
        ],
        "foreign_keys": [
            {"col": "parent_id", "ref_table": "categories", "ref_col": "id"}
        ],
    },
    {
        "name": "reviews",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "user_id", "type": "INT", "nullable": False},
            {"name": "product_id", "type": "INT", "nullable": False},
            {"name": "rating", "type": "INT", "nullable": False},
            {"name": "body", "type": "TEXT", "nullable": True},
            {"name": "created_at", "type": "TIMESTAMP", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "user_id", "ref_table": "users", "ref_col": "id"},
            {"col": "product_id", "ref_table": "products", "ref_col": "id"},
        ],
    },
    {
        "name": "sessions",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "user_id", "type": "INT", "nullable": False},
            {"name": "started_at", "type": "TIMESTAMP", "nullable": False},
            {"name": "duration_seconds", "type": "INT", "nullable": True},
        ],
        "foreign_keys": [
            {"col": "user_id", "ref_table": "users", "ref_col": "id"}
        ],
    },
    {
        "name": "promotions",
        "columns": [
            {"name": "id", "type": "INT", "nullable": False},
            {"name": "code", "type": "VARCHAR", "nullable": False},
            {"name": "discount_pct", "type": "DECIMAL", "nullable": False},
            {"name": "product_id", "type": "INT", "nullable": True},
            {"name": "expires_at", "type": "TIMESTAMP", "nullable": False},
        ],
        "foreign_keys": [
            {"col": "product_id", "ref_table": "products", "ref_col": "id"}
        ],
    },
]

# Naive corpus: verbose documentation chunk per table (~400-600 tokens each).
# This mirrors what real database documentation looks like — the kind of chunk
# a naive vector retriever returns from an ingested doc corpus.
def _make_naive_chunk(table: dict) -> str:
    name = table["name"]
    cols = table["columns"]
    fks = table.get("foreign_keys", [])

    col_lines = "\n".join(
        f"  - {c['name']} ({c['type']}{'  [NOT NULL]' if not c['nullable'] else ''}): "
        f"Stores the {c['name'].replace('_', ' ')} value for each {name} record. "
        f"This column {'cannot be null and must always' if not c['nullable'] else 'may'} "
        f"contain a valid {c['type']} value."
        for c in cols
    )
    fk_lines = "\n".join(
        f"  - {fk['col']} → {fk['ref_table']}.{fk['ref_col']}: "
        f"Links each {name} record to its corresponding {fk['ref_table']} record. "
        f"Always join on this column when combining {name} with {fk['ref_table']}."
        for fk in fks
    ) or "  (none)"

    usage_examples = "\n".join([
        f"  Example 1: SELECT * FROM {name} WHERE id = 42;",
        f"  Example 2: SELECT COUNT(*) FROM {name};",
        f"  Example 3: SELECT * FROM {name} ORDER BY id DESC LIMIT 10;",
    ])

    return (
        f"TABLE: {name}\n"
        f"Purpose: The {name} table is the primary store for "
        f"{name.replace('_', ' ')} data in the system. It is queried frequently "
        f"for reporting, user-facing features, and analytics workloads.\n\n"
        f"Columns:\n{col_lines}\n\n"
        f"Foreign key relationships:\n{fk_lines}\n\n"
        f"Performance notes: Ensure indexes exist on all foreign key columns and "
        f"any column used in WHERE clauses. Avoid SELECT * in production; "
        f"specify only required columns to reduce I/O. When joining {name} "
        f"with other tables, prefer the indexed foreign key columns listed above.\n\n"
        f"Sample queries:\n{usage_examples}\n\n"
        f"Schema definition:\n"
        f"CREATE TABLE {name} (\n"
        + "\n".join(
            f"  {c['name']} {c['type']}{' NOT NULL' if not c['nullable'] else ''},"
            for c in cols
        )
        + "\n);\n"
    )

NAIVE_CORPUS = {t["name"]: _make_naive_chunk(t) for t in SCHEMA}

# 20 benchmark queries covering single-table, join, and aggregation patterns
QUERIES = [
    ("q01", "How many orders did user 42 place in the last 30 days?",
     ["orders", "users"]),
    ("q02", "Show all products with inventory_count below 10",
     ["products"]),
    ("q03", "What is the total revenue from completed orders this month?",
     ["orders"]),
    ("q04", "List the top 5 users by number of orders placed",
     ["orders", "users"]),
    ("q05", "Which products have never been ordered?",
     ["products", "order_items"]),
    ("q06", "What is the average order value by payment method?",
     ["orders", "payments"]),
    ("q07", "Find all reviews with a rating below 3 for electronics products",
     ["reviews", "products", "categories"]),
    ("q08", "How many active sessions are running right now?",
     ["sessions"]),
    ("q09", "Which promotions expire in the next 7 days?",
     ["promotions"]),
    ("q10", "Show the shipping addresses for all orders with status 'pending'",
     ["orders", "addresses", "users"]),
    ("q11", "What is the total quantity sold for each product category?",
     ["order_items", "products", "categories"]),
    ("q12", "List users who placed an order but never left a review",
     ["orders", "reviews", "users"]),
    ("q13", "What is the most popular product in the 'Electronics' category?",
     ["order_items", "products", "categories"]),
    ("q14", "Show the payment method breakdown for orders over $500",
     ["payments", "orders"]),
    ("q15", "Which users have used a promotion code more than once?",
     ["orders", "promotions"]),
    ("q16", "Find the average session duration for premium plan users",
     ["sessions", "users"]),
    ("q17", "How many distinct products were ordered alongside product 77?",
     ["order_items"]),
    ("q18", "Show the refund rate (failed payments / total payments) by month",
     ["payments"]),
    ("q19", "List all order_items for orders placed by users in California",
     ["order_items", "orders", "users", "addresses"]),
    ("q20", "What is the average rating for products ordered more than 100 times?",
     ["reviews", "order_items", "products"]),
]


# ---------------------------------------------------------------------------
# Keyword-based mock vector store (no external model needed)
# ---------------------------------------------------------------------------

class _KeywordVectorStore:
    """Returns top-k naive corpus chunks based on word overlap with the query."""

    def __init__(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus

    async def query(self, text: str, k: int = 5, filter=None, hybrid_alpha=None):
        from harness.core.protocols import VectorHit
        words = set(text.lower().split())
        scored = []
        for name, chunk in self._corpus.items():
            chunk_words = set(chunk.lower().split())
            score = len(words & chunk_words) / max(len(words), 1)
            scored.append((score, name, chunk))
        scored.sort(reverse=True)
        return [
            VectorHit(id=name, text=chunk, score=score)
            for score, name, chunk in scored[:k]
        ]


class _NullEmbedder:
    """Embedder that returns zero vectors (GraphRAG uses vector store as a fallback only)."""
    model = "null"
    dimensions = 1

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run() -> None:
    from harness.core.context import AgentContext
    from harness.memory.graph import NetworkXGraphMemory
    from harness.memory.graph_rag import GraphRAGEngine

    # Build graph
    graph = NetworkXGraphMemory()
    embedder = _NullEmbedder()
    vector_store = _KeywordVectorStore(NAIVE_CORPUS)
    # max_rendered_paths=8 keeps only the highest-scored paths (realistic production setting)
    engine = GraphRAGEngine(
        graph=graph, vector_store=vector_store, embedder=embedder, max_rendered_paths=8
    )

    # Create a dummy context
    ctx = AgentContext(
        run_id="bench-graphrag",
        tenant_id="bench",
        agent_type="sql",
        task="benchmark",
        memory=MagicMock(),
        workspace_path=Path("/tmp/bench_graphrag"),
        max_steps=50,
        max_tokens=100_000,
        timeout_seconds=300.0,
    )

    # Populate schema graph
    await engine.populate_schema(SCHEMA, ctx)

    # Seed 20 query history nodes to simulate a live system
    seed_sqls = [
        ("SELECT COUNT(*) FROM orders WHERE user_id = ?", ["orders"], True),
        ("SELECT * FROM products WHERE inventory_count < 5", ["products"], True),
        ("SELECT SUM(total_usd) FROM orders WHERE status='completed'", ["orders"], True),
        ("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id LIMIT 5", ["orders", "users"], True),
        ("SELECT p.* FROM products p LEFT JOIN order_items oi ON p.id=oi.product_id WHERE oi.id IS NULL", ["products", "order_items"], True),
        ("SELECT payment_method, AVG(total_usd) FROM orders JOIN payments USING(id)", ["orders", "payments"], True),
        ("SELECT * FROM reviews WHERE rating < 3", ["reviews", "products"], False),
        ("SELECT COUNT(*) FROM sessions WHERE duration_seconds IS NULL", ["sessions"], True),
        ("SELECT * FROM promotions WHERE expires_at < NOW() + INTERVAL 7 DAY", ["promotions"], True),
        ("SELECT a.* FROM addresses a JOIN orders o ON a.user_id=o.user_id WHERE o.status='pending'", ["orders", "addresses"], True),
    ]
    for i, (sql, tables, ok) in enumerate(seed_sqls):
        await engine.record_query(
            query_sql=sql,
            tables_used=tables,
            run_id=f"seed-{i}",
            tenant_id="bench",
            success=ok,
            latency_ms=float(50 + i * 10),
        )

    results = []
    total_naive = 0
    total_graphrag = 0

    print(f"\n{'Query':>4}  {'Tables needed':<35}  {'Naive tokens':>12}  {'GraphRAG tokens':>15}  {'Strategy'}")
    print("-" * 90)

    for qid, query, expected_tables in QUERIES:
        t0 = time.perf_counter()
        # vector_k=0: pure graph traversal — no vector supplement (measures graph alone)
        result = await engine.retrieve(query, ctx, max_hops=2, vector_k=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Naive: top-5 vector chunks concatenated (realistic baseline)
        naive_hits = await vector_store.query(query, k=5)
        naive_text = "\n\n".join(h.text for h in naive_hits)
        naive_tokens = max(1, len(naive_text) // 4)

        graphrag_tokens = result.total_tokens_estimate

        total_naive += naive_tokens
        total_graphrag += graphrag_tokens
        savings_pct = (1 - graphrag_tokens / naive_tokens) * 100 if naive_tokens > 0 else 0

        # Coverage: did GraphRAG surface all expected tables?
        ctx_lower = (result.graph_context + " ".join(result.vector_context)).lower()
        covered = sum(1 for t in expected_tables if t.lower() in ctx_lower)
        coverage_pct = covered / len(expected_tables) * 100

        print(
            f"{qid}  {', '.join(expected_tables):<35}  {naive_tokens:>12}  "
            f"{graphrag_tokens:>15}  {result.strategy}  "
            f"cov={coverage_pct:.0f}%  {elapsed_ms:.1f}ms"
        )

        results.append({
            "query_id": qid,
            "query": query,
            "expected_tables": expected_tables,
            "naive_tokens": naive_tokens,
            "graphrag_tokens": graphrag_tokens,
            "savings_pct": round(savings_pct, 1),
            "strategy": result.strategy,
            "coverage_pct": round(coverage_pct, 1),
            "retrieval_ms": round(elapsed_ms, 2),
        })

    avg_naive = total_naive / len(QUERIES)
    avg_graphrag = total_graphrag / len(QUERIES)
    overall_savings = (1 - avg_graphrag / avg_naive) * 100

    strategies = {}
    for r in results:
        strategies[r["strategy"]] = strategies.get(r["strategy"], 0) + 1

    avg_coverage = sum(r["coverage_pct"] for r in results) / len(results)

    summary = {
        "naive_avg_tokens": round(avg_naive, 1),
        "graphrag_avg_tokens": round(avg_graphrag, 1),
        "overall_savings_pct": round(overall_savings, 1),
        "avg_coverage_pct": round(avg_coverage, 1),
        "strategy_breakdown": strategies,
        "query_count": len(QUERIES),
        "schema_tables": len(SCHEMA),
    }

    print("\n" + "=" * 90)
    print(f"SUMMARY")
    print(f"  Naive vector search  avg tokens : {avg_naive:.1f}")
    print(f"  GraphRAG             avg tokens : {avg_graphrag:.1f}")
    print(f"  Token savings                   : {overall_savings:.1f}%")
    print(f"  Average table coverage          : {avg_coverage:.1f}%")
    print(f"  Strategy breakdown              : {strategies}")

    output = {
        "benchmark": "graphrag_token_efficiency",
        "summary": summary,
        "per_query": results,
    }
    out_path = RESULTS_DIR / "graphrag_token_efficiency.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
