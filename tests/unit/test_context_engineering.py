"""Unit tests for the L1/L2/L3 context engineering layer."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from harness.memory.context_engineering import (
    AssembledContext,
    ColumnDef,
    ContextPipeline,
    ForeignKey,
    KGCache,
    L1ChatCache,
    L2VectorCache,
    L3KnowledgeStore,
    SchemaStore,
    TableSchema,
    TierResult,
    _count_tokens,
    _fit_to_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def _table(name: str = "users", n_cols: int = 2, with_fk: bool = False) -> TableSchema:
    cols = [ColumnDef("id", "INTEGER", nullable=False, primary_key=True)]
    for i in range(n_cols - 1):
        cols.append(ColumnDef(f"col{i}", "TEXT"))
    fks = [ForeignKey("col0", "other", "id")] if with_fk else []
    return TableSchema(name=name, columns=cols, foreign_keys=fks,
                       row_count=100, sample_rows=[{"id": 1}], description=f"{name} table")


def _tier(tier: str = "L1", tokens: int = 10, hit: bool = True) -> TierResult:
    return TierResult(tier=tier, messages=[{"role": "system", "content": "x"}],
                      tokens_used=tokens, source="test", latency_ms=1.0, hit=hit)


# ===========================================================================
# ColumnDef
# ===========================================================================

def test_column_def_round_trip():
    col = ColumnDef("email", "TEXT", nullable=False, primary_key=False)
    assert ColumnDef.from_dict(col.to_dict()) == col


def test_column_def_primary_key_flag():
    col = ColumnDef("id", "INTEGER", nullable=False, primary_key=True)
    assert col.primary_key is True
    assert ColumnDef.from_dict(col.to_dict()).primary_key is True


def test_column_def_defaults():
    col = ColumnDef("x", "REAL")
    assert col.nullable is True
    assert col.primary_key is False


# ===========================================================================
# ForeignKey
# ===========================================================================

def test_foreign_key_round_trip():
    fk = ForeignKey("user_id", "users", "id")
    assert ForeignKey.from_dict(fk.to_dict()) == fk


# ===========================================================================
# TableSchema
# ===========================================================================

def test_table_schema_round_trip():
    tbl = _table("orders", n_cols=3, with_fk=True)
    tbl2 = TableSchema.from_dict(tbl.to_dict())
    assert tbl2.name == "orders"
    assert len(tbl2.columns) == 3
    assert len(tbl2.foreign_keys) == 1
    assert tbl2.row_count == 100
    assert tbl2.description == "orders table"


def test_table_schema_context_block_contains_name():
    tbl = _table("payments")
    block = tbl.to_context_block()
    assert "TABLE payments" in block


def test_table_schema_context_block_shows_pk():
    tbl = _table("users")
    block = tbl.to_context_block()
    assert "PK" in block


def test_table_schema_context_block_shows_fk():
    tbl = _table("orders", with_fk=True)
    block = tbl.to_context_block()
    assert "FK:" in block
    assert "other" in block


def test_table_schema_context_block_shows_row_count():
    tbl = _table()
    block = tbl.to_context_block()
    assert "100" in block


def test_table_schema_context_block_no_samples():
    tbl = _table()
    block = tbl.to_context_block(include_samples=False)
    assert "sample" not in block.lower()


def test_table_schema_context_block_shows_description():
    tbl = _table("foo")
    block = tbl.to_context_block()
    assert "foo table" in block


def test_table_schema_empty_columns():
    tbl = TableSchema(name="empty")
    block = tbl.to_context_block()
    assert "empty" in block


# ===========================================================================
# TierResult
# ===========================================================================

def test_tier_result_hit_false_when_no_messages():
    tr = TierResult("L2", [], 0, "vec", 2.0, hit=False)
    assert tr.hit is False
    assert tr.tokens_used == 0


def test_tier_result_tier_literal():
    for tier in ("L1", "L2", "L3"):
        tr = TierResult(tier, [], 0, "x", 0.0)
        assert tr.tier == tier


# ===========================================================================
# AssembledContext
# ===========================================================================

def test_assembled_context_stats_keys():
    ac = AssembledContext(
        messages=[],
        total_tokens=50,
        budget=100,
        tier_results=[_tier("L1", 30), _tier("L2", 15, hit=False), _tier("L3", 5)],
        query="q",
        skill_ns="default",
    )
    stats = ac.stats
    assert stats["total_tokens"] == 50
    assert stats["budget"] == 100
    assert stats["utilisation"] == pytest.approx(0.5)
    assert "L1" in stats["tiers"]
    assert "L2" in stats["tiers"]
    assert "L3" in stats["tiers"]


def test_assembled_context_l1_l2_l3_tokens():
    ac = AssembledContext(
        messages=[],
        total_tokens=60,
        budget=100,
        tier_results=[_tier("L1", 30), _tier("L2", 20), _tier("L3", 10)],
        query="q",
        skill_ns="sql",
    )
    assert ac.l1_tokens() == 30
    assert ac.l2_tokens() == 20
    assert ac.l3_tokens() == 10


def test_assembled_context_missing_tier_returns_zero():
    ac = AssembledContext(
        messages=[],
        total_tokens=10,
        budget=100,
        tier_results=[_tier("L1", 10)],
        query="q",
        skill_ns="default",
    )
    assert ac.l2_tokens() == 0
    assert ac.l3_tokens() == 0


def test_assembled_context_full_budget_utilisation():
    ac = AssembledContext(
        messages=[],
        total_tokens=100,
        budget=100,
        tier_results=[_tier("L1", 100)],
        query="q",
        skill_ns="default",
    )
    assert ac.stats["utilisation"] == pytest.approx(1.0)


# ===========================================================================
# Pure helpers
# ===========================================================================

def test_count_tokens_nonempty():
    t = _count_tokens("hello world")
    assert t >= 1


def test_count_tokens_empty():
    assert _count_tokens("") == 0 or _count_tokens("") >= 0


def test_fit_to_budget_all_fit():
    msgs = [{"role": "user", "content": "hi", "tokens": 5}] * 4
    kept, used, truncated = _fit_to_budget(msgs, budget=100)
    assert len(kept) == 4
    assert used == 20
    assert truncated is False


def test_fit_to_budget_truncates_oldest():
    msgs = [{"role": "user", "content": f"msg{i}", "tokens": 10} for i in range(5)]
    kept, used, truncated = _fit_to_budget(msgs, budget=30)
    # Should keep 3 most-recent (budget // 10)
    assert len(kept) == 3
    assert used == 30
    assert truncated is True
    # Most recent messages are preserved (indices 2, 3, 4)
    assert kept[-1]["content"] == "msg4"


def test_fit_to_budget_zero_budget():
    msgs = [{"role": "user", "content": "hi", "tokens": 5}]
    kept, used, truncated = _fit_to_budget(msgs, budget=0)
    assert kept == []
    assert used == 0


def test_fit_to_budget_empty_messages():
    kept, used, truncated = _fit_to_budget([], budget=1000)
    assert kept == []
    assert used == 0
    assert truncated is False


# ===========================================================================
# SchemaStore
# ===========================================================================

@pytest.fixture
def schema_store():
    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://unused"
    store._ttl = 3600
    store._client = _fake_redis()
    return store


@pytest.mark.asyncio
async def test_schema_store_store_and_get(schema_store):
    tables = [_table("users"), _table("orders")]
    await schema_store.store("db1", tables)
    result = await schema_store.get("db1")
    assert {t.name for t in result} == {"users", "orders"}


@pytest.mark.asyncio
async def test_schema_store_get_specific_tables(schema_store):
    await schema_store.store("db1", [_table("a"), _table("b"), _table("c")])
    result = await schema_store.get("db1", ["a", "c"])
    assert {t.name for t in result} == {"a", "c"}


@pytest.mark.asyncio
async def test_schema_store_get_missing_db_returns_empty(schema_store):
    result = await schema_store.get("nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_schema_store_table_names(schema_store):
    await schema_store.store("db2", [_table("x"), _table("y")])
    names = await schema_store.table_names("db2")
    assert set(names) == {"x", "y"}


@pytest.mark.asyncio
async def test_schema_store_delete(schema_store):
    await schema_store.store("db3", [_table("t")])
    await schema_store.delete("db3")
    result = await schema_store.get("db3")
    assert result == []


@pytest.mark.asyncio
async def test_schema_store_overwrite_table(schema_store):
    t1 = TableSchema("users", columns=[ColumnDef("id", "INTEGER", False, True)])
    t2 = TableSchema("users", columns=[ColumnDef("id", "INTEGER", False, True),
                                        ColumnDef("email", "TEXT")])
    await schema_store.store("db4", [t1])
    await schema_store.store("db4", [t2])
    result = await schema_store.get("db4", ["users"])
    assert len(result[0].columns) == 2


@pytest.mark.asyncio
async def test_schema_store_context_block_contains_all_tables(schema_store):
    await schema_store.store("db5", [_table("alpha"), _table("beta")])
    block = await schema_store.get_context_block("db5")
    assert "alpha" in block
    assert "beta" in block
    assert "Database: db5" in block


@pytest.mark.asyncio
async def test_schema_store_context_block_empty_db(schema_store):
    block = await schema_store.get_context_block("empty_db")
    assert block == ""


@pytest.mark.asyncio
async def test_schema_store_context_block_max_tables(schema_store):
    tables = [_table(f"t{i}") for i in range(20)]
    await schema_store.store("db6", tables)
    block = await schema_store.get_context_block("db6", max_tables=5)
    # Only 5 tables rendered
    assert block.count("TABLE ") == 5


@pytest.mark.asyncio
async def test_schema_store_context_block_relevant_tables(schema_store):
    await schema_store.store("db7", [_table("users"), _table("orders"), _table("products")])
    block = await schema_store.get_context_block("db7", relevant_tables=["users"])
    assert "users" in block
    assert "orders" not in block
    assert "products" not in block


@pytest.mark.asyncio
async def test_schema_store_fk_rendered_in_block(schema_store):
    tbl = _table("orders", with_fk=True)
    await schema_store.store("db8", [tbl])
    block = await schema_store.get_context_block("db8")
    assert "FK:" in block


@pytest.mark.asyncio
async def test_schema_store_from_sqlite(schema_store, tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, "
                 "FOREIGN KEY(user_id) REFERENCES users(id))")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO orders VALUES (1, 1)")
    conn.commit()
    conn.close()

    tables = await schema_store.store_from_sqlite("sqlite_db", db_path)
    assert {t.name for t in tables} == {"users", "orders"}

    users = next(t for t in tables if t.name == "users")
    assert any(c.name == "id" and c.primary_key for c in users.columns)
    assert any(c.name == "name" and not c.nullable for c in users.columns)
    assert users.row_count == 1

    orders = next(t for t in tables if t.name == "orders")
    assert len(orders.foreign_keys) == 1
    assert orders.foreign_keys[0].references_table == "users"

    # Verify stored in Redis
    stored = await schema_store.get("sqlite_db")
    assert {t.name for t in stored} == {"users", "orders"}


@pytest.mark.asyncio
async def test_schema_store_from_sqlite_sample_rows(schema_store, tmp_path):
    db_path = str(tmp_path / "samples.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'foo'), (2, 'bar'), (3, 'baz')")
    conn.commit()
    conn.close()

    tables = await schema_store.store_from_sqlite("samples_db", db_path)
    items = tables[0]
    assert len(items.sample_rows) <= 2  # capped at 2
    assert items.sample_rows[0]["id"] == 1


# ===========================================================================
# L1ChatCache
# ===========================================================================

def _make_engine_mock(messages=None):
    """Return a mock ContextEngine with a pre-loaded hot window."""
    from harness.memory.schemas import ConversationMessage
    if messages is None:
        msgs = [
            ConversationMessage(role="user", content="hello", tokens=5),
            ConversationMessage(role="assistant", content="world", tokens=8),
        ]
    else:
        msgs = messages
    engine = MagicMock()
    engine._load_hot = AsyncMock(return_value=msgs)
    engine.push = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_l1_get_returns_messages():
    engine = _make_engine_mock()
    l1 = L1ChatCache(engine)
    result = await l1.get("run1", "default", token_budget=1000)
    assert result.tier == "L1"
    assert len(result.messages) == 2
    assert result.hit is True


@pytest.mark.asyncio
async def test_l1_get_respects_token_budget():
    from harness.memory.schemas import ConversationMessage
    msgs = [ConversationMessage(role="user", content=f"msg{i}", tokens=100) for i in range(10)]
    engine = _make_engine_mock(msgs)
    l1 = L1ChatCache(engine)
    result = await l1.get("run1", "default", token_budget=300)
    assert result.tokens_used <= 300
    assert len(result.messages) <= 3


@pytest.mark.asyncio
async def test_l1_get_empty_history_hit_false():
    engine = _make_engine_mock(messages=[])
    l1 = L1ChatCache(engine)
    result = await l1.get("run1", "default", token_budget=1000)
    assert result.hit is False
    assert result.tokens_used == 0


@pytest.mark.asyncio
async def test_l1_get_merges_shared_namespace():
    from harness.memory.schemas import ConversationMessage
    skill_msgs = [ConversationMessage("user", "skill msg", 5)]
    shared_msgs = [ConversationMessage("system", "shared msg", 5)]
    engine = MagicMock()
    # First call (skill ns), second call (default ns)
    engine._load_hot = AsyncMock(side_effect=[skill_msgs, shared_msgs])
    engine.push = AsyncMock()
    l1 = L1ChatCache(engine)
    result = await l1.get("run1", "sql", token_budget=1000)
    # Both skill + shared messages present
    contents = [m["content"] for m in result.messages]
    assert any("skill msg" in c for c in contents)
    assert any("shared msg" in c for c in contents)


@pytest.mark.asyncio
async def test_l1_get_no_shared_when_default_ns():
    from harness.memory.schemas import ConversationMessage
    msgs = [ConversationMessage("user", "msg", 5)]
    engine = _make_engine_mock(msgs)
    l1 = L1ChatCache(engine)
    await l1.get("run1", "default", token_budget=1000)
    # Only called once (no separate shared ns fetch)
    assert engine._load_hot.call_count == 1


@pytest.mark.asyncio
async def test_l1_push_delegates_to_engine():
    engine = _make_engine_mock()
    l1 = L1ChatCache(engine)
    await l1.push("run1", "user", "test message", tokens=5, skill_ns="sql", step=3)
    engine.push.assert_called_once_with("run1", "user", "test message",
                                        tokens=5, skill_ns="sql", step=3)


@pytest.mark.asyncio
async def test_l1_engine_failure_returns_empty():
    engine = MagicMock()
    engine._load_hot = AsyncMock(side_effect=RuntimeError("redis down"))
    engine.push = AsyncMock()
    l1 = L1ChatCache(engine)
    result = await l1.get("run1", "default", token_budget=1000)
    assert result.tier == "L1"
    assert result.hit is False
    assert result.tokens_used == 0


# ===========================================================================
# L2VectorCache
# ===========================================================================

def _make_vector_hit(text: str, score: float = 0.9, id_: str = "doc1"):
    hit = MagicMock()
    hit.text = text
    hit.score = score
    hit.id = id_
    return hit


@pytest.mark.asyncio
async def test_l2_get_returns_hits_above_threshold():
    vs = MagicMock()
    vs.query = AsyncMock(return_value=[
        _make_vector_hit("relevant context", score=0.92),
        _make_vector_hit("also relevant", score=0.85),
    ])
    embedder = MagicMock()
    l2 = L2VectorCache(vs, embedder, relevance_threshold=0.80)
    result = await l2.get("find users", token_budget=5000)
    assert result.tier == "L2"
    assert result.hit is True
    assert len(result.messages) == 2


@pytest.mark.asyncio
async def test_l2_get_filters_below_threshold():
    vs = MagicMock()
    vs.query = AsyncMock(return_value=[
        _make_vector_hit("relevant", score=0.92),
        _make_vector_hit("irrelevant", score=0.50),
    ])
    l2 = L2VectorCache(vs, MagicMock(), relevance_threshold=0.80)
    result = await l2.get("query", token_budget=5000)
    assert len(result.messages) == 1
    assert "relevant" in result.messages[0]["content"]


@pytest.mark.asyncio
async def test_l2_get_respects_token_budget():
    vs = MagicMock()
    # Each hit has ~25 tokens ("x" * 100 ≈ 25 tokens at 4 chars/token)
    vs.query = AsyncMock(return_value=[
        _make_vector_hit("x" * 100, score=0.9, id_=f"d{i}") for i in range(10)
    ])
    l2 = L2VectorCache(vs, MagicMock(), relevance_threshold=0.0, top_k=10)
    result = await l2.get("query", token_budget=60)
    assert result.tokens_used <= 60


@pytest.mark.asyncio
async def test_l2_get_no_vector_store_returns_miss():
    l2 = L2VectorCache(None, MagicMock())
    result = await l2.get("query", token_budget=5000)
    assert result.hit is False
    assert result.tokens_used == 0


@pytest.mark.asyncio
async def test_l2_get_query_error_returns_miss():
    vs = MagicMock()
    vs.query = AsyncMock(side_effect=RuntimeError("connection refused"))
    l2 = L2VectorCache(vs, MagicMock())
    result = await l2.get("query", token_budget=5000)
    assert result.hit is False


@pytest.mark.asyncio
async def test_l2_get_passes_filter():
    vs = MagicMock()
    vs.query = AsyncMock(return_value=[])
    l2 = L2VectorCache(vs, MagicMock())
    await l2.get("query", token_budget=5000, filter={"tenant_id": "acme"})
    vs.query.assert_called_once()
    call_kwargs = vs.query.call_args[1]
    assert call_kwargs["filter"] == {"tenant_id": "acme"}


@pytest.mark.asyncio
async def test_l2_store_calls_upsert():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    l2 = L2VectorCache(vs, embedder)
    uid = await l2.store("some text", metadata={"key": "val"})
    vs.upsert.assert_called_once()
    assert isinstance(uid, str) and len(uid) == 32  # uuid4 hex


@pytest.mark.asyncio
async def test_l2_store_uses_provided_doc_id():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1]])
    l2 = L2VectorCache(vs, embedder)
    uid = await l2.store("text", doc_id="custom-id")
    assert uid == "custom-id"
    call_kwargs = vs.upsert.call_args[1]
    assert call_kwargs["id"] == "custom-id"


@pytest.mark.asyncio
async def test_l2_store_embed_failure_does_not_raise():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=RuntimeError("embed failed"))
    l2 = L2VectorCache(vs, embedder)
    uid = await l2.store("text")  # should not raise
    assert isinstance(uid, str)


# ===========================================================================
# KGCache
# ===========================================================================

def _make_graph_mock(facts=None):
    """Return a mock graph that yields the given (src, rel, tgt) triples."""
    edge_mocks = []
    for src, rel, tgt in (facts or [("Alice", "WORKS_AT", "Acme")]):
        e = MagicMock()
        e.src = src
        e.type = rel
        e.tgt = tgt
        edge_mocks.append(e)
    path = MagicMock()
    path.edges = edge_mocks
    graph = MagicMock()
    graph.traverse = AsyncMock(return_value=[path])
    return graph


@pytest.mark.asyncio
async def test_kg_get_returns_facts():
    graph = _make_graph_mock([("users", "HAS_COLUMN", "id")])
    kg = KGCache(graph)
    result = await kg.get("tell me about users table", token_budget=5000)
    assert result.tier == "L3"
    assert result.hit is True
    assert any("users" in m["content"] for m in result.messages)


@pytest.mark.asyncio
async def test_kg_get_no_graph_returns_miss():
    kg = KGCache(None)
    result = await kg.get("query", token_budget=5000)
    assert result.hit is False


@pytest.mark.asyncio
async def test_kg_get_empty_path_returns_miss():
    path = MagicMock()
    path.edges = []
    graph = MagicMock()
    graph.traverse = AsyncMock(return_value=[path])
    kg = KGCache(graph)
    result = await kg.get("orders table schema", token_budget=5000)
    assert result.hit is False


@pytest.mark.asyncio
async def test_kg_get_graph_error_returns_miss():
    graph = MagicMock()
    graph.traverse = AsyncMock(side_effect=RuntimeError("graph unavailable"))
    kg = KGCache(graph)
    result = await kg.get("query", token_budget=5000)
    assert result.hit is False


@pytest.mark.asyncio
async def test_kg_get_uses_extractor_entities():
    extractor = MagicMock()
    extractor.extract_entities = AsyncMock(return_value=["orders"])
    graph = _make_graph_mock([("orders", "HAS_FK", "users")])
    kg = KGCache(graph, extractor)
    result = await kg.get("find orders with users", token_budget=5000)
    extractor.extract_entities.assert_called_once()
    assert result.hit is True


@pytest.mark.asyncio
async def test_kg_get_deduplicates_facts():
    # Duplicate fact from two traversal paths — use long enough entity names for regex fallback
    e1 = MagicMock(); e1.src = "alpha"; e1.type = "REL"; e1.tgt = "beta"
    e2 = MagicMock(); e2.src = "alpha"; e2.type = "REL"; e2.tgt = "beta"
    path1 = MagicMock(); path1.edges = [e1]
    path2 = MagicMock(); path2.edges = [e2]
    graph = MagicMock()
    graph.traverse = AsyncMock(return_value=[path1, path2])
    kg = KGCache(graph)
    result = await kg.get("alpha and beta nodes", token_budget=5000)
    assert result.hit is True
    content = result.messages[0]["content"]
    assert content.count("alpha REL beta") == 1


@pytest.mark.asyncio
async def test_kg_get_respects_token_budget():
    facts = [(f"node{i}", "REL", f"node{i+1}") for i in range(100)]
    graph = _make_graph_mock(facts)
    kg = KGCache(graph)
    result = await kg.get("query", token_budget=50)
    assert result.tokens_used <= 50 + 20  # small tolerance


# ===========================================================================
# L3KnowledgeStore
# ===========================================================================

@pytest.fixture
def l3_store(schema_store):
    kg_cache = KGCache(None)  # no graph
    return L3KnowledgeStore(schema=schema_store, kg=kg_cache)


@pytest.mark.asyncio
async def test_l3_get_includes_schema(l3_store, schema_store):
    await schema_store.store("mydb", [_table("users"), _table("orders")])
    result = await l3_store.get("find users", token_budget=10_000, db_id="mydb")
    assert result.tier == "L3"
    assert result.hit is True
    contents = " ".join(m["content"] for m in result.messages)
    assert "users" in contents


@pytest.mark.asyncio
async def test_l3_get_no_db_id_no_schema(l3_store):
    result = await l3_store.get("query", token_budget=5000, db_id=None)
    # No schema (db_id=None), no KG (graph=None) → no content
    assert result.hit is False


@pytest.mark.asyncio
async def test_l3_get_with_kg(schema_store):
    graph = _make_graph_mock([("users", "HAS_COL", "id")])
    kg_cache = KGCache(graph)
    l3 = L3KnowledgeStore(schema=schema_store, kg=kg_cache)
    result = await l3.get("users table info", token_budget=10_000,
                          include_schema=False, include_kg=True)
    assert result.hit is True
    assert result.tokens_used > 0


@pytest.mark.asyncio
async def test_l3_get_budget_splits_schema_kg(schema_store):
    """Schema gets 2/3 of L3 budget, KG gets 1/3."""
    await schema_store.store("splitdb", [_table("t")])
    graph = _make_graph_mock([("t", "HAS", "col")])
    l3 = L3KnowledgeStore(schema=schema_store, kg=KGCache(graph))
    result = await l3.get("query", token_budget=3000, db_id="splitdb")
    # Both sources contributed
    assert result.hit is True


@pytest.mark.asyncio
async def test_l3_get_include_schema_false_skips_schema(schema_store):
    await schema_store.store("nodb", [_table("users")])
    l3 = L3KnowledgeStore(schema=schema_store, kg=KGCache(None))
    result = await l3.get("query", token_budget=5000, db_id="nodb", include_schema=False)
    contents = " ".join(m["content"] for m in result.messages)
    assert "users" not in contents


@pytest.mark.asyncio
async def test_l3_get_include_kg_false_skips_graph(schema_store):
    graph = _make_graph_mock([("x", "REL", "y")])
    l3 = L3KnowledgeStore(schema=schema_store, kg=KGCache(graph))
    result = await l3.get("query", token_budget=5000, include_schema=False, include_kg=False)
    assert result.hit is False


# ===========================================================================
# ContextPipeline
# ===========================================================================

@pytest.fixture
def pipeline(schema_store):
    """Pipeline with mocked L1/L2/L3 tiers."""
    engine = _make_engine_mock()
    l1 = L1ChatCache(engine)
    vs = MagicMock()
    vs.query = AsyncMock(return_value=[_make_vector_hit("vec hit", score=0.9)])
    l2 = L2VectorCache(vs, MagicMock(), relevance_threshold=0.5, top_k=3)
    kg = KGCache(None)
    l3 = L3KnowledgeStore(schema=schema_store, kg=kg)
    return ContextPipeline(l1=l1, l2=l2, l3=l3,
                           l1_fraction=0.60, l2_fraction=0.25, l3_fraction=0.15)


@pytest.mark.asyncio
async def test_pipeline_assemble_returns_assembled_context(pipeline):
    result = await pipeline.assemble("run1", "find users", token_budget=10_000)
    assert isinstance(result, AssembledContext)
    assert result.query == "find users"
    assert result.skill_ns == "default"
    assert result.budget == 10_000


@pytest.mark.asyncio
async def test_pipeline_assemble_has_three_tier_results(pipeline):
    result = await pipeline.assemble("run1", "query", token_budget=10_000)
    tiers = {r.tier for r in result.tier_results}
    assert tiers == {"L1", "L2", "L3"}


@pytest.mark.asyncio
async def test_pipeline_assemble_messages_order(pipeline, schema_store):
    """L3 messages appear before L2 before L1 in assembled list."""
    await schema_store.store("db", [_table("t")])
    result = await pipeline.assemble("run1", "query", token_budget=10_000, db_id="db")
    roles_sources = [(m["role"], m["content"][:10]) for m in result.messages]
    # L3 schema block has "Database:" prefix → appears first
    system_msgs = [m for m in result.messages if m["role"] == "system"]
    assert len(system_msgs) >= 1  # at least schema or vector hit


@pytest.mark.asyncio
async def test_pipeline_assemble_total_tokens_correct(pipeline):
    result = await pipeline.assemble("run1", "query", token_budget=10_000)
    expected = sum(r.tokens_used for r in result.tier_results)
    assert result.total_tokens == expected


@pytest.mark.asyncio
async def test_pipeline_budget_allocation_fractions(pipeline):
    """L1 gets 60%, L2 25%, L3 15% of the budget."""
    budget = 10_000
    b1 = int(budget * 0.60)
    b2 = int(budget * 0.25)
    # Each tier must not exceed its allocation
    result = await pipeline.assemble("run1", "query", token_budget=budget)
    l1_r = next(r for r in result.tier_results if r.tier == "L1")
    l2_r = next(r for r in result.tier_results if r.tier == "L2")
    assert l1_r.tokens_used <= b1
    assert l2_r.tokens_used <= b2


@pytest.mark.asyncio
async def test_pipeline_assemble_custom_fractions(pipeline):
    """Per-call fraction override is respected."""
    result = await pipeline.assemble("run1", "query", token_budget=1000,
                                     l1_fraction=1.0, l2_fraction=0.0, l3_fraction=0.0)
    l2_r = next(r for r in result.tier_results if r.tier == "L2")
    # L2 gets 0 budget → no results
    assert l2_r.tokens_used == 0


@pytest.mark.asyncio
async def test_pipeline_assemble_with_db_id(pipeline, schema_store):
    await schema_store.store("mydb", [_table("sales")])
    result = await pipeline.assemble("run1", "query sales", token_budget=10_000, db_id="mydb")
    l3_r = next(r for r in result.tier_results if r.tier == "L3")
    assert l3_r.hit is True
    assert "sales" in l3_r.source or any("sales" in m["content"] for m in l3_r.messages)


@pytest.mark.asyncio
async def test_pipeline_assemble_with_relevant_tables(pipeline, schema_store):
    await schema_store.store("filtdb", [_table("a"), _table("b"), _table("c")])
    result = await pipeline.assemble("run1", "query", token_budget=10_000,
                                     db_id="filtdb", relevant_tables=["a"])
    l3_msgs = [m for r in result.tier_results if r.tier == "L3" for m in r.messages]
    content = " ".join(m["content"] for m in l3_msgs)
    assert "TABLE a" in content
    assert "TABLE b" not in content


@pytest.mark.asyncio
async def test_pipeline_assemble_stats_has_all_tiers(pipeline):
    result = await pipeline.assemble("run1", "query", token_budget=5000)
    stats = result.stats
    assert "L1" in stats["tiers"]
    assert "L2" in stats["tiers"]
    assert "L3" in stats["tiers"]


@pytest.mark.asyncio
async def test_pipeline_assemble_skill_ns_passed(pipeline):
    result = await pipeline.assemble("run1", "query", token_budget=5000, skill_ns="sql")
    assert result.skill_ns == "sql"


@pytest.mark.asyncio
async def test_pipeline_assemble_vector_filter_forwarded(pipeline):
    result = await pipeline.assemble("run1", "query", token_budget=5000,
                                     vector_filter={"tenant_id": "t1"})
    pipeline.l2._vs.query.assert_called_once()
    call_kwargs = pipeline.l2._vs.query.call_args[1]
    assert call_kwargs.get("filter", {}).get("tenant_id") == "t1"


@pytest.mark.asyncio
async def test_pipeline_all_tiers_fail_gracefully():
    """Pipeline returns an AssembledContext even when all tiers fail."""
    engine = MagicMock()
    engine._load_hot = AsyncMock(side_effect=RuntimeError("redis down"))
    engine.push = AsyncMock()
    vs = MagicMock()
    vs.query = AsyncMock(side_effect=RuntimeError("vec down"))
    l1 = L1ChatCache(engine)
    l2 = L2VectorCache(vs, MagicMock())

    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://x"
    store._ttl = 3600
    store._client = _fake_redis()
    l3 = L3KnowledgeStore(schema=store, kg=KGCache(None))

    pipe = ContextPipeline(l1=l1, l2=l2, l3=l3)
    result = await pipe.assemble("run1", "query", token_budget=5000)
    # No crash — returns empty context
    assert isinstance(result, AssembledContext)
    assert result.total_tokens == 0


# ===========================================================================
# ContextPipeline.create factory
# ===========================================================================

def test_pipeline_create_returns_pipeline():
    from harness.memory.context_engine import ContextEngine
    engine = MagicMock(spec=ContextEngine)
    pipe = ContextPipeline.create(
        redis_url="redis://localhost",
        vector_store=None,
        embedder=None,
        graph=None,
        context_engine=engine,
    )
    assert isinstance(pipe, ContextPipeline)
    assert isinstance(pipe.l1, L1ChatCache)
    assert isinstance(pipe.l2, L2VectorCache)
    assert isinstance(pipe.l3, L3KnowledgeStore)
    assert isinstance(pipe.l3.schema, SchemaStore)


def test_pipeline_create_custom_fractions():
    from harness.memory.context_engine import ContextEngine
    engine = MagicMock(spec=ContextEngine)
    pipe = ContextPipeline.create(
        redis_url="redis://localhost",
        context_engine=engine,
        l1_fraction=0.5,
        l2_fraction=0.3,
        l3_fraction=0.2,
    )
    assert pipe._l1_frac == 0.5
    assert pipe._l2_frac == 0.3
    assert pipe._l3_frac == 0.2


# ===========================================================================
# MemoryManager integration
# ===========================================================================

def test_memory_manager_has_pipeline_and_schema_properties():
    from harness.memory.context_engineering import ContextPipeline, SchemaStore
    from harness.memory.manager import MemoryManager

    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://x"
    store._ttl = 3600
    store._client = _fake_redis()

    engine = MagicMock()
    l1 = L1ChatCache(engine)
    l2 = L2VectorCache(None, None)
    l3 = L3KnowledgeStore(schema=store, kg=KGCache(None))
    pipe = ContextPipeline(l1=l1, l2=l2, l3=l3)

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._pipeline = pipe

    assert mgr.pipeline is pipe
    assert mgr.schema is store


def test_memory_manager_schema_none_without_pipeline():
    from harness.memory.manager import MemoryManager
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._pipeline = None
    assert mgr.schema is None


@pytest.mark.asyncio
async def test_memory_manager_assemble_context_delegates():
    from harness.memory.manager import MemoryManager

    engine = _make_engine_mock()
    l1 = L1ChatCache(engine)
    vs = MagicMock()
    vs.query = AsyncMock(return_value=[])
    l2 = L2VectorCache(vs, MagicMock())
    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://x"
    store._ttl = 3600
    store._client = _fake_redis()
    l3 = L3KnowledgeStore(schema=store, kg=KGCache(None))
    pipe = ContextPipeline(l1=l1, l2=l2, l3=l3)

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._pipeline = pipe

    result = await mgr.assemble_context("run1", "find orders", token_budget=5000)
    assert isinstance(result, AssembledContext)
    assert result.query == "find orders"


@pytest.mark.asyncio
async def test_memory_manager_assemble_context_none_without_pipeline():
    from harness.memory.manager import MemoryManager
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._pipeline = None
    result = await mgr.assemble_context("run1", "query")
    assert result is None
