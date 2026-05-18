"""Unit tests for AriaSql — standalone SQL agent with self-correction."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from pathlib import Path

from harness.agents.nexus_sql import (
    NexusSql as AriaSql,
    _db_id,
    _extract_sql,
    _select_relevant_tables,
    _introspect_schema,
)
from harness.memory.context_engineering import SchemaStore


def _make_ctx(task="count employees", metadata=None):
    """Create a minimal AgentContext for unit tests."""
    from harness.core.context import AgentContext
    import tempfile
    return AgentContext.create(
        tenant_id="test",
        agent_type="ariasql",
        task=task,
        memory=None,
        workspace_path=Path(tempfile.mkdtemp()),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redis():
    return fakeredis.FakeRedis(decode_responses=True)


def _schema_store():
    store = SchemaStore.__new__(SchemaStore)
    store._redis_url = "redis://unused"
    store._ttl = 3600
    store._client = _redis()
    return store


def _sqlite_db(tmp_path: Path) -> str:
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL)")
    conn.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget REAL)")
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?)", [
        (1, "Alice", "Engineering", 90000),
        (2, "Bob",   "Sales",       70000),
        (3, "Carol", "Engineering", 85000),
    ])
    conn.executemany("INSERT INTO departments VALUES (?,?,?)", [
        (1, "Engineering", 500000),
        (2, "Sales",       300000),
    ])
    conn.commit()
    conn.close()
    return db


class _MockLLM:
    """Returns a fixed SQL response at temperature=0."""
    def __init__(self, sql="SELECT COUNT(*) FROM employees"):
        self._sql = sql
        self.call_count = 0

    async def complete(self, messages, **kwargs):
        assert kwargs.get("temperature", 0.0) == 0.0, "Must use temperature=0"
        self.call_count += 1
        r = MagicMock()
        r.content = self._sql
        return r


class _MockVerifier:
    """Returns a fixed VerificationResult."""
    def __init__(self, reward=0.9, verdict="correct", feedback="ok"):
        self._reward = reward
        self._verdict = verdict
        self._feedback = feedback
        self.call_count = 0

    async def verify(self, task, action, result=None, gold=None, **kwargs):
        self.call_count += 1
        from harness.improvement.rlvr.verifiers import VerificationResult, VerificationStep
        return VerificationResult(
            overall_reward=self._reward,
            verdict=self._verdict,
            steps=[VerificationStep("test", self._reward >= 0.5, self._reward, self._feedback)],
            feedback_for_agent=self._feedback,
        )


# ===========================================================================
# _db_id
# ===========================================================================

def test_db_id_from_path():
    assert _db_id("/data/mydb.sqlite") == "mydb"
    assert _db_id("/tmp/test.db") == "test"


def test_db_id_empty_string():
    assert _db_id("") == "db"


def test_db_id_no_extension():
    assert _db_id("/path/employees") == "employees"


# ===========================================================================
# _extract_sql
# ===========================================================================

def test_extract_sql_plain():
    assert _extract_sql("SELECT * FROM users") == "SELECT * FROM users"


def test_extract_sql_markdown_fences():
    raw = "```sql\nSELECT * FROM users\n```"
    result = _extract_sql(raw)
    assert "SELECT" in result.upper()
    assert "```" not in result


def test_extract_sql_with_explanation():
    raw = "Here is the SQL:\nSELECT COUNT(*) FROM users WHERE active=1"
    result = _extract_sql(raw)
    assert "SELECT" in result.upper()


def test_extract_sql_with_semicolon():
    result = _extract_sql("SELECT 1; DROP TABLE users")
    assert "DROP" not in result.upper()


def test_extract_sql_empty_fallback():
    result = _extract_sql("I cannot generate SQL for this request.")
    assert isinstance(result, str)


def test_extract_sql_with_clause():
    sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
    assert _extract_sql(sql) == sql


# ===========================================================================
# _select_relevant_tables
# ===========================================================================

def test_select_relevant_tables_small_db():
    tables = ["users", "orders", "products"]
    # Small enough — return None (all tables)
    result = _select_relevant_tables("count users", tables, max_tables=5)
    assert result is None


def test_select_relevant_tables_large_db():
    tables = [f"table_{i}" for i in range(50)] + ["users", "orders", "payments"]
    result = _select_relevant_tables("count users and orders", tables, max_tables=5)
    assert result is not None
    assert len(result) <= 5
    # Relevant tables should be near the top
    assert any("users" in t or "orders" in t for t in result)


def test_select_relevant_tables_no_match():
    tables = [f"table_{i}" for i in range(20)]
    result = _select_relevant_tables("xyz query", tables, max_tables=5)
    assert result is not None
    assert len(result) == 5


# ===========================================================================
# _introspect_schema
# ===========================================================================

@pytest.mark.asyncio
async def test_introspect_schema_returns_table_names(tmp_path):
    db = _sqlite_db(tmp_path)
    schema = await _introspect_schema(db, "count employees")
    assert "employees" in schema
    assert "departments" in schema


@pytest.mark.asyncio
async def test_introspect_schema_includes_columns(tmp_path):
    db = _sqlite_db(tmp_path)
    schema = await _introspect_schema(db, "query")
    assert "salary" in schema or "name" in schema


@pytest.mark.asyncio
async def test_introspect_schema_bad_path():
    schema = await _introspect_schema("/nonexistent/path.db", "query")
    assert "failed" in schema.lower() or "schema" in schema.lower()


# ===========================================================================
# AriaSql — schema context
# ===========================================================================

@pytest.mark.asyncio
async def test_ariasql_schema_context_from_store(tmp_path):
    db = _sqlite_db(tmp_path)
    store = _schema_store()
    await store.store_from_sqlite("testdb", db)

    agent = AriaSql(llm_provider=_MockLLM(), schema_store=store)
    ctx = await agent._schema_context("count employees", db, "testdb")
    assert "employees" in ctx
    assert "departments" in ctx


@pytest.mark.asyncio
async def test_ariasql_schema_context_auto_indexes(tmp_path):
    db = _sqlite_db(tmp_path)
    store = _schema_store()
    # Do NOT pre-index — agent should auto-index on first call

    agent = AriaSql(llm_provider=_MockLLM(), schema_store=store)
    ctx = await agent._schema_context("count employees", db, "testdb2")
    assert "employees" in ctx


@pytest.mark.asyncio
async def test_ariasql_schema_context_fallback_introspect(tmp_path):
    db = _sqlite_db(tmp_path)
    # No schema store — should fallback to live introspection
    agent = AriaSql(llm_provider=_MockLLM(), schema_store=None)
    ctx = await agent._schema_context("count employees", db, "testdb3")
    assert "employees" in ctx


# ===========================================================================
# AriaSql — generate_sql
# ===========================================================================

@pytest.mark.asyncio
async def test_generate_sql_returns_string():
    llm = _MockLLM("SELECT COUNT(*) FROM employees")
    agent = AriaSql(llm_provider=llm)
    sql = await agent.generate_sql("how many employees?")
    assert isinstance(sql, str)
    assert "SELECT" in sql.upper()


@pytest.mark.asyncio
async def test_generate_sql_uses_temperature_zero():
    llm = _MockLLM("SELECT 1")
    agent = AriaSql(llm_provider=llm)
    await agent.generate_sql("test")
    assert llm.call_count >= 1


@pytest.mark.asyncio
async def test_generate_sql_no_correction_when_high_score():
    llm = _MockLLM("SELECT COUNT(*) FROM employees")
    verifier = _MockVerifier(reward=0.95, verdict="correct")
    agent = AriaSql(llm_provider=llm, verifier=verifier, correction_threshold=0.60)
    sql = await agent.generate_sql("count employees")
    # High score → no correction → LLM called exactly once
    assert llm.call_count == 1
    assert verifier.call_count == 1


@pytest.mark.asyncio
async def test_generate_sql_triggers_correction_on_low_score():
    llm = _MockLLM("SELECT COUNT(*) FROM employees")
    verifier = _MockVerifier(reward=0.30, verdict="incorrect", feedback="wrong table")
    agent = AriaSql(
        llm_provider=llm, verifier=verifier,
        max_retries=2, correction_threshold=0.60,
    )
    await agent.generate_sql("count employees")
    # Initial + 2 correction retries = 3 LLM calls, 3 verify calls
    assert llm.call_count == 3
    assert verifier.call_count == 3


@pytest.mark.asyncio
async def test_generate_sql_returns_best_sql_across_retries():
    call_count = 0
    sqls = [
        "SELECT wrong FROM nowhere",
        "SELECT COUNT(*) FROM employees",
        "SELECT id FROM employees",
    ]
    rewards = [0.1, 0.9, 0.5]

    class ProgressiveLLM:
        async def complete(self, messages, **kwargs):
            nonlocal call_count
            r = MagicMock()
            r.content = sqls[min(call_count, len(sqls) - 1)]
            call_count += 1
            return r

    call_v = 0

    class ProgressiveVerifier:
        async def verify(self, task, action, **kwargs):
            nonlocal call_v
            from harness.improvement.rlvr.verifiers import VerificationResult, VerificationStep
            reward = rewards[min(call_v, len(rewards) - 1)]
            call_v += 1
            return VerificationResult(
                overall_reward=reward,
                verdict="correct" if reward >= 0.8 else "incorrect",
                steps=[VerificationStep("test", reward >= 0.5, reward, "feedback")],
                feedback_for_agent="try again",
            )

    agent = AriaSql(
        llm_provider=ProgressiveLLM(),
        verifier=ProgressiveVerifier(),
        max_retries=2, correction_threshold=0.60,
    )
    sql = await agent.generate_sql("count employees")
    # Best SQL is the one with reward 0.9 = sqls[1]
    assert "COUNT" in sql.upper()


@pytest.mark.asyncio
async def test_generate_sql_stops_after_first_good_score():
    llm = _MockLLM("SELECT COUNT(*) FROM employees")

    rewards_iter = iter([0.3, 0.95])  # first bad, second good

    class OnceVerifier:
        call_count = 0
        async def verify(self, task, action, **kwargs):
            from harness.improvement.rlvr.verifiers import VerificationResult, VerificationStep
            r = next(rewards_iter)
            OnceVerifier.call_count += 1
            return VerificationResult(
                overall_reward=r,
                verdict="correct" if r >= 0.8 else "incorrect",
                steps=[VerificationStep("t", r >= 0.5, r, "ok")],
                feedback_for_agent="ok",
            )

    agent = AriaSql(
        llm_provider=llm, verifier=OnceVerifier(),
        max_retries=3, correction_threshold=0.60,
    )
    await agent.generate_sql("test")
    assert OnceVerifier.call_count == 2  # stopped after second call (score=0.95)


@pytest.mark.asyncio
async def test_generate_sql_with_db_path(tmp_path):
    db = _sqlite_db(tmp_path)
    llm = _MockLLM("SELECT COUNT(*) FROM employees")
    agent = AriaSql(llm_provider=llm)
    sql = await agent.generate_sql("how many employees?", db_path=db)
    assert "SELECT" in sql.upper()


@pytest.mark.asyncio
async def test_generate_sql_with_schema_store_and_db(tmp_path):
    db = _sqlite_db(tmp_path)
    store = _schema_store()
    llm = _MockLLM("SELECT * FROM employees LIMIT 5")
    agent = AriaSql(llm_provider=llm, schema_store=store)

    sql = await agent.generate_sql("list employees", db_path=db, db_id="emp_db")
    assert isinstance(sql, str)
    # Schema was indexed
    tables = await store.table_names("emp_db")
    assert "employees" in tables


@pytest.mark.asyncio
async def test_generate_sql_llm_failure_returns_fallback():
    class FailLLM:
        async def complete(self, messages, **kwargs):
            raise RuntimeError("api error")

    agent = AriaSql(llm_provider=FailLLM())
    sql = await agent.generate_sql("count employees")
    assert sql == "SELECT 1"  # fallback


# ===========================================================================
# AriaSql — no verifier path
# ===========================================================================

@pytest.mark.asyncio
async def test_generate_sql_no_verifier_no_retries():
    llm = _MockLLM("SELECT 1")
    agent = AriaSql(llm_provider=llm, verifier=None)
    sql = await agent.generate_sql("test")
    assert llm.call_count == 1  # no verify → no retry


# ===========================================================================
# AriaSqlAgent — harness agent (unit-level, no full run loop)
# ===========================================================================

@pytest.mark.asyncio
async def test_aria_sql_agent_preload_schema(tmp_path):
    from harness.agents.aria_sql import AriaSqlAgent

    db = _sqlite_db(tmp_path)
    store = _schema_store()

    agent = AriaSqlAgent.__new__(AriaSqlAgent)
    agent._llm_router = None
    agent._memory = None
    agent._tool_registry = None
    agent._safety_pipeline = None
    agent._step_tracer = None
    agent._mlflow_tracer = None
    agent._failure_tracker = None
    agent._audit_logger = None
    agent._event_bus = None
    agent._cost_tracker = None
    agent._checkpoint_manager = None
    agent._message_bus = None
    agent._online_monitor = None
    agent._prompt_manager = None
    agent._trace_recorder = None
    agent._feedback_last_id = {}

    ctx = _make_ctx("count employees", {
        "db_path": db,
        "db_id": "emp_test",
        "schema_store": store,
    })

    await agent._preload_schema_context(ctx)

    assert ctx.metadata.get("_schema_preloaded") is True
    tables = await store.table_names("emp_test")
    assert "employees" in tables


@pytest.mark.asyncio
async def test_aria_sql_agent_relevant_tables_small_db(tmp_path):
    from harness.agents.aria_sql import AriaSqlAgent

    db = _sqlite_db(tmp_path)
    store = _schema_store()
    await store.store_from_sqlite("small_db", db)

    agent = AriaSqlAgent.__new__(AriaSqlAgent)
    ctx = _make_ctx("count employees")
    relevant = await agent._relevant_tables(ctx, "small_db", store)
    assert relevant is None  # 2 tables ≤ max_tables=12 → inject all


@pytest.mark.asyncio
async def test_aria_sql_agent_relevant_tables_large_db(tmp_path):
    from harness.agents.aria_sql import AriaSqlAgent
    from harness.memory.context_engineering import TableSchema, ColumnDef

    store = _schema_store()
    big_tables = [TableSchema(name=f"table_{i}", columns=[ColumnDef("id", "INTEGER")]) for i in range(30)]
    big_tables.append(TableSchema(name="employees", columns=[ColumnDef("id", "INTEGER"), ColumnDef("name", "TEXT")]))
    await store.store("big_db", big_tables)

    agent = AriaSqlAgent.__new__(AriaSqlAgent)
    ctx = _make_ctx("count employees in department")
    relevant = await agent._relevant_tables(ctx, "big_db", store)
    assert relevant is not None
    assert len(relevant) <= 12


def test_aria_sql_agent_build_system_prompt_contains_schema():
    from harness.agents.aria_sql import AriaSqlAgent

    agent = AriaSqlAgent.__new__(AriaSqlAgent)
    ctx = _make_ctx("count rows", {
        "db_id": "mydb",
        "_cached_schema_block": "TABLE employees\n  id INTEGER",
    })
    prompt = agent.build_system_prompt(ctx)
    assert "NexusSql" in prompt
    assert "employees" in prompt


def test_aria_sql_agent_build_system_prompt_no_schema():
    from harness.agents.aria_sql import AriaSqlAgent

    agent = AriaSqlAgent.__new__(AriaSqlAgent)
    ctx = _make_ctx("query")
    prompt = agent.build_system_prompt(ctx)
    assert "NexusSql" in prompt
    assert "SELECT" in prompt


# ===========================================================================
# bench_bird.py ariasql_agent integration
# ===========================================================================

@pytest.mark.asyncio
async def test_ariasql_mock_fallback_when_no_llm():
    """When no LLM is configured, AriaSql falls back to SELECT 1."""
    class FailLLM:
        async def complete(self, messages, **kwargs):
            raise RuntimeError("no api key")

    agent = AriaSql(llm_provider=FailLLM(), verifier=None)
    sql = await agent.generate_sql("count active employees")
    assert sql == "SELECT 1"
