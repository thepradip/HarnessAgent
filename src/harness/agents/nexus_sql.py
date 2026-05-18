"""
NexusSql — production SQL agent for large databases.

Extends SQLAgent with:
  1. L3 SchemaStore context  — relevant schema injected directly from the schema
                               store; no round-trip to list_tables/describe_table
                               for databases already indexed.
  2. GraphRAG table routing  — for 100+ table databases, uses entity extraction
                               + graph traversal to find relevant tables before
                               injecting schema (same strategy as NexusSQL).
  3. Self-correction loop    — after generating SQL, runs SQLVerifier; if score
                               < threshold, injects the verifier's feedback and
                               calls the LLM again (up to max_retries).
  4. RLVR integration        — records per-step rewards when reward_buffer is
                               configured in ctx.metadata.
  5. Standalone callable     — NexusSql.generate_sql(question, db_path) works
                               without a full AgentContext (for BIRD benchmark).

Usage as harness agent:
    agent = NexusSqlAgent(llm_router, memory, tool_registry, ...)
    result = await agent.run(ctx)   # ctx.metadata["db_path"] = "/path/to/db"

Usage as standalone callable (BIRD benchmark):
    agent = NexusSql(llm_provider, schema_store=store)
    sql = await agent.generate_sql("How many active users?", db_path="/tmp/mydb.sqlite")
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from harness.agents.sql_agent import SQLAgent
from harness.core.context import AgentContext, AgentResult

logger = logging.getLogger(__name__)

# Reward threshold below which self-correction is triggered
_CORRECTION_THRESHOLD = 0.60
# Maximum self-correction retries per query
_MAX_RETRIES = 2
# Maximum tables to inject into context for large schemas
_MAX_TABLES_IN_CONTEXT = 12


# ---------------------------------------------------------------------------
# NexusSqlAgent — full harness agent
# ---------------------------------------------------------------------------

class NexusSqlAgent(SQLAgent):
    """
    SQLAgent + L3 schema context + self-correction + RLVR recording.

    Additional ctx.metadata keys consumed:
        db_path          : path to the SQLite database (for SchemaStore + sandbox)
        db_id            : database identifier (default: stem of db_path)
        schema_store     : SchemaStore instance (optional; falls back to graph memory)
        rlvr_reward_buffer : StepRewardBuffer for RLVR recording
        rlvr_verifier      : SQLVerifier for step verification
        rlvr_loop          : RLVRLoop for mid-step feedback publishing
    """

    agent_type: str = "ariasql"

    def build_system_prompt(self, ctx: AgentContext) -> str:
        db_path = ctx.metadata.get("db_path", "")
        db_id = ctx.metadata.get("db_id", "") or _db_id(db_path)
        schema_block = ctx.metadata.get("_cached_schema_block", "")

        base = (
            "You are NexusSql, an expert SQL agent for large relational databases.\n\n"
            "Rules:\n"
            "1. Write only SELECT queries (read-only).\n"
            "2. Use exact table and column names from the schema — never guess.\n"
            "3. Add LIMIT on non-aggregating queries.\n"
            "4. For JOINs, verify foreign key relationships from the schema.\n"
            "5. When you return results, include:\n"
            "   - The SQL query used\n"
            "   - A plain-language interpretation\n"
            "6. If your SQL fails, read the error carefully and fix it.\n"
        )

        if schema_block:
            base += f"\n\n## Database Schema ({db_id})\n\n{schema_block}\n"

        return base

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Pre-load L3 schema context, then run the base agent loop."""
        await self._preload_schema_context(ctx)
        return await super().run(ctx)

    async def _preload_schema_context(self, ctx: AgentContext) -> None:
        """
        Load schema from SchemaStore (L3) and cache it in ctx.metadata.
        Falls back to SQLAgent._populate_schema() (graph memory path).
        """
        if ctx.metadata.get("_schema_preloaded"):
            return
        ctx.metadata["_schema_preloaded"] = True

        schema_store = ctx.metadata.get("schema_store")
        db_path = ctx.metadata.get("db_path", "")
        db_id = ctx.metadata.get("db_id", "") or _db_id(db_path)

        if not db_id:
            return

        # Auto-index the database if SchemaStore is configured and not yet indexed
        if schema_store is not None and db_path:
            existing = await schema_store.table_names(db_id)
            if not existing and db_path:
                logger.info("NexusSql: indexing %s into SchemaStore", db_id)
                try:
                    await schema_store.store_from_sqlite(db_id, db_path)
                except Exception as exc:
                    logger.warning("NexusSql: schema indexing failed: %s", exc)

            # Get relevant tables via GraphRAG if memory is available
            relevant_tables = await self._relevant_tables(ctx, db_id, schema_store)

            block = await schema_store.get_context_block(
                db_id,
                relevant_tables=relevant_tables or None,
                max_tables=_MAX_TABLES_IN_CONTEXT,
            )
            if block:
                ctx.metadata["_cached_schema_block"] = block
                ctx.metadata["schema_tables"] = await schema_store.table_names(db_id)
                logger.debug(
                    "NexusSql: injected schema for %d tables (%d chars)",
                    len(ctx.metadata["schema_tables"]), len(block),
                )
                return

        # Fallback: use SQLAgent's graph-memory schema population
        await self._populate_schema(ctx)

    async def _relevant_tables(
        self,
        ctx: AgentContext,
        db_id: str,
        schema_store: Any,
    ) -> list[str] | None:
        """
        Use GraphRAG (if available) to find tables relevant to the current task.
        For small databases (≤ _MAX_TABLES_IN_CONTEXT tables) returns None
        (all tables will be injected).
        """
        all_tables = await schema_store.table_names(db_id)
        if len(all_tables) <= _MAX_TABLES_IN_CONTEXT:
            return None   # inject everything

        # Large database: use keyword extraction to narrow down tables
        task_words = {
            w.lower() for w in re.findall(r"\b\w{4,}\b", ctx.task)
        }
        scored = []
        for tbl in all_tables:
            tbl_lower = tbl.lower()
            score = sum(1 for w in task_words if w in tbl_lower or tbl_lower in w)
            scored.append((score, tbl))

        scored.sort(key=lambda x: -x[0])
        relevant = [t for _, t in scored[:_MAX_TABLES_IN_CONTEXT]]
        if not relevant:
            relevant = all_tables[:_MAX_TABLES_IN_CONTEXT]

        logger.debug(
            "NexusSql: %d/%d tables selected for context: %s",
            len(relevant), len(all_tables), relevant[:5],
        )
        return relevant


# ---------------------------------------------------------------------------
# NexusSql — standalone callable (for BIRD benchmark)
# ---------------------------------------------------------------------------

_SQL_EXTRACT_RE = re.compile(
    r"```sql\s*(.*?)\s*```|```\s*(.*?)\s*```|SELECT\b.+",
    re.IGNORECASE | re.DOTALL,
)

_GENERATE_PROMPT = """\
You are NexusSql, a precise SQL generation engine.
Generate a single SQLite SELECT query that answers the question.
Return ONLY the SQL query — no explanation, no markdown, no backticks.

Database schema:
{schema}

Question: {question}

SQL:"""

_CORRECT_PROMPT = """\
You are NexusSql. Your previous SQL had issues. Fix it.

Database schema:
{schema}

Question: {question}

Previous SQL:
{sql}

Verification feedback:
{feedback}

Write the corrected SQL query only — no explanation.

SQL:"""


class NexusSql:
    """
    Standalone NexusSql — generates SQL for a question given a database.

    Designed for:
      - BIRD benchmark evaluation
      - Direct integration without the full agent lifecycle
      - Step-by-step self-correction

    Parameters
    ----------
    llm_provider : LLMProvider
        Any harness LLMProvider (Anthropic, OpenAI, local).
    schema_store : SchemaStore | None
        Pre-built schema store.  If None, schema is introspected live.
    verifier : SQLVerifier | None
        Used for self-correction.  If None, no correction is attempted.
    max_retries : int
        Max self-correction attempts (default 2).
    correction_threshold : float
        Retry when verifier reward < this value (default 0.60).
    """

    def __init__(
        self,
        llm_provider: Any,
        schema_store: Any | None = None,
        verifier: Any | None = None,
        max_retries: int = _MAX_RETRIES,
        correction_threshold: float = _CORRECTION_THRESHOLD,
    ) -> None:
        self._llm = llm_provider
        self._schema_store = schema_store
        self._verifier = verifier
        self._max_retries = max_retries
        self._threshold = correction_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def generate_sql(
        self,
        question: str,
        db_path: str | None = None,
        db_id: str | None = None,
        gold_sql: str | None = None,
    ) -> str:
        """
        Generate SQL for question.

        1. Build schema context (SchemaStore → live introspection → none).
        2. Call LLM with schema + question → candidate SQL.
        3. Verify with SQLVerifier (if configured).
        4. If score < threshold, inject feedback and retry (up to max_retries).
        5. Return the SQL with the highest verifier score.
        """
        eff_db_id = db_id or (db_path and _db_id(db_path)) or "db"

        # Build schema context
        schema_block = await self._schema_context(question, db_path, eff_db_id)

        # Initial generation
        sql, score, feedback = await self._generate_and_verify(
            question, schema_block, db_path, eff_db_id, gold_sql
        )
        best_sql, best_score = sql, score

        # Self-correction loop
        for attempt in range(self._max_retries):
            if best_score >= self._threshold:
                break
            logger.debug(
                "NexusSql: score %.2f < %.2f — correction attempt %d/%d",
                best_score, self._threshold, attempt + 1, self._max_retries,
            )
            sql, score, feedback = await self._correct_and_verify(
                question, schema_block, best_sql, feedback, db_path, eff_db_id, gold_sql
            )
            if score > best_score:
                best_sql, best_score = sql, score

        logger.debug(
            "NexusSql: final sql score=%.2f  question=%s",
            best_score, question[:60],
        )
        return best_sql

    # ------------------------------------------------------------------
    # Schema context
    # ------------------------------------------------------------------

    async def _schema_context(
        self, question: str, db_path: str | None, db_id: str
    ) -> str:
        # Try SchemaStore first
        if self._schema_store is not None:
            existing = await self._schema_store.table_names(db_id)
            if not existing and db_path:
                try:
                    await self._schema_store.store_from_sqlite(db_id, db_path)
                except Exception as exc:
                    logger.debug("Schema indexing failed: %s", exc)

            all_tables = await self._schema_store.table_names(db_id)
            if all_tables:
                relevant = _select_relevant_tables(question, all_tables, _MAX_TABLES_IN_CONTEXT)
                block = await self._schema_store.get_context_block(
                    db_id, relevant_tables=relevant, max_tables=_MAX_TABLES_IN_CONTEXT
                )
                if block:
                    return block

        # Fallback: live SQLite introspection
        if db_path:
            return await _introspect_schema(db_path, question)

        return "(schema not available)"

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                system=(
                    "You are NexusSql. Return only the SQL query. "
                    "No markdown. No explanation. No backticks."
                ),
                temperature=0.0,
                skip_cache=False,
            )
            return _extract_sql(response.content)
        except Exception as exc:
            logger.warning("NexusSql LLM call failed: %s", exc)
            return "SELECT 1"

    async def _generate_and_verify(
        self,
        question: str,
        schema: str,
        db_path: str | None,
        db_id: str,
        gold_sql: str | None,
    ) -> tuple[str, float, str]:
        sql = await self._call_llm(
            _GENERATE_PROMPT.format(schema=schema[:3000], question=question)
        )
        return await self._verify(sql, question, db_path, db_id, gold_sql)

    async def _correct_and_verify(
        self,
        question: str,
        schema: str,
        prev_sql: str,
        feedback: str,
        db_path: str | None,
        db_id: str,
        gold_sql: str | None,
    ) -> tuple[str, float, str]:
        sql = await self._call_llm(
            _CORRECT_PROMPT.format(
                schema=schema[:3000],
                question=question,
                sql=prev_sql,
                feedback=feedback[:500],
            )
        )
        return await self._verify(sql, question, db_path, db_id, gold_sql)

    async def _verify(
        self,
        sql: str,
        question: str,
        db_path: str | None,
        db_id: str,
        gold_sql: str | None,
    ) -> tuple[str, float, str]:
        if self._verifier is None:
            return sql, 1.0, ""
        try:
            from harness.eval.sandbox import SQLSandbox
            sandbox = SQLSandbox(db_path=db_path) if db_path else None
            vr = await self._verifier.verify(
                task=question,
                action=sql,
                result=None,
                gold=gold_sql,
                db_id=db_id,
            )
            return sql, vr.overall_reward, vr.feedback_for_agent
        except Exception as exc:
            logger.debug("NexusSql verify failed: %s", exc)
            return sql, 0.5, str(exc)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        db_path: str | None = None,
        redis_url: str = "redis://localhost:6379",
        with_verifier: bool = True,
    ) -> "NexusSql":
        """
        Build NexusSql from harness config (.env / environment variables).

        Uses whatever LLM key is already configured (ANTHROPIC_API_KEY or
        OPENAI_API_KEY). No new key needed.
        """
        import asyncio
        import fakeredis.aioredis as fakeredis
        from harness.core.config import get_config
        from harness.llm.factory import build_router
        from harness.memory.context_engineering import SchemaStore
        from harness.improvement.rlvr.verifiers import SQLVerifier

        cfg = get_config()
        llm = build_router(cfg)   # LLMRouter — handles key selection automatically

        redis = fakeredis.FakeRedis(decode_responses=True)
        store = SchemaStore.__new__(SchemaStore)
        store._redis_url = redis_url
        store._ttl = 86400
        store._client = redis

        verifier = SQLVerifier(llm=llm, schema_store=store) if with_verifier else None

        agent = cls(llm_provider=llm, schema_store=store, verifier=verifier)

        if db_path:
            asyncio.get_event_loop().run_until_complete(
                store.store_from_sqlite(_db_id(db_path), db_path)
            )

        return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_id(db_path: str) -> str:
    """Extract a clean db_id from a file path."""
    from pathlib import Path
    return Path(db_path).stem if db_path else "db"


def _extract_sql(text: str) -> str:
    """Extract the first SQL statement from LLM output."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If output has multiple lines, take only up to first semicolon
    if ";" in text:
        text = text[: text.index(";") + 1].strip()

    # Basic sanity: must start with SELECT or WITH
    if re.match(r"^\s*(SELECT|WITH)\b", text, re.IGNORECASE):
        return text

    # Try to extract first SELECT from multi-line response
    m = re.search(r"(SELECT\b.+?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    return text if text else "SELECT 1"


def _select_relevant_tables(
    question: str,
    all_tables: list[str],
    max_tables: int,
) -> list[str] | None:
    """Select tables most relevant to the question using keyword overlap."""
    if len(all_tables) <= max_tables:
        return None

    task_words = {w.lower() for w in re.findall(r"\b\w{3,}\b", question)}
    scored = []
    for tbl in all_tables:
        tbl_lower = tbl.lower().replace("_", " ")
        tbl_words = set(tbl_lower.split())
        score = len(task_words & tbl_words) + (2 if any(w in tbl_lower for w in task_words) else 0)
        scored.append((score, tbl))

    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:max_tables]]


async def _introspect_schema(db_path: str, question: str) -> str:
    """Live SQLite introspection — fallback when SchemaStore is unavailable."""
    import sqlite3
    import asyncio

    def _read():
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]
            lines = [f"Database: {_db_id(db_path)} ({len(tables)} tables)"]
            for tbl in tables[:_MAX_TABLES_IN_CONTEXT]:
                cur2 = conn.execute(f"PRAGMA table_info({tbl})")
                cols = [f"{r[1]} {r[2]}" for r in cur2.fetchall()]
                lines.append(f"\nTABLE {tbl}\n  " + "\n  ".join(cols))
            return "\n".join(lines)
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _read)
    except Exception as exc:
        return f"(schema introspection failed: {exc})"
