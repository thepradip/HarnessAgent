"""Production SQL tools for HarnessAgent agents.

All tools are read-only by default. Connection pool is shared across tool instances.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from harness.core.context import AgentContext, ToolResult
from harness.core.errors import FailureClass, ToolError

logger = logging.getLogger(__name__)

# Regex to detect non-SELECT DML/DDL statements
_WRITE_STMT_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|CALL|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

# Write/DDL keywords that must not appear *anywhere* in a read-only statement
# (catches e.g. ``WITH t AS (SELECT 1) DELETE FROM users`` which bypasses the
# first-keyword prefix check above). Checked after stripping string literals
# and comments so legitimate SELECTs containing these words as data pass.
_WRITE_WORD_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE"
    r"|REPLACE|EXEC|EXECUTE|CALL|ATTACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

# String literals ('' / "" with doubled-quote escapes) and SQL comments
_SQL_STRING_OR_COMMENT_RE = re.compile(
    r"'(?:[^']|'')*'"        # single-quoted string literal
    r"|\"(?:[^\"]|\"\")*\""  # double-quoted identifier/string
    r"|--[^\n]*"             # line comment
    r"|/\*.*?\*/",           # block comment
    re.DOTALL,
)

# Valid SQL identifier (used for table_name and schema arguments)
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Regex to detect absence of LIMIT clause
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _strip_strings_and_comments(sql: str) -> str:
    """Replace string literals and comments with spaces for keyword scanning."""
    return _SQL_STRING_OR_COMMENT_RE.sub(" ", sql)


def _find_write_keyword(sql: str) -> str | None:
    """Return the first write/DDL keyword found anywhere in the SQL, or None.

    String literals and comments are stripped first so a SELECT containing
    e.g. ``WHERE note = 'please DELETE me'`` is not rejected.
    """
    match = _WRITE_WORD_RE.search(_strip_strings_and_comments(sql))
    return match.group(1).upper() if match else None


@dataclass
class SQLConnectionConfig:
    """Configuration for an async SQL database connection."""

    connection_string: str  # SQLAlchemy async URL e.g. "postgresql+asyncpg://..."
    read_only: bool = True
    max_rows: int = 1000
    query_timeout: float = 30.0


class _SharedPool:
    """Lazily initialised async SQLAlchemy engine shared across tools."""

    def __init__(self, config: SQLConnectionConfig) -> None:
        self._config = config
        self._engine: Any = None

    async def get_engine(self) -> Any:
        """Return or create the async SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy.ext.asyncio import create_async_engine

                connect_args: dict[str, Any] = {}
                if self._config.query_timeout:
                    connect_args["command_timeout"] = self._config.query_timeout

                self._engine = create_async_engine(
                    self._config.connection_string,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    connect_args=connect_args,
                )
                logger.info("Created async SQL engine for %s", self._config.connection_string.split("@")[-1])
            except Exception as exc:
                raise ToolError(
                    f"Failed to create database engine: {exc}",
                    tool_name="sql_pool",
                    failure_class=FailureClass.TOOL_EXEC_ERROR,
                    context={"error": str(exc)},
                ) from exc
        return self._engine

    async def execute_query(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return rows as a list of dicts."""
        import sqlalchemy

        engine = await self.get_engine()
        async with engine.connect() as conn:
            # Engine-level read-only enforcement where the dialect supports it:
            # for Postgres, make the implicit transaction READ ONLY so any
            # write that slips past the keyword checks is rejected by the DB.
            if self._config.read_only and _is_postgres(self._config.connection_string):
                try:
                    await conn.execute(sqlalchemy.text("SET TRANSACTION READ ONLY"))
                except Exception as exc:
                    logger.debug("SET TRANSACTION READ ONLY failed: %s", exc)
            result = await conn.execute(
                sqlalchemy.text(sql), params or {}
            )
            if result.returns_rows:
                keys = list(result.keys())
                rows = [dict(zip(keys, row)) for row in result.fetchall()]
                return rows
            return []

    async def close(self) -> None:
        """Dispose the engine and its connection pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


def _add_limit(sql: str, limit: int) -> str:
    """Append a LIMIT clause if one is not already present."""
    if not _LIMIT_RE.search(sql):
        return f"{sql.rstrip(';')} LIMIT {limit}"
    return sql


def _is_sqlite(connection_string: str) -> bool:
    return "sqlite" in connection_string.lower()


def _is_postgres(connection_string: str) -> bool:
    return "postgres" in connection_string.lower() or "pg" in connection_string.lower()


class ExecuteQueryTool:
    """Execute a SQL SELECT query against the configured database."""

    name = "execute_sql"
    description = (
        "Execute a SQL query against the database. "
        "Returns results as a list of dicts. "
        "Only SELECT queries are allowed in read-only mode."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The SQL query to execute."},
            "limit": {
                "type": "integer",
                "default": 100,
                "maximum": 1000,
                "description": "Maximum number of rows to return.",
            },
        },
        "required": ["sql"],
    }
    timeout_seconds: float = 30.0

    def __init__(self, pool: _SharedPool, config: SQLConnectionConfig) -> None:
        self._pool = pool
        self._config = config

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Execute the SQL query and return rows as a list of dicts."""
        sql: str = args["sql"].strip()
        limit: int = min(int(args.get("limit", 100)), self._config.max_rows)

        # Reject non-SELECT in read-only mode
        if self._config.read_only:
            if _WRITE_STMT_RE.match(sql):
                error = (
                    "Only SELECT queries are permitted in read-only mode. "
                    f"Received: {sql[:80]!r}"
                )
                return ToolResult(data=None, error=error)
            # Catch write keywords buried anywhere in the statement
            # (e.g. WITH t AS (SELECT 1) DELETE FROM users)
            write_kw = _find_write_keyword(sql)
            if write_kw is not None:
                error = (
                    f"Write keyword '{write_kw}' is not permitted in read-only mode. "
                    f"Received: {sql[:80]!r}"
                )
                return ToolResult(data=None, error=error)

        # Add LIMIT if missing on SELECT
        if sql.upper().lstrip().startswith("SELECT") and not _LIMIT_RE.search(sql):
            sql = _add_limit(sql, limit)

        import time as _time
        table_names = _extract_table_names(sql)
        t0 = _time.monotonic()
        error_message: str | None = None

        try:
            rows = await self._pool.execute_query(sql)
            if len(rows) > limit:
                rows = rows[:limit]
            latency_ms = (_time.monotonic() - t0) * 1000

            # Record successful query in the knowledge graph
            if ctx.memory is not None:
                try:
                    graph_rag = getattr(ctx.memory, "_graph_rag", None)
                    if graph_rag is not None:
                        await graph_rag.record_query(
                            query_sql=sql,
                            tables_used=table_names,
                            run_id=ctx.run_id,
                            tenant_id=ctx.tenant_id,
                            success=True,
                            latency_ms=latency_ms,
                        )
                    else:
                        # Fallback: add simple graph facts
                        for table_name in table_names:
                            await ctx.memory.add_fact(
                                f"query:{ctx.run_id}", "uses", f"table:{table_name}"
                            )
                except Exception as mem_exc:
                    logger.debug("Graph record_query failed: %s", mem_exc)

            return ToolResult(
                data=rows,
                metadata={"row_count": len(rows), "sql": sql, "latency_ms": round(latency_ms, 1)},
            )
        except ToolError:
            raise
        except Exception as exc:
            error_message = str(exc)
            latency_ms = (_time.monotonic() - t0) * 1000
            logger.exception("SQL execution failed: %s", exc)

            # Record failed query in graph so future retrievals surface the error pattern
            if ctx.memory is not None:
                try:
                    graph_rag = getattr(ctx.memory, "_graph_rag", None)
                    if graph_rag is not None:
                        await graph_rag.record_query(
                            query_sql=sql,
                            tables_used=table_names,
                            run_id=ctx.run_id,
                            tenant_id=ctx.tenant_id,
                            success=False,
                            error_message=error_message,
                            latency_ms=latency_ms,
                        )
                except Exception:
                    pass

            return ToolResult(
                data=None,
                error=f"Query execution failed: {exc}",
            )


class ListTablesTool:
    """List all tables in the database with their names and optional descriptions."""

    name = "list_tables"
    description = (
        "List all tables (and views) in the database with their row counts."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "schema": {
                "type": "string",
                "description": "Database schema name (default: public for Postgres).",
            }
        },
        "required": [],
    }
    timeout_seconds: float = 15.0

    def __init__(self, pool: _SharedPool, config: SQLConnectionConfig) -> None:
        self._pool = pool
        self._config = config

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Query information_schema to list all tables."""
        schema = args.get("schema", "public")

        # Sanitize schema to prevent injection
        if not _IDENTIFIER_RE.match(schema):
            return ToolResult(data=None, error=f"Invalid schema name: {schema!r}")

        if _is_sqlite(self._config.connection_string):
            sql = (
                "SELECT name AS table_name, 'table' AS table_type "
                "FROM sqlite_master WHERE type IN ('table','view') "
                "ORDER BY name"
            )
        else:
            sql = (
                "SELECT table_name, table_type, table_schema "
                f"FROM information_schema.tables "
                f"WHERE table_schema = '{schema}' "
                "ORDER BY table_name"
            )

        try:
            rows = await self._pool.execute_query(sql)
            return ToolResult(
                data=rows,
                metadata={"table_count": len(rows), "schema": schema},
            )
        except Exception as exc:
            logger.exception("list_tables failed: %s", exc)
            return ToolResult(data=None, error=f"list_tables failed: {exc}")


class DescribeTableTool:
    """Show columns, types, constraints, and sample count for a table."""

    name = "describe_table"
    description = (
        "Show columns, data types, constraints, and approximate row count "
        "for a specific table."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to describe.",
            },
            "schema": {
                "type": "string",
                "description": "Database schema (default: public).",
            },
        },
        "required": ["table_name"],
    }
    timeout_seconds: float = 15.0

    def __init__(self, pool: _SharedPool, config: SQLConnectionConfig) -> None:
        self._pool = pool
        self._config = config

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Query information_schema for column metadata."""
        table_name = args["table_name"].strip()
        schema = args.get("schema", "public")

        # Sanitize table_name and schema to prevent injection
        if not _IDENTIFIER_RE.match(table_name):
            return ToolResult(
                data=None,
                error=f"Invalid table name: {table_name!r}",
            )
        if not _IDENTIFIER_RE.match(schema):
            return ToolResult(data=None, error=f"Invalid schema name: {schema!r}")

        try:
            if _is_sqlite(self._config.connection_string):
                # Use PRAGMA for SQLite
                columns_sql = f"PRAGMA table_info({table_name})"
                columns = await self._pool.execute_query(columns_sql)
                col_info = [
                    {
                        "column_name": c.get("name"),
                        "data_type": c.get("type"),
                        "is_nullable": "NO" if c.get("notnull") else "YES",
                        "column_default": c.get("dflt_value"),
                        "is_primary_key": bool(c.get("pk")),
                    }
                    for c in columns
                ]
                count_sql = f"SELECT COUNT(*) AS cnt FROM {table_name}"
                count_rows = await self._pool.execute_query(count_sql)
                row_count = count_rows[0].get("cnt", "unknown") if count_rows else "unknown"
            else:
                columns_sql = (
                    "SELECT column_name, data_type, is_nullable, column_default, "
                    "character_maximum_length "
                    "FROM information_schema.columns "
                    f"WHERE table_name = '{table_name}' "
                    f"AND table_schema = '{schema}' "
                    "ORDER BY ordinal_position"
                )
                col_info = await self._pool.execute_query(columns_sql)

                # Foreign key info
                fk_sql = (
                    "SELECT kcu.column_name, ccu.table_name AS foreign_table, "
                    "ccu.column_name AS foreign_column "
                    "FROM information_schema.table_constraints AS tc "
                    "JOIN information_schema.key_column_usage AS kcu "
                    "ON tc.constraint_name = kcu.constraint_name "
                    "JOIN information_schema.constraint_column_usage AS ccu "
                    "ON ccu.constraint_name = tc.constraint_name "
                    f"WHERE tc.table_name = '{table_name}' "
                    "AND tc.constraint_type = 'FOREIGN KEY'"
                )
                try:
                    fk_rows = await self._pool.execute_query(fk_sql)
                except Exception:
                    fk_rows = []

                # Approx row count
                try:
                    count_rows = await self._pool.execute_query(
                        f"SELECT COUNT(*) AS cnt FROM {schema}.{table_name}"
                    )
                    row_count = count_rows[0].get("cnt", "unknown") if count_rows else "unknown"
                except Exception:
                    row_count = "unknown"

                return ToolResult(
                    data={
                        "table_name": table_name,
                        "schema": schema,
                        "columns": col_info,
                        "foreign_keys": fk_rows,
                        "approximate_row_count": row_count,
                    }
                )

            return ToolResult(
                data={
                    "table_name": table_name,
                    "columns": col_info,
                    "approximate_row_count": row_count,
                }
            )

        except Exception as exc:
            logger.exception("describe_table failed for '%s': %s", table_name, exc)
            return ToolResult(data=None, error=f"describe_table failed: {exc}")


class SampleRowsTool:
    """Return a small sample of rows from a table to understand data shape."""

    name = "sample_rows"
    description = (
        "Return a small sample of rows (default 5) from a table to understand "
        "the data shape and example values."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to sample.",
            },
            "n": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Number of sample rows to return.",
            },
            "schema": {
                "type": "string",
                "description": "Database schema (default: public).",
            },
        },
        "required": ["table_name"],
    }
    timeout_seconds: float = 15.0

    def __init__(self, pool: _SharedPool, config: SQLConnectionConfig) -> None:
        self._pool = pool
        self._config = config

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Fetch n random/first rows from the table."""
        table_name = args["table_name"].strip()
        n = min(int(args.get("n", 5)), 20)
        schema = args.get("schema", "public")

        if not _IDENTIFIER_RE.match(table_name):
            return ToolResult(data=None, error=f"Invalid table name: {table_name!r}")
        if not _IDENTIFIER_RE.match(schema):
            return ToolResult(data=None, error=f"Invalid schema name: {schema!r}")

        try:
            if _is_sqlite(self._config.connection_string):
                sql = f"SELECT * FROM {table_name} LIMIT {n}"
            else:
                sql = f"SELECT * FROM {schema}.{table_name} LIMIT {n}"

            rows = await self._pool.execute_query(sql)
            return ToolResult(
                data=rows,
                metadata={"table_name": table_name, "sampled_rows": len(rows)},
            )
        except Exception as exc:
            logger.exception("sample_rows failed for '%s': %s", table_name, exc)
            return ToolResult(data=None, error=f"sample_rows failed: {exc}")


def _extract_table_names(sql: str) -> list[str]:
    """Extract table names referenced in a SQL query (simple heuristic)."""
    # Match FROM and JOIN clauses
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.IGNORECASE
    )
    matches = pattern.findall(sql)
    # Strip schema prefix if present
    tables = []
    for m in matches:
        parts = m.split(".")
        tables.append(parts[-1])
    return list(set(tables))


def build_sql_tools(config: SQLConnectionConfig) -> list:
    """Build all SQL tools sharing a single connection pool.

    Returns a list of [ExecuteQueryTool, ListTablesTool, DescribeTableTool, SampleRowsTool].
    """
    pool = _SharedPool(config)
    return [
        ExecuteQueryTool(pool=pool, config=config),
        ListTablesTool(pool=pool, config=config),
        DescribeTableTool(pool=pool, config=config),
        SampleRowsTool(pool=pool, config=config),
    ]
