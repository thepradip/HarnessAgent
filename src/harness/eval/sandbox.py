"""EvalSandbox protocol and pluggable backends for any agentic app evaluation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of a sandboxed action execution."""

    output: Any              # structured result (rows dict, stdout str, return value, response body)
    raw_text: str            # serialised for scorer/LLM consumption
    execution_time_ms: float
    error: str | None = None
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.error is None


@runtime_checkable
class EvalSandbox(Protocol):
    """Execute an action and return a SandboxResult. Any backend implements this."""

    async def execute(self, action: str, **kwargs: Any) -> SandboxResult: ...

    async def is_available(self) -> bool: ...


# ---------------------------------------------------------------------------
# SQL backend
# ---------------------------------------------------------------------------

class SQLSandbox:
    """Read-only SQL execution against SQLite or PostgreSQL for evaluation."""

    def __init__(
        self,
        db_path: str | None = None,
        timeout: float = 30.0,
        max_rows: int = 1_000,
    ) -> None:
        self._db_path = db_path
        self._timeout = timeout
        self._max_rows = max_rows

    async def execute(self, action: str, db_path: str | None = None, **_: Any) -> SandboxResult:
        path = db_path or self._db_path
        if not path:
            return SandboxResult(output={}, raw_text="", execution_time_ms=0.0,
                                 error="No db_path provided to SQLSandbox")

        try:
            self._reject_unsafe(action)
        except Exception as exc:
            return SandboxResult(output={}, raw_text="",
                                 execution_time_ms=0.0,
                                 error=str(exc))

        loop = asyncio.get_running_loop()
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_sync, path, action),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            return SandboxResult(
                output={}, raw_text="",
                execution_time_ms=self._timeout * 1000,
                error=f"SQL execution timed out after {self._timeout}s",
            )
        except Exception as exc:
            return SandboxResult(
                output={}, raw_text="",
                execution_time_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )

        elapsed = (time.monotonic() - start) * 1000
        truncated = result["row_count"] > self._max_rows
        rows = result["rows"][:self._max_rows]
        output = {"columns": result["columns"], "rows": rows, "row_count": result["row_count"]}
        raw = self._to_text(result["columns"], rows, truncated)
        return SandboxResult(output=output, raw_text=raw,
                             execution_time_ms=elapsed, truncated=truncated)

    def _run_sync(self, db_path: str, sql: str) -> dict:
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cur = conn.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            rows = [list(r) for r in cur.fetchall()]
            return {"columns": columns, "rows": rows, "row_count": len(rows)}
        finally:
            conn.close()

    @staticmethod
    def _reject_unsafe(sql: str) -> None:
        try:
            import sqlglot
            stmts = sqlglot.parse(sql)
            for stmt in stmts:
                if stmt is None:
                    continue
                kind = stmt.key.upper()
                if kind in {"CREATE", "DROP", "ALTER", "INSERT", "UPDATE", "DELETE",
                             "TRUNCATE", "REPLACE", "MERGE", "GRANT", "REVOKE"}:
                    from harness.core.errors import SafetyViolation
                    raise SafetyViolation(
                        f"SQLSandbox: DDL/DML not allowed in eval ({kind})",
                        guard_source="sql_sandbox",
                    )
        except ImportError:
            # Fallback: simple keyword check
            upper = sql.strip().upper()
            for kw in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ", "TRUNCATE "):
                if upper.startswith(kw) or f" {kw}" in upper:
                    from harness.core.errors import SafetyViolation
                    raise SafetyViolation(
                        f"SQLSandbox: DDL/DML not allowed in eval",
                        guard_source="sql_sandbox",
                    )

    @staticmethod
    def _to_text(columns: list[str], rows: list[list], truncated: bool) -> str:
        if not columns:
            return "(no results)"
        header = " | ".join(columns)
        sep = "-" * len(header)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        suffix = "\n...[truncated]" if truncated else ""
        return f"{header}\n{sep}\n{body}{suffix}"

    async def is_available(self) -> bool:
        return self._db_path is not None


# ---------------------------------------------------------------------------
# Code backend
# ---------------------------------------------------------------------------

class CodeSandbox:
    """Python/shell execution for evaluation. Wraps DockerSandbox with fallback."""

    def __init__(self, timeout: float = 30.0, workload: str = "general") -> None:
        self._timeout = timeout
        self._workload = workload

    async def execute(self, action: str, language: str = "python", **_: Any) -> SandboxResult:
        from pathlib import Path
        from harness.filesystem.sandbox import DockerSandbox, RestrictedPythonExecutor, memory_for_workload

        start = time.monotonic()

        if await DockerSandbox.is_available():
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                docker = DockerSandbox(
                    timeout=self._timeout,
                    memory_limit=memory_for_workload(self._workload),
                )
                res = await docker.run_code(action, ws)
                elapsed = (time.monotonic() - start) * 1000
                output = {"stdout": res.stdout, "stderr": res.stderr, "exit_code": res.exit_code}
                raw = res.stdout or res.stderr
                if res.oom_killed:
                    error = "OOM: container exceeded memory limit"
                elif not res.success:
                    error = f"exit_code={res.exit_code}: {res.stderr[:300]}"
                else:
                    error = None
                return SandboxResult(output=output, raw_text=raw,
                                     execution_time_ms=elapsed, error=error)

        executor = RestrictedPythonExecutor(timeout=self._timeout)
        res = await executor.run_code(action)
        elapsed = (time.monotonic() - start) * 1000
        output = {"stdout": res.stdout, "stderr": res.stderr, "exit_code": res.exit_code}
        raw = res.stdout or res.stderr
        error = None if res.success else f"exit_code={res.exit_code}: {res.stderr[:300]}"
        return SandboxResult(output=output, raw_text=raw,
                             execution_time_ms=elapsed, error=error)

    async def is_available(self) -> bool:
        from harness.filesystem.sandbox import DockerSandbox
        return await DockerSandbox.is_available()


# ---------------------------------------------------------------------------
# Tool-call backend
# ---------------------------------------------------------------------------

class ToolCallSandbox:
    """Execute a named tool from the ToolRegistry and capture its output."""

    def __init__(self, tool_registry: Any, ctx: Any) -> None:
        self._registry = tool_registry
        self._ctx = ctx

    async def execute(self, action: str, args: dict | None = None, **_: Any) -> SandboxResult:
        from harness.core.context import ToolCall
        start = time.monotonic()
        call = ToolCall(name=action, args=args or {})
        try:
            result = await self._registry.execute(self._ctx, call)
            elapsed = (time.monotonic() - start) * 1000
            raw = result.to_text() if hasattr(result, "to_text") else str(result.data)
            error = result.error if result.is_error else None
            return SandboxResult(output=result.data, raw_text=raw,
                                 execution_time_ms=elapsed, error=error)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(output=None, raw_text="",
                                 execution_time_ms=elapsed, error=str(exc))

    async def is_available(self) -> bool:
        return self._registry is not None


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------

class HttpSandbox:
    """Call an external API endpoint and capture the response."""

    def __init__(self, base_url: str = "", timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def execute(
        self,
        action: str,
        method: str = "GET",
        body: dict | None = None,
        headers: dict | None = None,
        **_: Any,
    ) -> SandboxResult:
        try:
            import httpx
        except ImportError:
            return SandboxResult(output=None, raw_text="",
                                 execution_time_ms=0.0,
                                 error="httpx not installed; pip install httpx")

        url = f"{self._base_url}/{action.lstrip('/')}" if self._base_url else action
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(method, url, json=body, headers=headers or {})
            elapsed = (time.monotonic() - start) * 1000
            try:
                output = resp.json()
                raw = str(output)
            except Exception:
                output = resp.text
                raw = resp.text
            error = None if resp.is_success else f"HTTP {resp.status_code}"
            return SandboxResult(output=output, raw_text=raw,
                                 execution_time_ms=elapsed, error=error)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(output=None, raw_text="",
                                 execution_time_ms=elapsed, error=str(exc))

    async def is_available(self) -> bool:
        try:
            import httpx  # noqa: F401
            return True
        except ImportError:
            return False
