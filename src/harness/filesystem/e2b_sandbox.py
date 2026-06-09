"""E2B cloud sandbox backend for code execution.

`E2BSandbox` runs agent-generated code in an `E2B <https://e2b.dev>`_ cloud
micro-VM instead of a local Docker container. It is a **drop-in for**
:class:`~harness.filesystem.sandbox.SessionDockerSandbox` — same async
context-manager lifecycle, same ``run_code(code, timeout) -> SandboxResult``
signature, same ``is_available()`` classmethod — so ``RunCodeTool`` (which reads
``ctx.metadata["docker_session"]``) works against it with no changes.

Select it with ``SANDBOX_PROVIDER=e2b`` + ``E2B_API_KEY=...``. Requires the
``e2b`` extra: ``pip install agent-haas[e2b]``.

A third provider (Modal, Daytona, …) can be added the same way: implement the
three-method surface and register it in the selector in ``agents/base.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from harness.filesystem.sandbox import SandboxError, SandboxResult

logger = logging.getLogger(__name__)


class E2BSandbox:
    """Persistent E2B cloud sandbox, mirroring SessionDockerSandbox's surface."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str | None = None,
        timeout: float = 30.0,
        workspace_path: Path | None = None,
    ) -> None:
        self._api_key = api_key or None
        self._template = template or None
        self._timeout = timeout
        self._workspace = workspace_path  # accepted for interface parity; unused
        self._sandbox: Any = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    @staticmethod
    async def is_available() -> bool:
        """True when the e2b SDK is importable and an API key is configured."""
        try:
            import e2b_code_interpreter  # noqa: F401
        except ImportError:
            return False
        import os

        # api_key is read by the SDK from E2B_API_KEY; treat its presence as "available"
        return bool(os.environ.get("E2B_API_KEY"))

    def _ensure_sdk(self) -> Any:
        try:
            from e2b_code_interpreter import AsyncSandbox
        except ImportError as exc:
            raise SandboxError(
                "e2b-code-interpreter is required for E2BSandbox.\n"
                "Install: pip install e2b-code-interpreter\n"
                "     or: pip install agent-haas[e2b]"
            ) from exc
        return AsyncSandbox

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> E2BSandbox:
        sandbox_cls = self._ensure_sdk()
        create_kwargs: dict[str, Any] = {"timeout": int(self._timeout)}
        if self._api_key:
            create_kwargs["api_key"] = self._api_key
        if self._template:
            create_kwargs["template"] = self._template
        try:
            self._sandbox = await sandbox_cls.create(**create_kwargs)
        except Exception as exc:
            raise SandboxError(f"Failed to create E2B sandbox: {exc}") from exc
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.kill()
            except Exception as exc:
                logger.warning("Failed to kill E2B sandbox: %s", exc)
            self._sandbox = None

    # ------------------------------------------------------------------
    # Public API — matches SessionDockerSandbox.run_code
    # ------------------------------------------------------------------
    async def run_code(self, code: str, timeout: float | None = None) -> SandboxResult:
        if self._sandbox is None:
            raise SandboxError(
                "E2BSandbox is not running — use as an async context manager"
            )
        start = time.monotonic()
        try:
            execution = await self._sandbox.run_code(
                code, timeout=int(timeout or self._timeout)
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000.0
            timed_out = "timeout" in str(exc).lower()
            return SandboxResult(
                stdout="",
                stderr=f"E2B execution failed: {exc}",
                exit_code=124 if timed_out else 1,
                timed_out=timed_out,
                execution_time_ms=elapsed,
            )
        elapsed = (time.monotonic() - start) * 1000.0

        logs = getattr(execution, "logs", None)
        stdout = "".join(getattr(logs, "stdout", []) or []) if logs else ""
        stderr = "".join(getattr(logs, "stderr", []) or []) if logs else ""

        error = getattr(execution, "error", None)
        if error is not None:
            tb = getattr(error, "traceback", None) or getattr(error, "value", "") or str(error)
            stderr = (stderr + "\n" + str(tb)).strip()
            exit_code = 1
        else:
            exit_code = 0

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=False,
            execution_time_ms=elapsed,
        )
