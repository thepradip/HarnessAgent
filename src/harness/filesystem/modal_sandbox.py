"""Modal cloud sandbox backend for code execution.

`ModalSandbox` runs agent-generated code in a `Modal <https://modal.com>`_
serverless container. Like :class:`~harness.filesystem.e2b_sandbox.E2BSandbox`,
it is a **drop-in for**
:class:`~harness.filesystem.sandbox.SessionDockerSandbox` — same async
context-manager lifecycle, ``run_code(code, timeout) -> SandboxResult``, and
``is_available()`` — so ``RunCodeTool`` (which reads
``ctx.metadata["docker_session"]``) works against it unchanged.

Select it with ``SANDBOX_PROVIDER=modal``; auth comes from ``MODAL_TOKEN_ID`` /
``MODAL_TOKEN_SECRET`` (the standard Modal env vars). Requires the ``modal``
extra: ``pip install agent-haas[modal]``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from harness.filesystem.sandbox import SandboxError, SandboxResult

logger = logging.getLogger(__name__)


class ModalSandbox:
    """Persistent Modal sandbox, mirroring SessionDockerSandbox's surface."""

    def __init__(
        self,
        *,
        app_name: str = "agent-haas-sandbox",
        timeout: float = 30.0,
        workspace_path: Path | None = None,
    ) -> None:
        self._app_name = app_name
        self._timeout = timeout
        self._workspace = workspace_path  # accepted for interface parity; unused
        self._sandbox: Any = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    @staticmethod
    async def is_available() -> bool:
        """True when the modal SDK is importable and Modal tokens are set."""
        try:
            import modal  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))

    def _ensure_sdk(self) -> Any:
        try:
            import modal
        except ImportError as exc:
            raise SandboxError(
                "modal is required for ModalSandbox.\n"
                "Install: pip install modal\n"
                "     or: pip install agent-haas[modal]"
            ) from exc
        return modal

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> ModalSandbox:
        modal = self._ensure_sdk()
        try:
            app = await modal.App.lookup.aio(self._app_name, create_if_missing=True)
            self._sandbox = await modal.Sandbox.create.aio(app=app)
        except Exception as exc:
            raise SandboxError(f"Failed to create Modal sandbox: {exc}") from exc
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate.aio(wait=True)
            except Exception as exc:
                logger.warning("Failed to terminate Modal sandbox: %s", exc)
            self._sandbox = None

    # ------------------------------------------------------------------
    # Public API — matches SessionDockerSandbox.run_code
    # ------------------------------------------------------------------
    async def run_code(self, code: str, timeout: float | None = None) -> SandboxResult:
        if self._sandbox is None:
            raise SandboxError(
                "ModalSandbox is not running — use as an async context manager"
            )
        start = time.monotonic()
        try:
            proc = await self._sandbox.exec.aio(
                "python", "-c", code, timeout=int(timeout or self._timeout)
            )
            await proc.wait.aio()
            stdout = await proc.stdout.read.aio()
            stderr = await proc.stderr.read.aio()
            exit_code = int(getattr(proc, "returncode", 0) or 0)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000.0
            timed_out = "timeout" in str(exc).lower()
            return SandboxResult(
                stdout="",
                stderr=f"Modal execution failed: {exc}",
                exit_code=124 if timed_out else 1,
                timed_out=timed_out,
                execution_time_ms=elapsed,
            )
        elapsed = (time.monotonic() - start) * 1000.0
        return SandboxResult(
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
            timed_out=exit_code == 137,  # SIGKILL — Modal kills on timeout
            execution_time_ms=elapsed,
        )
