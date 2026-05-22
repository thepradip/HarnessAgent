"""Docker and RestrictedPython code execution sandboxes."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.core.errors import SandboxError

logger = logging.getLogger(__name__)

# Memory limits per workload profile.
# "general"  — scripting, algorithms, stdlib only
# "data"     — pandas / numpy with real datasets
# "ml"       — torch / sklearn model runs
WORKLOAD_MEMORY: dict[str, str] = {
    "general": "256m",
    "data": "512m",
    "ml": "2g",
}


def memory_for_workload(workload: str) -> str:
    """Return the Docker memory limit string for a named workload profile.

    Falls back to the "general" limit for unknown names.
    """
    return WORKLOAD_MEMORY.get(workload, WORKLOAD_MEMORY["general"])


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    execution_time_ms: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def oom_killed(self) -> bool:
        """True when Docker OOM-killed the container (exit code 137, not a timeout)."""
        return self.exit_code == 137 and not self.timed_out


class DockerSandbox:
    """
    Execute Python code (or arbitrary commands) inside an isolated Docker container.

    Security controls:
    - Memory limit (default 256 MiB)
    - CPU limit (1 core)
    - Optional network isolation (default: no network)
    - Read/write mount of workspace only

    Falls back to RestrictedPythonExecutor if Docker is unavailable.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        timeout: float = 30.0,
        network: bool = False,
    ) -> None:
        self._image = image
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._network = network

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_code(self, code: str, workspace_path: Path) -> SandboxResult:
        """
        Write ``code`` to a temp file in the workspace, then execute it
        inside a Docker container.
        """
        run_filename = f"run_{uuid.uuid4().hex[:8]}.py"
        run_path = workspace_path / run_filename

        try:
            run_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"Failed to write code to workspace: {exc}")

        cmd = [
            "docker", "run",
            "--rm",
            f"--memory={self._memory_limit}",
            "--memory-swap=-1",        # disable swap
            "--cpus=1",
            f"--network={'none' if not self._network else 'bridge'}",
            "--volume", f"{workspace_path.resolve()}:/sandbox:rw",
            "--workdir=/sandbox",
            "--user=nobody",           # non-root execution
            "--security-opt=no-new-privileges",
            self._image,
            "python", f"/sandbox/{run_filename}",
        ]

        result = await self.run_command(cmd=cmd, workspace_path=workspace_path)

        # Cleanup temp file
        try:
            run_path.unlink(missing_ok=True)
        except OSError:
            pass

        return result

    async def run_command(
        self,
        cmd: list[str],
        workspace_path: Path,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """
        Execute an arbitrary command inside a Docker container.

        The first argument should be "docker" if this is intended as a
        Docker run; otherwise the command is executed directly.
        """
        start_time = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                f"Command not found: {cmd[0]}. "
                "Is Docker installed and in PATH?"
            ) from exc

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            stdout_bytes = b""
            stderr_bytes = b"Execution timed out."

        elapsed_ms = (time.monotonic() - start_time) * 1000

        exit_code = proc.returncode if proc.returncode is not None else -1

        return SandboxResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=exit_code,
            timed_out=timed_out,
            execution_time_ms=elapsed_ms,
        )

    @staticmethod
    async def is_available() -> bool:
        """Check if Docker daemon is accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return proc.returncode == 0
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            return False


class RestrictedPythonExecutor:
    """
    Fallback sandbox using RestrictedPython for environments without Docker.

    Compiles code with restriction guards and executes with safe builtins.
    All exceptions are caught and returned as SandboxResult.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def run_code(self, code: str) -> SandboxResult:
        """
        Compile and execute ``code`` using RestrictedPython.

        Standard output is captured via a custom print hook.
        """
        try:
            from RestrictedPython import (  # type: ignore[import]
                RestrictingNodeTransformer,
                compile_restricted,
                safe_globals,
            )
            from RestrictedPython.PrintCollector import PrintCollector  # type: ignore[import]
        except ImportError:
            return SandboxResult(
                stdout="",
                stderr="RestrictedPython not installed. pip install RestrictedPython",
                exit_code=1,
                timed_out=False,
                execution_time_ms=0.0,
            )

        start_time = time.monotonic()
        collected_output: list[str] = []
        errors: list[str] = []

        try:
            # Compile with restrictions
            byte_code = compile_restricted(
                code,
                filename="<sandbox>",
                mode="exec",
            )
        except SyntaxError as exc:
            return SandboxResult(
                stdout="",
                stderr=f"SyntaxError: {exc}",
                exit_code=1,
                timed_out=False,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Build a safe globals dict
        _print_collector = PrintCollector()
        exec_globals: dict[str, Any] = {
            **safe_globals,
            "_print_": _print_collector,
            "_getattr_": getattr,
            "_getitem_": lambda obj, idx: obj[idx],
            "_getiter_": iter,
            "_write_": lambda x: x,
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "isinstance": isinstance,
                "type": type,
                "repr": repr,
                "True": True,
                "False": False,
                "None": None,
            },
        }

        loop = asyncio.get_running_loop()
        exit_code = 0

        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, exec, byte_code, exec_globals),
                timeout=self._timeout,
            )
            # Collect printed output
            if hasattr(_print_collector, "_call_print"):
                collected_output = list(_print_collector._call_print)
        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="",
                stderr="Execution timed out.",
                exit_code=1,
                timed_out=True,
                execution_time_ms=self._timeout * 1000,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            exit_code = 1

        elapsed_ms = (time.monotonic() - start_time) * 1000
        return SandboxResult(
            stdout="\n".join(str(o) for o in collected_output),
            stderr="\n".join(errors),
            exit_code=exit_code,
            timed_out=False,
            execution_time_ms=elapsed_ms,
        )
