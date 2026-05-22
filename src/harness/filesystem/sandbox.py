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
    - Optional gVisor / Kata kernel isolation via ``runtime``

    Falls back to RestrictedPythonExecutor if Docker is unavailable.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        timeout: float = 30.0,
        network: bool = False,
        runtime: str = "runc",
    ) -> None:
        self._image = image
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._network = network
        self._runtime = runtime

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
        ]
        if self._runtime != "runc":
            cmd += [f"--runtime={self._runtime}"]
        cmd += [
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


class SessionDockerSandbox:
    """Reuses a single Docker container across multiple ``run_code`` calls.

    Each ``docker run`` costs 2–5 seconds of cold-start overhead. This class
    starts one container per session and uses ``docker exec`` for subsequent
    calls, eliminating that overhead entirely.

    State (installed packages, created files in /sandbox) persists across
    calls within the same session — the agent can pip-install in one step
    and import in the next.

    Usage::

        async with SessionDockerSandbox(workspace_path, ...) as sandbox:
            r1 = await sandbox.run_code("import sys; print(sys.version)")
            r2 = await sandbox.run_code("x = 1 + 1; print(x)")

    The container is stopped (and auto-removed via ``--rm``) on ``__aexit__``
    even when an exception is raised.

    OOM handling: if the container is OOM-killed (exit 137), the next
    ``run_code`` call will raise ``SandboxError`` with a clear message so the
    caller can decide whether to start a new session.
    """

    _STOP_TIMEOUT = 5  # seconds given to docker stop before SIGKILL

    def __init__(
        self,
        workspace_path: Path,
        image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        timeout: float = 30.0,
        network: bool = False,
        runtime: str = "runc",
    ) -> None:
        self._workspace = workspace_path
        self._image = image
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._network = network
        self._runtime = runtime
        self._container_id: str | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SessionDockerSandbox":
        await self._start_container()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._stop_container()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_code(self, code: str, timeout: float | None = None) -> SandboxResult:
        """Execute ``code`` inside the running container via ``docker exec``.

        Raises ``SandboxError`` when the session has not been started or the
        container has died (e.g. OOM-killed between calls).
        """
        if self._container_id is None:
            raise SandboxError(
                "SessionDockerSandbox is not running — use as an async context manager"
            )

        run_filename = f"run_{uuid.uuid4().hex[:8]}.py"
        run_path = self._workspace / run_filename

        try:
            run_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"Failed to write code to workspace: {exc}")

        exec_timeout = timeout if timeout is not None else self._timeout
        cmd = [
            "docker", "exec",
            self._container_id,
            "python", f"/sandbox/{run_filename}",
        ]

        start_time = time.monotonic()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxError("docker not found in PATH") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=exec_timeout
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

        # Detect container death: docker exec returns 1 with "No such container"
        if (
            exit_code == 1
            and not timed_out
            and b"No such container" in stderr_bytes
        ):
            self._container_id = None  # mark dead so callers know
            raise SandboxError(
                "Session container died (possibly OOM-killed). Start a new session."
            )

        try:
            run_path.unlink(missing_ok=True)
        except OSError:
            pass

        return SandboxResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=exit_code,
            timed_out=timed_out,
            execution_time_ms=elapsed_ms,
        )

    @property
    def is_alive(self) -> bool:
        """True when the container has been started and not yet stopped."""
        return self._container_id is not None

    @staticmethod
    async def is_available() -> bool:
        """Check whether the Docker daemon is reachable."""
        return await DockerSandbox.is_available()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _start_container(self) -> None:
        cmd = [
            "docker", "run",
            "--detach",                # returns container ID, doesn't block
            "--rm",                    # auto-remove on stop
            f"--memory={self._memory_limit}",
            "--memory-swap=-1",
            "--cpus=1",
            f"--network={'none' if not self._network else 'bridge'}",
            "--volume", f"{self._workspace.resolve()}:/sandbox:rw",
            "--workdir=/sandbox",
            "--user=nobody",
            "--security-opt=no-new-privileges",
        ]
        if self._runtime != "runc":
            cmd.append(f"--runtime={self._runtime}")
        cmd += [self._image, "sleep", "infinity"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except FileNotFoundError as exc:
            raise SandboxError("docker not found in PATH") from exc
        except asyncio.TimeoutError as exc:
            raise SandboxError("docker run timed out while starting session container") from exc

        if proc.returncode != 0:
            raise SandboxError(
                f"Failed to start session container: {stderr.decode().strip()}"
            )

        self._container_id = stdout.decode().strip()
        logger.debug("SessionDockerSandbox started: %s", self._container_id[:12])

    async def _stop_container(self) -> None:
        cid = self._container_id
        self._container_id = None
        if cid is None:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", f"--time={self._STOP_TIMEOUT}", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=self._STOP_TIMEOUT + 5.0)
            logger.debug("SessionDockerSandbox stopped: %s", cid[:12])
        except Exception as exc:
            logger.warning("SessionDockerSandbox stop failed for %s: %s", cid[:12], exc)


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
