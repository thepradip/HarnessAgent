"""Code execution and analysis tools for HarnessAgent agents."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from harness.core.context import AgentContext, ToolResult
from harness.core.errors import FailureClass, ToolError

logger = logging.getLogger(__name__)

_OOM_EXIT_CODE = 137
_OOM_ERROR = "OOM: container exceeded memory limit"


def _is_oom(result: dict[str, Any]) -> bool:
    return result.get("exit_code") == _OOM_EXIT_CODE and not result.get("timed_out", False)


class RunCodeTool:
    """Execute Python code in a sandbox.

    Attempts DockerSandbox first; falls back to RestrictedPythonExecutor
    if Docker is unavailable.
    """

    name = "run_python"
    description = (
        "Execute Python code in a sandboxed environment. "
        "Returns stdout, stderr, and exit code."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "number",
                "default": 30.0,
                "description": "Execution timeout in seconds.",
            },
        },
        "required": ["code"],
    }
    timeout_seconds: float = 60.0

    def __init__(
        self,
        docker_sandbox: Any | None = None,
        restricted_executor: Any | None = None,
        workspace_manager: Any | None = None,
    ) -> None:
        self._docker_sandbox = docker_sandbox
        self._restricted_executor = restricted_executor
        self._workspace_manager = workspace_manager

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Run the provided Python code and return the output."""
        code: str = args["code"]
        timeout: float = float(args.get("timeout", 30.0))

        # Prefer a persistent session container when one is live for this run
        session = ctx.metadata.get("docker_session") if ctx.metadata else None
        if session is not None and getattr(session, "is_alive", False):
            try:
                res = await session.run_code(code, timeout=timeout)
                result_data = {
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "exit_code": res.exit_code,
                    "timed_out": res.timed_out,
                }
                if _is_oom(result_data):
                    return ToolResult(data=None, error=_OOM_ERROR)
                await self._save_output_to_workspace(ctx, code, result_data)
                return ToolResult(data=result_data)
            except Exception as exc:
                logger.warning(
                    "Session exec failed, falling back to per-call sandbox: %s", exc
                )

        # Try DockerSandbox (per-call, cold start each time)
        if self._docker_sandbox is not None:
            try:
                result_data = await self._run_in_docker(ctx, code, timeout)
                if _is_oom(result_data):
                    return ToolResult(data=None, error=_OOM_ERROR)
                await self._save_output_to_workspace(ctx, code, result_data)
                return ToolResult(data=result_data)
            except Exception as exc:
                logger.warning(
                    "DockerSandbox execution failed, falling back to RestrictedPython: %s",
                    exc,
                )

        # Fall back to RestrictedPythonExecutor
        if self._restricted_executor is not None:
            try:
                result_data = await self._run_restricted(ctx, code, timeout)
                if _is_oom(result_data):
                    return ToolResult(data=None, error=_OOM_ERROR)
                await self._save_output_to_workspace(ctx, code, result_data)
                return ToolResult(data=result_data)
            except Exception as exc:
                logger.exception("RestrictedPython execution failed: %s", exc)
                return ToolResult(
                    data=None, error=f"Code execution failed: {exc}"
                )

        # Fallback: subprocess-based execution in temp dir
        try:
            result_data = await self._run_subprocess(ctx, code, timeout)
            if _is_oom(result_data):
                return ToolResult(data=None, error=_OOM_ERROR)
            await self._save_output_to_workspace(ctx, code, result_data)
            return ToolResult(data=result_data)
        except Exception as exc:
            logger.exception("Subprocess code execution failed: %s", exc)
            return ToolResult(data=None, error=f"Code execution failed: {exc}")

    async def _run_in_docker(
        self, ctx: AgentContext, code: str, timeout: float
    ) -> dict[str, Any]:
        """Execute code via DockerSandbox (one container per call)."""
        res = await self._docker_sandbox.run_code(code, ctx.workspace_path)
        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "exit_code": res.exit_code,
            "timed_out": res.timed_out,
        }

    async def _run_restricted(
        self, ctx: AgentContext, code: str, timeout: float
    ) -> dict[str, Any]:
        """Execute code via RestrictedPythonExecutor."""
        res = await self._restricted_executor.run_code(code)
        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "exit_code": res.exit_code,
            "timed_out": res.timed_out,
        }

    async def _run_subprocess(
        self, ctx: AgentContext, code: str, timeout: float
    ) -> dict[str, Any]:
        """Execute Python code in a subprocess (last resort, use sparingly)."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, dir=str(ctx.workspace_path)
        ) as tmp_file:
            tmp_file.write(code)
            script_path = tmp_file.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ctx.workspace_path),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout}s",
                    "exit_code": -1,
                    "timed_out": True,
                }

            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode,
                "timed_out": False,
            }
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    async def _save_output_to_workspace(
        self, ctx: AgentContext, code: str, result: dict[str, Any]
    ) -> None:
        """Save the executed code and its output to the workspace for traceability."""
        try:
            output_path = ctx.workspace_path / f"run_{ctx.step_count}_output.txt"
            content = (
                f"# Code executed at step {ctx.step_count}\n"
                f"# Exit code: {result.get('exit_code')}\n\n"
                f"## Code\n```python\n{code}\n```\n\n"
                f"## Stdout\n{result.get('stdout', '')}\n\n"
                f"## Stderr\n{result.get('stderr', '')}\n"
            )
            output_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to save code output to workspace: %s", exc)


class LintCodeTool:
    """Lint Python code using ruff and return any issues found."""

    name = "lint_code"
    description = (
        "Lint Python code with ruff. Returns a list of issues (line, column, code, message)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to lint.",
            },
            "select": {
                "type": "string",
                "default": "E,F,W",
                "description": "Comma-separated ruff rule codes to enable.",
            },
        },
        "required": ["code"],
    }
    timeout_seconds: float = 30.0

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Write code to a temp file, run ruff, and return the findings."""
        code: str = args["code"]
        select: str = args.get("select", "E,F,W")

        with tempfile.NamedTemporaryFile(
            suffix=".py",
            mode="w",
            delete=False,
            dir=str(ctx.workspace_path),
        ) as tmp_file:
            tmp_file.write(code)
            script_path = tmp_file.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff",
                "check",
                "--select",
                select,
                "--no-cache",
                "--output-format",
                "json",
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(data=None, error="ruff lint timed out")

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            # ruff returns exit 0 for no issues, 1 for issues, 2+ for errors
            if proc.returncode is not None and proc.returncode >= 2:
                return ToolResult(
                    data=None,
                    error=f"ruff exited with code {proc.returncode}: {stderr_text}",
                )

            # Parse JSON output
            import json as _json
            issues: list[dict[str, Any]] = []
            if stdout_text.strip():
                try:
                    raw_issues = _json.loads(stdout_text)
                    for issue in raw_issues:
                        issues.append(
                            {
                                "line": issue.get("location", {}).get("row"),
                                "column": issue.get("location", {}).get("column"),
                                "code": issue.get("code"),
                                "message": issue.get("message"),
                                "fix": issue.get("fix"),
                            }
                        )
                except _json.JSONDecodeError:
                    # Fall back to plain text parsing
                    issues = [{"raw": line} for line in stdout_text.splitlines() if line.strip()]

            return ToolResult(
                data={
                    "issues": issues,
                    "issue_count": len(issues),
                    "clean": len(issues) == 0,
                },
                metadata={"select_rules": select},
            )
        except FileNotFoundError:
            return ToolResult(
                data=None,
                error="ruff not found. Install with: pip install ruff",
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass


class ApplyPatchTool:
    """Apply a unified diff patch to a file in the agent workspace."""

    name = "apply_patch"
    description = (
        "Apply a unified diff patch to a file in the agent workspace. "
        "The patch must be in standard unified diff format."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to patch, relative to workspace.",
            },
            "patch": {
                "type": "string",
                "description": "Unified diff patch content.",
            },
            "dry_run": {
                "type": "boolean",
                "default": False,
                "description": "If true, check the patch without applying.",
            },
        },
        "required": ["path", "patch"],
    }
    timeout_seconds: float = 30.0

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Write the patch to a temp file and apply it via the patch command."""
        rel_path: str = args["path"]
        patch_content: str = args["patch"]
        dry_run: bool = bool(args.get("dry_run", False))

        # Resolve the target path within workspace (prevent path traversal)
        workspace = ctx.workspace_path
        target_path = (workspace / rel_path).resolve()
        if not str(target_path).startswith(str(workspace.resolve())):
            return ToolResult(
                data=None,
                error=f"Path '{rel_path}' escapes workspace boundary.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=".patch",
            mode="w",
            delete=False,
            dir=str(workspace),
        ) as patch_file:
            patch_file.write(patch_content)
            patch_path = patch_file.name

        try:
            cmd = ["patch", "--unified", str(target_path), patch_path]
            if dry_run:
                cmd.append("--dry-run")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(data=None, error="patch command timed out")

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                return ToolResult(
                    data=None,
                    error=f"patch failed (exit {proc.returncode}): {stderr_text or stdout_text}",
                )

            return ToolResult(
                data={
                    "success": True,
                    "dry_run": dry_run,
                    "path": rel_path,
                    "output": stdout_text,
                }
            )
        finally:
            try:
                os.unlink(patch_path)
            except OSError:
                pass
