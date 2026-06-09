"""Unit tests for the E2B sandbox backend (SDK mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness.filesystem.e2b_sandbox import E2BSandbox
from harness.filesystem.sandbox import SandboxError


@pytest.mark.asyncio
async def test_run_code_requires_active_session():
    sb = E2BSandbox()
    with pytest.raises(SandboxError):
        await sb.run_code("print(1)")


@pytest.mark.asyncio
async def test_is_available_false_without_key(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    assert await E2BSandbox.is_available() is False


@pytest.mark.asyncio
async def test_run_code_maps_stdout_success():
    sb = E2BSandbox()
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["hello\n"], stderr=[]), error=None
    )
    sb._sandbox = SimpleNamespace(run_code=AsyncMock(return_value=execution))

    res = await sb.run_code("print('hello')")
    assert res.stdout == "hello\n"
    assert res.exit_code == 0
    assert res.success is True
    assert res.timed_out is False


@pytest.mark.asyncio
async def test_run_code_maps_error_to_nonzero():
    sb = E2BSandbox()
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=[], stderr=[]),
        error=SimpleNamespace(traceback="Traceback: NameError: x not defined"),
    )
    sb._sandbox = SimpleNamespace(run_code=AsyncMock(return_value=execution))

    res = await sb.run_code("x")
    assert res.exit_code == 1
    assert res.success is False
    assert "NameError" in res.stderr


@pytest.mark.asyncio
async def test_run_code_timeout_maps_to_124():
    sb = E2BSandbox()
    sb._sandbox = SimpleNamespace(
        run_code=AsyncMock(side_effect=RuntimeError("execution timeout exceeded"))
    )
    res = await sb.run_code("while True: pass")
    assert res.timed_out is True
    assert res.exit_code == 124


@pytest.mark.asyncio
async def test_run_code_generic_exception_maps_to_1():
    sb = E2BSandbox()
    sb._sandbox = SimpleNamespace(run_code=AsyncMock(side_effect=RuntimeError("boom")))
    res = await sb.run_code("print(1)")
    assert res.exit_code == 1
    assert res.timed_out is False
    assert "boom" in res.stderr
