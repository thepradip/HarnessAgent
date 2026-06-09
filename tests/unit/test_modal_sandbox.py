"""Unit tests for the Modal sandbox backend (SDK mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness.filesystem.modal_sandbox import ModalSandbox
from harness.filesystem.sandbox import SandboxError


def _proc(stdout="", stderr="", returncode=0):
    return SimpleNamespace(
        wait=SimpleNamespace(aio=AsyncMock(return_value=None)),
        stdout=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=stdout))),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=stderr))),
        returncode=returncode,
    )


@pytest.mark.asyncio
async def test_run_code_requires_active_session():
    sb = ModalSandbox()
    with pytest.raises(SandboxError):
        await sb.run_code("print(1)")


@pytest.mark.asyncio
async def test_is_available_false_without_tokens(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert await ModalSandbox.is_available() is False


@pytest.mark.asyncio
async def test_run_code_maps_stdout_success():
    sb = ModalSandbox()
    sb._sandbox = SimpleNamespace(
        exec=SimpleNamespace(aio=AsyncMock(return_value=_proc(stdout="hi\n", returncode=0)))
    )
    res = await sb.run_code("print('hi')")
    assert res.stdout == "hi\n"
    assert res.exit_code == 0
    assert res.success is True
    assert res.timed_out is False


@pytest.mark.asyncio
async def test_run_code_maps_nonzero_exit():
    sb = ModalSandbox()
    sb._sandbox = SimpleNamespace(
        exec=SimpleNamespace(
            aio=AsyncMock(return_value=_proc(stderr="NameError: x", returncode=1))
        )
    )
    res = await sb.run_code("x")
    assert res.exit_code == 1
    assert res.success is False
    assert "NameError" in res.stderr


@pytest.mark.asyncio
async def test_run_code_sigkill_marks_timed_out():
    sb = ModalSandbox()
    sb._sandbox = SimpleNamespace(
        exec=SimpleNamespace(aio=AsyncMock(return_value=_proc(returncode=137)))
    )
    res = await sb.run_code("while True: pass")
    assert res.exit_code == 137
    assert res.timed_out is True


@pytest.mark.asyncio
async def test_run_code_exception_maps_to_error():
    sb = ModalSandbox()
    sb._sandbox = SimpleNamespace(
        exec=SimpleNamespace(aio=AsyncMock(side_effect=RuntimeError("boom")))
    )
    res = await sb.run_code("print(1)")
    assert res.exit_code == 1
    assert "boom" in res.stderr
