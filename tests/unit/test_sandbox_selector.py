"""Tests for the SANDBOX_PROVIDER selector in BaseAgent._start_docker_session.

Verifies the provider chosen by config/metadata is the one instantiated and
stored at ctx.metadata["docker_session"] (the key RunCodeTool reads), and that
an unavailable provider falls back to no session.
"""

from __future__ import annotations

import uuid

import pytest

from harness.agents.base import BaseAgent
from harness.core.context import AgentContext


class _FakeSandbox:
    """Stands in for any provider sandbox: available, no-op lifecycle."""

    available = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    async def is_available(cls) -> bool:
        return cls.available

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


def _ctx(tmp_path, provider):
    return AgentContext(
        run_id=uuid.uuid4().hex,
        tenant_id="t",
        agent_type="t",
        task="t",
        memory=None,
        workspace_path=tmp_path / "ws",
        metadata={"sandbox_session": True, "sandbox_provider": provider},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, module_attr",
    [
        ("docker", "harness.filesystem.sandbox.SessionDockerSandbox"),
        ("e2b", "harness.filesystem.e2b_sandbox.E2BSandbox"),
        ("modal", "harness.filesystem.modal_sandbox.ModalSandbox"),
    ],
)
async def test_selector_instantiates_the_configured_provider(
    monkeypatch, tmp_path, provider, module_attr
):
    # Distinct fake class per provider so the assertion can't accidentally pass.
    fake = type(f"Fake_{provider}", (_FakeSandbox,), {"available": True})
    monkeypatch.setattr(module_attr, fake)

    agent = object.__new__(BaseAgent)  # method touches no self attrs
    ctx = _ctx(tmp_path, provider)
    await BaseAgent._start_docker_session(agent, ctx)

    session = ctx.metadata.get("docker_session")
    assert isinstance(session, fake), f"{provider} selector did not pick {module_attr}"


@pytest.mark.asyncio
async def test_unavailable_provider_falls_back_to_no_session(monkeypatch, tmp_path):
    fake = type("FakeUnavailable", (_FakeSandbox,), {"available": False})
    monkeypatch.setattr("harness.filesystem.e2b_sandbox.E2BSandbox", fake)

    agent = object.__new__(BaseAgent)
    ctx = _ctx(tmp_path, "e2b")
    await BaseAgent._start_docker_session(agent, ctx)

    assert ctx.metadata.get("docker_session") is None


@pytest.mark.asyncio
async def test_disabled_session_starts_nothing(monkeypatch, tmp_path):
    fake = type("FakeNever", (_FakeSandbox,), {"available": True})
    monkeypatch.setattr("harness.filesystem.e2b_sandbox.E2BSandbox", fake)

    agent = object.__new__(BaseAgent)
    ctx = AgentContext(
        run_id=uuid.uuid4().hex, tenant_id="t", agent_type="t", task="t",
        memory=None, workspace_path=tmp_path / "ws",
        metadata={"sandbox_session": False, "sandbox_provider": "e2b"},
    )
    await BaseAgent._start_docker_session(agent, ctx)

    assert ctx.metadata.get("docker_session") is None
