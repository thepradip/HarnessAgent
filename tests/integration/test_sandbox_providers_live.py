"""Live smoke tests for the cloud sandbox providers.

These actually run code in the real E2B / Modal services, so they are skipped
unless the provider's credentials are present in the environment (and its SDK
installed). With no creds they show as an explicit SKIP — they never fail CI
for missing secrets, and they give a real end-to-end check when you do have
credentials configured.
"""

from __future__ import annotations

import os

import pytest

E2B_READY = bool(os.environ.get("E2B_API_KEY"))
MODAL_READY = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


@pytest.mark.skipif(not E2B_READY, reason="E2B_API_KEY not set — live E2B test skipped")
@pytest.mark.asyncio
async def test_e2b_runs_real_code():
    from harness.filesystem.e2b_sandbox import E2BSandbox

    async with E2BSandbox(api_key=os.environ["E2B_API_KEY"]) as sb:
        res = await sb.run_code("print('hi from e2b')")
    assert res.exit_code == 0, res.stderr
    assert "hi from e2b" in res.stdout


@pytest.mark.skipif(not MODAL_READY, reason="MODAL_TOKEN_ID/SECRET not set — live Modal test skipped")
@pytest.mark.asyncio
async def test_modal_runs_real_code():
    from harness.filesystem.modal_sandbox import ModalSandbox

    async with ModalSandbox() as sb:
        res = await sb.run_code("print('hi from modal')")
    assert res.exit_code == 0, res.stderr
    assert "hi from modal" in res.stdout
