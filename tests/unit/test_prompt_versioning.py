"""Unit tests for prompt versioning — PromptStore and PromptManager."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from harness.prompts.schemas import PromptVersion
from harness.prompts.store import PromptStore
from harness.prompts.manager import PromptManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(redis_client) -> PromptStore:
    return PromptStore(redis=redis_client)


async def _seed_versions(store: PromptStore, agent_type: str, n: int) -> list[PromptVersion]:
    """Create n versions, promoting the last one."""
    versions = []
    for i in range(n):
        v = await store.create_version(
            agent_type=agent_type,
            content=f"Prompt content v{i+1}",
            created_by="test",
        )
        versions.append(v)
    # Promote last
    await store.promote(versions[-1].version_id)
    return versions


# ---------------------------------------------------------------------------
# create_version + version numbering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_version_increments_version_number(redis_client):
    store = _store(redis_client)
    v1 = await store.create_version("sql", "prompt v1")
    v2 = await store.create_version("sql", "prompt v2")
    v3 = await store.create_version("sql", "prompt v3")
    assert v1.version_number == 1
    assert v2.version_number == 2
    assert v3.version_number == 3


@pytest.mark.asyncio
async def test_version_number_never_reused_after_delete(redis_client):
    # Regression: zcard+1 would reuse a number after a delete (count shrinks),
    # colliding in the version_number-scored index. The monotonic counter must
    # keep advancing.
    store = _store(redis_client)
    v1 = await store.create_version("sql", "v1")
    v2 = await store.create_version("sql", "v2")
    await store.delete("sql", v2.version_id)
    v3 = await store.create_version("sql", "v3")
    assert v3.version_number == 3  # not 2 (which would collide with v2's slot)
    assert v3.version_number != v1.version_number


@pytest.mark.asyncio
async def test_version_numbers_independent_per_agent_type(redis_client):
    store = _store(redis_client)
    sql_v1 = await store.create_version("sql", "sql prompt")
    code_v1 = await store.create_version("code", "code prompt")
    sql_v2 = await store.create_version("sql", "sql prompt 2")
    assert sql_v1.version_number == 1
    assert code_v1.version_number == 1
    assert sql_v2.version_number == 2


# ---------------------------------------------------------------------------
# promote + get_active
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_makes_version_active(redis_client):
    store = _store(redis_client)
    versions = await _seed_versions(store, "sql", 3)
    active = await store.get_active("sql")
    assert active is not None
    assert active.version_id == versions[-1].version_id
    assert active.active is True


@pytest.mark.asyncio
async def test_promote_deactivates_previous(redis_client):
    store = _store(redis_client)
    v1 = await store.create_version("sql", "v1")
    v2 = await store.create_version("sql", "v2")

    await store.promote(v1.version_id)
    await store.promote(v2.version_id)

    # Fetch v1 — should no longer be active
    v1_fresh = await store.get(v1.version_id, "sql")
    assert v1_fresh.active is False

    active = await store.get_active("sql")
    assert active.version_id == v2.version_id


@pytest.mark.asyncio
async def test_get_active_returns_none_when_no_versions(redis_client):
    store = _store(redis_client)
    active = await store.get_active("research")
    assert active is None


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rollback_one_step(redis_client):
    store = _store(redis_client)
    v1 = await store.create_version("sql", "v1")
    v2 = await store.create_version("sql", "v2")
    v3 = await store.create_version("sql", "v3")
    await store.promote(v3.version_id)

    rolled = await store.rollback("sql", steps=1)
    assert rolled.version_id == v2.version_id
    assert rolled.active is True


@pytest.mark.asyncio
async def test_rollback_two_steps(redis_client):
    store = _store(redis_client)
    versions = await _seed_versions(store, "sql", 3)

    rolled = await store.rollback("sql", steps=2)
    assert rolled.version_id == versions[0].version_id


@pytest.mark.asyncio
async def test_rollback_raises_when_not_enough_versions(redis_client):
    store = _store(redis_client)
    await _seed_versions(store, "sql", 1)
    with pytest.raises(ValueError, match="Cannot roll back"):
        await store.rollback("sql", steps=1)


# ---------------------------------------------------------------------------
# update_score
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_score_persists(redis_client):
    store = _store(redis_client)
    v = await store.create_version("sql", "prompt")
    assert v.score is None

    await store.update_score("sql", v.version_id, 0.87)

    refreshed = await store.get(v.version_id, "sql")
    assert refreshed is not None
    assert abs(refreshed.score - 0.87) < 1e-6


@pytest.mark.asyncio
async def test_update_score_clamps_to_0_1(redis_client):
    store = _store(redis_client)
    v = await store.create_version("sql", "prompt")

    await store.update_score("sql", v.version_id, 1.5)
    refreshed = await store.get(v.version_id, "sql")
    assert refreshed.score == 1.0

    await store.update_score("sql", v.version_id, -0.5)
    refreshed = await store.get(v.version_id, "sql")
    assert refreshed.score == 0.0


@pytest.mark.asyncio
async def test_update_score_noop_on_missing_version(redis_client):
    """Should not raise if version doesn't exist."""
    store = _store(redis_client)
    await store.update_score("sql", "nonexistent_id", 0.9)  # must not raise


# ---------------------------------------------------------------------------
# get_performance_history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_performance_history_chronological(redis_client):
    store = _store(redis_client)
    v1 = await store.create_version("sql", "v1", created_by="human")
    v2 = await store.create_version("sql", "v2", created_by="hermes")
    v3 = await store.create_version("sql", "v3", created_by="hermes")

    await store.update_score("sql", v1.version_id, 0.6)
    await store.update_score("sql", v2.version_id, 0.75)
    await store.update_score("sql", v3.version_id, 0.85)

    history = await store.get_performance_history("sql")

    assert len(history) == 3
    # Chronological: v1 first, v3 last
    assert history[0]["version_number"] == 1
    assert history[-1]["version_number"] == 3
    assert abs(history[0]["score"] - 0.6) < 1e-6
    assert abs(history[-1]["score"] - 0.85) < 1e-6


@pytest.mark.asyncio
async def test_get_performance_history_empty(redis_client):
    store = _store(redis_client)
    history = await store.get_performance_history("unknown_agent")
    assert history == []


@pytest.mark.asyncio
async def test_get_performance_history_includes_active_flag(redis_client):
    store = _store(redis_client)
    v1 = await store.create_version("sql", "v1")
    v2 = await store.create_version("sql", "v2")
    await store.promote(v2.version_id)

    history = await store.get_performance_history("sql")
    active_entries = [h for h in history if h["active"]]
    assert len(active_entries) == 1
    assert active_entries[0]["version_number"] == 2


# ---------------------------------------------------------------------------
# PromptManager — apply_patch + rollback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_manager_apply_patch_append(redis_client):
    store = _store(redis_client)
    manager = PromptManager(store=store)

    v = await store.create_version("sql", "Original prompt.")
    await store.promote(v.version_id)

    patch = type("Patch", (), {
        "agent_type": "sql",
        "op": "append",
        "path": "",
        "value": "Always use LIMIT 100.",
        "patch_id": uuid.uuid4().hex,
        "rationale": "test",
        "based_on_errors": [],
    })()

    new_version = await manager.apply_patch(patch)
    assert "Original prompt." in new_version.content
    assert "Always use LIMIT 100." in new_version.content
    assert new_version.active is True


@pytest.mark.asyncio
async def test_prompt_manager_apply_patch_set(redis_client):
    store = _store(redis_client)
    manager = PromptManager(store=store)

    v = await store.create_version("sql", "Old content.")
    await store.promote(v.version_id)

    patch = type("Patch", (), {
        "agent_type": "sql",
        "op": "set",
        "path": "",
        "value": "Completely new prompt.",
        "patch_id": uuid.uuid4().hex,
        "rationale": "full rewrite",
        "based_on_errors": [],
    })()

    new_version = await manager.apply_patch(patch)
    assert new_version.content == "Completely new prompt."


@pytest.mark.asyncio
async def test_prompt_manager_rollback_restores_previous(redis_client):
    store = _store(redis_client)
    manager = PromptManager(store=store)

    v1 = await store.create_version("sql", "Version one.")
    await store.promote(v1.version_id)
    v2 = await store.create_version("sql", "Version two.")
    await store.promote(v2.version_id)

    rolled = await manager.rollback("sql", steps=1)
    assert "Version one" in rolled.content

    # Cache should be cleared
    prompt_text = await manager.get_prompt("sql")
    assert "Version one" in prompt_text


@pytest.mark.asyncio
async def test_prompt_manager_cache_invalidated_after_apply(redis_client):
    store = _store(redis_client)
    manager = PromptManager(store=store)

    v = await store.create_version("sql", "Cached prompt.")
    await store.promote(v.version_id)

    # Prime the cache
    _ = await manager.get_prompt("sql")

    patch = type("Patch", (), {
        "agent_type": "sql",
        "op": "append",
        "path": "",
        "value": "New addition.",
        "patch_id": uuid.uuid4().hex,
        "rationale": "",
        "based_on_errors": [],
    })()

    await manager.apply_patch(patch)

    # Cache should have been cleared — should return new content
    prompt_text = await manager.get_prompt("sql")
    assert "New addition." in prompt_text


@pytest.mark.asyncio
async def test_prompt_manager_fallback_to_default(redis_client):
    store = _store(redis_client)
    manager = PromptManager(store=store)

    # No version stored — should return hardcoded default
    prompt = await manager.get_prompt("sql")
    assert len(prompt) > 20
    assert "SQL" in prompt or "sql" in prompt.lower()


@pytest.mark.asyncio
async def test_prompt_manager_generic_fallback_for_unknown_type(redis_client):
    store = _store(redis_client)
    manager = PromptManager(store=store)

    prompt = await manager.get_prompt("banana_agent")
    assert "banana_agent" in prompt or "helpful" in prompt.lower()


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_versions_newest_first(redis_client):
    store = _store(redis_client)
    await _seed_versions(store, "sql", 5)

    versions = await store.list_versions("sql", limit=5)
    assert len(versions) == 5
    # Newest first
    assert versions[0].version_number > versions[-1].version_number


@pytest.mark.asyncio
async def test_list_versions_respects_limit(redis_client):
    store = _store(redis_client)
    await _seed_versions(store, "sql", 10)

    versions = await store.list_versions("sql", limit=3)
    assert len(versions) == 3
