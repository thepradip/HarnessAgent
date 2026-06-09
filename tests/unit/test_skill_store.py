"""Tests for SkillStore, SkillArtifact, SkillCapture, and helpers."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import fakeredis.aioredis as fakeredis

from harness.tools.skill_store import (
    RedFlagKind,
    SkillArtifact,
    SkillCapture,
    SkillHealthReport,
    SkillStore,
    SkillType,
    ValidationStatus,
    _validate_artifact,
    check_requirements,
    detect_flags,
    format_skills_for_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _redis():
    return fakeredis.FakeRedis(decode_responses=True)


def _skill(
    *,
    title="Paginated table renderer",
    description="Renders a paginated HTML table from a list of dicts",
    content="def render_table(rows, page=0, page_size=10):\n    return rows[page*page_size:(page+1)*page_size]",
    skill_type=SkillType.CODE,
    tenant_id="t1",
    score=0.85,
    language="python",
    tags=None,
    requirements=None,
    **kwargs,
) -> SkillArtifact:
    return SkillArtifact(
        skill_id=uuid.uuid4().hex,
        tenant_id=tenant_id,
        skill_type=skill_type,
        title=title,
        description=description,
        content=content,
        language=language,
        tags=tags or ["ui", "table"],
        requirements=requirements or {"python": ">=3.9"},
        score=score,
        **kwargs,
    )


# ===========================================================================
# SkillArtifact — data model
# ===========================================================================

def test_artifact_round_trip():
    s = _skill()
    s2 = SkillArtifact.from_dict(s.to_dict())
    assert s2.skill_id == s.skill_id
    assert s2.title == s.title
    assert s2.content == s.content
    assert s2.skill_type == SkillType.CODE
    assert s2.validation_status == ValidationStatus.UNVALIDATED
    assert s2.requirements == s.requirements


def test_artifact_round_trip_preserves_datetimes():
    s = _skill()
    s.last_validated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    s2 = SkillArtifact.from_dict(s.to_dict())
    assert s2.last_validated_at == s.last_validated_at


def test_artifact_content_hash_deterministic():
    s = _skill(content="x = 1 + 1")
    assert s.content_hash == s.content_hash
    assert len(s.content_hash) == 16


def test_artifact_content_hash_differs_for_different_content():
    s1 = _skill(content="x = 1")
    s2 = _skill(content="x = 2")
    assert s1.content_hash != s2.content_hash


def test_artifact_is_stale_true_when_old():
    s = _skill()
    s.last_validated_at = datetime.now(timezone.utc) - timedelta(days=31)
    s.validation_status = ValidationStatus.VALID
    assert s.is_stale(stale_days=30) is True


def test_artifact_is_stale_false_when_recent():
    s = _skill()
    s.last_validated_at = datetime.now(timezone.utc) - timedelta(days=5)
    assert s.is_stale(stale_days=30) is False


def test_artifact_is_stale_false_when_never_validated():
    s = _skill()
    s.last_validated_at = None
    assert s.is_stale() is False


def test_artifact_from_dict_ignores_unknown_fields():
    d = _skill().to_dict()
    d["unknown_future_field"] = "should be ignored"
    s = SkillArtifact.from_dict(d)
    assert s.title is not None


# ===========================================================================
# _validate_artifact
# ===========================================================================

def test_validate_raises_on_empty_title():
    s = _skill(title="   ")
    with pytest.raises(ValueError, match="title"):
        _validate_artifact(s)


def test_validate_raises_on_empty_description():
    s = _skill(description="")
    with pytest.raises(ValueError, match="description"):
        _validate_artifact(s)


def test_validate_raises_on_short_content():
    s = _skill(content="x=1")
    with pytest.raises(ValueError, match="content too short"):
        _validate_artifact(s)


def test_validate_raises_on_empty_skill_id():
    s = _skill()
    s.skill_id = ""
    with pytest.raises(ValueError, match="skill_id"):
        _validate_artifact(s)


def test_validate_raises_on_empty_tenant_id():
    s = _skill()
    s.tenant_id = ""
    with pytest.raises(ValueError, match="tenant_id"):
        _validate_artifact(s)


# ===========================================================================
# check_requirements
# ===========================================================================

def test_check_requirements_compatible():
    mismatches = check_requirements({"pandas": ">=2.0"}, {"pandas": "2.1.0"})
    assert mismatches == []


def test_check_requirements_missing_package():
    mismatches = check_requirements({"torch": ">=2.0"}, {})
    assert len(mismatches) == 1
    assert "torch" in mismatches[0]


def test_check_requirements_version_mismatch_equality():
    mismatches = check_requirements({"pandas": "==1.5.0"}, {"pandas": "2.0.0"})
    assert len(mismatches) == 1
    assert "pandas" in mismatches[0]


def test_check_requirements_empty_skill_reqs():
    mismatches = check_requirements({}, {"pandas": "2.0.0"})
    assert mismatches == []


def test_check_requirements_empty_env():
    mismatches = check_requirements({"pandas": ">=2.0"}, {})
    assert len(mismatches) == 1


def test_check_requirements_empty_spec_skipped():
    mismatches = check_requirements({"some-lib": ""}, {"some-lib": "1.0"})
    assert mismatches == []


# ===========================================================================
# detect_flags
# ===========================================================================

def test_detect_flags_broken():
    s = _skill()
    s.validation_status = ValidationStatus.BROKEN
    flags = detect_flags(s)
    assert any(f.kind == RedFlagKind.BROKEN for f in flags)
    assert all(f.severity == "high" for f in flags if f.kind == RedFlagKind.BROKEN)


def test_detect_flags_stale():
    s = _skill()
    s.validation_status = ValidationStatus.VALID
    s.last_validated_at = datetime.now(timezone.utc) - timedelta(days=40)
    flags = detect_flags(s, stale_days=30)
    assert any(f.kind == RedFlagKind.STALE for f in flags)
    stale_flag = next(f for f in flags if f.kind == RedFlagKind.STALE)
    assert stale_flag.severity == "medium"


def test_detect_flags_stale_not_raised_for_broken():
    # Broken already covers the problem; stale shouldn't double-flag
    s = _skill()
    s.validation_status = ValidationStatus.BROKEN
    s.last_validated_at = datetime.now(timezone.utc) - timedelta(days=60)
    flags = detect_flags(s, stale_days=30)
    assert not any(f.kind == RedFlagKind.STALE for f in flags)


def test_detect_flags_low_quality_high_use():
    s = _skill(score=0.1)
    s.use_count = 10
    flags = detect_flags(s)
    assert any(f.kind == RedFlagKind.LOW_QUALITY_HIGH_USE for f in flags)


def test_detect_flags_requirement_mismatch_in_metadata():
    s = _skill()
    s.metadata["requirement_mismatches"] = ["pandas: requires >=2.0, got 1.5"]
    flags = detect_flags(s)
    assert any(f.kind == RedFlagKind.REQUIREMENT_MISMATCH for f in flags)


def test_detect_flags_clean():
    s = _skill(score=0.9)
    s.validation_status = ValidationStatus.VALID
    s.last_validated_at = datetime.now(timezone.utc)
    flags = detect_flags(s)
    assert flags == []


# ===========================================================================
# format_skills_for_context
# ===========================================================================

def test_format_skills_empty():
    assert format_skills_for_context([]) == ""


def test_format_skills_includes_title_and_content():
    s = _skill()
    out = format_skills_for_context([s])
    assert s.title in out
    assert "def render_table" in out


def test_format_skills_stale_adds_note():
    s = _skill()
    s.validation_status = ValidationStatus.STALE
    out = format_skills_for_context([s])
    assert "STALE" in out


def test_format_skills_includes_requirements():
    s = _skill(requirements={"pandas": ">=2.0"})
    out = format_skills_for_context([s])
    assert "pandas" in out


def test_format_skills_respects_max_chars():
    skills = [_skill(content="x" * 2000) for _ in range(5)]
    out = format_skills_for_context(skills, max_chars=500)
    assert len(out) <= 600  # some header overhead allowed


# ===========================================================================
# SkillStore — CRUD
# ===========================================================================

@pytest.mark.asyncio
async def test_store_save_and_get():
    store = SkillStore(_redis())
    s = _skill()
    await store.save(s)
    loaded = await store.get(s.skill_id)
    assert loaded is not None
    assert loaded.title == s.title
    assert loaded.content == s.content


@pytest.mark.asyncio
async def test_store_get_missing_returns_none():
    store = SkillStore(_redis())
    result = await store.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_store_get_corrupt_returns_none():
    r = _redis()
    store = SkillStore(r)
    await r.set("harness:skill:bad", "{invalid json!!")
    result = await store.get("bad")
    assert result is None


@pytest.mark.asyncio
async def test_store_save_clamps_score_above_one():
    store = SkillStore(_redis())
    s = _skill(score=2.5)
    await store.save(s)
    loaded = await store.get(s.skill_id)
    assert loaded.score == 1.0


@pytest.mark.asyncio
async def test_store_save_clamps_score_below_zero():
    store = SkillStore(_redis())
    s = _skill(score=-0.5)
    await store.save(s)
    loaded = await store.get(s.skill_id)
    assert loaded.score == 0.0


@pytest.mark.asyncio
async def test_store_save_overwrites_existing():
    store = SkillStore(_redis())
    s = _skill()
    await store.save(s)
    s.title = "Updated title"
    await store.save(s)
    loaded = await store.get(s.skill_id)
    assert loaded.title == "Updated title"


@pytest.mark.asyncio
async def test_store_delete():
    r = _redis()
    store = SkillStore(r)
    s = _skill()
    await store.save(s)
    assert await store.get(s.skill_id) is not None
    await store.delete(s.skill_id, s.tenant_id)
    assert await store.get(s.skill_id) is None


@pytest.mark.asyncio
async def test_store_delete_nonexistent_no_error():
    store = SkillStore(_redis())
    await store.delete("ghost-id", "t1")  # must not raise


@pytest.mark.asyncio
async def test_store_save_raises_on_invalid():
    store = SkillStore(_redis())
    s = _skill(title="")
    with pytest.raises(ValueError):
        await store.save(s)


# ===========================================================================
# SkillStore — retrieval (no vector store)
# ===========================================================================

@pytest.mark.asyncio
async def test_store_retrieve_empty_returns_empty():
    store = SkillStore(_redis())
    results = await store.retrieve_relevant("some query", "t1")
    assert results == []


@pytest.mark.asyncio
async def test_store_retrieve_fallback_to_index():
    store = SkillStore(_redis())  # no memory_manager
    s1 = _skill(score=0.9)
    s2 = _skill(score=0.7, title="Another skill", description="Does something else",
                content="def other(): return 42")
    await store.save(s1)
    await store.save(s2)
    results = await store.retrieve_relevant("render table", "t1", k=5)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_store_retrieve_excludes_broken():
    store = SkillStore(_redis())
    s = _skill()
    s.validation_status = ValidationStatus.BROKEN
    await store.save(s)
    results = await store.retrieve_relevant("anything", "t1", exclude_broken=True)
    assert all(r.skill_id != s.skill_id for r in results)


@pytest.mark.asyncio
async def test_store_retrieve_includes_broken_when_flag_false():
    store = SkillStore(_redis())
    s = _skill()
    s.validation_status = ValidationStatus.BROKEN
    await store.save(s)
    results = await store.retrieve_relevant("anything", "t1", exclude_broken=False)
    assert any(r.skill_id == s.skill_id for r in results)


@pytest.mark.asyncio
async def test_store_retrieve_filters_low_score():
    store = SkillStore(_redis())
    s = _skill(score=0.1)
    await store.save(s)
    results = await store.retrieve_relevant("anything", "t1", min_score=0.5)
    assert all(r.skill_id != s.skill_id for r in results)


@pytest.mark.asyncio
async def test_store_retrieve_filters_by_skill_type():
    store = SkillStore(_redis())
    code_skill = _skill(skill_type=SkillType.CODE)
    approach_skill = _skill(
        skill_type=SkillType.APPROACH,
        title="Batch approach",
        description="How to batch process data",
        content="Process in chunks of 1000 rows for memory efficiency.",
    )
    await store.save(code_skill)
    await store.save(approach_skill)
    results = await store.retrieve_relevant(
        "batch process", "t1", skill_types=[SkillType.APPROACH]
    )
    assert all(r.skill_type == SkillType.APPROACH for r in results)


# ===========================================================================
# SkillStore — usage tracking
# ===========================================================================

@pytest.mark.asyncio
async def test_store_record_use_increments():
    store = SkillStore(_redis())
    s = _skill()
    await store.save(s)
    await store.record_use(s.skill_id, s.tenant_id)
    loaded = await store.get(s.skill_id)
    assert loaded.use_count == 1


@pytest.mark.asyncio
async def test_store_record_use_missing_skill_no_error():
    store = SkillStore(_redis())
    await store.record_use("ghost-id", "t1")  # must not raise


# ===========================================================================
# SkillStore — tenant isolation
# ===========================================================================

@pytest.mark.asyncio
async def test_store_get_denies_cross_tenant_access():
    """A tenant that learns another tenant's skill_id cannot read the artifact."""
    store = SkillStore(_redis())
    s = _skill(tenant_id="t1")
    await store.save(s)

    assert await store.get(s.skill_id, tenant_id="t2") is None
    assert (await store.get(s.skill_id, tenant_id="t1")).skill_id == s.skill_id


@pytest.mark.asyncio
async def test_store_get_without_tenant_skips_check():
    """tenant_id=None preserves the legacy no-check behaviour."""
    store = SkillStore(_redis())
    s = _skill(tenant_id="t1")
    await store.save(s)
    assert (await store.get(s.skill_id)).skill_id == s.skill_id


@pytest.mark.asyncio
async def test_store_record_use_denies_cross_tenant():
    """record_use from the wrong tenant must not modify the artifact."""
    store = SkillStore(_redis())
    s = _skill(tenant_id="t1")
    await store.save(s)

    await store.record_use(s.skill_id, "t2")  # wrong tenant — ignored
    loaded = await store.get(s.skill_id, tenant_id="t1")
    assert loaded.use_count == 0

    await store.record_use(s.skill_id, "t1")  # owner — counted
    loaded = await store.get(s.skill_id, tenant_id="t1")
    assert loaded.use_count == 1


@pytest.mark.asyncio
async def test_store_delete_denies_cross_tenant():
    """delete from the wrong tenant must leave the artifact intact."""
    store = SkillStore(_redis())
    s = _skill(tenant_id="t1")
    await store.save(s)

    await store.delete(s.skill_id, "t2")  # wrong tenant — ignored
    assert await store.get(s.skill_id, tenant_id="t1") is not None

    await store.delete(s.skill_id, "t1")
    assert await store.get(s.skill_id) is None


@pytest.mark.asyncio
async def test_store_update_validation_denies_cross_tenant():
    """update_validation with the wrong tenant returns None and changes nothing."""
    store = SkillStore(_redis())
    s = _skill(tenant_id="t1")
    await store.save(s)

    result = await store.update_validation(
        s.skill_id, ValidationStatus.BROKEN, tenant_id="t2"
    )
    assert result is None
    loaded = await store.get(s.skill_id, tenant_id="t1")
    assert loaded.validation_status == ValidationStatus.UNVALIDATED


# ===========================================================================
# SkillStore — validation
# ===========================================================================

@pytest.mark.asyncio
async def test_update_validation_sets_status():
    store = SkillStore(_redis())
    s = _skill()
    await store.save(s)
    updated = await store.update_validation(s.skill_id, ValidationStatus.VALID)
    assert updated.validation_status == ValidationStatus.VALID
    assert updated.last_validated_at is not None


@pytest.mark.asyncio
async def test_update_validation_broken_on_req_mismatch():
    store = SkillStore(_redis())
    s = _skill(requirements={"pandas": ">=2.0"})
    await store.save(s)
    updated = await store.update_validation(
        s.skill_id,
        ValidationStatus.VALID,
        env_requirements={"pandas": "1.5.0"},
    )
    assert updated.validation_status == ValidationStatus.BROKEN
    assert "requirement_mismatches" in updated.metadata


@pytest.mark.asyncio
async def test_update_validation_compatible_requirements_stays_valid():
    store = SkillStore(_redis())
    s = _skill(requirements={"pandas": ">=2.0"})
    await store.save(s)
    updated = await store.update_validation(
        s.skill_id,
        ValidationStatus.VALID,
        env_requirements={"pandas": "2.1.0"},
    )
    assert updated.validation_status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_update_validation_auto_stale():
    store = SkillStore(_redis(), stale_days=30)
    s = _skill()
    s.last_validated_at = datetime.now(timezone.utc) - timedelta(days=31)
    await store.save(s)
    # Manually set last_validated_at to old before update
    loaded = await store.get(s.skill_id)
    loaded.last_validated_at = datetime.now(timezone.utc) - timedelta(days=31)
    await store.save(loaded)
    updated = await store.update_validation(loaded.skill_id, ValidationStatus.VALID)
    # After update, last_validated_at is NOW, so NOT stale
    assert updated.last_validated_at is not None
    assert updated.validation_status == ValidationStatus.VALID


@pytest.mark.asyncio
async def test_update_validation_missing_skill_returns_none():
    store = SkillStore(_redis())
    result = await store.update_validation("ghost", ValidationStatus.VALID)
    assert result is None


# ===========================================================================
# SkillStore — health report
# ===========================================================================

@pytest.mark.asyncio
async def test_health_report_empty_store():
    store = SkillStore(_redis())
    report = await store.health_report("t1")
    assert report.total_skills == 0
    assert report.has_issues is False


@pytest.mark.asyncio
async def test_health_report_counts():
    store = SkillStore(_redis())
    valid_skill = _skill(score=0.9)
    valid_skill.validation_status = ValidationStatus.VALID
    valid_skill.last_validated_at = datetime.now(timezone.utc)
    broken_skill = _skill(title="Broken one", description="This skill is broken",
                          content="import this_does_not_exist")
    broken_skill.validation_status = ValidationStatus.BROKEN
    await store.save(valid_skill)
    await store.save(broken_skill)
    report = await store.health_report("t1")
    assert report.total_skills == 2
    assert report.valid == 1
    assert report.broken == 1


@pytest.mark.asyncio
async def test_health_report_red_flags_broken():
    store = SkillStore(_redis())
    s = _skill()
    s.validation_status = ValidationStatus.BROKEN
    await store.save(s)
    report = await store.health_report("t1")
    assert report.has_issues is True
    assert any(f.kind == RedFlagKind.BROKEN for f in report.red_flags)


@pytest.mark.asyncio
async def test_health_report_red_flags_stale():
    store = SkillStore(_redis(), stale_days=30)
    s = _skill()
    s.validation_status = ValidationStatus.VALID
    s.last_validated_at = datetime.now(timezone.utc) - timedelta(days=40)
    await store.save(s)
    report = await store.health_report("t1")
    assert any(f.kind == RedFlagKind.STALE for f in report.red_flags)


@pytest.mark.asyncio
async def test_health_report_red_flags_low_quality_high_use():
    store = SkillStore(_redis())
    s = _skill(score=0.1)
    s.use_count = 10
    await store.save(s)
    report = await store.health_report("t1")
    assert any(f.kind == RedFlagKind.LOW_QUALITY_HIGH_USE for f in report.red_flags)


@pytest.mark.asyncio
async def test_health_report_no_flags_for_healthy_skill():
    store = SkillStore(_redis())
    s = _skill(score=0.9)
    s.validation_status = ValidationStatus.VALID
    s.last_validated_at = datetime.now(timezone.utc)
    await store.save(s)
    report = await store.health_report("t1")
    assert report.has_issues is False


# ===========================================================================
# SkillStore — duplicate detection
# ===========================================================================

@pytest.mark.asyncio
async def test_is_duplicate_no_memory_returns_false():
    store = SkillStore(_redis())  # no memory manager
    result = await store.is_duplicate("any description", "t1")
    assert result is False


@pytest.mark.asyncio
async def test_is_duplicate_memory_high_similarity_returns_true():
    mock_memory = AsyncMock()
    hit = MagicMock()
    hit.score = 0.95
    hit.metadata = {"skill_id": "existing-id"}
    mock_memory.recall = AsyncMock(return_value=[hit])
    store = SkillStore(_redis(), memory_manager=mock_memory)
    result = await store.is_duplicate("some description", "t1", threshold=0.9)
    assert result is True


@pytest.mark.asyncio
async def test_is_duplicate_memory_low_similarity_returns_false():
    mock_memory = AsyncMock()
    hit = MagicMock()
    hit.score = 0.5
    hit.metadata = {"skill_id": "existing-id"}
    mock_memory.recall = AsyncMock(return_value=[hit])
    store = SkillStore(_redis(), memory_manager=mock_memory)
    result = await store.is_duplicate("some description", "t1", threshold=0.9)
    assert result is False


@pytest.mark.asyncio
async def test_is_duplicate_memory_error_returns_false():
    mock_memory = AsyncMock()
    mock_memory.recall = AsyncMock(side_effect=RuntimeError("vector store down"))
    store = SkillStore(_redis(), memory_manager=mock_memory)
    result = await store.is_duplicate("description", "t1")
    assert result is False  # fail-open: treat as not duplicate


# ===========================================================================
# SkillCapture — auto-save gate
# ===========================================================================

@pytest.mark.asyncio
async def test_capture_below_score_returns_none():
    store = SkillStore(_redis())
    capture = SkillCapture(store, min_score=0.8)
    result = await capture.capture(
        title="T", description="D", content="x = 1 + 1 + 1",
        skill_type=SkillType.CODE, tenant_id="t1", score=0.5,
    )
    assert result is None


@pytest.mark.asyncio
async def test_capture_duplicate_returns_none():
    mock_memory = AsyncMock()
    hit = MagicMock()
    hit.score = 0.95
    hit.metadata = {"skill_id": "existing"}
    mock_memory.recall = AsyncMock(return_value=[hit])
    store = SkillStore(_redis(), memory_manager=mock_memory)
    capture = SkillCapture(store, min_score=0.8, novelty_threshold=0.9)
    result = await capture.capture(
        title="T", description="D", content="x = 1 + 1 + 1",
        skill_type=SkillType.CODE, tenant_id="t1", score=0.9,
    )
    assert result is None


@pytest.mark.asyncio
async def test_capture_saves_valid_skill():
    store = SkillStore(_redis())
    capture = SkillCapture(store, min_score=0.8)
    result = await capture.capture(
        title="Batch insert helper",
        description="Inserts rows in batches of 1000 to avoid memory pressure",
        content="def batch_insert(rows, conn, batch=1000):\n    for i in range(0, len(rows), batch):\n        conn.executemany(sql, rows[i:i+batch])",
        skill_type=SkillType.CODE,
        tenant_id="t1",
        score=0.9,
        language="python",
        requirements={"psycopg2": ">=2.9"},
        run_id="run-abc",
    )
    assert result is not None
    assert result.score == 0.9
    assert result.language == "python"
    loaded = await store.get(result.skill_id)
    assert loaded is not None


@pytest.mark.asyncio
async def test_capture_with_approach_type():
    store = SkillStore(_redis())
    capture = SkillCapture(store, min_score=0.7)
    result = await capture.capture(
        title="Pagination strategy",
        description="Use cursor-based pagination instead of OFFSET for large datasets",
        content="Always use WHERE id > last_seen_id LIMIT N instead of OFFSET. Reason: OFFSET scans skipped rows; cursor pagination is O(1) at any depth.",
        skill_type=SkillType.APPROACH,
        tenant_id="t1",
        score=0.85,
    )
    assert result is not None
    assert result.skill_type == SkillType.APPROACH


# ===========================================================================
# BaseAgent._retrieve_skills integration (smoke test)
# ===========================================================================

@pytest.mark.asyncio
async def test_base_agent_retrieve_skills_noop_without_store(agent_context):
    from harness.agents.base import BaseAgent
    from unittest.mock import AsyncMock, MagicMock

    agent = BaseAgent(
        llm_router=MagicMock(), memory_manager=None, tool_registry=None,
        safety_pipeline=None, step_tracer=None, mlflow_tracer=None,
        failure_tracker=None, audit_logger=None, event_bus=None,
        cost_tracker=None, checkpoint_manager=None,
    )
    ctx = agent_context()
    result = await agent._retrieve_skills(ctx)
    assert result == ""


@pytest.mark.asyncio
async def test_base_agent_retrieve_skills_returns_formatted(agent_context):
    from harness.agents.base import BaseAgent
    from unittest.mock import AsyncMock, MagicMock

    r = _redis()
    store = SkillStore(r)
    s = _skill()
    await store.save(s)

    agent = BaseAgent(
        llm_router=MagicMock(), memory_manager=None, tool_registry=None,
        safety_pipeline=None, step_tracer=None, mlflow_tracer=None,
        failure_tracker=None, audit_logger=None, event_bus=None,
        cost_tracker=None, checkpoint_manager=None,
    )
    ctx = agent_context()
    ctx.metadata["skill_store"] = store
    result = await agent._retrieve_skills(ctx)
    # May be empty (no vector search) or non-empty (index fallback)
    # What matters: no exception and returns a string
    assert isinstance(result, str)
