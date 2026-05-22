"""Skill artifact store — versioned, validated, vector-searchable reuse library.

Skills are reusable artifacts (code, approach, full codebase patterns, monitoring
configs) that agents can retrieve instead of regenerating common work.

Storage:
    harness:skill:{skill_id}            — SkillArtifact JSON  (TTL 90 d)
    harness:skill_index:{tenant_id}     — sorted set  skill_id → relevance score
    harness:skill_flags:{tenant_id}     — set of skill_ids with active red flags
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger(__name__)

_SKILL_TTL = 86400 * 90   # 90 days
_STALE_DAYS = 30
_MIN_CONTENT_LEN = 10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SkillType(str, Enum):
    CODE = "code"
    APPROACH = "approach"
    CODEBASE = "codebase"
    MONITORING = "monitoring"


class ValidationStatus(str, Enum):
    VALID = "valid"
    STALE = "stale"
    BROKEN = "broken"
    UNVALIDATED = "unvalidated"


class RedFlagKind(str, Enum):
    BROKEN = "broken"
    STALE = "stale"
    LOW_QUALITY_HIGH_USE = "low_quality_high_use"
    REQUIREMENT_MISMATCH = "requirement_mismatch"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkillArtifact:
    """A versioned, validated reusable artifact with dependency metadata."""

    skill_id: str
    tenant_id: str
    skill_type: SkillType
    title: str
    description: str        # natural-language summary — gets embedded for search
    content: str            # actual code / approach / docs
    language: str | None = None             # python, typescript, sql, bash, text
    tags: list[str] = field(default_factory=list)
    requirements: dict[str, str] = field(default_factory=dict)  # {"pandas": ">=2.0"}
    metadata: dict[str, Any] = field(default_factory=dict)      # free-form extras
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = 0.5
    use_count: int = 0
    last_validated_at: datetime | None = None
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    run_id: str | None = None   # producing agent run

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def is_stale(self, stale_days: int = _STALE_DAYS) -> bool:
        """True when last validation is older than stale_days.

        Returns False (not stale) when the skill has never been validated —
        that is ValidationStatus.UNVALIDATED, a distinct state.
        """
        if self.last_validated_at is None:
            return False
        return self.last_validated_at < datetime.now(timezone.utc) - timedelta(days=stale_days)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "tenant_id": self.tenant_id,
            "skill_type": self.skill_type.value,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "language": self.language,
            "tags": self.tags,
            "requirements": self.requirements,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "score": self.score,
            "use_count": self.use_count,
            "last_validated_at": (
                self.last_validated_at.isoformat() if self.last_validated_at else None
            ),
            "validation_status": self.validation_status.value,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SkillArtifact":
        d = dict(d)
        for f in ("created_at", "updated_at"):
            if isinstance(d.get(f), str):
                d[f] = datetime.fromisoformat(d[f])
        lv = d.get("last_validated_at")
        if isinstance(lv, str):
            d["last_validated_at"] = datetime.fromisoformat(lv)
        if isinstance(d.get("skill_type"), str):
            d["skill_type"] = SkillType(d["skill_type"])
        if isinstance(d.get("validation_status"), str):
            d["validation_status"] = ValidationStatus(d["validation_status"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RedFlag:
    skill_id: str
    title: str
    kind: RedFlagKind
    detail: str
    severity: Literal["low", "medium", "high"]


@dataclass
class SkillHealthReport:
    total_skills: int
    valid: int
    stale: int
    broken: int
    unvalidated: int
    red_flags: list[RedFlag] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.red_flags)


# ---------------------------------------------------------------------------
# Requirement checking helpers
# ---------------------------------------------------------------------------

def check_requirements(
    skill_reqs: dict[str, str],
    env_reqs: dict[str, str],
) -> list[str]:
    """Return mismatch descriptions. Empty list = fully compatible.

    ``env_reqs`` is a flat dict of {package_name: installed_version},
    e.g. from ``pip freeze`` parsed into a dict.
    """
    mismatches: list[str] = []
    for pkg, required_spec in skill_reqs.items():
        if not required_spec:
            continue
        env_version = env_reqs.get(pkg)
        if env_version is None:
            mismatches.append(
                f"'{pkg}' not present in environment (skill requires {required_spec})"
            )
            continue
        msg = _version_mismatch(pkg, required_spec, env_version)
        if msg:
            mismatches.append(msg)
    return mismatches


def _version_mismatch(pkg: str, spec: str, actual: str) -> str | None:
    """Return description string if actual version doesn't satisfy spec, else None."""
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore[import]
        from packaging.version import Version          # type: ignore[import]
        if Version(actual) not in SpecifierSet(spec):
            return f"'{pkg}': requires {spec}, installed {actual}"
        return None
    except Exception:
        # packaging unavailable or malformed spec — fall back to equality check only
        if spec.startswith("==") and spec[2:].strip() != actual:
            return f"'{pkg}': requires =={spec[2:].strip()}, installed {actual}"
    return None


# ---------------------------------------------------------------------------
# Red-flag detection
# ---------------------------------------------------------------------------

def detect_flags(skill: SkillArtifact, stale_days: int = _STALE_DAYS) -> list[RedFlag]:
    """Return all active red flags for a skill artifact."""
    flags: list[RedFlag] = []

    if skill.validation_status == ValidationStatus.BROKEN:
        detail = ", ".join(skill.metadata.get("requirement_mismatches", ["validation failed"]))
        flags.append(RedFlag(
            skill_id=skill.skill_id,
            title=skill.title,
            kind=RedFlagKind.BROKEN,
            detail=detail or "validation failed",
            severity="high",
        ))

    if skill.is_stale(stale_days) and skill.validation_status != ValidationStatus.BROKEN:
        flags.append(RedFlag(
            skill_id=skill.skill_id,
            title=skill.title,
            kind=RedFlagKind.STALE,
            detail=f"Not validated in >{stale_days} days",
            severity="medium",
        ))

    if skill.use_count > SkillStore.HIGH_USE_THRESHOLD and skill.score < SkillStore.LOW_QUALITY_THRESHOLD:
        flags.append(RedFlag(
            skill_id=skill.skill_id,
            title=skill.title,
            kind=RedFlagKind.LOW_QUALITY_HIGH_USE,
            detail=f"score={skill.score:.2f}, use_count={skill.use_count}",
            severity="medium",
        ))

    if skill.metadata.get("requirement_mismatches"):
        # Only add if not already covered by BROKEN flag
        if not any(f.kind == RedFlagKind.REQUIREMENT_MISMATCH for f in flags):
            flags.append(RedFlag(
                skill_id=skill.skill_id,
                title=skill.title,
                kind=RedFlagKind.REQUIREMENT_MISMATCH,
                detail=str(skill.metadata["requirement_mismatches"]),
                severity="high",
            ))

    return flags


# ---------------------------------------------------------------------------
# SkillStore
# ---------------------------------------------------------------------------

class SkillStore:
    """Redis-backed versioned skill store with optional vector search.

    Vector search is performed via ``memory_manager.recall()`` when available;
    falls back to a relevance-sorted Redis index otherwise.
    """

    SIMILARITY_THRESHOLD: float = 0.9    # above → duplicate, skip auto-save
    LOW_QUALITY_THRESHOLD: float = 0.3
    HIGH_USE_THRESHOLD: int = 5

    def __init__(
        self,
        redis: Any,
        memory_manager: Any | None = None,
        stale_days: int = _STALE_DAYS,
    ) -> None:
        self._redis = redis
        self._memory = memory_manager
        self._stale_days = stale_days

    # ------------------------------------------------------------------
    # Key builders
    # ------------------------------------------------------------------

    @staticmethod
    def _key(skill_id: str) -> str:
        return f"harness:skill:{skill_id}"

    @staticmethod
    def _index_key(tenant_id: str) -> str:
        return f"harness:skill_index:{tenant_id}"

    @staticmethod
    def _flags_key(tenant_id: str) -> str:
        return f"harness:skill_flags:{tenant_id}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save(self, skill: SkillArtifact) -> SkillArtifact:
        """Persist a skill. Validates, clamps score, updates timestamps."""
        _validate_artifact(skill)
        skill.score = max(0.0, min(1.0, skill.score))
        skill.updated_at = datetime.now(timezone.utc)

        await self._redis.setex(
            self._key(skill.skill_id),
            _SKILL_TTL,
            json.dumps(skill.to_dict(), ensure_ascii=False, default=str),
        )

        # Sorted index: score × log(use_count+1) keeps popular+quality skills up top
        relevance = skill.score * math.log1p(skill.use_count)
        await self._redis.zadd(
            self._index_key(skill.tenant_id),
            {skill.skill_id: relevance},
        )

        # Vector index (best-effort — metadata still saved if this fails)
        if self._memory is not None:
            try:
                await self._memory.remember(
                    text=f"{skill.title}: {skill.description}",
                    metadata={
                        "skill_id": skill.skill_id,
                        "tenant_id": skill.tenant_id,
                        "skill_type": skill.skill_type.value,
                        "language": skill.language or "",
                        "tags": ",".join(skill.tags),
                        "validation_status": skill.validation_status.value,
                    },
                    tenant_id=skill.tenant_id,
                )
            except Exception as exc:
                logger.warning("SkillStore: vector index failed (metadata saved): %s", exc)

        await self._update_flags(skill)
        logger.info(
            "Skill saved: %s (%s) type=%s tenant=%s",
            skill.skill_id, skill.title, skill.skill_type.value, skill.tenant_id,
        )
        return skill

    async def get(self, skill_id: str) -> SkillArtifact | None:
        """Load by ID. Returns None on missing, corrupt, or Redis error."""
        try:
            raw = await self._redis.get(self._key(skill_id))
            if raw is None:
                return None
            data = json.loads(raw if isinstance(raw, str) else raw.decode())
            return SkillArtifact.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("SkillStore.get(%s) corrupt data: %s", skill_id, exc)
            return None
        except Exception as exc:
            logger.error("SkillStore.get(%s) error: %s", skill_id, exc)
            return None

    async def delete(self, skill_id: str, tenant_id: str) -> None:
        """Remove artifact, index entry, and flag entry. Never raises."""
        try:
            await self._redis.delete(self._key(skill_id))
            await self._redis.zrem(self._index_key(tenant_id), skill_id)
            await self._redis.srem(self._flags_key(tenant_id), skill_id)
        except Exception as exc:
            logger.warning("SkillStore.delete(%s) failed: %s", skill_id, exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve_relevant(
        self,
        query: str,
        tenant_id: str,
        k: int = 5,
        skill_types: list[SkillType] | None = None,
        min_score: float = 0.2,
        exclude_broken: bool = True,
    ) -> list[SkillArtifact]:
        """Return top-k skills semantically relevant to query.

        Tries vector search first; falls back to relevance-sorted Redis index.
        Filters: broken status, minimum quality score, skill type whitelist.
        """
        candidates: list[SkillArtifact] = []

        if self._memory is not None:
            try:
                hits = await self._memory.recall(
                    query=query,
                    k=k * 3,
                    filter={"tenant_id": tenant_id},
                )
                for hit in hits:
                    skill_id = hit.metadata.get("skill_id")
                    if not skill_id:
                        continue
                    skill = await self.get(skill_id)
                    if skill is None:
                        continue
                    if exclude_broken and skill.validation_status == ValidationStatus.BROKEN:
                        continue
                    if skill.score < min_score:
                        continue
                    if skill_types and skill.skill_type not in skill_types:
                        continue
                    candidates.append(skill)
                    if len(candidates) >= k:
                        break
            except Exception as exc:
                logger.warning("SkillStore: vector search failed, falling back: %s", exc)

        if not candidates:
            try:
                raw_ids = await self._redis.zrevrange(
                    self._index_key(tenant_id), 0, k * 2 - 1
                )
                for sid in raw_ids:
                    sid_str = sid if isinstance(sid, str) else sid.decode()
                    skill = await self.get(sid_str)
                    if skill is None:
                        continue
                    if exclude_broken and skill.validation_status == ValidationStatus.BROKEN:
                        continue
                    if skill.score < min_score:
                        continue
                    if skill_types and skill.skill_type not in skill_types:
                        continue
                    candidates.append(skill)
                    if len(candidates) >= k:
                        break
            except Exception as exc:
                logger.warning("SkillStore: index fallback failed: %s", exc)

        return candidates[:k]

    async def is_duplicate(
        self,
        description: str,
        tenant_id: str,
        threshold: float | None = None,
    ) -> bool:
        """True if a near-identical skill already exists in the vector store."""
        if self._memory is None:
            return False
        thr = threshold if threshold is not None else self.SIMILARITY_THRESHOLD
        try:
            hits = await self._memory.recall(
                query=description, k=1, filter={"tenant_id": tenant_id}
            )
            return bool(hits and hits[0].score >= thr)
        except Exception as exc:
            logger.debug("is_duplicate check failed (treating as not duplicate): %s", exc)
            return False

    # ------------------------------------------------------------------
    # Usage & validation
    # ------------------------------------------------------------------

    async def record_use(self, skill_id: str, tenant_id: str) -> None:
        """Increment use_count and refresh the index score."""
        skill = await self.get(skill_id)
        if skill is None:
            return
        skill.use_count += 1
        await self.save(skill)

    async def update_validation(
        self,
        skill_id: str,
        status: ValidationStatus,
        env_requirements: dict[str, str] | None = None,
    ) -> SkillArtifact | None:
        """Set validation status, check requirements, auto-detect staleness.

        ``env_requirements`` — {package: installed_version} from the runtime
        environment (e.g. parsed from pip freeze). If provided, any
        requirement mismatch forces status to BROKEN regardless of ``status``.
        """
        skill = await self.get(skill_id)
        if skill is None:
            return None

        skill.validation_status = status
        skill.last_validated_at = datetime.now(timezone.utc)

        if env_requirements and skill.requirements:
            mismatches = check_requirements(skill.requirements, env_requirements)
            if mismatches:
                skill.validation_status = ValidationStatus.BROKEN
                skill.metadata["requirement_mismatches"] = mismatches
                logger.warning(
                    "Skill %s (%s) has requirement mismatches: %s",
                    skill_id, skill.title, mismatches,
                )

        # Auto-detect staleness after the explicit status is set
        if skill.is_stale(self._stale_days) and skill.validation_status == ValidationStatus.VALID:
            skill.validation_status = ValidationStatus.STALE

        await self.save(skill)
        return skill

    # ------------------------------------------------------------------
    # Health / dashboard
    # ------------------------------------------------------------------

    async def health_report(self, tenant_id: str) -> SkillHealthReport:
        """Scan all skills and return counts + red flags for the dashboard."""
        try:
            all_ids = await self._redis.zrange(self._index_key(tenant_id), 0, -1)
        except Exception as exc:
            logger.warning("SkillStore.health_report scan failed: %s", exc)
            return SkillHealthReport(0, 0, 0, 0, 0)

        counts: dict[str, int] = {"valid": 0, "stale": 0, "broken": 0, "unvalidated": 0}
        red_flags: list[RedFlag] = []

        for sid in all_ids:
            sid_str = sid if isinstance(sid, str) else sid.decode()
            skill = await self.get(sid_str)
            if skill is None:
                continue
            counts[skill.validation_status.value] = counts.get(skill.validation_status.value, 0) + 1
            red_flags.extend(detect_flags(skill, self._stale_days))

        return SkillHealthReport(
            total_skills=sum(counts.values()),
            valid=counts["valid"],
            stale=counts["stale"],
            broken=counts["broken"],
            unvalidated=counts["unvalidated"],
            red_flags=red_flags,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _update_flags(self, skill: SkillArtifact) -> None:
        flags = detect_flags(skill, self._stale_days)
        key = self._flags_key(skill.tenant_id)
        try:
            if flags:
                await self._redis.sadd(key, skill.skill_id)
            else:
                await self._redis.srem(key, skill.skill_id)
        except Exception as exc:
            logger.debug("_update_flags failed: %s", exc)


# ---------------------------------------------------------------------------
# SkillCapture — auto-save gate
# ---------------------------------------------------------------------------

class SkillCapture:
    """Captures reusable skills from successful runs.

    Two gates:
    1. Quality: ``score >= min_score``
    2. Novelty: cosine similarity to nearest existing skill < ``novelty_threshold``
    """

    def __init__(
        self,
        skill_store: SkillStore,
        min_score: float = 0.8,
        novelty_threshold: float = 0.9,
    ) -> None:
        self._store = skill_store
        self._min_score = min_score
        self._novelty_threshold = novelty_threshold

    async def capture(
        self,
        *,
        title: str,
        description: str,
        content: str,
        skill_type: SkillType,
        tenant_id: str,
        score: float,
        language: str | None = None,
        tags: list[str] | None = None,
        requirements: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> SkillArtifact | None:
        """Save if quality and novelty gates pass. Returns None if skipped."""
        if score < self._min_score:
            logger.debug(
                "Skill capture skipped: score %.2f < threshold %.2f",
                score, self._min_score,
            )
            return None

        if await self._store.is_duplicate(description, tenant_id, self._novelty_threshold):
            logger.debug("Skill capture skipped: near-duplicate exists for '%s'", title)
            return None

        skill = SkillArtifact(
            skill_id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            skill_type=skill_type,
            title=title,
            description=description,
            content=content,
            language=language,
            tags=tags or [],
            requirements=requirements or {},
            metadata=metadata or {},
            score=score,
            run_id=run_id,
        )
        return await self._store.save(skill)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_skills_for_context(skills: list[SkillArtifact], max_chars: int = 4_000) -> str:
    """Render relevant skills as a compact context block for agent injection.

    Truncates at ``max_chars`` to avoid blowing the context window.
    Stale skills are labelled so the agent knows to verify before reusing.
    """
    if not skills:
        return ""

    parts = ["[Relevant skills from library]"]
    total = len(parts[0])

    for skill in skills:
        stale_note = " [STALE — verify before reuse]" if skill.validation_status == ValidationStatus.STALE else ""
        req_note = ""
        if skill.requirements:
            req_note = f"\n    Requirements: {json.dumps(skill.requirements)}"
        block = (
            f"\n--- {skill.title} ({skill.skill_type.value}){stale_note} ---"
            f"{req_note}"
            f"\n{skill.content}"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts) if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# Validation helper (internal)
# ---------------------------------------------------------------------------

def _validate_artifact(skill: SkillArtifact) -> None:
    if not skill.skill_id:
        raise ValueError("skill_id is required")
    if not skill.tenant_id:
        raise ValueError("tenant_id is required")
    if not skill.title.strip():
        raise ValueError("title cannot be empty")
    if not skill.description.strip():
        raise ValueError("description cannot be empty")
    if len(skill.content) < _MIN_CONTENT_LEN:
        raise ValueError(
            f"content too short: {len(skill.content)} chars (minimum {_MIN_CONTENT_LEN})"
        )
