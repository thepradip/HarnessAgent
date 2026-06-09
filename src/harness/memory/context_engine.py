"""
ContextEngine — paged, skill-isolated context management for agent runs.

Capabilities
------------
Offload     : cold messages are grouped into pages, compressed, embedded,
              and evicted from the hot Redis window to the vector store.
Compress    : LLM (preferred) or extractive fallback; keeps tool results,
              errors, and decisions; discards filler.
Select      : per-query semantic retrieval of cold pages; only pages
              relevant to the current action are re-injected.
Isolate     : every skill namespace (sql, code, search, default …)
              maintains a separate hot-window key so context from one
              skill does not pollute another.
Evaluate    : after each LLM + tool round-trip a rule-based scorer
              (goal_progress, tool_relevance, confidence) stores an
              ActionRecord in Redis.  No LLM call needed per action.
Sub-agents  : a parent can slice its context (relevant cold pages +
              recent hot messages) into a child's hot window with a
              token budget cap; the child's result is injected back as
              a single compressed tool message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from harness.core.protocols import EmbeddingProvider, VectorStore
from harness.memory.embedder import estimate_tokens
from harness.memory.schemas import ConversationMessage

logger = logging.getLogger(__name__)

# ── Redis key prefixes ────────────────────────────────────────────────────────
_HOT_PFX = "harness:hot:"        # LIST  – per (run_id, skill_ns), newest-first
_PAGE_PFX = "harness:page:"      # HASH  – ContextPage data
_PAGES_SET_PFX = "harness:pages:" # ZSET  – page_ids scored by importance
_ACTIONS_PFX = "harness:actions:" # LIST  – ActionRecord per run_id

# ── Tuning constants ──────────────────────────────────────────────────────────
_PAGE_TOKEN_TARGET = 2_000        # target tokens per offloaded page
_PAGE_TTL = 86_400                # 24 h TTL for cold pages
_ACTIONS_TTL = 86_400             # 24 h TTL for action logs
_HOT_MAX_MSGS = 200               # message count that triggers offload check
_HOT_TTL = 86_400                 # 24 h TTL for hot windows (prevent unbounded growth)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ContextPage:
    """A compressed segment of evicted conversation history."""
    page_id: str
    run_id: str
    skill_ns: str
    step_start: int
    step_end: int
    token_count: int
    importance: float             # 0–1; higher = keep longer in index
    created_at: datetime
    summary: str                  # compressed text (LLM or extractive)
    raw_count: int                # number of original messages in page


@dataclass
class ActionRecord:
    """Scored record of one agent action (one LLM call + optional tool call)."""
    action_id: str
    run_id: str
    step: int
    skill_ns: str
    llm_preview: str              # first 300 chars of LLM response
    tool_name: str | None
    goal_progress: float          # 0–1
    tool_relevance: float         # 0–1
    confidence: float             # 0–1
    composite_score: float        # 0.5·goal + 0.3·tool + 0.2·conf
    is_error: bool
    timestamp: datetime


@dataclass
class BuiltContext:
    """Assembled context window ready to pass to an LLM."""
    messages: list[ConversationMessage]
    total_tokens: int
    hot_tokens: int
    cold_tokens: int
    pages_retrieved: int
    skill_ns: str
    truncated: bool


@dataclass
class SubAgentSlice:
    """Context slice handed from a parent agent to a child agent."""
    parent_run_id: str
    child_run_id: str
    task: str
    messages: list[ConversationMessage]
    token_budget: int
    page_ids: list[str]


@dataclass
class ActionSummary:
    """Aggregated action metrics for a run."""
    total_actions: int
    avg_score: float
    min_score: float
    max_score: float
    error_count: int
    low_score_count: int
    by_skill: dict[str, dict[str, Any]] = field(default_factory=dict)


# ── Main engine ───────────────────────────────────────────────────────────────

class ContextEngine:
    """
    Advanced context engineering for agent runs.

    Hot window (Redis LIST per run_id:skill_ns)
        ↓  offload when > offload_threshold * max_hot_tokens
    Cold pages (Redis HASH + vector store)
        ↑  retrieved by semantic search for each LLM call

    Usage (typical)
    ---------------
    engine = await ContextEngine.create(redis_url, vector_store, embedder, llm)

    # Push each message
    await engine.push(run_id, "user", "…", skill_ns="sql")

    # Build context before LLM call
    ctx = await engine.build_context(run_id, query="list active users", skill_ns="sql")

    # Score the action after LLM + tool round-trip
    action = await engine.evaluate_action(run_id, step=3, goal="list users",
                                           llm_content="…", tool_name="execute_sql",
                                           tool_result="[(1,'Alice'),…]")

    # Slice context for a child agent
    slice_ = await engine.slice_for_subagent(parent_id, child_id, task, budget=8_000)

    # Inject child result back into parent
    await engine.inject_subagent_result(parent_id, child_id, result_summary)
    """

    def __init__(
        self,
        max_hot_tokens: int,
        redis_url: str,
        vector_store: VectorStore | None,
        embedder: EmbeddingProvider | None,
        summarizer: Any | None = None,      # LLMProvider used for compression
        reserve_output: int = 2_000,
        offload_threshold: float = 0.80,    # offload when hot > threshold * max
        cold_pages_per_query: int = 3,
    ) -> None:
        self._max_hot = max_hot_tokens
        self._redis_url = redis_url
        self._pool: ConnectionPool | None = None
        self._client: aioredis.Redis | None = None
        self._vector_store = vector_store
        self._embedder = embedder
        self._summarizer = summarizer
        self._reserve = reserve_output
        self._threshold = offload_threshold
        self._cold_k = cold_pages_per_query
        # Serialize offloads per hot-key so concurrent pushes can't double-trim
        # or duplicate pages.
        self._offload_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── Connection ────────────────────────────────────────────────────────────

    async def _redis(self) -> aioredis.Redis:
        if self._client is None:
            self._pool = aioredis.ConnectionPool.from_url(
                self._redis_url, max_connections=20, decode_responses=True
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None

    # ── Push ─────────────────────────────────────────────────────────────────

    async def push(
        self,
        run_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        skill_ns: str = "default",
        step: int = 0,
    ) -> None:
        """Append a message to the hot window; trigger offload if needed."""
        if tokens <= 0:
            tokens = estimate_tokens(content)

        r = await self._redis()
        raw = json.dumps({
            "role": role,
            "content": content,
            "tokens": tokens,
            "skill_ns": skill_ns,
            "step": step,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        hot_key = self._hot_key(run_id, skill_ns)
        await r.lpush(hot_key, raw)
        await r.expire(hot_key, _HOT_TTL)
        await self._maybe_offload(run_id, skill_ns)

    # ── Build context ─────────────────────────────────────────────────────────

    async def build_context(
        self,
        run_id: str,
        query: str,
        skill_ns: str = "default",
        token_budget: int | None = None,
        include_shared: bool = True,
    ) -> BuiltContext:
        """
        Assemble an LLM-ready context window for the current query.

        Layout (oldest → newest):
          [cold page summaries for this skill]
          [shared/default hot messages if skill != default]
          [skill-specific hot messages]
        """
        budget = (token_budget or self._max_hot) - self._reserve

        # 1. Hot messages for this skill
        hot_msgs = await self._load_hot(run_id, skill_ns)

        # 2. Hot messages from shared namespace (if different skill active)
        if include_shared and skill_ns != "default":
            shared = await self._load_hot(run_id, "default")
            hot_msgs = _merge_by_step(hot_msgs, shared)

        hot_tokens = sum(m.tokens for m in hot_msgs)

        # 3. Cold page retrieval — give cold at most half the remaining budget
        remaining = max(0, budget - hot_tokens)
        cold_msgs: list[ConversationMessage] = []
        pages_retrieved = 0

        if remaining > 500 and self._vector_store is not None:
            cold_budget = min(remaining // 2, 8_000)
            cold_msgs, pages_retrieved = await self._retrieve_cold_pages(
                run_id=run_id,
                query=query,
                skill_ns=skill_ns,
                token_budget=cold_budget,
            )

        # 4. Merge and fit to budget
        merged = cold_msgs + hot_msgs
        fitted, total_tokens, truncated = _sliding_fit(merged, budget)

        cold_tokens = sum(m.tokens for m in fitted if m.role == "system"
                          and m.content.startswith("[Context page"))

        return BuiltContext(
            messages=fitted,
            total_tokens=total_tokens,
            hot_tokens=total_tokens - cold_tokens,
            cold_tokens=cold_tokens,
            pages_retrieved=pages_retrieved,
            skill_ns=skill_ns,
            truncated=truncated,
        )

    # ── Action evaluation ─────────────────────────────────────────────────────

    async def evaluate_action(
        self,
        run_id: str,
        step: int,
        goal: str,
        llm_content: str,
        tool_name: str | None = None,
        tool_result: str | None = None,
        is_error: bool = False,
        skill_ns: str = "default",
    ) -> ActionRecord:
        """
        Score one agent action (rule-based, no LLM call).

        Scores
        ------
        goal_progress  : keyword overlap between response and goal text
        tool_relevance : 0.9 on success, 0.1 on error, 0.5 when no tool used
        confidence     : hedging-word detector on LLM response
        composite      : 0.5·goal + 0.3·tool + 0.2·confidence
        """
        gp = _score_goal_progress(llm_content, goal)
        tr = _score_tool_relevance(tool_name, tool_result, is_error)
        cf = _score_confidence(llm_content)
        composite = round(0.5 * gp + 0.3 * tr + 0.2 * cf, 4)

        record = ActionRecord(
            action_id=uuid.uuid4().hex,
            run_id=run_id,
            step=step,
            skill_ns=skill_ns,
            llm_preview=llm_content[:300],
            tool_name=tool_name,
            goal_progress=gp,
            tool_relevance=tr,
            confidence=cf,
            composite_score=composite,
            is_error=is_error,
            timestamp=datetime.now(timezone.utc),
        )

        r = await self._redis()
        key = f"{_ACTIONS_PFX}{run_id}"
        await r.rpush(key, json.dumps({
            "action_id": record.action_id,
            "step": record.step,
            "skill_ns": record.skill_ns,
            "tool_name": record.tool_name,
            "goal_progress": gp,
            "tool_relevance": tr,
            "confidence": cf,
            "composite_score": composite,
            "is_error": is_error,
            "llm_preview": record.llm_preview,
            "ts": record.timestamp.isoformat(),
        }))
        await r.expire(key, _ACTIONS_TTL)

        return record

    async def get_action_log(
        self,
        run_id: str,
        only_errors: bool = False,
        min_score: float | None = None,
    ) -> list[ActionRecord]:
        """Return scored action records for this run (all or filtered)."""
        r = await self._redis()
        raw_items = await r.lrange(f"{_ACTIONS_PFX}{run_id}", 0, -1)
        records: list[ActionRecord] = []
        for raw in raw_items:
            try:
                d = json.loads(raw)
                rec = ActionRecord(
                    action_id=d["action_id"],
                    run_id=run_id,
                    step=d["step"],
                    skill_ns=d.get("skill_ns", "default"),
                    llm_preview=d.get("llm_preview", ""),
                    tool_name=d.get("tool_name"),
                    goal_progress=d["goal_progress"],
                    tool_relevance=d["tool_relevance"],
                    confidence=d["confidence"],
                    composite_score=d["composite_score"],
                    is_error=d.get("is_error", False),
                    timestamp=datetime.fromisoformat(d["ts"]),
                )
                if only_errors and not rec.is_error:
                    continue
                if min_score is not None and rec.composite_score < min_score:
                    continue
                records.append(rec)
            except Exception:
                pass
        return records

    async def get_action_summary(self, run_id: str) -> ActionSummary:
        """Aggregate action metrics for a completed run."""
        records = await self.get_action_log(run_id)
        if not records:
            return ActionSummary(0, 0.0, 0.0, 0.0, 0, 0)

        scores = [r.composite_score for r in records]
        return ActionSummary(
            total_actions=len(records),
            avg_score=round(sum(scores) / len(scores), 4),
            min_score=round(min(scores), 4),
            max_score=round(max(scores), 4),
            error_count=sum(1 for r in records if r.is_error),
            low_score_count=sum(1 for r in records if r.composite_score < 0.4),
            by_skill=_group_by_skill(records),
        )

    # ── Sub-agent context bridge ──────────────────────────────────────────────

    async def slice_for_subagent(
        self,
        parent_run_id: str,
        child_run_id: str,
        task: str,
        token_budget: int,
        skill_ns: str = "default",
    ) -> SubAgentSlice:
        """
        Create a focused context slice for a child agent.

        Cold budget = token_budget // 2 (semantic search)
        Hot budget  = token_budget // 2 (most recent relevant messages)
        The slice is pushed into the child's hot window as pre-loaded context.
        """
        cold_msgs, page_ids = await self._retrieve_cold_pages(
            run_id=parent_run_id,
            query=task,
            skill_ns=skill_ns,
            token_budget=token_budget // 2,
        )

        parent_hot = await self._load_hot(parent_run_id, skill_ns)
        hot_budget = token_budget // 2
        selected_hot: list[ConversationMessage] = []
        used = 0
        for msg in reversed(parent_hot):          # newest first
            if used + msg.tokens > hot_budget:
                break
            selected_hot.insert(0, msg)
            used += msg.tokens

        injected = cold_msgs + selected_hot

        r = await self._redis()
        child_key = self._hot_key(child_run_id, skill_ns)
        # `injected` is chronological (oldest → newest), but hot lists are
        # newest-first (LPUSH convention; _load_hot reverses on read). RPUSH in
        # reversed order so index 0 ends up newest and the child sees the
        # pre-loaded context in correct chronological order end-to-end.
        for msg in reversed(injected):
            await r.rpush(child_key, json.dumps({
                "role": msg.role,
                "content": msg.content,
                "tokens": msg.tokens,
                "skill_ns": skill_ns,
                "step": 0,
                "ts": datetime.now(timezone.utc).isoformat(),
                "_from_parent": parent_run_id,
            }))
        if injected:
            await r.expire(child_key, _HOT_TTL)

        return SubAgentSlice(
            parent_run_id=parent_run_id,
            child_run_id=child_run_id,
            task=task,
            messages=injected,
            token_budget=token_budget,
            page_ids=page_ids,
        )

    async def inject_subagent_result(
        self,
        parent_run_id: str,
        child_run_id: str,
        result_summary: str,
        skill_ns: str = "default",
    ) -> None:
        """
        Inject child's result into parent's hot window as a single
        compressed tool message — keeps parent context clean.
        """
        tokens = estimate_tokens(result_summary)
        r = await self._redis()
        parent_key = self._hot_key(parent_run_id, skill_ns)
        await r.lpush(
            parent_key,
            json.dumps({
                "role": "tool",
                "content": f"[SubAgent:{child_run_id[:8]}] {result_summary}",
                "tokens": tokens,
                "skill_ns": skill_ns,
                "step": -1,
                "ts": datetime.now(timezone.utc).isoformat(),
                "_from_child": child_run_id,
            })
        )
        await r.expire(parent_key, _HOT_TTL)

    # ── Offload internals ─────────────────────────────────────────────────────

    async def _maybe_offload(self, run_id: str, skill_ns: str) -> None:
        """Evict the oldest page of messages when the hot window is over threshold.

        Serialized per hot-key by an asyncio.Lock and trimmed relative to the
        TAIL of the list, so concurrent LPUSHes that occur during the offload
        cannot shift the indices we trim (which would otherwise silently delete
        non-offloaded messages or duplicate pages).
        """
        r = await self._redis()
        hot_key = self._hot_key(run_id, skill_ns)

        # Cheap pre-check outside the lock to avoid contention on the common path.
        if await r.llen(hot_key) < _HOT_MAX_MSGS:
            return

        async with self._offload_locks[hot_key]:
            # Re-read inside the lock: another offload (or pushes) may have run.
            count = await r.llen(hot_key)
            if count < _HOT_MAX_MSGS:
                return

            # Full token scan (only runs when near threshold)
            all_raw: list[str] = await r.lrange(hot_key, 0, -1)
            hot_tokens = _total_tokens(all_raw)

            if hot_tokens < self._max_hot * self._threshold:
                return

            # Identify oldest messages to offload (tail of the list).
            # LPUSH → index 0 = newest, index -1 = oldest.
            to_offload_raw: list[str] = []
            offload_tokens = 0
            # Walk from the end (oldest) inward.
            for raw in reversed(all_raw):
                d = json.loads(raw)
                t = d.get("tokens", estimate_tokens(d.get("content", "")))
                to_offload_raw.append(raw)
                offload_tokens += t
                if offload_tokens >= _PAGE_TOKEN_TARGET:
                    break

            if not to_offload_raw:
                return

            to_offload_raw.reverse()   # chronological order
            await self._offload_page(
                r, hot_key, run_id, skill_ns, to_offload_raw, offload_tokens
            )

    async def _offload_page(
        self,
        r: aioredis.Redis,
        hot_key: str,
        run_id: str,
        skill_ns: str,
        to_offload_raw: list[str],
        offload_tokens: int,
    ) -> None:
        """Compress + embed + store a page, then trim the offloaded tail entries.

        Must be called while holding the per-key offload lock.
        """
        msgs = _raw_to_msgs(to_offload_raw)

        # Compress
        summary = await self._compress(msgs)

        # Embed
        embedding: list[float] | None = None
        if self._embedder is not None:
            try:
                embedding = (await self._embedder.embed([summary]))[0]
            except Exception as exc:
                logger.debug("Page embedding failed: %s", exc)

        # Build page
        steps = [json.loads(raw).get("step", 0) for raw in to_offload_raw]
        page = ContextPage(
            page_id=uuid.uuid4().hex,
            run_id=run_id,
            skill_ns=skill_ns,
            step_start=min(steps),
            step_end=max(steps),
            token_count=offload_tokens,
            importance=_importance_score(to_offload_raw),
            created_at=datetime.now(timezone.utc),
            summary=summary,
            raw_count=len(to_offload_raw),
        )

        await self._store_page(r, page)

        # Upsert into vector store for semantic retrieval
        if embedding is not None and self._vector_store is not None:
            try:
                await self._vector_store.upsert(
                    id=f"page:{page.page_id}",
                    text=summary,
                    metadata={
                        "type": "context_page",
                        "run_id": run_id,
                        "skill_ns": skill_ns,
                        "page_id": page.page_id,
                        "step_start": page.step_start,
                        "step_end": page.step_end,
                        "importance": page.importance,
                    },
                    embedding=embedding,
                )
            except Exception as exc:
                logger.debug("Vector page upsert failed: %s", exc)

        # Trim hot list relative to the TAIL: the offloaded entries are the
        # oldest `n` items (tail). Trimming with a negative stop index keeps the
        # newest `len - n` items even if other coroutines LPUSHed new (newest)
        # messages while this offload was in flight — those new items stay at
        # the head and are never dropped.
        n = len(to_offload_raw)
        # LTRIM key 0 -(n+1) keeps indices [0 .. -(n+1)], i.e. drops the last n.
        await r.ltrim(hot_key, 0, -(n + 1))
        await r.expire(hot_key, _HOT_TTL)

        logger.debug(
            "Offloaded page %s: %d msgs / %d tokens (steps %d–%d) [%s/%s]",
            page.page_id[:8], page.raw_count, offload_tokens,
            page.step_start, page.step_end, run_id[:8], skill_ns,
        )

    async def _retrieve_cold_pages(
        self,
        run_id: str,
        query: str,
        skill_ns: str,
        token_budget: int,
    ) -> tuple[list[ConversationMessage], list[str]]:
        """Semantic search across offloaded pages for this run."""
        if self._vector_store is None:
            return [], []
        try:
            hits = await self._vector_store.query(
                text=query,
                k=self._cold_k,
                filter={"run_id": run_id, "type": "context_page"},
            )
        except Exception as exc:
            logger.debug("Cold page query failed: %s", exc)
            return [], []

        msgs: list[ConversationMessage] = []
        page_ids: list[str] = []
        used = 0

        for hit in hits:
            ns = hit.metadata.get("skill_ns", "default")
            if ns not in (skill_ns, "default"):
                continue
            tokens = estimate_tokens(hit.text)
            if used + tokens > token_budget:
                continue
            s = hit.metadata.get("step_start", 0)
            e = hit.metadata.get("step_end", 0)
            msgs.append(ConversationMessage(
                role="system",
                content=f"[Context page, steps {s}–{e}]\n{hit.text}",
                tokens=tokens,
            ))
            page_ids.append(hit.metadata.get("page_id", hit.id))
            used += tokens

        return msgs, page_ids

    async def _compress(self, messages: list[ConversationMessage]) -> str:
        """LLM compression with extractive fallback."""
        if self._summarizer is not None:
            try:
                formatted = "\n".join(
                    f"{m.role.upper()}: {m.content[:600]}" for m in messages
                )
                resp = await self._summarizer.complete(
                    messages=[{
                        "role": "user",
                        "content": (
                            "Compress the following conversation into a dense 3-5 sentence "
                            "summary. Preserve all tool results, decisions, errors, and "
                            "key facts. Omit pleasantries and filler.\n\n" + formatted
                        ),
                    }],
                    max_tokens=300,
                )
                return resp.content.strip()
            except Exception as exc:
                logger.debug("LLM compressor failed, using extractive: %s", exc)

        # Extractive fallback: prefer tool results and errors; sample every 3rd msg
        kept: list[str] = []
        for i, m in enumerate(messages):
            is_important = (
                m.role == "tool"
                or "error" in m.content.lower()
                or "failed" in m.content.lower()
                or "result" in m.content.lower()
            )
            if is_important or i % 3 == 0:
                kept.append(f"{m.role}: {m.content[:250]}")

        return "Summary: " + " | ".join(kept) if kept else "(context omitted)"

    async def _store_page(self, r: aioredis.Redis, page: ContextPage) -> None:
        page_key = f"{_PAGE_PFX}{page.page_id}"
        pages_set = f"{_PAGES_SET_PFX}{page.run_id}:{page.skill_ns}"

        await r.hset(page_key, mapping={
            "page_id":    page.page_id,
            "run_id":     page.run_id,
            "skill_ns":   page.skill_ns,
            "step_start": page.step_start,
            "step_end":   page.step_end,
            "token_count": page.token_count,
            "importance": page.importance,
            "summary":    page.summary,
            "raw_count":  page.raw_count,
            "created_at": page.created_at.isoformat(),
        })
        await r.expire(page_key, _PAGE_TTL)
        await r.zadd(pages_set, {page.page_id: page.importance})
        await r.expire(pages_set, _PAGE_TTL)

    async def _load_hot(
        self,
        run_id: str,
        skill_ns: str,
        max_msgs: int = 100,
    ) -> list[ConversationMessage]:
        """Load hot messages in chronological order (oldest first)."""
        r = await self._redis()
        raw_items: list[str] = await r.lrange(
            self._hot_key(run_id, skill_ns), 0, max_msgs - 1
        )
        msgs: list[ConversationMessage] = []
        for raw in reversed(raw_items):   # LPUSH → reversed = chronological
            try:
                d = json.loads(raw)
                msgs.append(ConversationMessage(
                    role=d["role"],
                    content=d["content"],
                    tokens=d.get("tokens", 0),
                ))
            except Exception:
                pass
        return msgs

    # ── Key helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _hot_key(run_id: str, skill_ns: str) -> str:
        return f"{_HOT_PFX}{run_id}:{skill_ns}"

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        redis_url: str,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        summarizer: Any | None = None,
        max_hot_tokens: int = 80_000,
        reserve_output: int = 2_000,
        offload_threshold: float = 0.80,
        cold_pages_per_query: int = 3,
    ) -> "ContextEngine":
        """Build a ContextEngine from plain config values."""
        return cls(
            max_hot_tokens=max_hot_tokens,
            redis_url=redis_url,
            vector_store=vector_store,
            embedder=embedder,
            summarizer=summarizer,
            reserve_output=reserve_output,
            offload_threshold=offload_threshold,
            cold_pages_per_query=cold_pages_per_query,
        )


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _sliding_fit(
    messages: list[ConversationMessage],
    budget: int,
) -> tuple[list[ConversationMessage], int, bool]:
    """Keep the most recent messages that fit within budget."""
    kept: list[ConversationMessage] = []
    used = 0
    for msg in reversed(messages):
        t = msg.tokens if msg.tokens > 0 else estimate_tokens(msg.content)
        if used + t <= budget:
            kept.insert(0, msg)
            used += t
        else:
            break
    return kept, used, len(kept) < len(messages)


def _merge_by_step(
    a: list[ConversationMessage],
    b: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Concatenate two already-chronological message lists."""
    return a + b


def _total_tokens(raw_items: list[str]) -> int:
    total = 0
    for raw in raw_items:
        try:
            d = json.loads(raw)
            total += d.get("tokens", estimate_tokens(d.get("content", "")))
        except Exception:
            pass
    return total


def _raw_to_msgs(raw_items: list[str]) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for raw in raw_items:
        try:
            d = json.loads(raw)
            msgs.append(ConversationMessage(
                role=d["role"],
                content=d["content"],
                tokens=d.get("tokens", 0),
            ))
        except Exception:
            pass
    return msgs


def _importance_score(raw_items: list[str]) -> float:
    """Estimate how important a group of messages is for retention."""
    score = 0.5
    for raw in raw_items:
        try:
            d = json.loads(raw)
            content = d.get("content", "").lower()
            if d.get("role") == "tool":
                score += 0.10
            if any(kw in content for kw in ("error", "failed", "exception")):
                score += 0.15
            if any(kw in content for kw in ("result", "found", "success", "completed")):
                score += 0.08
        except Exception:
            pass
    return min(1.0, score)


def _score_goal_progress(llm_content: str, goal: str) -> float:
    """Keyword overlap between LLM response and the task goal."""
    if not goal:
        return 0.5
    goal_words = {w for w in goal.lower().split() if len(w) > 3}
    if not goal_words:
        return 0.5
    content_words = set(llm_content.lower().split())
    overlap = len(goal_words & content_words) / len(goal_words)
    return min(1.0, overlap * 2.0)


def _score_tool_relevance(
    tool_name: str | None,
    tool_result: str | None,
    is_error: bool,
) -> float:
    if tool_name is None:
        return 0.5
    if is_error:
        return 0.1
    if tool_result and len(tool_result) > 20:
        return 0.9
    return 0.6


def _score_confidence(llm_content: str) -> float:
    """Detect hedging / affirming language to estimate expressed confidence."""
    low = llm_content.lower()
    hedges = ["i'm not sure", "i think", "possibly", "maybe", "might be",
              "uncertain", "i believe", "could be", "perhaps", "not certain"]
    affirms = ["the answer is", "found", "completed", "result:", "done",
               "here is", "confirmed", "successfully", "the output is"]
    score = 0.6
    score -= sum(0.08 for h in hedges if h in low)
    score += sum(0.08 for a in affirms if a in low)
    return max(0.1, min(1.0, score))


def _group_by_skill(records: list[ActionRecord]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        g = groups.setdefault(r.skill_ns, {"count": 0, "total_score": 0.0, "errors": 0})
        g["count"] += 1
        g["total_score"] += r.composite_score
        if r.is_error:
            g["errors"] += 1
    for g in groups.values():
        if g["count"]:
            g["avg_score"] = round(g.pop("total_score") / g["count"], 3)
    return groups
