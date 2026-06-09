# Changelog

All notable changes to `agent-haas` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
semantic versioning (pre-1.0: minor = features/behavior changes, patch = fixes).

## [0.4.0] — 2026-06-10

A large correctness, security, and enforcement release. ~70 reviewed bugs fixed
across every module; three advertised guarantees that were previously dead code
now actually enforce. Test suite grew from 1084 to 1363 passing.

### Added
- **Cost budget enforcement** — the agent loop now calls `CostTracker.check_budget`
  before each LLM call; an over-budget tenant stops with the new
  `FailureClass.BUDGET_COST`. Toggle via `enforce_cost_budget` /
  `ENFORCE_COST_BUDGET`. Infra errors fail open.
- **API rate limiting** — `RateLimitMiddleware` is registered and enforces
  `rate_limit_rpm` per tenant; fails open on Redis outage, exempts health/docs
  paths. Toggle via `rate_limit_enabled` / `RATE_LIMIT_ENABLED`.
- **Operator cancellation** — `DELETE /runs/{id}` now interrupts a *running*
  agent (not just pending runs); the loop raises `RunCancelled`
  (`FailureClass.CANCELLED`) when the persisted status flips.
- New public surface: `FailureClass.BUDGET_COST`, `FailureClass.CANCELLED`,
  `RunCancelled` exception, config flags `enforce_cost_budget` /
  `rate_limit_enabled`.

### Fixed — critical
- Multi-turn tool use was broken end-to-end: tool results and assistant
  tool-call turns are now translated to each provider's wire format
  (Anthropic `tool_use`/`tool_result`, OpenAI `tool_calls`/`tool_call_id`,
  local/Bedrock folded into user turns). Real-time feedback is injected as a
  user turn instead of a dropped/ rejected system turn.
- `_llm_span`/`_mlflow_context` no longer swallow body exceptions — LLM and
  budget failures keep their failure class instead of becoming a generator
  `RuntimeError`.
- Worker agent factory `NameError` that broke every in-process run; the worker
  now wires the live Redis client (event streaming + cost tracking work).
- Scheduler no longer persists the `AgentBlackboard` object into run metadata
  (it crashed `json.dumps` and failed every multi-agent subtask).
- Inter-agent messaging request/reply race (lost replies) and infinite hang on
  zero replies; `SentenceTransformerEmbedder` crash on every uncached call.
- OpenAI reasoning models (o-series / gpt-5) now receive `tools`.

### Fixed — security
- Cross-tenant access on feedback, HITL approve/reject, span, and step-stream
  endpoints now enforce run/tenant ownership.
- Refuse to start (and reject JWT auth) in prod with the default JWT secret.
- SQL tools: `schema` argument validated; read-only gate scans for write
  keywords anywhere in the statement; Postgres `SET TRANSACTION READ ONLY`.
- LLM-generated code no longer silently falls back to unsandboxed host
  execution (opt-in via `HARNESS_ALLOW_UNSANDBOXED`).
- `ApplyPatchTool` path-traversal check; MCP servers get an env allowlist and
  their transports are closed (subprocess no longer leaks).
- Tool safety pipeline fails closed on guard errors; skill store enforces
  tenant ownership.

### Fixed — correctness (eval / self-improvement)
- RLVR advantage split no longer cancels out the rolling baseline.
- `score_execution_match` is symmetric (supersets of gold no longer score 1.0).
- LLM-judge unavailability scores 0.0 instead of auto-passing at the 0.5
  threshold; exact-match is numeric/word-boundary aware.
- Online monitor compares windowed error rates, not cumulative counts; GEPA
  uses a real train/val split; evaluator restores the exact prior version;
  prompt versions use an atomic counter (no reuse after delete).

### Fixed — correctness (memory / llm / api)
- ContextEngine offload is tail-relative and per-key locked (no lost messages);
  Weaviate filter/score fixes; qdrant lazy-embedder; Cypher injection guard;
  crash-safe checkpoint writes; Docker container kill on timeout; E2B session
  TTL.
- `router.stream` only falls back before the first token; semantic cache
  restricted to single-turn; `sk-proj`/`sk-svcacct` key redaction; tenant
  safety policies no longer expire after 30 days.
- Run enqueue moved off the event loop with proper failure signaling;
  `cost_tracker`/`rate_limiter` key + rollback fixes; NexusSQL verifier gets its
  sandbox; eval routes require a tenant; shared TraceRecorder; single global
  tracer provider.

### Docs
- Documented budgets, rate limiting, and cancellation on the site; added the
  `DELETE /runs/{id}` endpoint to the REST API reference.

## [0.3.2] and earlier

See the git history. 0.3.x covered the sandbox provider work (E2B, Modal),
messaging/tools test coverage, and the initial public release of the harness.

[0.4.0]: https://github.com/thepradip/HarnessAgent/releases/tag/v0.4.0
