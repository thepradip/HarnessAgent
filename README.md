# Agent Harness as a Service

Production-grade agent harness for building, running, observing, and self-improving AI agents. Bring your own framework or use the native SQL and Code agents. Memory, safety, tracing, and failure recovery come included.

[![PyPI](https://img.shields.io/pypi/v/agent-haas?color=blue&label=PyPI)](https://pypi.org/project/agent-haas/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-1087%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Models](https://img.shields.io/badge/LLMs-Claude%20%7C%20GPT--5%20%7C%20vLLM%20%7C%20llama.cpp-purple)](https://thepradip.github.io/HarnessAgent/)

---

```bash
pip install agent-haas                                    # core ~50 MB — no torch, no SDKs
pip install agent-haas[anthropic,api,vector,workers,sql]  # recommended ~300 MB
pip install agent-haas[recommended]                       # same as above, one flag
pip install agent-haas[all]                               # everything except torch ~500 MB
pip install agent-haas[all,embed-full]                    # + sentence-transformers + torch ~2 GB
```

---

![HarnessAgent Architecture](assets/architecture.png)

---

## What is this?

Think about what actually happens when you run an AI agent in production. The LLM call needs to work. It needs to not cost $500 a day. It needs to not loop forever when the API is slow. It needs to remember context from three messages ago — and intelligently *discard* context that no longer matters. It needs to not crash your app when one provider goes down. And when it does fail, it needs to tell you *exactly why*.

HarnessAgent handles all of that.

| What you see | What happens under the hood |
|---|---|
| AI answers your question | Picks the healthiest LLM, checks the budget, falls back if the provider fails |
| AI runs a SQL query | Validates input schema, checks safety rules, executes in a TOOL span, logs the result |
| AI remembers past context | Hot window in Redis → paged cold storage → vector DB; only relevant pages re-injected |
| AI finds relevant info fast | GraphRAG: entity extraction + weighted BFS traversal, 83% fewer tokens than naive vector search |
| AI gets better after failures | Hermes loop: samples errors → LLM patch → eval replay → auto-apply + rollback if regression |
| One provider goes down | Circuit breaker opens after 5 failures, auto-recovers after 60 seconds |
| Run fails | Full span tree (RUN → LLM → TOOL → GUARDRAIL) queryable via `GET /runs/{id}/trace` |
| Long agent session | Older messages auto-compressed + offloaded to vector store; retrieved semantically per query |
| Agent run crashes mid-step | Checkpoint saves state + full history every 10 steps and on any exit; resume from exact step |
| Code agent calls run_python 10x | One container started at run start; all calls use `docker exec` — zero cold start overhead |
| API key leaks into LLM output | SecretScanner detects and redacts before key enters history, checkpoints, or traces |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client / SDK                                │
│              REST API  ·  SSE streaming  ·  Python SDK              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                       Orchestration                                 │
│          AgentRunner  ·  Planner (DAG)  ·  Scheduler  ·  HITL       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        Agent Layer                                  │
│   BaseAgent (run loop)  ·  CodeAgent  ·  SQLAgent                   │
│   LangGraph  ·  AutoGen  ·  CrewAI  ·  OpenClaw adapters            │
└────┬──────────────┬──────────────┬────────────────┬─────────────────┘
     │              │              │                │
┌────▼────┐  ┌──────▼──────┐  ┌───▼────────┐  ┌───▼──────────────────┐
│  Memory │  │  LLM Router │  │   Tools    │  │  Safety + Security   │
│         │  │             │  │            │  │                      │
│Context  │  │Claude · GPT │  │ToolRegistry│  │Guardrail pipeline    │
│Engine   │  │vLLM · SGLang│  │SQL · Code  │  │Policy enforcement    │
│GraphRAG │  │llama.cpp    │  │File · MCP  │  │HITL approval         │
│VectorDB │  │Circuit breaker  │SkillStore  │  │SecretProvider (Vault)│
│LLM Cache│  │             │  │Session     │  │SecretScanner         │
└────┬────┘  └─────────────┘  │Sandbox     │  └──────────────────────┘
     │                        └────────────┘
┌────▼────────────────────────────────────────────────────────────────┐
│                       Observability                                 │
│  TraceRecorder  ·  MLflow  ·  OpenTelemetry  ·  Prometheus  ·  SSE  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ failures
┌──────────────────────────────▼──────────────────────────────────────┐
│                     Self-Improvement                                │
│     Hermes Loop  ·  PatchGenerator  ·  Evaluator  ·  SkillCapture   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **LLM Routing** | Claude, GPT-5, o4-mini, vLLM, SGLang, llama.cpp with health-aware fallback and circuit breaking |
| **Paged Context Engine** | Auto-offload cold messages to vector store; per-skill namespace isolation; action scoring per step |
| **GraphRAG** | 83% token reduction via weighted multi-hop graph traversal vs naive vector search |
| **Semantic LLM Cache** | Cosine-similarity response cache (threshold 0.97) backed by Redis + embeddings |
| **Hierarchical Span Tracing** | Every run produces a `RUN→LLM→TOOL→GUARDRAIL` span tree stored in Redis + JSONL; queryable via API |
| **Framework Adapters** | LangGraph, AutoGen, CrewAI — plug in without rewriting agent logic |
| **Safety Pipeline** | PII redaction, injection detection, tool policy, loop detection, budget enforcement |
| **Policy Enforcement** | Per-tenant: blocked tools, `allow_code_execution`, `allow_file_write` enforced before every tool call |
| **Hermes Self-Improvement** | Samples failures → LLM generates prompt patch → eval replay → auto-apply + regression rollback |
| **Human-in-the-Loop** | Agent pauses on risky tool calls, waits for Redis-polled approval, then continues or stops |
| **Session Sandbox** | One Docker container per run; `docker exec` for each call — eliminates 2–5s cold-start overhead |
| **Sandbox Workload Profiles** | `general` (256 MiB) / `data` (512 MiB) / `ml` (2 GiB) — set via `SANDBOX_WORKLOAD` |
| **gVisor / Kata Support** | Kernel-level isolation via `SANDBOX_RUNTIME=runsc` (gVisor) or `kata`; falls back to runc |
| **OOM Detection** | Exit code 137 surfaces as `"OOM: container exceeded memory limit"` not a silent crash |
| **Secret Provider** | `EnvSecretProvider` (zero migration) → `VaultSecretProvider` (HashiCorp) → `AWSSecretsProvider`; TTL cache; per-tenant path isolation |
| **Secret Scanner** | Detects Anthropic / OpenAI / GitHub / Slack / JWT / bearer tokens in LLM outputs; redacts before hitting history or checkpoints |
| **Skill Store** | Vector-indexed library of reusable code, approaches, and patterns; auto-injected into context; health report with red flags |
| **Skill Capture** | Auto-saves high-scoring (≥0.8) novel patterns from successful runs; novelty gate via cosine similarity |
| **Reliable Checkpoints** | Full conversation history saved every 10 steps and on every exit (crash, cancel, budget, success); resumes from exact message |
| **Multi-Agent DAG** | Planner decomposes tasks → DAG; Scheduler executes in parallel with back-pressure + handoff enrichment |
| **MCP Support** | Connect any MCP server over stdio or SSE; YAML config; environment variable interpolation |
| **Tool Result Cap** | Tool outputs capped at 8 k chars before entering agent history; `metadata["truncated"]` flag + `original_chars` for observability |
| **Eval Framework** | Dataset-driven evaluation with per-case diagnostics, failure stage classification, and optimization hints |
| **Audit Trail** | Append-only compliance log (PII-hashed payloads) with Redis stream + JSONL dual persistence |

---

## What's new

### Tool result size cap

Large tool outputs (full-table SQL dumps, verbose file reads) are automatically capped at **8,000 chars** before they enter agent history, preventing a single noisy tool call from consuming the entire context window.

```python
result = await registry.execute(ctx, call)

# Small result — passes through unchanged
result.data          # → {"rows": [...]}
result.metadata      # → {}

# Large result — automatically truncated
result.data          # → "col1 | col2\n...\n…[truncated — original output was 42,310 chars]"
result.metadata      # → {"truncated": True, "original_chars": 42310}
```

The truncation suffix shows the original byte count so you can decide whether to re-run with a narrower query or `LIMIT` clause. The `FailureCategory.OUTPUT_TRUNCATED` failure category + `top_hint()` will surface this automatically during eval.

---

### Secret vault & scanner

API keys never appear in agent context, traces, or checkpoints. The `SecretProvider` abstraction keeps credentials out of `AgentContext`, and `SecretScanner` redacts any that accidentally leak into LLM responses before they enter history.

```python
# Dev — zero migration, reads from os.environ / .env as before
from harness.security import get_secret
key = await get_secret("anthropic_api_key")

# Production — swap backend without changing any callsite
from harness.security import configure, VaultSecretProvider, CachedSecretProvider
configure(CachedSecretProvider(
    VaultSecretProvider(url="https://vault:8200", token="s.xxx"),
    ttl_seconds=300,
))

# Per-tenant isolation — acme/anthropic_api_key first, falls back to global
from harness.security import TenantSecretProvider, EnvSecretProvider
provider = TenantSecretProvider(EnvSecretProvider(), tenant_id=ctx.tenant_id)
```

The `SecretScanner` detects: Anthropic `sk-ant-`, OpenAI `sk-`, GitHub `ghp_` / `github_pat_`, Slack `xoxb-`/`xoxp-`, GitLab `glpat-`, JWTs, bearer tokens, and URL-embedded credentials. Detected keys are redacted before they touch history, memory, or the trace store.

---

### Session sandbox — zero cold-start

Every `run_python` call previously started a new Docker container (~2–5s each). For a CodeAgent doing 10 debug iterations, that was 20–50 seconds of pure overhead.

```python
# Enable session reuse — one container for the entire run
ctx.metadata["sandbox_session"] = True
# or: SANDBOX_SESSION_REUSE=true in .env

# The container starts at run begin, docker exec for each call, stops on run exit
# State persists: variables, pip-installed packages, written files — all available in the next call
```

Container death (OOM kill, crash) is detected and surfaces as `SandboxError("Session container died")` so the agent can report clearly rather than looping confused.

---

### Sandbox workload profiles + gVisor

```bash
# Memory limits per workload — set in .env or pass to DockerSandbox(memory_limit=...)
SANDBOX_WORKLOAD=general   # 256 MiB — scripting, algorithms (default)
SANDBOX_WORKLOAD=data      # 512 MiB — pandas / numpy with real datasets
SANDBOX_WORKLOAD=ml        # 2 GiB   — torch / sklearn model runs

# Kernel-level isolation (requires gVisor on host)
SANDBOX_RUNTIME=runsc      # gVisor — intercepts all syscalls before host kernel
SANDBOX_RUNTIME=kata       # Kata Containers — full lightweight VM per sandbox
```

OOM kills now surface a clear error (`OOM: container exceeded memory limit`) instead of an opaque exit code 137, in all three execution paths (session, per-call Docker, subprocess fallback).

---

### Skill store — reusable code and approach library

Agents retrieve relevant skills (code snippets, architectural approaches, monitoring patterns) from a vector-indexed library instead of regenerating common work from scratch. Token savings are significant for patterns used repeatedly.

```python
from harness.tools.skill_store import SkillStore, SkillCapture, SkillType

store = SkillStore(redis=redis_client, memory_manager=memory)

# Save a skill manually
await store.save(SkillArtifact(
    skill_id="batch-insert-001",
    tenant_id="acme",
    skill_type=SkillType.CODE,
    title="Batch insert helper",
    description="Inserts rows in 1000-row batches to avoid memory pressure",
    content="def batch_insert(rows, conn):\n    for i in range(0, len(rows), 1000):\n        conn.executemany(sql, rows[i:i+1000])",
    language="python",
    requirements={"psycopg2": ">=2.9"},
    score=0.9,
))

# Auto-capture from a successful run (score gate + novelty gate)
capture = SkillCapture(store, min_score=0.8)
await capture.capture(
    title="...", description="...", content="...",
    skill_type=SkillType.CODE, tenant_id="acme", score=0.92,
    run_id=ctx.run_id,
)

# Wire into an agent run — skills are auto-retrieved and injected into context
ctx.metadata["skill_store"] = store
```

**Dependency metadata** prevents silent breakage. Skills declare `requirements` (`{"pandas": ">=2.0"}`). On validation, `update_validation()` checks against the live environment and marks the skill `BROKEN` if a requirement is no longer satisfied.

**Dashboard red flags** — `health_report()` returns counts and a list of `RedFlag` entries:

| Flag | Severity | Trigger |
|---|---|---|
| `BROKEN` | high | Validation failed or requirement mismatch |
| `STALE` | medium | Not validated in >30 days |
| `LOW_QUALITY_HIGH_USE` | medium | `use_count > 5` and `score < 0.3` |
| `REQUIREMENT_MISMATCH` | high | Known incompatible requirement in metadata |

---

### Reliable checkpoints

Checkpoints previously saved a stub dict (missing history) via a broken method call. Now:

- **Full history saved** — `CheckpointManager.save(ctx, history)` serializes the complete conversation alongside step/token counts
- **Correct resume** — `load(run_id, tenant_id)` restores both counters and history; the loop resumes from the exact message where it stopped
- **Always-on** — `_save_checkpoint` is called in `finally` on every exit path: clean completion, budget exceeded, exception, and `CancelledError`

---

### Policy enforcement

Per-tenant policies now enforce `blocked_tools`, `allow_code_execution`, and `allow_file_write` at tool dispatch time, not just in configuration. All three checks run before HITL approval, raising `SafetyViolation(SAFETY_STEP)` immediately.

```python
from harness.safety.policies import HarnessPolicy

ctx.metadata["policy"] = HarnessPolicy(
    tenant_id="acme",
    blocked_tools=["drop_table", "delete_database"],
    allow_code_execution=False,   # blocks run_python, exec_*, run_* tools
    allow_file_write=False,        # blocks write_file, apply_patch, write_* tools
)
```

---

### Context Engine (paged context management)

Long-running agents no longer overflow or drop context blindly. The `ContextEngine` manages the hot window per skill namespace and evicts cold pages automatically:

- **Offload** — oldest `~2 000 tokens` are LLM-compressed and evicted to the vector store when the hot window exceeds 80% capacity
- **Select** — before each LLM call, relevant cold pages are retrieved by semantic search against the current query
- **Isolate** — each skill (`sql`, `code`, `search`) has its own Redis key; shared context is merged on demand
- **Evaluate** — every LLM + tool round-trip is scored (`goal_progress`, `tool_relevance`, `confidence`) and stored for Hermes sampling
- **Sub-agents** — parent can slice its context for a child agent within a token budget; child result is injected back as a single compressed message

---

### Hierarchical span tracing

Every agent run produces a queryable span tree persisted to Redis (48 h) and `logs/runs/{run_id}/trace.jsonl`:

```
run:sql_agent                        1 234 ms
  llm:call                   450 ms  1 200 tok  $0.002
  guardrail:output             12 ms  passed
  tool:execute_sql            180 ms  42 rows
  llm:call                   310 ms    800 tok  $0.001
```

Query it:
```bash
curl http://localhost:8000/runs/{run_id}/trace
```

---

## LLM Support

| Provider | Models | Tool Calling | Prompt Caching | Cost per 1M input tokens |
|---|---|---|---|---|
| Anthropic | Sonnet 4.6, Haiku 4.5, Opus 4.7 | Native | Yes | $0.25 – $15 |
| OpenAI | GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini, o1, o3, o4-mini | Native | Auto | $0.15 – $75 |
| vLLM | Any HuggingFace model | Native | No | Free (self-hosted) |
| SGLang | Any HuggingFace model | Native | No | Free (self-hosted) |
| llama.cpp | Any GGUF quantized model | ReAct text injection | No | Free (CPU / Metal) |
| Ollama | Any Ollama model | Native | No | Free (local) |

No GPU? llama.cpp runs on any Mac or CPU machine.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/thepradip/HarnessAgent.git
cd HarnessAgent
poetry install

# 2. Configure (set at least one API key, or a local model URL)
cp .env.example .env

# 3. Start infrastructure (Redis, Qdrant, Neo4j, MLflow, Prometheus, Grafana)
docker compose up -d

# 4. Start API and worker
make api      # terminal 1 — FastAPI on port 8000
make worker   # terminal 2 — async agent worker

# 5. Run your first agent
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "sql", "task": "How many users signed up this week?"}'

# Stream steps in real time
curl http://localhost:8000/runs/{run_id}/stream

# Inspect the full span trace
curl http://localhost:8000/runs/{run_id}/trace
```

No API key? Use llama.cpp locally:

```bash
# Put a GGUF model in ./models/ then:
docker compose --profile local-cpu up -d llamacpp
# Add to .env: LLAMACPP_BASE_URL=http://localhost:8080
```

### Minimal dev setup (no Docker)

```bash
brew install redis && brew services start redis
pip install agent-haas[anthropic,api,vector,observe,mcp]
uvicorn harness.api.main:create_app --factory --port 8000
# Open http://localhost:8000/ for the dashboard
```

---

## Python SDK

### Single agent

```python
from harness.core.context import AgentContext
from harness.agents.sql_agent import SQLAgent
from harness.observability.trace_recorder import TraceRecorder
from pathlib import Path

recorder = TraceRecorder.create(redis_url="redis://localhost:6379")

agent = SQLAgent(
    llm_router=llm_router,
    memory_manager=memory,
    tool_registry=registry,
    safety_pipeline=None,
    step_tracer=None,
    mlflow_tracer=mlflow_tracer,
    failure_tracker=failure_tracker,
    audit_logger=audit_logger,
    event_bus=event_bus,
    cost_tracker=cost_tracker,
    checkpoint_manager=checkpoint_manager,
    trace_recorder=recorder,
)

ctx = AgentContext.create(
    tenant_id="acme",
    agent_type="sql",
    task="List all tables and their row counts",
    memory=memory,
    workspace_path=Path("/workspaces/acme/run1"),
)

result = await agent.run(ctx)
print(result.output, result.cost_usd, result.steps)
```

### Session sandbox

```python
# Enable in context — one container per run, docker exec for each tool call
ctx.metadata["sandbox_session"] = True

# Or globally in .env:
# SANDBOX_SESSION_REUSE=true
# SANDBOX_WORKLOAD=data       # 512 MiB
# SANDBOX_RUNTIME=runsc       # gVisor kernel isolation
```

### Secret provider

```python
from harness.security import configure, VaultSecretProvider, CachedSecretProvider

# Configure once at startup — all get_secret() calls use it
configure(CachedSecretProvider(
    VaultSecretProvider(url="https://vault:8200", token=os.environ["VAULT_TOKEN"]),
    ttl_seconds=300,   # refresh every 5 minutes
))

# Per-tenant: tries acme/anthropic_api_key first, falls back to ANTHROPIC_API_KEY
from harness.security import TenantSecretProvider, EnvSecretProvider
ctx.metadata["secret_provider"] = TenantSecretProvider(
    EnvSecretProvider(), tenant_id=ctx.tenant_id
)
```

### Skill store

```python
from harness.tools.skill_store import SkillStore, SkillCapture, SkillType, ValidationStatus

store = SkillStore(redis=redis_client, memory_manager=memory)

# Wire into agent — skills auto-retrieved and injected into system prompt
ctx.metadata["skill_store"] = store

# Dashboard health check
report = await store.health_report(tenant_id="acme")
print(f"{report.total_skills} skills, {report.broken} broken")
for flag in report.red_flags:
    print(f"  [{flag.severity}] {flag.title}: {flag.detail}")

# Validate a skill against the current environment
await store.update_validation(
    skill_id="batch-insert-001",
    status=ValidationStatus.VALID,
    env_requirements={"psycopg2": "2.9.6", "python": "3.11.4"},
)
```

### Wrap an existing framework

```python
import harness

adapter = harness.wrap(my_langgraph_graph)
adapter.attach_harness(
    safety_pipeline=pipeline,
    cost_tracker=cost_tracker,
    audit_logger=audit_logger,
)

async for event in adapter.run_with_harness(ctx, {"input": "analyze sales data"}):
    print(event.event_type, event.payload)
```

### Multi-agent DAG

```python
from harness.orchestrator.planner import Planner
from harness.orchestrator.scheduler import Scheduler

planner = Planner(llm_provider=llm)
plan = await planner.plan(
    task="Fetch sales data, analyze trends, write a report",
    available_agents=["sql", "code"],
)

scheduler = Scheduler(agent_runner=runner)
results = await scheduler.execute_plan(plan, tenant_id="acme")
```

### Context Engine (paged context)

```python
from harness.memory.context_engine import ContextEngine

engine = ContextEngine.create(
    redis_url="redis://localhost:6379",
    vector_store=vector_store,
    embedder=embedder,
    summarizer=llm,
    max_hot_tokens=80_000,
    offload_threshold=0.80,
)

await engine.push(run_id, "user", "list all users", skill_ns="sql", step=1)
ctx_window = await engine.build_context(run_id, query="list users", skill_ns="sql")
action = await engine.evaluate_action(
    run_id, step=1, goal="list users",
    llm_content="I'll run SELECT * FROM users",
    tool_name="execute_sql", tool_result="42 rows",
)
```

---

## Use Cases

**SQL Data Agent** — Ask business questions in plain English. The agent reads your schema into a knowledge graph, writes safe SELECT queries, returns formatted results with PII redacted, and shows a full LLM→TOOL span trace.

**Code Assistant** — Give it a ticket or a spec. It reads your workspace, writes code, lints it, runs it in a session-reused Docker sandbox (zero cold-start), and fixes errors until it passes. Common patterns are automatically captured into the skill store.

**Research Agent** — Feed it documents or URLs. It ingests them into the vector store and knowledge graph, then answers multi-hop questions using GraphRAG.

**Multi-Agent Pipeline** — Chain specialists through the planner: researcher feeds coder, coder feeds reviewer. All share the same memory pool and produce a unified trace.

**Long-running Agent** — Sessions that span hundreds of steps use paged context: old turns are compressed and offloaded, only relevant pages are re-injected per query. Checkpoints ensure no progress is lost on crash.

**Existing Framework** — Already using LangGraph, AutoGen, or CrewAI? Drop your graph or crew into the adapter. You get traces, cost tracking, circuit breaking, and safety without rewriting agent logic.

---

## Project Structure

```
HarnessAgent/
├── src/harness/
│   ├── agents/            # BaseAgent loop, SQLAgent, CodeAgent
│   ├── adapters/          # LangGraph, AutoGen, CrewAI wrappers
│   ├── api/               # FastAPI routes, JWT auth, SSE streaming
│   │   └── routes/
│   │       ├── runs.py    # POST /runs, GET /runs/{id}/stream
│   │       └── traces.py  # GET /runs/{id}/trace, /spans/{id}
│   ├── core/              # Config, protocols, error hierarchy, circuit breaker
│   ├── eval/              # Datasets, EvalRunner, EvalReport, diagnostics
│   ├── filesystem/
│   │   ├── sandbox.py          # DockerSandbox, SessionDockerSandbox, RestrictedPython
│   │   ├── checkpoint.py       # CheckpointManager — atomic save/load with full history
│   │   └── workspace.py        # Per-run workspace isolation
│   ├── improvement/       # HermesLoop, ErrorCollector, Evaluator, OnlineMonitor, RLVR
│   ├── ingestion/         # PDF/HTML/MD loaders, chunker, extraction
│   ├── llm/               # Anthropic, OpenAI, local providers, router, SemanticCache
│   ├── memory/
│   │   ├── context_engine.py   # Paged offload + skill isolation + action scoring
│   │   ├── manager.py          # Unified memory interface
│   │   ├── graph_rag.py        # Weighted multi-hop retrieval
│   │   ├── short_term.py       # Redis conversation history
│   │   └── backends/           # Chroma, Qdrant, Weaviate
│   ├── messaging/         # Redis Streams inter-agent bus
│   ├── observability/
│   │   ├── trace_schema.py     # TraceSpan, SpanKind, SpanStatus, TraceView
│   │   ├── trace_recorder.py   # Span lifecycle — Redis + JSONL persistence
│   │   ├── tracer.py           # OpenTelemetry integration
│   │   ├── mlflow_tracer.py    # MLflow experiment tracking
│   │   ├── failures.py         # StepFailure, FailureTracker
│   │   ├── metrics.py          # Prometheus counters / histograms / gauges
│   │   ├── audit.py            # Append-only compliance log
│   │   └── event_bus.py        # Redis Pub/Sub for SSE
│   ├── orchestrator/      # AgentRunner, Planner, Scheduler, HITLManager
│   ├── prompts/           # Versioned prompt store, patch application
│   ├── safety/            # Guardrail pipeline factory and per-tenant policies
│   ├── security/
│   │   ├── secrets.py          # SecretProvider — Env / Vault / AWS / Cached / Tenant
│   │   └── scanner.py          # SecretScanner — detects + redacts leaked API keys
│   ├── tools/
│   │   ├── registry.py         # ToolRegistry — schema validate + safety + timeout
│   │   ├── skills.py           # SkillRegistry — versioned prompt-based capabilities
│   │   ├── skill_store.py      # SkillStore — vector-indexed reuse library with health report
│   │   ├── code_tools.py       # RunCodeTool (session + docker + subprocess)
│   │   ├── file_tools.py       # ReadFileTool, WriteFileTool, ListWorkspaceTool
│   │   ├── sql_tools.py        # ExecuteQueryTool, ListTablesTool, DescribeTableTool
│   │   └── mcp_client.py       # MCPToolAdapter
│   └── workers/           # RQ agent worker, Hermes background scheduler
├── tests/
│   ├── unit/
│   │   ├── test_security.py           # 52 — SecretProvider, SecretScanner, pipeline integration
│   │   ├── test_skill_store.py        # 68 — SkillStore, SkillCapture, health report, red flags
│   │   ├── test_filesystem.py         # 43 — CheckpointManager, WorkspaceManager, sandbox
│   │   ├── test_agent_base.py         # 77 — checkpoint resume, policy, HITL, session sandbox
│   │   ├── test_safety.py             # 52 — HarnessPolicy, PolicyStore, pipeline
│   │   ├── test_eval_components.py    # 82 — CodeSandbox, OOM detection, SQL sandbox
│   │   ├── test_trace_schema.py       # 27 — TraceSpan schema
│   │   ├── test_trace_recorder.py     # 31 — span lifecycle, Redis, JSONL
│   │   ├── test_context_engine.py     # 60 — offload, scoring, sub-agents
│   │   └── ...                        # 990 tests total
│   └── integration/
├── ui/
│   ├── dashboard.html     # Operator dashboard with Trace waterfall tab
│   └── docs.html          # Full technical reference (open in browser)
├── configs/               # Model capabilities, MCP server definitions
├── infra/                 # Prometheus, OTel collector, Grafana
├── docker-compose.yml     # Redis, Qdrant, Neo4j, MLflow, Grafana
├── Dockerfile
├── Makefile
└── pyproject.toml
```

---

## API Reference

### Runs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/runs` | Create and enqueue a run. Body: `{agent_type, task, metadata}` |
| `GET` | `/runs/{run_id}` | Retrieve run record |
| `GET` | `/runs` | List runs for tenant. Query: `limit`, `offset` |
| `DELETE` | `/runs/{run_id}` | Cancel a pending or running run |
| `GET` | `/runs/{run_id}/stream` | SSE stream of StepEvents. Terminates on `completed`/`failed` |

### Traces

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/runs/{run_id}/trace` | Full span hierarchy with aggregated tokens, cost, duration. 48 h TTL |
| `GET` | `/runs/spans/{span_id}` | Single span by ID |

### Trace response shape

```json
{
  "trace_id":             "ddda858ebe8f42b6...",
  "run_id":               "3f2a8c1e4d...",
  "agent_type":           "sql",
  "status":               "ok",
  "duration_ms":          1234,
  "total_input_tokens":   980,
  "total_output_tokens":  270,
  "total_cost_usd":       0.00031,
  "span_count":           6,
  "spans": [
    {
      "span_id":        "01f04e9413851d7f",
      "parent_span_id": null,
      "kind":           "run",
      "name":           "run:sql",
      "status":         "ok",
      "duration_ms":    1234,
      "input_preview":  "List all tables",
      "output_preview": "Found 7 tables..."
    }
  ]
}
```

---

## Observability

### Span kinds

| Kind | Emitted by | Contains |
|---|---|---|
| `run` | BaseAgent.run() | Full run duration, task, output |
| `llm` | _llm_span() | input/output tokens, model, cost, cached flag |
| `tool` | _execute_one() | tool name, args preview, result preview |
| `guardrail` | safety check | blocked or passed |
| `memory` | memory retrieval | query, tokens used |
| `handoff` | inter-agent message | sender, recipient |
| `eval` | EvalRunner | case id, score |

### Prometheus metrics

| Metric | Labels |
|---|---|
| `harness_agent_steps_total` | agent_type, tenant_id, status |
| `harness_tool_calls_total` | tool_name, agent_type, status |
| `harness_safety_blocks_total` | guard, agent_type, stage |
| `harness_active_runs` | agent_type |
| `harness_cost_usd_total` | tenant_id, model |
| `harness_llm_request_duration_seconds` | provider, model |
| `harness_hermes_patches_total` | agent_type, status |

### Dashboards

| Dashboard | URL | Credentials |
|---|---|---|
| Operator console + Trace waterfall | http://localhost:8000 | API key or dev mode |
| Technical docs | https://thepradip.github.io/HarnessAgent/ | — |
| MLflow Traces | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / harness_admin |
| Prometheus | http://localhost:9090 | — |

---

## Configuration

```bash
# LLM providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Local LLMs (no API key required)
VLLM_BASE_URL=http://localhost:8000
LLAMACPP_BASE_URL=http://localhost:8080

# Memory backends
VECTOR_BACKEND=chroma          # chroma | qdrant | weaviate
GRAPH_BACKEND=networkx         # networkx | neo4j
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # fastembed default (~100 MB, no torch)
EMBEDDING_BACKEND=fastembed              # fastembed (default) | sentence-transformers (+1.5 GB)

# Sandbox — code execution isolation
SANDBOX_WORKLOAD=general       # general (256m) | data (512m) | ml (2g)
SANDBOX_RUNTIME=runc           # runc (default) | runsc (gVisor) | kata (Kata Containers)
SANDBOX_SESSION_REUSE=false    # true = one container per run, docker exec per call

# Context engine tuning
CONTEXT_MAX_HOT_TOKENS=80000
CONTEXT_OFFLOAD_THRESHOLD=0.80
CONTEXT_COLD_PAGES=3
CONTEXT_RESERVE_OUTPUT=2000

# Hermes self-improvement
HERMES_AUTO_APPLY=false
HERMES_INTERVAL_SECONDS=3600
HERMES_MIN_ERRORS_TO_TRIGGER=5
HERMES_PATCH_SCORE_THRESHOLD=0.7

# Cost and safety
COST_BUDGET_USD_PER_TENANT=100.0
RATE_LIMIT_RPM=60
ENVIRONMENT=dev                # dev | staging | prod
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| API | FastAPI + uvicorn | Async, SSE for step streaming |
| LLM | anthropic + openai SDKs | Optional extras `[anthropic]` / `[openai]`; lazy-loaded |
| Tracing | TraceRecorder + Redis + JSONL | Hierarchical spans; 48 h live query; durable JSONL |
| OTel export | opentelemetry-sdk | Optional; exports to Jaeger / Tempo / Grafana Tempo |
| Short-term memory | Redis LIST | Conversation history per run |
| Context engine | Redis + VectorStore | Paged offload, skill namespaces, action scoring |
| Long-term memory | Qdrant / ChromaDB / Weaviate | Chroma for dev, Qdrant/Weaviate for prod |
| Knowledge graph | NetworkX / Neo4j | NetworkX in-process for dev, Neo4j for production |
| LLM cache | Redis + FastEmbedEmbedder | Cosine similarity at 0.97 threshold; ONNX, no torch |
| Experiment tracking | MLflow | LLM-native spans, eval metrics, prompt versioning |
| Metrics | Prometheus + Grafana | 15 pre-defined metrics |
| Safety | Guardrail + policy enforcement | 3-stage pipeline + per-tenant tool/exec/write gates |
| Secret management | `harness.security` | Env / Vault / AWS; TTL cache; per-tenant isolation; scanner |
| Sandbox | SessionDockerSandbox + RestrictedPython | Session reuse; workload profiles; gVisor/Kata opt-in |
| Skill store | Redis + VectorStore | Vector-indexed reuse library; dependency validation; red flags |
| Workers | RQ + Redis | Same Redis, no extra broker |
| Multi-agent | Planner + Scheduler | DAG with Kahn's algorithm, semaphore back-pressure |

---

## Testing

```bash
# Unit tests
PYTHONPATH=src python3 -m pytest tests/unit/

# Integration tests (SQLite, no Docker required)
PYTHONPATH=src python3 -m pytest tests/integration/

# Specific suites
PYTHONPATH=src python3 -m pytest tests/unit/test_security.py
PYTHONPATH=src python3 -m pytest tests/unit/test_skill_store.py
PYTHONPATH=src python3 -m pytest tests/unit/test_filesystem.py

# With coverage
PYTHONPATH=src python3 -m pytest tests/ --cov=src/harness --cov-report=term-missing
```

**Current: 1058 unit tests passing, 0 failures.**

| Test file | Tests | What it covers |
|---|---|---|
| `test_security.py` | 52 | SecretProvider (Env/Vault/AWS/Cached/Tenant), SecretScanner, pipeline integration |
| `test_skill_store.py` | 68 | SkillStore CRUD, retrieval, validation, health report, red flags, SkillCapture |
| `test_verifier.py` | 31 | PEV CodeExitCodeVerifier, ExpectedOutputVerifier, BaseAgent feedback injection |
| `test_harness_attribution.py` | 37 | HarnessComponent attribution, generate_retry_patch, generate_permission_patch |
| `test_filesystem.py` | 43 | CheckpointManager, WorkspaceManager, DockerSandbox runtime flag, SessionDockerSandbox |
| `test_agent_base.py` | 77 | Checkpoint resume, policy enforcement, HITL, skill retrieval, session sandbox |
| `test_safety.py` | 52 | HarnessPolicy, PolicyStore, pipeline factory |
| `test_eval_components.py` | 82 | CodeSandbox OOM, RunCodeTool paths, SQL/HTTP sandbox |
| `test_trace_schema.py` | 27 | TraceSpan, SpanKind, finish(), to_dict/from_dict, TraceView |
| `test_trace_recorder.py` | 31 | Span lifecycle, parent stack, set_llm_usage, context manager, JSONL |
| `test_context_engine.py` | 60 | Push, build_context, evaluate_action, sub-agent slice, scoring helpers |
| `test_agent_base_fixes.py` | 20 | _record_failure StepFailure fix, _llm_span sync CM fix, span wiring |
| Other unit tests | ~478 | BaseAgent lifecycle, tools, RLVR, messaging, MLflow, eval pipeline, GraphRAG |

---

## Future Scope

| Area | Feature | Expected Impact |
|---|---|---|
| **Security** | TLS-intercepting credential proxy (litellm-agent-platform pattern) — agents receive stub tokens; proxy swaps real keys at the network layer | Agents never see real credentials even in memory |
| **Sandbox** | `SandboxWarmPool` — pre-warmed container pool for instant allocation | Sub-100ms sandbox startup vs 2–5s current |
| **Tracing** | Export spans to OTel-native backends (Jaeger, Grafana Tempo) with full W3C TraceContext propagation | Full distributed trace across parent→child agents |
| **Token Efficiency** | Adaptive context compression — summarize stale history with a small model | 40–60% token reduction on long sessions |
| **Routing** | ML-based predictive model selection — learn per-task-type patterns | Eliminates over-provisioned Opus/GPT-5 usage |
| **Skill Store** | Automated staleness validation pipeline — re-run skill code in sandbox on schedule | Catches broken skills before agents rely on them |
| **Parallelism** | Streaming pipeline overlap — start tool execution while LLM still generating | Lower end-to-end step latency |
| **Hermes** | Cost-aware patch targeting — rank prompt candidates by token spend | Better ROI from self-improvement cycles |
| **Scheduling** | Fair-share multi-tenant scheduler — priority queues and resource caps | Predictable per-tenant cost and latency |

---

## Contributing

Fork, branch off `main`, write tests for anything new, open a PR.

```bash
git checkout -b feat/your-feature
PYTHONPATH=src python3 -m pytest tests/unit/
ruff check src/ tests/
```

Things that would be useful: new LLM provider adapters, additional vector backends, more tool integrations, Kubernetes Helm chart, and examples for specific use cases.

---

## License

MIT. See [LICENSE](LICENSE).

---

<p align="center">
  <a href="https://thepradip.github.io/HarnessAgent/">Technical Docs</a> &nbsp;|&nbsp;
  <a href="http://localhost:8000">Dashboard</a> &nbsp;|&nbsp;
  <a href="https://github.com/thepradip/HarnessAgent/issues">Issues</a>
</p>
