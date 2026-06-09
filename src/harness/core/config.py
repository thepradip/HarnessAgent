"""Pydantic-settings configuration for HarnessAgent."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -------------------------------------------------------------------------
    # LLM Provider Keys
    # -------------------------------------------------------------------------
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Comma-separated OpenAI models to register, e.g. "gpt-4o-mini,gpt-4o,o4-mini"
    openai_models: str = "gpt-4o-mini"
    # Optional Azure OpenAI base URL (leave empty for api.openai.com)
    openai_base_url: str = ""
    # Azure OpenAI (takes priority over openai_api_key when set)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""            # e.g. https://your-resource.openai.azure.com/
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_deployment: str = "gpt-5.2"  # deployment name in Azure portal
    # Local model endpoints (optional)
    vllm_base_url: str = ""
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    sglang_base_url: str = ""
    sglang_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llamacpp_base_url: str = ""

    # OpenAI-compatible vendors (enabled by setting the key). Optional per-vendor
    # <VENDOR>_BASE_URL / <VENDOR>_MODELS overrides are read from the environment.
    deepseek_api_key: str = ""
    together_api_key: str = ""
    fireworks_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    mistral_api_key: str = ""
    xai_api_key: str = ""

    # AWS Bedrock (set bedrock_enabled=true + AWS creds in the standard chain).
    # Model lists are "id:tier" comma-separated, e.g.
    #   BEDROCK_CLAUDE_MODELS="anthropic.claude-opus-4-7:premium,anthropic.claude-haiku-4-5:cheap"
    #   BEDROCK_CONVERSE_MODELS="meta.llama3-3-70b-instruct-v1:0:standard"
    bedrock_enabled: bool = False
    bedrock_region: str = "us-east-1"
    bedrock_claude_models: str = ""
    bedrock_converse_models: str = ""

    # Cost-aware routing: complexity scorer on by default; optional per-tenant
    # tier→model maps as a JSON string (ROUTING_TENANT_TIERS).
    routing_complexity_enabled: bool = True
    routing_tenant_tiers: str = ""

    # -------------------------------------------------------------------------
    # Vector Store
    # -------------------------------------------------------------------------
    vector_backend: Literal["chroma", "qdrant", "weaviate"] = "chroma"
    chroma_path: str = "/data/chroma"
    qdrant_url: str = "http://localhost:6333"
    weaviate_url: str = "http://localhost:8080"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # fastembed (default, ONNX ~100 MB) | sentence-transformers (torch ~1.5 GB)
    embedding_backend: str = "fastembed"

    # -------------------------------------------------------------------------
    # Graph Store
    # -------------------------------------------------------------------------
    graph_backend: Literal["networkx", "neo4j"] = "networkx"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "harnesspassword"

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379"

    # -------------------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "harness-agent"

    # -------------------------------------------------------------------------
    # OpenTelemetry
    # -------------------------------------------------------------------------
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "harness-agent"

    # -------------------------------------------------------------------------
    # LLM Defaults
    # -------------------------------------------------------------------------
    default_model: str = "claude-sonnet-4-6"

    # -------------------------------------------------------------------------
    # Workspace
    # -------------------------------------------------------------------------
    workspace_base_path: str = "/workspaces"

    # -------------------------------------------------------------------------
    # Sandbox
    # -------------------------------------------------------------------------
    # Workload profile for the code execution sandbox.
    # Controls Docker container memory limit:
    #   general → 256 MiB  (scripting, algorithms)
    #   data    → 512 MiB  (pandas / numpy with real datasets)
    #   ml      → 2 GiB    (torch / sklearn model runs)
    sandbox_workload: Literal["general", "data", "ml"] = "general"
    # Container runtime for code execution sandboxes.
    # "runc"  — default Docker runtime (no extra setup)
    # "runsc" — gVisor (kernel-level syscall interception, stronger isolation)
    # "kata"  — Kata Containers (lightweight VM per sandbox, strongest isolation)
    sandbox_runtime: str = "runc"
    # Reuse one container per agent run (eliminates per-call cold-start overhead).
    sandbox_session_reuse: bool = False
    # Where session code execution runs:
    #   "docker" — local Docker container (default; uses sandbox_runtime above)
    #   "e2b"    — E2B cloud micro-VM (set E2B_API_KEY; pip install agent-haas[e2b])
    #   "modal"  — Modal serverless container (set MODAL_TOKEN_ID/SECRET; agent-haas[modal])
    sandbox_provider: Literal["docker", "e2b", "modal"] = "docker"
    e2b_api_key: str = ""
    e2b_template: str = ""   # optional E2B template id; empty = SDK default
    modal_app_name: str = "agent-haas-sandbox"

    # -------------------------------------------------------------------------
    # Optional Agent Tool Backends
    # -------------------------------------------------------------------------
    sql_connection_string: str = ""

    # -------------------------------------------------------------------------
    # Hermes Self-Healing
    # -------------------------------------------------------------------------
    hermes_auto_apply: bool = False
    hermes_interval_seconds: float = 3600.0
    hermes_min_errors_to_trigger: int = 5
    hermes_patch_score_threshold: float = 0.7

    # Run the full evaluator-backed HermesLoop in the background worker: patches
    # are scored by replaying failing tasks (AgentRunner + Evaluator) before the
    # apply/rollback gate. Required for GEPA in production. When False the worker
    # uses the lightweight generate-and-queue path.
    hermes_use_evaluator: bool = False

    # Prompt-patch generation strategy:
    #   "heuristic" — the default LLM/error-analysis PatchGenerator (one targeted edit).
    #   "gepa"      — GEPA reflective prompt evolution (population + Pareto selection)
    #                 over the same Evaluator metric. Requires the ``gepa`` extra and
    #                 an Evaluator-backed loop; falls back to "heuristic" otherwise.
    hermes_strategy: str = "heuristic"
    # Max metric calls (candidate evaluations) GEPA may spend per cycle. Each call
    # replays one failing task through the Evaluator, so keep this modest.
    hermes_gepa_budget: int = 30
    # Max tokens for GEPA's reflection (teacher) LM when proposing a new prompt.
    hermes_gepa_reflection_max_tokens: int = 4096
    # Minimum sampled errors required before GEPA runs; below this it is not worth
    # the rollout cost and the loop falls back to the heuristic generator.
    hermes_gepa_min_train: int = 3
    # Log GEPA's optimization progress (params, per-step scores, summary) to MLflow
    # using the ``mlflow_tracking_uri`` / ``mlflow_experiment_name`` settings above.
    hermes_gepa_use_mlflow: bool = False

    # -------------------------------------------------------------------------
    # Auth
    # -------------------------------------------------------------------------
    jwt_secret_key: str = "change-me-in-production"

    # -------------------------------------------------------------------------
    # Budgets & Rate Limits
    # -------------------------------------------------------------------------
    cost_budget_usd_per_tenant: float = 100.0
    rate_limit_rpm: int = 60

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    worker_concurrency: int = 4


@lru_cache
def get_config() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
