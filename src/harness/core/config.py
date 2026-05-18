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

    # -------------------------------------------------------------------------
    # Vector Store
    # -------------------------------------------------------------------------
    vector_backend: Literal["chroma", "qdrant", "weaviate"] = "chroma"
    chroma_path: str = "/data/chroma"
    qdrant_url: str = "http://localhost:6333"
    weaviate_url: str = "http://localhost:8080"
    embedding_model: str = "all-MiniLM-L6-v2"

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
