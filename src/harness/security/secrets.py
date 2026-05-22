"""Secret provider abstraction — never expose real credentials to agent context.

Usage (dev, zero migration):
    from harness.security.secrets import get_secret
    key = await get_secret("anthropic_api_key")   # reads from os.environ

Usage (production, Vault):
    from harness.security.secrets import configure, VaultSecretProvider
    configure(VaultSecretProvider(url="https://vault:8200", token="s.xxx"))

Usage (per-tenant isolation):
    from harness.security.secrets import TenantSecretProvider, EnvSecretProvider
    provider = TenantSecretProvider(EnvSecretProvider(), tenant_id="acme")
    key = await provider.get("anthropic_api_key")
    # tries ACME_ANTHROPIC_API_KEY first, falls back to ANTHROPIC_API_KEY
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Patterns that identify likely API key values
_LIKELY_SECRET_PREFIXES = (
    "sk-ant-",      # Anthropic
    "sk-",          # OpenAI
    "ghp_",         # GitHub PAT
    "github_pat_",  # GitHub fine-grained PAT
    "xoxb-",        # Slack bot token
    "xoxp-",        # Slack user token
    "eyJ",          # JWT (base64 header)
    "glpat-",       # GitLab PAT
)
_MIN_SECRET_LEN = 20


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SecretNotFoundError(KeyError):
    """Raised when a named secret does not exist in the backing store."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SecretProvider(Protocol):
    """Retrieve secrets by name. Async so all backends are consistent."""

    async def get(self, name: str) -> str:
        """Return the secret value. Raises SecretNotFoundError if missing."""
        ...

    async def list_names(self) -> list[str]:
        """Return names of all available secrets (may be unsupported → [])."""
        ...


# ---------------------------------------------------------------------------
# EnvSecretProvider — dev / CI (zero infrastructure)
# ---------------------------------------------------------------------------

class EnvSecretProvider:
    """Reads secrets from environment variables or a provided dict.

    Name lookup is case-insensitive and also tries the UPPER_CASE form,
    so both ``anthropic_api_key`` and ``ANTHROPIC_API_KEY`` resolve.
    """

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env: dict[str, str] = env if env is not None else os.environ  # type: ignore[assignment]

    async def get(self, name: str) -> str:
        for candidate in (name, name.upper()):
            value = self._env.get(candidate)
            if value:
                return value
        raise SecretNotFoundError(
            f"Secret '{name}' not found in environment "
            f"(tried '{name}' and '{name.upper()}')"
        )

    async def list_names(self) -> list[str]:
        return list(self._env.keys())


# ---------------------------------------------------------------------------
# VaultSecretProvider — HashiCorp Vault KV v2
# ---------------------------------------------------------------------------

class VaultSecretProvider:
    """HashiCorp Vault KV v2 backend.

    Requires: ``pip install hvac``

    Secrets are stored at ``{mount}/{path_prefix}/{name}`` with the value
    key ``"value"``.  Example::

        vault kv put secret/harness/anthropic_api_key value=sk-ant-...
    """

    def __init__(
        self,
        url: str,
        token: str,
        mount: str = "secret",
        path_prefix: str = "harness",
    ) -> None:
        self._url = url
        self._token = token
        self._mount = mount
        self._path_prefix = path_prefix

    async def get(self, name: str) -> str:
        try:
            import hvac  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "VaultSecretProvider requires 'hvac': pip install hvac"
            ) from exc

        import asyncio

        def _read() -> str:
            client = hvac.Client(url=self._url, token=self._token)
            path = f"{self._path_prefix}/{name}"
            try:
                resp = client.secrets.kv.v2.read_secret_version(
                    path=path, mount_point=self._mount
                )
                value = resp["data"]["data"].get("value")
                if not value:
                    raise SecretNotFoundError(
                        f"Vault secret '{path}' exists but has no 'value' key"
                    )
                return value
            except hvac.exceptions.InvalidPath:
                raise SecretNotFoundError(
                    f"Secret '{path}' not found in Vault (mount={self._mount})"
                )

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _read)
        except SecretNotFoundError:
            raise
        except Exception as exc:
            raise SecretNotFoundError(f"Vault lookup failed for '{name}': {exc}") from exc

    async def list_names(self) -> list[str]:
        try:
            import hvac  # type: ignore[import]
            import asyncio

            def _list() -> list[str]:
                client = hvac.Client(url=self._url, token=self._token)
                resp = client.secrets.kv.v2.list_secrets(
                    path=self._path_prefix, mount_point=self._mount
                )
                return resp.get("data", {}).get("keys", [])

            return await asyncio.get_event_loop().run_in_executor(None, _list)
        except Exception as exc:
            logger.warning("VaultSecretProvider.list_names failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# AWSSecretsProvider — AWS Secrets Manager
# ---------------------------------------------------------------------------

class AWSSecretsProvider:
    """AWS Secrets Manager backend.

    Requires: ``pip install boto3``

    Secrets are stored as plain strings or JSON ``{"value": "..."}``.
    The ``name_prefix`` is prepended automatically: ``{prefix}/{name}``.
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        name_prefix: str = "harness",
        **boto3_kwargs: Any,
    ) -> None:
        self._region = region_name
        self._prefix = name_prefix
        self._boto3_kwargs = boto3_kwargs

    def _full_name(self, name: str) -> str:
        return f"{self._prefix}/{name}" if self._prefix else name

    async def get(self, name: str) -> str:
        try:
            import boto3  # type: ignore[import]
            import json as _json
        except ImportError as exc:
            raise ImportError(
                "AWSSecretsProvider requires 'boto3': pip install boto3"
            ) from exc

        import asyncio

        full_name = self._full_name(name)

        def _read() -> str:
            client = boto3.client(
                "secretsmanager",
                region_name=self._region,
                **self._boto3_kwargs,
            )
            try:
                resp = client.get_secret_value(SecretId=full_name)
            except client.exceptions.ResourceNotFoundException:
                raise SecretNotFoundError(
                    f"Secret '{full_name}' not found in AWS Secrets Manager"
                )
            raw = resp.get("SecretString", "")
            # Try JSON {"value": "..."}, fall back to raw string
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed.get("value") or parsed.get(name) or raw
            except (_json.JSONDecodeError, TypeError):
                pass
            return raw

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _read)
        except SecretNotFoundError:
            raise
        except Exception as exc:
            raise SecretNotFoundError(
                f"AWS Secrets Manager lookup failed for '{full_name}': {exc}"
            ) from exc

    async def list_names(self) -> list[str]:
        return []  # listing is expensive; not worth the API call


# ---------------------------------------------------------------------------
# CachedSecretProvider — in-memory TTL cache wrapper
# ---------------------------------------------------------------------------

class CachedSecretProvider:
    """Wraps any SecretProvider with an in-memory TTL cache.

    Avoids repeated round-trips to Vault / AWS on every secret access.
    Default TTL is 5 minutes — short enough to pick up rotated keys,
    long enough to avoid hammering the vault on every LLM call.
    """

    def __init__(
        self,
        provider: SecretProvider,
        ttl_seconds: int = 300,
    ) -> None:
        self._provider = provider
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[str, datetime]] = {}

    async def get(self, name: str) -> str:
        now = datetime.now(timezone.utc)
        if name in self._cache:
            value, expires_at = self._cache[name]
            if now < expires_at:
                return value
            del self._cache[name]

        value = await self._provider.get(name)
        self._cache[name] = (value, now + timedelta(seconds=self._ttl))
        return value

    async def list_names(self) -> list[str]:
        return await self._provider.list_names()

    def invalidate(self, name: str | None = None) -> None:
        """Remove one entry or flush the entire cache."""
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)


# ---------------------------------------------------------------------------
# TenantSecretProvider — per-tenant path isolation
# ---------------------------------------------------------------------------

class TenantSecretProvider:
    """Adds per-tenant secret path isolation.

    Lookup order:
    1. ``{tenant_id}/{name}``  (e.g. ``acme/anthropic_api_key``)
    2. ``{name}``              (global fallback)

    This means tenants can override global keys with their own credentials
    without requiring separate infrastructure per tenant.
    """

    def __init__(self, provider: SecretProvider, tenant_id: str) -> None:
        self._provider = provider
        self._tenant_id = tenant_id

    async def get(self, name: str) -> str:
        tenant_name = f"{self._tenant_id}/{name}"
        try:
            return await self._provider.get(tenant_name)
        except SecretNotFoundError:
            pass
        return await self._provider.get(name)

    async def list_names(self) -> list[str]:
        return await self._provider.list_names()


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_provider: SecretProvider | None = None


def configure(provider: SecretProvider) -> None:
    """Set the module-level default provider used by ``get_secret``."""
    global _default_provider
    _default_provider = provider
    logger.info("SecretProvider configured: %s", type(provider).__name__)


async def get_secret(name: str) -> str:
    """Get a secret using the configured provider (default: EnvSecretProvider).

    Convenience wrapper — use this in application code so the provider can
    be swapped at startup without changing call sites.
    """
    provider = _default_provider or EnvSecretProvider()
    return await provider.get(name)


def get_provider() -> SecretProvider:
    """Return the active provider (or a fresh EnvSecretProvider if unconfigured)."""
    return _default_provider or EnvSecretProvider()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def mask(value: str) -> str:
    """Return a safely masked version for logging.

    ``sk-ant-api03-abcd...wxyz`` → ``sk-ant-a[MASKED]wxyz``
    """
    if not value:
        return "[EMPTY]"
    if len(value) <= 8:
        return "[MASKED]"
    prefix = value[:8]
    suffix = value[-4:] if len(value) > 12 else ""
    return f"{prefix}[MASKED]{suffix}"


def is_likely_secret(value: str) -> bool:
    """Heuristic: return True if the string looks like an API key or token."""
    if len(value) < _MIN_SECRET_LEN:
        return False
    return value.startswith(_LIKELY_SECRET_PREFIXES)
