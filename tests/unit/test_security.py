"""Tests for SecretProvider, SecretScanner, and safety pipeline integration."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.security.secrets import (
    AWSSecretsProvider,
    CachedSecretProvider,
    EnvSecretProvider,
    SecretNotFoundError,
    TenantSecretProvider,
    VaultSecretProvider,
    configure,
    get_provider,
    get_secret,
    is_likely_secret,
    mask,
)
from harness.security.scanner import SecretMatch, SecretScanner, has_secrets, redact, scan


# ===========================================================================
# EnvSecretProvider
# ===========================================================================

@pytest.mark.asyncio
async def test_env_provider_reads_key():
    p = EnvSecretProvider(env={"ANTHROPIC_API_KEY": "sk-ant-real-key-abc"})
    assert await p.get("ANTHROPIC_API_KEY") == "sk-ant-real-key-abc"


@pytest.mark.asyncio
async def test_env_provider_case_insensitive_lookup():
    p = EnvSecretProvider(env={"ANTHROPIC_API_KEY": "sk-ant-real-key-abc"})
    # lowercase name → should find UPPER form
    assert await p.get("anthropic_api_key") == "sk-ant-real-key-abc"


@pytest.mark.asyncio
async def test_env_provider_missing_raises():
    p = EnvSecretProvider(env={})
    with pytest.raises(SecretNotFoundError):
        await p.get("missing_key")


@pytest.mark.asyncio
async def test_env_provider_empty_value_raises():
    p = EnvSecretProvider(env={"EMPTY": ""})
    with pytest.raises(SecretNotFoundError):
        await p.get("EMPTY")


@pytest.mark.asyncio
async def test_env_provider_list_names():
    p = EnvSecretProvider(env={"A": "1", "B": "2"})
    names = await p.list_names()
    assert "A" in names
    assert "B" in names


# ===========================================================================
# VaultSecretProvider
# ===========================================================================

@pytest.mark.asyncio
async def test_vault_provider_get_success():
    mock_hvac = MagicMock()
    mock_client = MagicMock()
    mock_hvac.Client.return_value = mock_client
    mock_client.secrets.kv.v2.read_secret_version.return_value = {
        "data": {"data": {"value": "sk-ant-vault-secret"}}
    }

    with patch.dict("sys.modules", {"hvac": mock_hvac}):
        p = VaultSecretProvider(url="https://vault:8200", token="s.test")
        result = await p.get("anthropic_api_key")

    assert result == "sk-ant-vault-secret"


@pytest.mark.asyncio
async def test_vault_provider_not_found_raises():
    mock_hvac = MagicMock()
    mock_client = MagicMock()
    mock_hvac.Client.return_value = mock_client
    mock_hvac.exceptions.InvalidPath = Exception  # so isinstance check works
    mock_client.secrets.kv.v2.read_secret_version.side_effect = Exception("InvalidPath")

    with patch.dict("sys.modules", {"hvac": mock_hvac}):
        p = VaultSecretProvider(url="https://vault:8200", token="s.test")
        with pytest.raises(SecretNotFoundError):
            await p.get("missing")


@pytest.mark.asyncio
async def test_vault_provider_missing_hvac_raises_import_error():
    with patch.dict("sys.modules", {"hvac": None}):
        p = VaultSecretProvider(url="https://vault:8200", token="s.test")
        with pytest.raises(ImportError, match="hvac"):
            await p.get("any_key")


# ===========================================================================
# AWSSecretsProvider
# ===========================================================================

@pytest.mark.asyncio
async def test_aws_provider_plain_string():
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.get_secret_value.return_value = {"SecretString": "sk-aws-secret"}

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        p = AWSSecretsProvider(region_name="us-east-1", name_prefix="harness")
        result = await p.get("openai_api_key")

    assert result == "sk-aws-secret"


@pytest.mark.asyncio
async def test_aws_provider_json_value():
    import json
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.get_secret_value.return_value = {
        "SecretString": json.dumps({"value": "sk-json-secret"})
    }

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        p = AWSSecretsProvider()
        result = await p.get("some_key")

    assert result == "sk-json-secret"


@pytest.mark.asyncio
async def test_aws_provider_not_found_raises():
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    class _NotFound(Exception):
        pass

    mock_client.exceptions.ResourceNotFoundException = _NotFound
    mock_client.get_secret_value.side_effect = _NotFound("not found")

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        p = AWSSecretsProvider()
        with pytest.raises(SecretNotFoundError):
            await p.get("missing")


# ===========================================================================
# CachedSecretProvider
# ===========================================================================

@pytest.mark.asyncio
async def test_cached_provider_returns_value():
    inner = EnvSecretProvider(env={"KEY": "val"})
    p = CachedSecretProvider(inner, ttl_seconds=60)
    assert await p.get("KEY") == "val"


@pytest.mark.asyncio
async def test_cached_provider_caches_on_second_call():
    inner = AsyncMock()
    inner.get = AsyncMock(return_value="cached-value")
    p = CachedSecretProvider(inner, ttl_seconds=60)
    await p.get("KEY")
    await p.get("KEY")
    assert inner.get.call_count == 1  # second call was from cache


@pytest.mark.asyncio
async def test_cached_provider_re_fetches_after_expiry():
    inner = AsyncMock()
    inner.get = AsyncMock(return_value="fresh-value")
    p = CachedSecretProvider(inner, ttl_seconds=1)

    await p.get("KEY")
    # Manually expire the cache entry
    p._cache["KEY"] = ("old-value", datetime.now(timezone.utc) - timedelta(seconds=10))
    await p.get("KEY")

    assert inner.get.call_count == 2


@pytest.mark.asyncio
async def test_cached_provider_invalidate_single():
    inner = AsyncMock()
    inner.get = AsyncMock(return_value="v")
    p = CachedSecretProvider(inner, ttl_seconds=60)
    await p.get("A")
    await p.get("B")
    p.invalidate("A")
    assert "A" not in p._cache
    assert "B" in p._cache


@pytest.mark.asyncio
async def test_cached_provider_invalidate_all():
    inner = AsyncMock()
    inner.get = AsyncMock(return_value="v")
    p = CachedSecretProvider(inner, ttl_seconds=60)
    await p.get("A")
    await p.get("B")
    p.invalidate()
    assert p._cache == {}


# ===========================================================================
# TenantSecretProvider
# ===========================================================================

@pytest.mark.asyncio
async def test_tenant_provider_uses_tenant_key_first():
    env = {
        "acme/anthropic_api_key": "tenant-key",
        "ANTHROPIC_API_KEY": "global-key",
    }
    p = TenantSecretProvider(EnvSecretProvider(env=env), tenant_id="acme")
    result = await p.get("anthropic_api_key")
    assert result == "tenant-key"


@pytest.mark.asyncio
async def test_tenant_provider_falls_back_to_global():
    env = {"ANTHROPIC_API_KEY": "global-key"}
    p = TenantSecretProvider(EnvSecretProvider(env=env), tenant_id="acme")
    result = await p.get("anthropic_api_key")
    assert result == "global-key"


@pytest.mark.asyncio
async def test_tenant_provider_raises_when_both_missing():
    p = TenantSecretProvider(EnvSecretProvider(env={}), tenant_id="acme")
    with pytest.raises(SecretNotFoundError):
        await p.get("missing_key")


# ===========================================================================
# Module-level configure / get_secret
# ===========================================================================

@pytest.mark.asyncio
async def test_configure_and_get_secret():
    import harness.security.secrets as _mod
    original = _mod._default_provider

    try:
        configure(EnvSecretProvider(env={"MY_KEY": "my-value"}))
        result = await get_secret("MY_KEY")
        assert result == "my-value"
    finally:
        _mod._default_provider = original


@pytest.mark.asyncio
async def test_get_secret_falls_back_to_env_when_unconfigured():
    import harness.security.secrets as _mod
    original = _mod._default_provider
    _mod._default_provider = None

    try:
        with patch.dict(os.environ, {"FALLBACK_KEY": "env-value"}):
            result = await get_secret("FALLBACK_KEY")
        assert result == "env-value"
    finally:
        _mod._default_provider = original


# ===========================================================================
# mask() and is_likely_secret()
# ===========================================================================

def test_mask_long_key():
    key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
    masked = mask(key)
    assert "[MASKED]" in masked
    assert key not in masked
    assert masked.startswith("sk-ant-a")


def test_mask_short_value():
    assert mask("abc") == "[MASKED]"


def test_mask_empty():
    assert mask("") == "[EMPTY]"


def test_is_likely_secret_anthropic():
    assert is_likely_secret("sk-ant-api03-abcdefghijklmnopqrst") is True


def test_is_likely_secret_openai():
    assert is_likely_secret("sk-abcdefghijklmnopqrstuvwxyz1234567890ab") is True


def test_is_likely_secret_github():
    assert is_likely_secret("ghp_abcdefghijklmnopqrstuvwxyz1234567") is True


def test_is_likely_secret_too_short():
    assert is_likely_secret("sk-short") is False


def test_is_likely_secret_normal_text():
    assert is_likely_secret("hello world this is a normal sentence") is False


# ===========================================================================
# SecretScanner — scan()
# ===========================================================================

def test_scanner_detects_anthropic_key():
    text = f"my key is sk-ant-api03-{'a' * 30} thanks"
    matches = scan(text)
    assert any(m.pattern_name == "anthropic" for m in matches)


def test_scanner_detects_openai_key():
    text = f"sk-{'a' * 48} is my key"
    matches = scan(text)
    assert any(m.pattern_name == "openai" for m in matches)


def test_scanner_detects_github_pat():
    text = f"token=ghp_{'a' * 36}"
    matches = scan(text)
    assert any(m.pattern_name == "github_ghp" for m in matches)


def test_scanner_detects_url_credentials():
    text = "postgresql://admin:supersecret123@localhost/db"
    matches = scan(text)
    assert any(m.pattern_name == "url_creds" for m in matches)


def test_scanner_detects_bearer_token():
    text = f"Authorization: Bearer {'a' * 25}"
    matches = scan(text)
    assert any(m.pattern_name == "bearer_token" for m in matches)


def test_scanner_detects_jwt():
    # Minimal valid JWT structure
    text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    matches = scan(text)
    assert any(m.pattern_name == "jwt" for m in matches)


def test_scanner_clean_text_no_matches():
    text = "The quick brown fox jumps over the lazy dog"
    assert scan(text) == []


def test_scanner_no_partial_overlap_duplicates():
    key = f"sk-ant-api03-{'b' * 30}"
    text = f"key1={key} and key2={key}"
    matches = scan(text)
    # Both instances should be found
    anthropic_matches = [m for m in matches if m.pattern_name == "anthropic"]
    assert len(anthropic_matches) == 2


# ===========================================================================
# SecretScanner — redact()
# ===========================================================================

def test_scanner_redacts_anthropic_key():
    key = f"sk-ant-api03-{'x' * 30}"
    result = redact(f"key={key}")
    assert key not in result
    assert "REDACTED" in result


def test_scanner_redacts_openai_key():
    key = f"sk-{'z' * 48}"
    result = redact(f"use {key} for auth")
    assert key not in result


def test_scanner_preserves_surrounding_text():
    key = f"sk-ant-api03-{'a' * 20}"
    result = redact(f"before {key} after")
    assert "before" in result
    assert "after" in result


def test_scanner_redacts_url_credentials():
    url = "redis://user:mypassword123@localhost:6379"
    result = redact(url)
    assert "mypassword123" not in result
    assert "REDACTED" in result


def test_scanner_clean_text_unchanged():
    text = "No secrets here, just a normal string."
    assert redact(text) == text


# ===========================================================================
# SecretScanner — scan_dict() and redact_dict()
# ===========================================================================

def test_scanner_scan_dict_finds_nested():
    scanner = SecretScanner()
    d = {
        "config": {
            "api_key": f"sk-ant-api03-{'n' * 25}",
            "model": "claude-3",
        }
    }
    matches = scanner.scan_dict(d)
    assert any(m.pattern_name == "anthropic" for m in matches)


def test_scanner_scan_dict_finds_in_list():
    scanner = SecretScanner()
    d = {"keys": [f"sk-{'k' * 48}", "normal"]}
    matches = scanner.scan_dict(d)
    assert len(matches) >= 1


def test_scanner_redact_dict():
    scanner = SecretScanner()
    key = f"sk-ant-api03-{'r' * 25}"
    d = {"auth": {"api_key": key, "model": "claude"}}
    result = scanner.redact_dict(d)
    assert key not in str(result)
    assert result["auth"]["model"] == "claude"


def test_scanner_redact_dict_preserves_non_strings():
    scanner = SecretScanner()
    d = {"count": 42, "enabled": True, "data": None}
    result = scanner.redact_dict(d)
    assert result == d


def test_scanner_redact_dict_handles_list():
    scanner = SecretScanner()
    key = f"sk-{'m' * 48}"
    lst = [key, "safe", 123]
    result = scanner.redact_dict(lst)
    assert key not in result[0]
    assert result[1] == "safe"
    assert result[2] == 123


def test_has_secrets_true():
    assert has_secrets(f"key=sk-ant-api03-{'h' * 25}") is True


def test_has_secrets_false():
    assert has_secrets("nothing secret here at all") is False


# ===========================================================================
# _HardConstraintPipeline — secret detection in check_output
# ===========================================================================

@pytest.mark.asyncio
async def test_pipeline_check_output_warns_on_leaked_key():
    from harness.safety.pipeline_factory import _HardConstraintPipeline
    pipeline = _HardConstraintPipeline()
    key = f"sk-ant-api03-{'w' * 25}"
    result = await pipeline.check_output({"content": f"Here is your key: {key}"})
    # Does NOT block (we redact, not block) but records the reason
    assert result.blocked is False
    assert "secret_detected" in (result.reason or "")


@pytest.mark.asyncio
async def test_pipeline_check_output_clean_content():
    from harness.safety.pipeline_factory import _HardConstraintPipeline
    pipeline = _HardConstraintPipeline()
    result = await pipeline.check_output({"content": "The answer is 42."})
    assert result.blocked is False
    assert "secret_detected" not in (result.reason or "")


def test_pipeline_redact_covers_both_pii_and_secrets():
    from harness.safety.pipeline_factory import _HardConstraintPipeline
    pipeline = _HardConstraintPipeline()
    key = f"sk-ant-api03-{'p' * 25}"
    text = f"SSN 123-45-6789 and key {key}"
    result = pipeline.redact(text)
    assert "123-45-6789" not in result
    assert key not in result
    assert "[SSN REDACTED]" in result
    assert "REDACTED" in result
