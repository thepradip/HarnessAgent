"""HarnessAgent security layer — secret management and output scanning."""

from harness.security.scanner import SecretMatch, SecretScanner, has_secrets, redact, scan
from harness.security.secrets import (
    AWSSecretsProvider,
    CachedSecretProvider,
    EnvSecretProvider,
    SecretNotFoundError,
    SecretProvider,
    TenantSecretProvider,
    VaultSecretProvider,
    configure,
    get_provider,
    get_secret,
    is_likely_secret,
    mask,
)

__all__ = [
    # Providers
    "SecretProvider",
    "EnvSecretProvider",
    "VaultSecretProvider",
    "AWSSecretsProvider",
    "CachedSecretProvider",
    "TenantSecretProvider",
    # Convenience
    "configure",
    "get_secret",
    "get_provider",
    # Utilities
    "SecretNotFoundError",
    "mask",
    "is_likely_secret",
    # Scanner
    "SecretScanner",
    "SecretMatch",
    "redact",
    "scan",
    "has_secrets",
]
