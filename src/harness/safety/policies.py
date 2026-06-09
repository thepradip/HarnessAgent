"""Per-tenant policy overrides for HarnessAgent.

Policies control budget limits, allowed agent types, tool restrictions,
PII handling, and HITL requirements on a per-tenant basis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_POLICY_KEY_PREFIX = "harness:policy:"


@dataclass
class HarnessPolicy:
    """Per-tenant policy overrides.

    All fields have sensible defaults that match the global config defaults.
    """

    tenant_id: str
    max_steps: int = 50
    max_tokens: int = 100_000
    max_cost_usd: float = 10.0
    allowed_agent_types: list[str] | None = None  # None = all agent types allowed
    blocked_tools: list[str] = field(default_factory=list)
    pii_redact: bool = True
    hitl_required_for: list[str] = field(default_factory=list)  # tool names needing approval
    hermes_auto_apply: bool = False
    # Additional capability flags
    allow_code_execution: bool = True
    allow_file_write: bool = True
    max_concurrent_runs: int = 5
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def allows_agent_type(self, agent_type: str) -> bool:
        """Return True if this tenant may use the specified agent type."""
        if self.allowed_agent_types is None:
            return True
        return agent_type in self.allowed_agent_types

    def requires_hitl(self, tool_name: str) -> bool:
        """Return True if human approval is required before running this tool."""
        return tool_name in self.hitl_required_for

    def is_tool_blocked(self, tool_name: str) -> bool:
        """Return True if this tool is blocked for this tenant."""
        return tool_name in self.blocked_tools

    def to_dict(self) -> dict[str, Any]:
        """Serialise the policy to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessPolicy":
        """Deserialise from a dict (as returned by to_dict)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


_DEFAULT_POLICY_TEMPLATE: dict[str, Any] = {
    "max_steps": 50,
    "max_tokens": 100_000,
    "max_cost_usd": 10.0,
    "allowed_agent_types": None,
    "blocked_tools": [],
    "pii_redact": True,
    "hitl_required_for": [],
    "hermes_auto_apply": False,
    "allow_code_execution": True,
    "allow_file_write": True,
    "max_concurrent_runs": 5,
    "custom_metadata": {},
}


class PolicyStore:
    """Redis-backed store for per-tenant HarnessPolicy objects.

    Policies are stored as JSON at ``harness:policy:{tenant_id}`` with no TTL —
    a tenant's restrictions (blocked_tools / HITL requirements) must never
    silently expire into the permissive default. A missing policy returns the
    default policy for that tenant.
    """

    def __init__(self, redis: Any) -> None:
        """Initialise with an async redis.asyncio.Redis client."""
        self._redis = redis

    def _key(self, tenant_id: str) -> str:
        return f"{_POLICY_KEY_PREFIX}{tenant_id}"

    async def get(self, tenant_id: str) -> HarnessPolicy:
        """Return the stored policy for tenant_id, or a default policy if not found."""
        try:
            raw = await self._redis.get(self._key(tenant_id))
            if raw is None:
                return self._default_policy(tenant_id)
            data = json.loads(raw)
            # Ensure tenant_id is always correct
            data["tenant_id"] = tenant_id
            return HarnessPolicy.from_dict(data)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to decode policy for tenant '%s': %s — using default",
                tenant_id,
                exc,
            )
            return self._default_policy(tenant_id)
        except Exception as exc:
            logger.warning(
                "PolicyStore.get failed for tenant '%s': %s — using default",
                tenant_id,
                exc,
            )
            return self._default_policy(tenant_id)

    async def set(self, policy: HarnessPolicy) -> None:
        """Store a policy, overwriting any existing policy for that tenant."""
        try:
            serialized = json.dumps(policy.to_dict(), default=str)
            # No TTL: policy records (blocked_tools / HITL) must persist until
            # explicitly changed or deleted — never expire into the default.
            await self._redis.set(
                self._key(policy.tenant_id),
                serialized,
            )
            logger.info("Stored policy for tenant '%s'", policy.tenant_id)
        except Exception as exc:
            logger.error(
                "PolicyStore.set failed for tenant '%s': %s", policy.tenant_id, exc
            )
            raise

    async def delete(self, tenant_id: str) -> None:
        """Remove the custom policy for a tenant (reverts to default)."""
        try:
            await self._redis.delete(self._key(tenant_id))
            logger.info("Deleted policy for tenant '%s'", tenant_id)
        except Exception as exc:
            logger.warning(
                "PolicyStore.delete failed for tenant '%s': %s", tenant_id, exc
            )

    async def list_tenants(self) -> list[str]:
        """Return a list of tenant_ids that have custom policies."""
        try:
            keys = await self._redis.keys(f"{_POLICY_KEY_PREFIX}*")
            prefix_len = len(_POLICY_KEY_PREFIX)
            return [
                k.decode() if isinstance(k, bytes) else k
                for k in [
                    k[prefix_len:] if isinstance(k, str) else k[prefix_len:]
                    for k in keys
                ]
            ]
        except Exception as exc:
            logger.warning("PolicyStore.list_tenants failed: %s", exc)
            return []

    @staticmethod
    def _default_policy(tenant_id: str) -> HarnessPolicy:
        """Construct a default policy for a tenant with no custom policy."""
        data = dict(_DEFAULT_POLICY_TEMPLATE)
        data["tenant_id"] = tenant_id
        return HarnessPolicy.from_dict(data)
