"""Checkpoint manager: atomic save/restore of AgentContext for crash recovery."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "checkpoint.json"
# NOTE: do NOT use a ``*.tmp`` suffix here — workspace cleanup(keep_artifacts=True)
# deletes *.tmp files and could clobber an in-flight checkpoint. ``.swap`` is
# outside the cleaned patterns.
_CHECKPOINT_TMP_SUFFIX = ".swap"


@dataclass
class CheckpointData:
    """Serialisable snapshot of an AgentContext's mutable state."""

    run_id: str
    tenant_id: str
    agent_type: str
    task: str
    step_count: int
    token_count: int
    started_at: datetime
    metadata: dict[str, Any]
    failed: bool
    failure_class: str | None
    history_snapshot: list[dict[str, Any]]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "step_count": self.step_count,
            "token_count": self.token_count,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
            "failed": self.failed,
            "failure_class": self.failure_class,
            "history_snapshot": self.history_snapshot,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointData":
        d = dict(d)
        for field_name in ("started_at", "created_at"):
            val = d.get(field_name)
            if isinstance(val, str):
                d[field_name] = datetime.fromisoformat(val)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """
    Saves and restores agent run state for crash recovery.

    Checkpoint files are written atomically:
    1. Serialise to a ``.tmp`` file.
    2. Rename to the final checkpoint filename (atomic on POSIX filesystems).

    The workspace layout:
        ``{workspace_base}/{tenant_id}/{run_id}/checkpoint.json``
    """

    def __init__(self, workspace_base: str | Path) -> None:
        self._workspace_base = Path(workspace_base)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(
        self,
        ctx: "AgentContext",
        history: list[Any],
    ) -> Path:
        """
        Serialise ``ctx`` and ``history`` to a checkpoint file.

        Uses atomic write (tmp → rename) to prevent partial checkpoint reads
        on crash.

        ``history`` should be a list of ConversationMessage objects or dicts.
        """
        checkpoint_dir = self._checkpoint_dir(ctx.run_id, ctx.tenant_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Serialise history entries
        history_snapshot: list[dict[str, Any]] = []
        for msg in history:
            if hasattr(msg, "to_dict"):
                history_snapshot.append(msg.to_dict())
            elif isinstance(msg, dict):
                history_snapshot.append(msg)
            else:
                history_snapshot.append(
                    {
                        "role": getattr(msg, "role", "unknown"),
                        "content": str(getattr(msg, "content", "")),
                        "tokens": getattr(msg, "tokens", 0),
                        "timestamp": getattr(
                            msg, "timestamp", datetime.now(timezone.utc)
                        ).isoformat()
                        if hasattr(getattr(msg, "timestamp", None), "isoformat")
                        else str(getattr(msg, "timestamp", "")),
                    }
                )

        data = CheckpointData(
            run_id=ctx.run_id,
            tenant_id=ctx.tenant_id,
            agent_type=ctx.agent_type,
            task=ctx.task,
            step_count=ctx.step_count,
            token_count=ctx.token_count,
            started_at=ctx.started_at,
            metadata=dict(ctx.metadata),
            failed=ctx.failed,
            failure_class=ctx.failure_class,
            history_snapshot=history_snapshot,
        )

        checkpoint_path = checkpoint_dir / _CHECKPOINT_FILENAME

        # Crash-safe atomic write:
        #   1. Write to a UNIQUE temp file in the same dir (mkstemp avoids the
        #      fixed-name race between concurrent saves).
        #   2. flush() + os.fsync() so the bytes hit disk before the rename.
        #   3. os.replace() (atomic on POSIX) to swap in the new checkpoint.
        # The temp suffix is ``.swap`` so workspace cleanup's *.tmp pattern can't
        # delete an in-flight checkpoint.
        serialized = json.dumps(data.to_dict(), indent=2, default=str)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(checkpoint_dir),
            prefix=f"{_CHECKPOINT_FILENAME}.",
            suffix=_CHECKPOINT_TMP_SUFFIX,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(serialized)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, checkpoint_path)
            logger.debug("Checkpoint saved: %s", checkpoint_path)
        except Exception as exc:
            # Clean up tmp file if the write/replace failed.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise RuntimeError(f"Checkpoint save failed: {exc}") from exc

        return checkpoint_path

    async def load(
        self,
        run_id: str,
        tenant_id: str,
    ) -> CheckpointData | None:
        """
        Load and deserialise the checkpoint for ``run_id``.

        Returns None if the checkpoint file does not exist.
        """
        checkpoint_path = self._checkpoint_path(run_id, tenant_id)

        if not checkpoint_path.exists():
            return None

        try:
            raw = checkpoint_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            checkpoint = CheckpointData.from_dict(data)
            logger.debug(
                "Checkpoint loaded: run_id=%s step=%d",
                run_id,
                checkpoint.step_count,
            )
            return checkpoint
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "Checkpoint for run %s is corrupt: %s", run_id, exc
            )
            return None

    async def exists(self, run_id: str, tenant_id: str) -> bool:
        """Return True if a valid checkpoint file exists for this run."""
        return self._checkpoint_path(run_id, tenant_id).exists()

    async def delete(self, run_id: str, tenant_id: str) -> None:
        """Remove the checkpoint file for this run."""
        checkpoint_path = self._checkpoint_path(run_id, tenant_id)
        try:
            checkpoint_path.unlink(missing_ok=True)
            logger.debug("Checkpoint deleted: %s", checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to delete checkpoint for %s: %s", run_id, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _checkpoint_dir(self, run_id: str, tenant_id: str) -> Path:
        return self._workspace_base / tenant_id / run_id

    def _checkpoint_path(self, run_id: str, tenant_id: str) -> Path:
        return self._checkpoint_dir(run_id, tenant_id) / _CHECKPOINT_FILENAME
