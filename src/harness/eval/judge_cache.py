"""In-memory LLM judge cache for eval runs — prevents re-scoring identical prompts."""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.eval.scorers import ScoreResult


class EvalJudgeCache:
    """Thread-safe in-memory cache keyed on sha256(prompt)."""

    def __init__(self) -> None:
        self._store: dict[str, "ScoreResult"] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> "ScoreResult | None":
        key = self._key(prompt)
        with self._lock:
            result = self._store.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def set(self, prompt: str, result: "ScoreResult") -> None:
        key = self._key(prompt)
        with self._lock:
            self._store[key] = result

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


# ---------------------------------------------------------------------------
# Global opt-in cache
# ---------------------------------------------------------------------------

_global_cache: EvalJudgeCache | None = None


def enable_judge_cache() -> None:
    """Enable global judge cache for the current process."""
    global _global_cache
    _global_cache = EvalJudgeCache()


def clear_judge_cache() -> None:
    """Clear the global judge cache (e.g. between test runs)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


def get_global_cache() -> EvalJudgeCache | None:
    return _global_cache
