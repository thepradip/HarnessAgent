"""Secret scanner — detect and redact leaked API keys in strings and dicts.

Used by the safety pipeline's output check to prevent credentials from
flowing into traces, logs, checkpoints, or LLM history.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns — ordered most-specific first to avoid partial matches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Pattern:
    name: str
    regex: re.Pattern[str]
    replacement: str


_PATTERNS: list[_Pattern] = [
    _Pattern("anthropic",    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),         "[ANTHROPIC_KEY REDACTED]"),
    _Pattern("openai",       re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}"), "[OPENAI_KEY REDACTED]"),
    _Pattern("github_pat",   re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),       "[GITHUB_PAT REDACTED]"),
    _Pattern("github_ghp",   re.compile(r"ghp_[A-Za-z0-9]{36,}"),              "[GITHUB_TOKEN REDACTED]"),
    _Pattern("slack_bot",    re.compile(r"xoxb-[0-9]+-[0-9]+-[A-Za-z0-9]+"),   "[SLACK_BOT_TOKEN REDACTED]"),
    _Pattern("slack_user",   re.compile(r"xoxp-[0-9]+-[0-9]+-[A-Za-z0-9]+"),   "[SLACK_USER_TOKEN REDACTED]"),
    _Pattern("gitlab_pat",   re.compile(r"glpat-[A-Za-z0-9\-_]{20,}"),         "[GITLAB_PAT REDACTED]"),
    _Pattern("url_creds",    re.compile(r"://[^:@\s/]{3,}:[^@\s]{8,}@"),       "://[CREDENTIALS REDACTED]@"),
    _Pattern("bearer_token", re.compile(r"(?i)Bearer\s+[A-Za-z0-9\-_\.]{20,}"), "Bearer [TOKEN REDACTED]"),
    _Pattern("jwt",          re.compile(r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"), "[JWT REDACTED]"),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecretMatch:
    """A detected secret in a scanned string."""
    pattern_name: str
    matched_text: str
    start: int
    end: int
    replacement: str


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class SecretScanner:
    """Detect and redact API keys and tokens in strings and dicts.

    Thread-safe (stateless after construction).
    """

    def __init__(self, extra_patterns: list[_Pattern] | None = None) -> None:
        self._patterns = list(_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def scan(self, text: str) -> list[SecretMatch]:
        """Return all secret matches found in ``text``."""
        matches: list[SecretMatch] = []
        for pat in self._patterns:
            for m in pat.regex.finditer(text):
                matches.append(SecretMatch(
                    pattern_name=pat.name,
                    matched_text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    replacement=pat.replacement,
                ))
        # Deduplicate overlapping matches — keep the one with the longest span
        matches.sort(key=lambda x: (x.start, -(x.end - x.start)))
        deduped: list[SecretMatch] = []
        last_end = -1
        for match in matches:
            if match.start >= last_end:
                deduped.append(match)
                last_end = match.end
        return deduped

    def redact(self, text: str) -> str:
        """Replace all detected secrets in ``text`` with redaction labels."""
        for pat in self._patterns:
            text = pat.regex.sub(pat.replacement, text)
        return text

    def scan_dict(self, d: Any, _path: str = "") -> list[SecretMatch]:
        """Recursively scan all string values in a dict/list structure."""
        results: list[SecretMatch] = []
        if isinstance(d, str):
            for match in self.scan(d):
                results.append(SecretMatch(
                    pattern_name=match.pattern_name,
                    matched_text=match.matched_text,
                    start=match.start,
                    end=match.end,
                    replacement=match.replacement,
                ))
        elif isinstance(d, dict):
            for key, value in d.items():
                results.extend(self.scan_dict(value, _path=f"{_path}.{key}"))
        elif isinstance(d, (list, tuple)):
            for i, item in enumerate(d):
                results.extend(self.scan_dict(item, _path=f"{_path}[{i}]"))
        return results

    def redact_dict(self, d: Any) -> Any:
        """Return a copy of ``d`` with all string values redacted.

        Preserves structure; non-string leaf values are unchanged.
        """
        if isinstance(d, str):
            return self.redact(d)
        if isinstance(d, dict):
            return {k: self.redact_dict(v) for k, v in d.items()}
        if isinstance(d, list):
            return [self.redact_dict(item) for item in d]
        if isinstance(d, tuple):
            return tuple(self.redact_dict(item) for item in d)
        return d

    def has_secrets(self, text: str) -> bool:
        """Return True if any secret pattern matches ``text``."""
        return any(pat.regex.search(text) for pat in self._patterns)


# ---------------------------------------------------------------------------
# Module-level singleton (zero-config for common use)
# ---------------------------------------------------------------------------

_default_scanner = SecretScanner()


def redact(text: str) -> str:
    """Redact secrets in ``text`` using the default scanner."""
    return _default_scanner.redact(text)


def scan(text: str) -> list[SecretMatch]:
    """Scan ``text`` for secrets using the default scanner."""
    return _default_scanner.scan(text)


def has_secrets(text: str) -> bool:
    """Return True if ``text`` contains any detectable secret."""
    return _default_scanner.has_secrets(text)
