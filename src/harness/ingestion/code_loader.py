"""Source-file loader for the code knowledge graph.

Walks a repository, filters out vendored / generated directories, detects the
language from the file extension, and returns each source file with a stable
content hash so the indexer can skip unchanged files on re-index.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories that are never useful in a code graph (vendored, generated,
# caches, virtualenvs). Matched against every path component.
_DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "site-packages",
        "dist",
        "build",
        ".eggs",
        "htmlcov",
        ".idea",
        ".vscode",
    }
)

# Extension → language map. Only languages listed here are loaded.
_EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

# Files larger than this are skipped — almost always generated/minified code
# that would bloat the graph without adding structure.
_DEFAULT_MAX_FILE_BYTES = 1_048_576  # 1 MB


@dataclass
class SourceFile:
    """A single source file loaded for code-graph extraction.

    Attributes:
        path:         Path relative to the repository root (POSIX separators).
        abs_path:     Absolute filesystem path.
        language:     Detected language (e.g. "python").
        content:      Full file text.
        content_hash: SHA-256 hex digest of the raw bytes — used by the
                      indexer to detect unchanged files on re-index.
        size_bytes:   File size in bytes.
        line_count:   Number of lines in the file.
    """

    path: str
    abs_path: str
    language: str
    content: str
    content_hash: str
    size_bytes: int
    line_count: int


def detect_language(path: Path) -> str | None:
    """Return the language for *path* based on its extension, or None."""
    return _EXTENSION_LANGUAGE.get(path.suffix.lower())


def _is_excluded(rel_parts: tuple[str, ...], exclude_dirs: frozenset[str]) -> bool:
    """True when any path component is an excluded directory name."""
    return any(part in exclude_dirs for part in rel_parts[:-1])


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source_files(
    root: Path | str,
    languages: set[str] | None = None,
    extra_exclude_dirs: set[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> list[SourceFile]:
    """Walk *root* and load every recognised source file.

    Args:
        root:               Repository root directory.
        languages:          Restrict to these languages (default: all known).
        extra_exclude_dirs: Directory names to exclude in addition to the
                            built-in list (.git, node_modules, __pycache__, …).
        exclude_globs:      fnmatch patterns applied to the relative path
                            (e.g. ``["*_pb2.py", "migrations/*"]``).
        max_file_bytes:     Skip files larger than this (default 1 MB).

    Returns:
        List of SourceFile objects sorted by relative path.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"Code loader root is not a directory: {root_path}")

    exclude_dirs = _DEFAULT_EXCLUDE_DIRS | frozenset(extra_exclude_dirs or ())
    patterns = exclude_globs or []
    files: list[SourceFile] = []
    skipped_large = 0
    skipped_binary = 0

    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue

        language = detect_language(path)
        if language is None:
            continue
        if languages is not None and language not in languages:
            continue

        rel = path.relative_to(root_path)
        rel_posix = rel.as_posix()
        if _is_excluded(rel.parts, exclude_dirs):
            continue
        if any(fnmatch.fnmatch(rel_posix, pattern) for pattern in patterns):
            continue

        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("Code loader could not read %s: %s", rel_posix, exc)
            continue

        if len(raw) > max_file_bytes:
            skipped_large += 1
            logger.debug(
                "Skipping %s: %d bytes exceeds max_file_bytes=%d",
                rel_posix,
                len(raw),
                max_file_bytes,
            )
            continue

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary += 1
            logger.debug("Skipping %s: not valid UTF-8", rel_posix)
            continue

        files.append(
            SourceFile(
                path=rel_posix,
                abs_path=str(path),
                language=language,
                content=content,
                content_hash=_hash_bytes(raw),
                size_bytes=len(raw),
                line_count=content.count("\n") + 1 if content else 0,
            )
        )

    logger.info(
        "Code loader: %d source files under %s (%d skipped large, %d skipped binary)",
        len(files),
        root_path,
        skipped_large,
        skipped_binary,
    )
    return files
