"""Tests for harness.ingestion.code_loader — repo walking, hashing, filtering."""

from __future__ import annotations

import pytest

from harness.ingestion.code_loader import (
    SourceFile,
    detect_language,
    load_source_files,
)


def _write(root, rel_path: str, content: str = "x = 1\n") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


# ===========================================================================
# Language detection
# ===========================================================================

def test_detect_language_python(tmp_path):
    f = tmp_path / "a.py"
    assert detect_language(f) == "python"


def test_detect_language_typescript(tmp_path):
    assert detect_language(tmp_path / "a.tsx") == "typescript"


def test_detect_language_unknown(tmp_path):
    assert detect_language(tmp_path / "a.txt") is None
    assert detect_language(tmp_path / "Makefile") is None


# ===========================================================================
# Walking & filtering
# ===========================================================================

def test_load_finds_source_files(tmp_path):
    _write(tmp_path, "app/main.py", "def run():\n    pass\n")
    _write(tmp_path, "app/util.py")
    _write(tmp_path, "README.md", "# no code")
    files = load_source_files(tmp_path)
    paths = [f.path for f in files]
    assert paths == ["app/main.py", "app/util.py"]  # sorted, md excluded


def test_load_excludes_default_dirs(tmp_path):
    _write(tmp_path, "app/main.py")
    _write(tmp_path, ".venv/lib/x.py")
    _write(tmp_path, "node_modules/pkg/y.py")
    _write(tmp_path, "app/__pycache__/z.py")
    _write(tmp_path, ".git/hooks/h.py")
    files = load_source_files(tmp_path)
    assert [f.path for f in files] == ["app/main.py"]


def test_load_extra_exclude_dirs(tmp_path):
    _write(tmp_path, "app/main.py")
    _write(tmp_path, "migrations/0001.py")
    files = load_source_files(tmp_path, extra_exclude_dirs={"migrations"})
    assert [f.path for f in files] == ["app/main.py"]


def test_load_exclude_globs(tmp_path):
    _write(tmp_path, "app/main.py")
    _write(tmp_path, "app/schema_pb2.py")
    files = load_source_files(tmp_path, exclude_globs=["*_pb2.py"])
    assert [f.path for f in files] == ["app/main.py"]


def test_load_language_filter(tmp_path):
    _write(tmp_path, "a.py")
    _write(tmp_path, "b.ts", "const x = 1;\n")
    files = load_source_files(tmp_path, languages={"python"})
    assert [f.path for f in files] == ["a.py"]


def test_load_skips_large_files(tmp_path):
    _write(tmp_path, "big.py", "# " + "x" * 100)
    _write(tmp_path, "small.py")
    files = load_source_files(tmp_path, max_file_bytes=50)
    assert [f.path for f in files] == ["small.py"]


def test_load_skips_non_utf8(tmp_path):
    (tmp_path / "bin.py").write_bytes(b"\xff\xfe\x00bad")
    _write(tmp_path, "ok.py")
    files = load_source_files(tmp_path)
    assert [f.path for f in files] == ["ok.py"]


def test_load_root_not_a_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        load_source_files(tmp_path / "missing")


# ===========================================================================
# SourceFile fields
# ===========================================================================

def test_source_file_fields(tmp_path):
    content = "def a():\n    return 1\n"
    _write(tmp_path, "m.py", content)
    (sf,) = load_source_files(tmp_path)
    assert isinstance(sf, SourceFile)
    assert sf.language == "python"
    assert sf.content == content
    assert sf.size_bytes == len(content.encode())
    assert sf.line_count == 3  # two lines + trailing newline
    assert len(sf.content_hash) == 64  # sha256 hex


def test_content_hash_deterministic_and_change_sensitive(tmp_path):
    _write(tmp_path, "m.py", "x = 1\n")
    (first,) = load_source_files(tmp_path)
    (again,) = load_source_files(tmp_path)
    assert first.content_hash == again.content_hash

    _write(tmp_path, "m.py", "x = 2\n")
    (changed,) = load_source_files(tmp_path)
    assert changed.content_hash != first.content_hash
