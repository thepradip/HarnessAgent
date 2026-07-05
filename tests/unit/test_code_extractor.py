"""Tests for harness.ingestion.code_extractor — Python fact extraction."""

from __future__ import annotations

import pytest

from harness.ingestion.code_extractor import (
    extract_code_facts,
    module_name_from_path,
    tree_sitter_available,
)
from harness.ingestion.code_loader import SourceFile


def _source(content: str, path: str = "app/sample.py") -> SourceFile:
    return SourceFile(
        path=path,
        abs_path=f"/repo/{path}",
        language="python",
        content=content,
        content_hash="deadbeef",
        size_bytes=len(content),
        line_count=content.count("\n") + 1,
    )


_SAMPLE = '''"""Sample module docstring."""

import os
import json as j
from pathlib import Path
from . import sibling
from ..pkg import thing


def top(a: int, b: str = "x", *args, key: bool = False, **kw) -> bool:
    """Top-level function."""
    inner_result = os.getenv("HOME")
    return bool(inner_result)


class Base:
    """Base class."""

    def save(self) -> None:
        """Persist."""
        top(1)


class User(Base, dict):
    """A user."""

    async def rename(self, new: str) -> str:
        self.save()
        return new

    def _sig_test(self, x, /, y=1, *, z: int = 2):
        pass
'''


# ===========================================================================
# module_name_from_path
# ===========================================================================

def test_module_name_simple():
    assert module_name_from_path("app/utils.py") == "app.utils"


def test_module_name_init_collapses_to_package():
    assert module_name_from_path("app/__init__.py") == "app"


def test_module_name_src_layout():
    assert module_name_from_path("src/harness/memory/graph.py") == "src.harness.memory.graph"


def test_module_name_pyi():
    assert module_name_from_path("app/types.pyi") == "app.types"


# ===========================================================================
# Symbols
# ===========================================================================

def test_module_docstring_extracted():
    facts = extract_code_facts(_source(_SAMPLE))
    assert facts.docstring == "Sample module docstring."
    assert facts.parse_error is None
    assert facts.module_name == "app.sample"


def test_symbols_and_kinds():
    facts = extract_code_facts(_source(_SAMPLE))
    by_name = {s.qualified_name: s for s in facts.symbols}
    assert by_name["top"].kind == "function"
    assert by_name["Base"].kind == "class"
    assert by_name["Base.save"].kind == "method"
    assert by_name["User"].kind == "class"
    assert by_name["User.rename"].kind == "method"
    assert by_name["Base.save"].parent == "Base"
    assert by_name["top"].parent == ""


def test_function_signature_full():
    facts = extract_code_facts(_source(_SAMPLE))
    top = next(s for s in facts.symbols if s.qualified_name == "top")
    assert top.signature == (
        'def top(a: int, b: str = "x", *args, key: bool = False, **kw) -> bool'
    ) or top.signature == (
        "def top(a: int, b: str = 'x', *args, key: bool = False, **kw) -> bool"
    )


def test_async_and_posonly_signatures():
    facts = extract_code_facts(_source(_SAMPLE))
    rename = next(s for s in facts.symbols if s.qualified_name == "User.rename")
    assert rename.signature == "async def rename(self, new: str) -> str"
    sig_test = next(s for s in facts.symbols if s.qualified_name == "User._sig_test")
    assert sig_test.signature == "def _sig_test(self, x, /, y=1, *, z: int = 2)"


def test_class_signature_with_bases():
    facts = extract_code_facts(_source(_SAMPLE))
    user = next(s for s in facts.symbols if s.qualified_name == "User")
    assert user.signature == "class User(Base, dict)"


def test_docstrings_first_line_only():
    facts = extract_code_facts(_source(_SAMPLE))
    save = next(s for s in facts.symbols if s.qualified_name == "Base.save")
    assert save.docstring == "Persist."


def test_line_ranges():
    facts = extract_code_facts(_source(_SAMPLE))
    base = next(s for s in facts.symbols if s.qualified_name == "Base")
    assert base.line < base.end_line
    assert _SAMPLE.splitlines()[base.line - 1].startswith("class Base")


# ===========================================================================
# Imports
# ===========================================================================

def test_imports_plain_aliased_from_and_relative():
    facts = extract_code_facts(_source(_SAMPLE))
    imports = {(i.module, tuple(i.names), i.alias, i.level) for i in facts.imports}
    assert ("os", (), None, 0) in imports
    assert ("json", (), "j", 0) in imports
    assert ("pathlib", ("Path",), None, 0) in imports
    assert ("", ("sibling",), None, 1) in imports       # from . import sibling
    assert ("pkg", ("thing",), None, 2) in imports      # from ..pkg import thing


# ===========================================================================
# Inheritance & calls
# ===========================================================================

def test_inheritance_edges():
    facts = extract_code_facts(_source(_SAMPLE))
    pairs = {(i.class_name, i.base_name) for i in facts.inherits}
    assert ("User", "Base") in pairs
    assert ("User", "dict") in pairs


def test_calls_collected_with_callers():
    facts = extract_code_facts(_source(_SAMPLE))
    pairs = {(c.caller, c.callee_name) for c in facts.calls}
    assert ("top", "os.getenv") in pairs
    assert ("top", "bool") in pairs
    assert ("Base.save", "top") in pairs
    assert ("User.rename", "self.save") in pairs


def test_calls_deduped():
    content = "def f():\n    g()\n    g()\n    g()\n\ndef g():\n    pass\n"
    facts = extract_code_facts(_source(content))
    assert len([c for c in facts.calls if c.callee_name == "g"]) == 1


# ===========================================================================
# Error handling
# ===========================================================================

def test_syntax_error_sets_parse_error():
    facts = extract_code_facts(_source("def broken(:\n"))
    assert facts.parse_error is not None
    assert "SyntaxError" in facts.parse_error
    assert facts.symbols == []


def test_unknown_language_skipped():
    sf = SourceFile(
        path="a.go", abs_path="/r/a.go", language="go",
        content="package main", content_hash="h", size_bytes=12, line_count=1,
    )
    facts = extract_code_facts(sf)
    assert facts.parse_error is not None
    assert facts.symbols == []


def test_invalid_parser_choice():
    with pytest.raises(ValueError, match="Unknown parser"):
        extract_code_facts(_source("x = 1"), parser="magic")


def test_tree_sitter_forced_without_install():
    if tree_sitter_available("python"):
        pytest.skip("tree-sitter installed; forced-import error not reachable")
    with pytest.raises(ImportError, match="code-graph"):
        extract_code_facts(_source("x = 1"), parser="tree-sitter")


# ===========================================================================
# tree-sitter parity (runs only when agent-haas[code-graph] is installed)
# ===========================================================================

@pytest.mark.skipif(
    not tree_sitter_available("python"), reason="tree-sitter not installed"
)
def test_tree_sitter_parity_with_ast():
    ast_facts = extract_code_facts(_source(_SAMPLE), parser="ast")
    ts_facts = extract_code_facts(_source(_SAMPLE), parser="tree-sitter")

    ast_symbols = {(s.qualified_name, s.kind) for s in ast_facts.symbols}
    ts_symbols = {(s.qualified_name, s.kind) for s in ts_facts.symbols}
    assert ast_symbols == ts_symbols

    ast_inherits = {(i.class_name, i.base_name) for i in ast_facts.inherits}
    ts_inherits = {(i.class_name, i.base_name) for i in ts_facts.inherits}
    assert ast_inherits == ts_inherits

    ast_calls = {(c.caller, c.callee_name) for c in ast_facts.calls}
    ts_calls = {(c.caller, c.callee_name) for c in ts_facts.calls}
    # tree-sitter must find at least the same core call edges
    assert {("Base.save", "top"), ("User.rename", "self.save")} <= ts_calls
    assert {("Base.save", "top"), ("User.rename", "self.save")} <= ast_calls
