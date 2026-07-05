"""Structural code-fact extraction for the code knowledge graph.

Two-tier strategy, mirroring the harness entity-extraction philosophy:

Tier 1 (Python):        stdlib ``ast`` parsing — always available, exact
                        signatures, docstrings, and line numbers.

Tier 2 (tree-sitter):   optional ``agent-haas[code-graph]`` extra. Used when
                        explicitly requested (``parser="tree-sitter"``) or for
                        non-Python languages once their grammar is installed.

The extractor is purely structural — no LLM calls. It produces symbols
(classes, functions, methods) and relations (imports, calls, inheritance)
that :class:`harness.memory.code_graph.CodeGraphIndexer` loads into the
knowledge graph.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from harness.ingestion.code_loader import SourceFile

logger = logging.getLogger(__name__)

_DOCSTRING_MAX_CHARS = 200


# ---------------------------------------------------------------------------
# Fact dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CodeSymbol:
    """A definition found in a source file (class, function, or method)."""

    name: str
    qualified_name: str          # e.g. "HaasLLM.chat" (module-relative)
    kind: str                    # "class" | "function" | "method"
    file_path: str               # repo-relative path
    line: int
    end_line: int
    signature: str               # e.g. "def chat(self, prompt: str) -> str"
    docstring: str = ""          # first line, capped at 200 chars
    parent: str = ""             # qualified name of enclosing class/function


@dataclass
class CodeImport:
    """An import statement in a source file."""

    module: str                  # imported module as written (may be relative)
    names: list[str] = field(default_factory=list)  # from-imports; [] for plain
    alias: str | None = None
    line: int = 0
    level: int = 0               # relative-import level (from ..x import y → 2)


@dataclass
class CodeCall:
    """A call site: *caller* invokes *callee_name* (raw dotted name)."""

    caller: str                  # qualified name of enclosing symbol ("" = module level)
    callee_name: str             # as written: "foo", "self.bar", "mod.Class.method"
    line: int = 0


@dataclass
class CodeInheritance:
    """A class inheriting from a base (raw dotted name)."""

    class_name: str              # qualified name of the subclass
    base_name: str               # as written: "BaseAgent", "abc.ABC"
    line: int = 0


@dataclass
class CodeFileFacts:
    """All structural facts extracted from one source file."""

    file_path: str
    language: str
    module_name: str             # dotted module path, e.g. "harness.memory.graph"
    docstring: str = ""
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[CodeImport] = field(default_factory=list)
    calls: list[CodeCall] = field(default_factory=list)
    inherits: list[CodeInheritance] = field(default_factory=list)
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def module_name_from_path(rel_path: str) -> str:
    """Derive a dotted module name from a repo-relative path.

    ``src/harness/memory/graph.py`` → ``src.harness.memory.graph``
    ``pkg/__init__.py``             → ``pkg``
    """
    parts = rel_path.split("/")
    last = parts[-1]
    for suffix in (".py", ".pyi"):
        if last.endswith(suffix):
            last = last[: -len(suffix)]
            break
    else:
        last = last.rsplit(".", 1)[0]
    if last == "__init__":
        parts = parts[:-1]
    else:
        parts[-1] = last
    return ".".join(p for p in parts if p)


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    line = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return line[:_DOCSTRING_MAX_CHARS]


def _dotted_name(node: ast.expr) -> str | None:
    """Render a Name/Attribute chain as a dotted string, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _annotation(node: ast.expr | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover — unparse is total on valid ASTs
        return ""


def _format_arguments(args: ast.arguments) -> str:
    """Render an ast.arguments node as a readable parameter list."""
    parts: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    pad = len(positional) - len(defaults)

    for i, arg in enumerate(positional):
        piece = arg.arg
        ann = _annotation(arg.annotation)
        if ann:
            piece += f": {ann}"
        if i >= pad:
            default = _annotation(defaults[i - pad])
            piece += f" = {default}" if ann else f"={default}"
        parts.append(piece)
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg is not None:
        piece = f"*{args.vararg.arg}"
        ann = _annotation(args.vararg.annotation)
        if ann:
            piece += f": {ann}"
        parts.append(piece)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        piece = arg.arg
        ann = _annotation(arg.annotation)
        if ann:
            piece += f": {ann}"
        if default_node is not None:
            default = _annotation(default_node)
            piece += f" = {default}" if ann else f"={default}"
        parts.append(piece)

    if args.kwarg is not None:
        piece = f"**{args.kwarg.arg}"
        ann = _annotation(args.kwarg.annotation)
        if ann:
            piece += f": {ann}"
        parts.append(piece)

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Tier 1: Python via stdlib ast
# ---------------------------------------------------------------------------


class _PythonFactVisitor(ast.NodeVisitor):
    """Collects symbols, calls, imports, and inheritance from a Python AST."""

    def __init__(self, facts: CodeFileFacts) -> None:
        self._facts = facts
        # Stack of (qualified_name, kind) for enclosing scopes.
        self._scope: list[tuple[str, str]] = []

    # -- scope helpers --------------------------------------------------

    def _qualify(self, name: str) -> str:
        if not self._scope:
            return name
        return f"{self._scope[-1][0]}.{name}"

    def _enclosing_symbol(self) -> str:
        return self._scope[-1][0] if self._scope else ""

    def _in_class(self) -> bool:
        return bool(self._scope) and self._scope[-1][1] == "class"

    # -- imports ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._facts.imports.append(
                CodeImport(
                    module=alias.name,
                    names=[],
                    alias=alias.asname,
                    line=node.lineno,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._facts.imports.append(
            CodeImport(
                module=node.module or "",
                names=[alias.name for alias in node.names],
                alias=None,
                line=node.lineno,
                level=node.level,
            )
        )

    # -- definitions -----------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualify(node.name)
        bases = [b for b in (_dotted_name(base) for base in node.bases) if b]
        base_str = f"({', '.join(bases)})" if bases else ""
        self._facts.symbols.append(
            CodeSymbol(
                name=node.name,
                qualified_name=qualified,
                kind="class",
                file_path=self._facts.file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=f"class {node.name}{base_str}",
                docstring=_first_line(ast.get_docstring(node)),
                parent=self._enclosing_symbol(),
            )
        )
        for base in bases:
            self._facts.inherits.append(
                CodeInheritance(
                    class_name=qualified, base_name=base, line=node.lineno
                )
            )
        self._scope.append((qualified, "class"))
        self.generic_visit(node)
        self._scope.pop()

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str
    ) -> None:
        qualified = self._qualify(node.name)
        kind = "method" if self._in_class() else "function"
        returns = _annotation(node.returns)
        signature = f"{prefix} {node.name}({_format_arguments(node.args)})"
        if returns:
            signature += f" -> {returns}"
        self._facts.symbols.append(
            CodeSymbol(
                name=node.name,
                qualified_name=qualified,
                kind=kind,
                file_path=self._facts.file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=signature,
                docstring=_first_line(ast.get_docstring(node)),
                parent=self._enclosing_symbol(),
            )
        )
        self._scope.append((qualified, kind))
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "def")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "async def")

    # -- calls -------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        callee = _dotted_name(node.func)
        if callee:
            self._facts.calls.append(
                CodeCall(
                    caller=self._enclosing_symbol(),
                    callee_name=callee,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


def _extract_python_ast(source: SourceFile) -> CodeFileFacts:
    """Extract facts from a Python file using the stdlib ast module."""
    facts = CodeFileFacts(
        file_path=source.path,
        language="python",
        module_name=module_name_from_path(source.path),
    )
    try:
        tree = ast.parse(source.content, filename=source.path)
    except SyntaxError as exc:
        facts.parse_error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        logger.warning("Code extractor: %s failed to parse: %s", source.path, exc)
        return facts

    facts.docstring = _first_line(ast.get_docstring(tree))
    _PythonFactVisitor(facts).visit(tree)
    _dedupe_calls(facts)
    return facts


def _dedupe_calls(facts: CodeFileFacts) -> None:
    """Collapse repeated (caller, callee) pairs, keeping the first line."""
    seen: set[tuple[str, str]] = set()
    unique: list[CodeCall] = []
    for call in facts.calls:
        key = (call.caller, call.callee_name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    facts.calls = unique


# ---------------------------------------------------------------------------
# Tier 2: tree-sitter (optional extra: agent-haas[code-graph])
# ---------------------------------------------------------------------------


def tree_sitter_available(language: str) -> bool:
    """True when tree-sitter and a grammar for *language* are importable."""
    if language != "python":
        return False  # grammars for other languages are not bundled yet
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
    except ImportError:
        return False
    return True


def _extract_python_tree_sitter(source: SourceFile) -> CodeFileFacts:
    """Extract facts from a Python file using tree-sitter.

    Kept behaviourally aligned with :func:`_extract_python_ast`; the ast tier
    remains authoritative for Python. This path exists so multi-language
    support can grow behind the same interface.
    """
    import tree_sitter
    import tree_sitter_python

    facts = CodeFileFacts(
        file_path=source.path,
        language="python",
        module_name=module_name_from_path(source.path),
    )
    parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))
    tree = parser.parse(source.content.encode("utf-8"))
    root = tree.root_node
    if root.has_error:
        # tree-sitter is error-tolerant; note it but keep what parsed.
        facts.parse_error = "tree-sitter reported syntax errors"

    def text(node: object) -> str:
        return node.text.decode("utf-8") if node is not None else ""  # type: ignore[attr-defined]

    def docstring_of(body: object) -> str:
        if body is None:
            return ""
        for child in body.named_children:  # type: ignore[attr-defined]
            if child.type == "expression_statement" and child.named_children:
                first = child.named_children[0]
                if first.type == "string":
                    raw = text(first)
                    stripped = raw.strip("\"'")
                    return _first_line(stripped)
            break
        return ""

    facts.docstring = docstring_of(root)

    def walk(node: object, scope: list[tuple[str, str]]) -> None:
        for child in node.named_children:  # type: ignore[attr-defined]
            node_type = child.type
            target = child
            if node_type == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is None:
                    continue
                target = inner
                node_type = inner.type

            if node_type == "class_definition":
                name = text(target.child_by_field_name("name"))
                qualified = f"{scope[-1][0]}.{name}" if scope else name
                bases: list[str] = []
                superclasses = target.child_by_field_name("superclasses")
                if superclasses is not None:
                    for arg in superclasses.named_children:
                        if arg.type in ("identifier", "attribute"):
                            bases.append(text(arg))
                base_str = f"({', '.join(bases)})" if bases else ""
                facts.symbols.append(
                    CodeSymbol(
                        name=name,
                        qualified_name=qualified,
                        kind="class",
                        file_path=source.path,
                        line=target.start_point[0] + 1,
                        end_line=target.end_point[0] + 1,
                        signature=f"class {name}{base_str}",
                        docstring=docstring_of(target.child_by_field_name("body")),
                        parent=scope[-1][0] if scope else "",
                    )
                )
                for base in bases:
                    facts.inherits.append(
                        CodeInheritance(
                            class_name=qualified,
                            base_name=base,
                            line=target.start_point[0] + 1,
                        )
                    )
                walk(
                    target.child_by_field_name("body") or target,
                    [*scope, (qualified, "class")],
                )

            elif node_type == "function_definition":
                name = text(target.child_by_field_name("name"))
                qualified = f"{scope[-1][0]}.{name}" if scope else name
                kind = "method" if scope and scope[-1][1] == "class" else "function"
                params = text(target.child_by_field_name("parameters"))
                return_type = text(target.child_by_field_name("return_type"))
                is_async = any(c.type == "async" for c in target.children)
                prefix = "async def" if is_async else "def"
                inner_params = params[1:-1] if params.startswith("(") else params
                signature = f"{prefix} {name}({inner_params})"
                if return_type:
                    signature += f" -> {return_type}"
                facts.symbols.append(
                    CodeSymbol(
                        name=name,
                        qualified_name=qualified,
                        kind=kind,
                        file_path=source.path,
                        line=target.start_point[0] + 1,
                        end_line=target.end_point[0] + 1,
                        signature=signature,
                        docstring=docstring_of(target.child_by_field_name("body")),
                        parent=scope[-1][0] if scope else "",
                    )
                )
                walk(
                    target.child_by_field_name("body") or target,
                    [*scope, (qualified, kind)],
                )

            elif node_type == "import_statement":
                for item in child.named_children:
                    if item.type == "dotted_name":
                        facts.imports.append(
                            CodeImport(
                                module=text(item),
                                line=child.start_point[0] + 1,
                            )
                        )
                    elif item.type == "aliased_import":
                        module_node = item.child_by_field_name("name")
                        alias_node = item.child_by_field_name("alias")
                        facts.imports.append(
                            CodeImport(
                                module=text(module_node),
                                alias=text(alias_node) or None,
                                line=child.start_point[0] + 1,
                            )
                        )

            elif node_type == "import_from_statement":
                module_node = child.child_by_field_name("module_name")
                module_text = text(module_node)
                level = len(module_text) - len(module_text.lstrip("."))
                names = [
                    text(item)
                    for item in child.named_children[1:]
                    if item.type in ("dotted_name", "identifier", "aliased_import")
                ]
                facts.imports.append(
                    CodeImport(
                        module=module_text.lstrip("."),
                        names=names,
                        line=child.start_point[0] + 1,
                        level=level,
                    )
                )

            elif node_type == "call":
                fn = child.child_by_field_name("function")
                if fn is not None and fn.type in ("identifier", "attribute"):
                    facts.calls.append(
                        CodeCall(
                            caller=scope[-1][0] if scope else "",
                            callee_name=text(fn),
                            line=child.start_point[0] + 1,
                        )
                    )
                walk(child, scope)

            else:
                walk(child, scope)

    walk(root, [])
    _dedupe_calls(facts)
    return facts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_code_facts(source: SourceFile, parser: str = "auto") -> CodeFileFacts:
    """Extract structural facts from *source*.

    Args:
        source: A SourceFile from :func:`harness.ingestion.code_loader.load_source_files`.
        parser: ``"auto"`` (default — stdlib ast for Python),
                ``"ast"`` (force stdlib ast, Python only), or
                ``"tree-sitter"`` (force tree-sitter; needs
                ``pip install agent-haas[code-graph]``).

    Returns:
        CodeFileFacts. When the file cannot be parsed, ``parse_error`` is set
        and the fact lists are empty — the indexer records the failure instead
        of aborting the whole run.
    """
    if parser not in ("auto", "ast", "tree-sitter"):
        raise ValueError(
            f"Unknown parser '{parser}'. Valid choices: 'auto', 'ast', 'tree-sitter'."
        )

    if source.language == "python":
        if parser == "tree-sitter":
            if not tree_sitter_available("python"):
                raise ImportError(
                    "tree-sitter is required for parser='tree-sitter'. "
                    "Install with: pip install agent-haas[code-graph]"
                )
            return _extract_python_tree_sitter(source)
        return _extract_python_ast(source)

    # Non-Python languages need a tree-sitter grammar (not bundled yet).
    logger.debug(
        "Code extractor: no parser for language '%s' (%s) — skipping",
        source.language,
        source.path,
    )
    return CodeFileFacts(
        file_path=source.path,
        language=source.language,
        module_name=module_name_from_path(source.path),
        parse_error=f"No parser available for language '{source.language}'",
    )
