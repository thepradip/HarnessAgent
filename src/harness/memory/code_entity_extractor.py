"""Code-identifier extraction for CodeGraphRAG — three-tier strategy.

Tier 1 (best):   high-precision regex — backticked names, file paths, dotted
                 paths, CamelCase and snake_case identifiers. Code names have
                 much more surface signal than SQL entities, so a careful
                 regex tier resolves most queries with zero cost.

Tier 2 (good):   LLM extraction — for fuzzy natural-language questions
                 ("where do we retry failed provider calls?"). Tiny prompt,
                 cheapest available model. Skipped if no provider configured.

Tier 3 (fallback): generic identifier regex with stopword filtering.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Words that are never code entities in a question about code.
_STOP = {
    "the", "and", "for", "are", "not", "that", "this", "with", "from",
    "have", "all", "was", "been", "can", "will", "how", "what", "where",
    "which", "when", "who", "why", "but", "does", "did", "into", "about",
    "list", "get", "show", "give", "find", "return", "returns", "use",
    "used", "uses", "using", "make", "makes", "made", "work", "works",
    "code", "codebase", "file", "files", "function", "functions", "method",
    "methods", "class", "classes", "module", "modules", "import", "imports",
    "call", "calls", "called", "caller", "callers", "define", "defined",
    "definition", "implement", "implements", "implementation", "inherit",
    "inherits", "subclass", "test", "tests", "bug", "error", "errors",
    "fix", "run", "add", "new", "change", "changed", "look", "line",
    "there", "here", "python", "repo", "repository", "graph", "any",
}

# Tier-1 patterns, in descending confidence order.
_BACKTICK_RE = re.compile(r"`([^`]{2,80})`")
_PATH_RE = re.compile(r"\b([\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|go|java|rs))\b")
_DOTTED_RE = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b")
_CAMEL_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b")
_SNAKE_RE = re.compile(r"\b([a-z_][a-z0-9]*_[a-z0-9_]+)\b")
_CALL_RE = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")

# Tier-3 fallback.
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")


def _dedupe(entities: list[str]) -> list[str]:
    """Dedupe (case-insensitive), drop stopwords, and drop substrings.

    Substring suppression keeps "HaasLLM.chat" but drops the "HaasLLM" the
    CamelCase pattern would also emit — the dotted form already anchors the
    class via fuzzy matching, so the fragment is pure noise.
    """
    seen: set[str] = set()
    out: list[str] = []
    for entity in entities:
        key = entity.lower()
        if key in seen or key in _STOP:
            continue
        if any(key != kept.lower() and key in kept.lower() for kept in out):
            continue
        seen.add(key)
        out.append(entity)
    return out


def extract_code_identifiers(query: str) -> list[str]:
    """Tier 1: pull code-shaped tokens out of *query* with precise regexes.

    Ordered by confidence: backticked > file paths > dotted paths >
    call-syntax names > CamelCase > snake_case.
    """
    entities: list[str] = []
    entities.extend(m.group(1) for m in _BACKTICK_RE.finditer(query))
    entities.extend(m.group(1) for m in _PATH_RE.finditer(query))
    entities.extend(m.group(1) for m in _DOTTED_RE.finditer(query))
    entities.extend(m.group(1) for m in _CALL_RE.finditer(query))
    entities.extend(m.group(1) for m in _CAMEL_RE.finditer(query))
    entities.extend(m.group(1) for m in _SNAKE_RE.finditer(query))
    return _dedupe(entities)


async def extract_from_nl(query: str, llm_provider: Any | None) -> list[str]:
    """Tier 2: ask the LLM for code symbol names in a natural-language query.

    Returns an empty list if the LLM is unavailable or fails — never raises.
    """
    if llm_provider is None:
        return []

    prompt = (
        "Extract code symbol names (function names, class names, method names, "
        "module or file names) mentioned or implied in this question about a "
        "codebase. Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Question: {query}\n\nSymbols:"
    )
    try:
        response = await llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        content = response.content.strip()
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE
        ).strip()
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return _dedupe([str(e) for e in parsed if e])
    except Exception as exc:
        logger.debug("LLM code entity extraction failed: %s", exc)
    return []


def extract_from_regex(query: str) -> list[str]:
    """Tier 3: generic identifier fallback, stopword-filtered. Noisy but total."""
    return _dedupe([m.group(1) for m in _IDENTIFIER_RE.finditer(query)])


async def extract_code_entities(
    query: str,
    llm_provider: Any | None = None,
) -> list[str]:
    """Main entry point. Precise regex first, LLM second, generic regex last.

    Args:
        query:        The user question (natural language, possibly containing
                      code identifiers, paths, or backticked names).
        llm_provider: Optional LLM provider for fuzzy NL extraction.

    Returns:
        Deduplicated list of candidate symbol/file names in confidence order.
    """
    code_entities = extract_code_identifiers(query)
    if code_entities:
        logger.debug("Code entity extraction via regex tier-1: %s", code_entities)
        return code_entities

    if llm_provider is not None:
        nl_entities = await extract_from_nl(query, llm_provider)
        if nl_entities:
            logger.debug("Code entity extraction via LLM: %s", nl_entities)
            return nl_entities

    fallback = extract_from_regex(query)
    logger.debug("Code entity extraction via regex fallback: %s", fallback)
    return fallback
