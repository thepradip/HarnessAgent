"""Tests for harness.memory.code_entity_extractor — three-tier extraction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from harness.memory.code_entity_extractor import (
    extract_code_entities,
    extract_code_identifiers,
    extract_from_regex,
)

# ===========================================================================
# Tier 1: precise regex
# ===========================================================================

def test_backticked_names():
    assert extract_code_identifiers("what does `LLMRouter.complete` do?") == [
        "LLMRouter.complete"
    ]


def test_file_paths():
    entities = extract_code_identifiers("open src/harness/memory/graph.py please")
    assert "src/harness/memory/graph.py" in entities


def test_dotted_paths():
    entities = extract_code_identifiers("who calls HaasLLM.chat here")
    assert "HaasLLM.chat" in entities


def test_camel_case():
    entities = extract_code_identifiers("where is CodeGraphIndexer used")
    assert "CodeGraphIndexer" in entities


def test_snake_case():
    entities = extract_code_identifiers("find extract_code_facts and its tests")
    assert "extract_code_facts" in entities


def test_call_syntax():
    entities = extract_code_identifiers("why does validate() fail sometimes")
    assert "validate" in entities


def test_plain_english_yields_nothing_tier1():
    assert extract_code_identifiers("why is the app slow on startup") == []


def test_dedupe_and_stopwords():
    entities = extract_code_identifiers(
        "`run_query` calls run_query which calls run_query"
    )
    assert entities == ["run_query"]


# ===========================================================================
# Tier 2: LLM (mocked)
# ===========================================================================

async def test_llm_tier_used_for_plain_english():
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=MagicMock(content=json.dumps(["LLMRouter", "retry_call"]))
    )
    entities = await extract_code_entities(
        "where do we retry failed provider requests", llm_provider=provider
    )
    assert entities == ["LLMRouter", "retry_call"]
    provider.complete.assert_awaited_once()


async def test_llm_tier_skipped_when_tier1_matches():
    provider = MagicMock()
    provider.complete = AsyncMock()
    entities = await extract_code_entities(
        "who calls `HaasLLM.chat`", llm_provider=provider
    )
    assert entities == ["HaasLLM.chat"]
    provider.complete.assert_not_awaited()


async def test_llm_failure_falls_back_to_regex():
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
    entities = await extract_code_entities(
        "where do we retry failed provider requests", llm_provider=provider
    )
    # Tier 3 fallback: stopword-filtered identifiers
    assert "retry" in entities
    assert "where" not in entities


async def test_llm_markdown_fences_stripped():
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=MagicMock(content='```json\n["Router"]\n```')
    )
    entities = await extract_code_entities("something vague", llm_provider=provider)
    assert entities == ["Router"]


# ===========================================================================
# Tier 3: fallback regex
# ===========================================================================

def test_fallback_filters_stopwords():
    entities = extract_from_regex("how does the parser work with files")
    assert "parser" in entities
    assert "how" not in entities
    assert "files" not in entities


async def test_no_provider_plain_english_uses_fallback():
    entities = await extract_code_entities("why is startup slow")
    assert "startup" in entities
    assert "slow" in entities
