"""Entity and relationship extraction for populating the knowledge graph."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    """A subject-predicate-object triple extracted from text.

    Attributes:
        subject:         The entity acting or being described.
        predicate:       The relationship or attribute.
        object_:         The target entity or value.
        confidence:      Confidence score in [0, 1].
        source_chunk_id: ID of the Chunk this fact was extracted from.
    """

    subject: str
    predicate: str
    object_: str
    confidence: float = 1.0
    source_chunk_id: str = ""


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are an information extraction expert. Extract structured knowledge triples "
    "from text as (subject, predicate, object) tuples. Focus on entities, relationships, "
    "and important attributes. Output only valid JSON."
)

_EXTRACT_PROMPT_TEMPLATE = """\
Extract all subject-predicate-object triples from the following text.

Text:
---
{text}
---

The text is split into numbered chunks marked "[Chunk N/M]".
Output a JSON array where each element has keys: "subject", "predicate", "object",
"confidence" (0.0-1.0), and "chunk" (the 1-based chunk number the fact came from).
Focus on:
- Named entities (people, organizations, products, technologies)
- Relationships (is_a, has, uses, belongs_to, created_by, located_in)
- Key facts and attributes

Example output:
[
  {{"subject": "Python", "predicate": "is_a", "object": "programming language", "confidence": 1.0, "chunk": 1}},
  {{"subject": "Guido van Rossum", "predicate": "created", "object": "Python", "confidence": 1.0, "chunk": 2}}
]

JSON array:"""


async def extract_facts(
    chunks: list[Chunk],
    llm_provider: Any,
    batch_size: int = 5,
) -> list[ExtractedFact]:
    """Extract subject-predicate-object triples from chunks using an LLM.

    Processes chunks in batches to reduce LLM calls.  Each batch combines
    multiple chunks with clear separators.  Parses the response as a JSON
    array of triple dicts.

    Args:
        chunks:       List of Chunk objects to extract from.
        llm_provider: An LLMProvider-compatible object.
        batch_size:   Number of chunks to combine per LLM call.

    Returns:
        List of ExtractedFact objects across all chunks.
    """
    all_facts: list[ExtractedFact] = []

    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        combined_text = "\n\n---\n\n".join(
            f"[Chunk {i + 1}/{len(batch)}]\n{c.content}" for i, c in enumerate(batch)
        )
        # Truncate to avoid context overflows (approx 8000 chars ≈ 2000 tokens)
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "\n...[truncated]"

        prompt = _EXTRACT_PROMPT_TEMPLATE.format(text=combined_text)
        try:
            response = await llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                system=_EXTRACT_SYSTEM,
            )
            raw = response.content.strip()

            # Strip markdown code fences
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                logger.warning("LLM extraction returned non-list; skipping batch")
                continue

            # Map chunk IDs to batch chunks for attribution
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                obj = str(item.get("object", "")).strip()
                if not subject or not predicate or not obj:
                    continue
                confidence = float(item.get("confidence", 1.0))
                confidence = max(0.0, min(1.0, confidence))

                # Attribute to the chunk the LLM cited (1-based index into the
                # batch). Fall back to the first chunk only when the index is
                # missing or out of range, instead of mis-attributing every fact
                # in the batch to batch[0].
                source_chunk_id = batch[0].chunk_id if batch else ""
                try:
                    chunk_idx = int(item.get("chunk", 0)) - 1
                    if batch and 0 <= chunk_idx < len(batch):
                        source_chunk_id = batch[chunk_idx].chunk_id
                except (TypeError, ValueError):
                    pass

                all_facts.append(
                    ExtractedFact(
                        subject=subject,
                        predicate=predicate,
                        object_=obj,
                        confidence=confidence,
                        source_chunk_id=source_chunk_id,
                    )
                )

        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse LLM extraction response for batch starting at %d: %s",
                batch_start,
                exc,
            )
        except Exception as exc:
            logger.error(
                "LLM extraction call failed for batch starting at %d: %s",
                batch_start,
                exc,
            )

    logger.debug("Extracted %d facts from %d chunks", len(all_facts), len(chunks))
    return all_facts


# ---------------------------------------------------------------------------
# Structural SQL schema extraction (no LLM required)
# ---------------------------------------------------------------------------


def extract_sql_schema_facts(tables_info: list[dict]) -> list[ExtractedFact]:
    """Extract structural knowledge triples from SQL schema metadata.

    Creates facts like:
    - (table_name, "has_column", column_name)
    - (table_name, "column_type", "col_name:type")
    - (table_name, "joins", other_table) when FK relationships are detected
    - (table_name, "has_primary_key", pk_column)

    Args:
        tables_info: List of dicts, each with keys:
            - "table_name" (str)
            - "columns" (list[dict] with "name", "type", optional "primary_key", "foreign_key")
            - Optional "foreign_keys" (list[dict] with "column", "references_table", "references_column")

    Returns:
        List of ExtractedFact objects.
    """
    facts: list[ExtractedFact] = []

    for table_info in tables_info:
        table_name = table_info.get("table_name", "")
        if not table_name:
            continue

        columns = table_info.get("columns", [])
        for col in columns:
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            if not col_name:
                continue

            facts.append(
                ExtractedFact(
                    subject=table_name,
                    predicate="has_column",
                    object_=col_name,
                    confidence=1.0,
                )
            )

            if col_type:
                facts.append(
                    ExtractedFact(
                        subject=table_name,
                        predicate="column_type",
                        object_=f"{col_name}:{col_type}",
                        confidence=1.0,
                    )
                )

            # Primary key detection
            if col.get("primary_key") or col.get("is_primary_key"):
                facts.append(
                    ExtractedFact(
                        subject=table_name,
                        predicate="has_primary_key",
                        object_=col_name,
                        confidence=1.0,
                    )
                )

        # Foreign key / join detection
        foreign_keys = table_info.get("foreign_keys", [])
        for fk in foreign_keys:
            fk_col = fk.get("column", "")
            ref_table = fk.get("references_table", "")
            ref_col = fk.get("references_column", "")
            if fk_col and ref_table:
                facts.append(
                    ExtractedFact(
                        subject=table_name,
                        predicate="joins",
                        object_=ref_table,
                        confidence=1.0,
                    )
                )
                if ref_col:
                    facts.append(
                        ExtractedFact(
                            subject=table_name,
                            predicate="foreign_key",
                            object_=f"{fk_col} -> {ref_table}.{ref_col}",
                            confidence=1.0,
                        )
                    )

        # Also check inline column foreign key hints
        for col in columns:
            col_fk = col.get("foreign_key", "")
            if col_fk and "." in col_fk:
                ref_parts = col_fk.split(".", 1)
                ref_table = ref_parts[0]
                facts.append(
                    ExtractedFact(
                        subject=table_name,
                        predicate="joins",
                        object_=ref_table,
                        confidence=0.9,
                    )
                )

    logger.debug(
        "Extracted %d schema facts from %d tables", len(facts), len(tables_info)
    )
    return facts
