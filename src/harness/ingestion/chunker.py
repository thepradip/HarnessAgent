"""Text chunker for HarnessAgent — recursive splitting with overlap."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A text chunk produced by the chunker.

    Attributes:
        chunk_id:       Unique identifier (UUID hex).
        doc_id:         ID of the parent Document.
        content:        The text content of this chunk.
        start_char:     Start character offset in the original document.
        end_char:       End character offset in the original document.
        metadata:       Inherited and augmented metadata dict.
        token_estimate: Approximate token count (len(content) // 4).
    """

    chunk_id: str
    doc_id: str
    content: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0 and self.content:
            self.token_estimate = len(self.content) // 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count as len(text) // 4."""
    return len(text) // 4


def _hard_split(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
    doc_id: str,
    base_offset: int,
    metadata: dict,
) -> list[Chunk]:
    """Last-resort character-level splitter.

    Splits text into chunks of chunk_size_chars with overlap_chars overlap.
    """
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        content = text[start:end]
        if content.strip():
            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4().hex,
                    doc_id=doc_id,
                    content=content,
                    start_char=base_offset + start,
                    end_char=base_offset + end,
                    metadata={**metadata},
                    token_estimate=_estimate_tokens(content),
                )
            )
        # Clamp the step to at least 1 char so a pathological config
        # (overlap_chars >= chunk_size_chars) cannot stall or, worse, discard
        # the remainder of the text by stepping backwards.
        step = max(1, chunk_size_chars - overlap_chars)
        start += step
    return chunks


def _recursive_split(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    doc_id: str,
    base_offset: int,
    metadata: dict,
    depth: int = 0,
) -> list[Chunk]:
    """Recursively split text into chunks using successively finer boundaries.

    Levels:
        0 → split on double newline  (paragraphs)
        1 → split on single newline  (lines)
        2 → split on ". "            (sentences)
        3 → hard character split

    Args:
        text:               The text to split.
        chunk_size_tokens:  Target chunk size in tokens.
        overlap_tokens:     Overlap in tokens between consecutive chunks.
        doc_id:             Parent document ID.
        base_offset:        Character offset of *text* within the original doc.
        metadata:           Metadata to attach to each chunk.
        depth:              Current recursion depth (0-3).

    Returns:
        List of Chunk objects.
    """
    chunk_size_chars = chunk_size_tokens * 4  # approximation
    overlap_chars = overlap_tokens * 4

    if not text.strip():
        return []

    if _estimate_tokens(text) <= chunk_size_tokens:
        return [
            Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=doc_id,
                content=text,
                start_char=base_offset,
                end_char=base_offset + len(text),
                metadata={**metadata},
                token_estimate=_estimate_tokens(text),
            )
        ]

    # Choose separator by depth
    separators = ["\n\n", "\n", ". ", ""]
    sep = separators[min(depth, len(separators) - 1)]

    if sep == "":
        # Hard split fallback
        return _hard_split(
            text, chunk_size_chars, overlap_chars, doc_id, base_offset, metadata
        )

    # Split text on separator, keeping an eye on offsets
    parts: list[tuple[str, int]] = []  # (text, offset_in_full_text)
    current_pos = 0
    if sep:
        while True:
            idx = text.find(sep, current_pos)
            if idx == -1:
                parts.append((text[current_pos:], current_pos))
                break
            parts.append((text[current_pos: idx + len(sep)], current_pos))
            current_pos = idx + len(sep)
    else:
        parts = [(text, 0)]

    # Merge parts into chunks respecting the token budget
    chunks: list[Chunk] = []
    current_tokens = 0
    current_parts: list[tuple[str, int]] = []

    for part_text, part_offset in parts:
        part_tokens = _estimate_tokens(part_text)

        if current_tokens + part_tokens > chunk_size_tokens and current_parts:
            # Flush current accumulation
            combined = "".join(p for p, _ in current_parts)
            chunk_offset = base_offset + current_parts[0][1]

            if _estimate_tokens(combined) > chunk_size_tokens and depth < 3:
                # Recurse deeper
                sub_chunks = _recursive_split(
                    combined,
                    chunk_size_tokens,
                    overlap_tokens,
                    doc_id,
                    chunk_offset,
                    metadata,
                    depth=depth + 1,
                )
                chunks.extend(sub_chunks)
            elif combined.strip():
                chunks.append(
                    Chunk(
                        chunk_id=uuid.uuid4().hex,
                        doc_id=doc_id,
                        content=combined,
                        start_char=chunk_offset,
                        end_char=chunk_offset + len(combined),
                        metadata={**metadata},
                        token_estimate=_estimate_tokens(combined),
                    )
                )

            # Compute overlap carry-over
            overlap_text = ""
            overlap_acc_tokens = 0
            for p_text, p_offset in reversed(current_parts):
                p_tok = _estimate_tokens(p_text)
                if overlap_acc_tokens + p_tok <= overlap_tokens:
                    overlap_text = p_text + overlap_text
                    overlap_acc_tokens += p_tok
                else:
                    break

            if overlap_text:
                # The overlap start offset
                overlap_start_in_text = combined.rfind(overlap_text)
                if overlap_start_in_text >= 0:
                    overlap_offset = chunk_offset + overlap_start_in_text
                else:
                    overlap_offset = chunk_offset + len(combined) - len(overlap_text)
                current_parts = [(overlap_text, overlap_offset - base_offset)]
                current_tokens = overlap_acc_tokens
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append((part_text, part_offset))
        current_tokens += part_tokens

    # Flush remaining
    if current_parts:
        combined = "".join(p for p, _ in current_parts)
        chunk_offset = base_offset + current_parts[0][1]
        if _estimate_tokens(combined) > chunk_size_tokens and depth < 3:
            sub_chunks = _recursive_split(
                combined,
                chunk_size_tokens,
                overlap_tokens,
                doc_id,
                chunk_offset,
                metadata,
                depth=depth + 1,
            )
            chunks.extend(sub_chunks)
        elif combined.strip():
            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4().hex,
                    doc_id=doc_id,
                    content=combined,
                    start_char=chunk_offset,
                    end_char=chunk_offset + len(combined),
                    metadata={**metadata},
                    token_estimate=_estimate_tokens(combined),
                )
            )

    return chunks


def chunk_text(
    content: str,
    chunk_size: int = 500,
    overlap: int = 50,
    doc_id: str = "",
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """Split *content* into overlapping semantic chunks.

    Uses a recursive text splitter:
        1. Try splitting on \\n\\n (paragraph boundaries).
        2. If still too large: split on \\n (line boundaries).
        3. If still too large: split on ". " (sentence boundaries).
        4. Last resort: hard character split.

    Args:
        content:    The text to split.
        chunk_size: Target chunk size in tokens (1 token ≈ 4 chars).
        overlap:    Overlap between consecutive chunks in tokens.
        doc_id:     ID of the parent document.
        metadata:   Metadata to attach to each chunk.

    Returns:
        List of Chunk objects.
    """
    if not content or not content.strip():
        return []

    effective_doc_id = doc_id or uuid.uuid4().hex
    effective_metadata = metadata or {}
    return _recursive_split(
        text=content,
        chunk_size_tokens=chunk_size,
        overlap_tokens=overlap,
        doc_id=effective_doc_id,
        base_offset=0,
        metadata=effective_metadata,
    )


def chunk_documents(
    docs: list,
    chunk_size: int = 500,
    overlap: int = 50,
    **kwargs,
) -> list[Chunk]:
    """Chunk all documents in *docs*, preserving each document's metadata.

    Args:
        docs:       List of Document objects.
        chunk_size: Target chunk size in tokens.
        overlap:    Overlap in tokens.
        **kwargs:   Additional keyword arguments forwarded to chunk_text().

    Returns:
        Flat list of Chunk objects across all documents.
    """
    all_chunks: list[Chunk] = []
    for doc in docs:
        doc_meta = {**getattr(doc, "metadata", {})}
        doc_meta.setdefault("file_type", getattr(doc, "file_type", ""))
        doc_meta.setdefault("source_path", getattr(doc, "source_path", ""))

        chunks = chunk_text(
            content=doc.content,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc.id,
            metadata=doc_meta,
        )
        all_chunks.extend(chunks)
    return all_chunks
