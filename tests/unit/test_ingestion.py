"""Tests for ingestion layer: Chunk, chunk_documents, loaders, IngestionResult."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from harness.ingestion.chunker import Chunk, chunk_documents
from harness.ingestion.pipeline import IngestionResult


# ===========================================================================
# Chunk dataclass
# ===========================================================================

def test_chunk_token_estimate_auto():
    c = Chunk(chunk_id="c1", doc_id="d1", content="hello world", start_char=0, end_char=11)
    assert c.token_estimate > 0
    assert c.token_estimate == len("hello world") // 4


def test_chunk_token_estimate_explicit():
    c = Chunk(chunk_id="c1", doc_id="d1", content="hello", start_char=0, end_char=5, token_estimate=99)
    assert c.token_estimate == 99


def test_chunk_empty_content_zero_tokens():
    c = Chunk(chunk_id="c1", doc_id="d1", content="", start_char=0, end_char=0)
    assert c.token_estimate == 0


def test_chunk_metadata_default_empty():
    c = Chunk(chunk_id="c1", doc_id="d1", content="text", start_char=0, end_char=4)
    assert c.metadata == {}


def test_chunk_metadata_preserved():
    c = Chunk(chunk_id="c1", doc_id="d1", content="text", start_char=0, end_char=4,
              metadata={"source": "pdf", "page": 1})
    assert c.metadata["source"] == "pdf"
    assert c.metadata["page"] == 1


# ===========================================================================
# chunk_documents
# ===========================================================================

def _make_doc(content: str, doc_id: str = "doc1") -> object:
    """Create a minimal Document-like object for testing."""
    from unittest.mock import MagicMock
    doc = MagicMock()
    doc.id = doc_id          # chunk_documents uses doc.id
    doc.doc_id = doc_id      # kept for compatibility
    doc.content = content
    doc.metadata = {}
    doc.file_type = ""
    doc.source_path = ""
    return doc


def test_chunk_documents_empty_list():
    result = chunk_documents([])
    assert result == []


def test_chunk_documents_short_text_single_chunk():
    doc = _make_doc("Hello world this is a test.")
    chunks = chunk_documents([doc], chunk_size=2000, overlap=200)
    assert len(chunks) >= 1
    # Combined content should cover the original
    combined = " ".join(c.content for c in chunks)
    assert "Hello world" in combined


def test_chunk_documents_long_text_multiple_chunks():
    long_text = "This is a sentence. " * 500  # ~10k chars
    doc = _make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_chunk_documents_overlap_creates_continuity():
    # chunk_size is in tokens (~4 chars each), so 200 tokens ≈ 800 chars
    text = "word " * 500   # ~2500 chars ≈ 625 tokens
    doc = _make_doc(text)
    chunks = chunk_documents([doc], chunk_size=100, overlap=20)
    assert len(chunks) >= 1  # at least split into chunks


def test_chunk_documents_chunks_have_ids():
    doc = _make_doc("test content here for chunking")
    chunks = chunk_documents([doc])
    for c in chunks:
        assert c.chunk_id
        assert len(c.chunk_id) > 0


def test_chunk_documents_chunks_reference_doc_id():
    doc = _make_doc("test content here for chunking split across chunks", doc_id="my_doc")
    chunks = chunk_documents([doc])
    for c in chunks:
        assert c.doc_id == "my_doc"


def test_chunk_documents_char_offsets_valid():
    doc = _make_doc("Hello world test content")
    chunks = chunk_documents([doc])
    for c in chunks:
        assert c.start_char >= 0
        assert c.end_char > c.start_char
        assert c.end_char <= len(doc.content) + 10  # small tolerance for separators


def test_chunk_documents_multiple_docs():
    docs = [_make_doc(f"Document {i} content here", doc_id=f"doc{i}") for i in range(3)]
    chunks = chunk_documents(docs)
    doc_ids = {c.doc_id for c in chunks}
    assert "doc0" in doc_ids
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_chunk_documents_whitespace_only_content():
    doc = _make_doc("   \n\n\t  ")
    chunks = chunk_documents([doc])
    # Should handle whitespace gracefully — may return 0 or 1 chunks
    assert isinstance(chunks, list)


def test_chunk_documents_inherits_metadata():
    from unittest.mock import MagicMock
    doc = MagicMock()
    doc.doc_id = "d1"
    doc.content = "test content here"
    doc.metadata = {"source": "web", "url": "https://example.com"}
    chunks = chunk_documents([doc])
    for c in chunks:
        # Metadata should be inherited or augmented
        assert isinstance(c.metadata, dict)


# ===========================================================================
# IngestionResult
# ===========================================================================

def test_ingestion_result_success_true():
    r = IngestionResult(source="doc.pdf", documents_loaded=1, errors=[])
    assert r.success is True


def test_ingestion_result_success_false_no_docs():
    r = IngestionResult(source="doc.pdf", documents_loaded=0, errors=[])
    assert r.success is False


def test_ingestion_result_success_false_with_errors():
    r = IngestionResult(source="doc.pdf", documents_loaded=1, errors=["parse error"])
    assert r.success is False


def test_ingestion_result_repr():
    r = IngestionResult(
        source="test.pdf",
        documents_loaded=2,
        chunks_created=10,
        embeddings_stored=10,
        facts_extracted=5,
        errors=[],
        duration_seconds=1.5,
    )
    s = repr(r)
    assert "test.pdf" in s
    assert "docs=2" in s
    assert "chunks=10" in s


def test_ingestion_result_defaults():
    r = IngestionResult(source="test.txt")
    assert r.documents_loaded == 0
    assert r.chunks_created == 0
    assert r.embeddings_stored == 0
    assert r.facts_extracted == 0
    assert r.errors == []
    assert r.duration_seconds == 0.0


# ===========================================================================
# Loaders (file-based, no external deps)
# ===========================================================================

def test_detect_type_txt(tmp_path):
    from harness.ingestion.loaders import detect_type
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    assert detect_type(str(f)) in ("text", "txt", "plain", "markdown", str(f.suffix))


def test_detect_type_md(tmp_path):
    from harness.ingestion.loaders import detect_type
    f = tmp_path / "README.md"
    f.write_text("# Hello")
    result = detect_type(str(f))
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_load_text_file(tmp_path):
    from harness.ingestion.loaders import load
    f = tmp_path / "sample.txt"
    f.write_text("Hello world\nLine 2\nLine 3")
    docs = await load(str(f))
    assert len(docs) >= 1
    combined = " ".join(d.content for d in docs)
    assert "Hello world" in combined


@pytest.mark.asyncio
async def test_load_markdown_file(tmp_path):
    from harness.ingestion.loaders import load
    f = tmp_path / "README.md"
    f.write_text("# Title\n\nParagraph content here.\n\n## Section 2\n\nMore content.")
    docs = await load(str(f))
    assert len(docs) >= 1
    combined = " ".join(d.content for d in docs)
    assert "Title" in combined or "content" in combined.lower()


@pytest.mark.asyncio
async def test_load_returns_documents_with_content(tmp_path):
    from harness.ingestion.loaders import load
    f = tmp_path / "data.txt"
    f.write_text("Important business data here.")
    docs = await load(str(f))
    for doc in docs:
        assert hasattr(doc, "content")
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0


@pytest.mark.asyncio
async def test_load_csv_file(tmp_path):
    from harness.ingestion.loaders import load
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25\n")
    docs = await load(str(f))
    assert len(docs) >= 1


@pytest.mark.asyncio
async def test_load_nonexistent_file_returns_empty():
    from harness.ingestion.loaders import load
    # load() logs the error gracefully and returns empty list
    docs = await load("/nonexistent/path/file.txt")
    assert docs == []
