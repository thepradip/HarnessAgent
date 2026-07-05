"""Ingestion module — document loading, chunking, entity extraction, pipeline."""

from harness.ingestion.chunker import Chunk, chunk_documents, chunk_text
from harness.ingestion.code_extractor import (
    CodeFileFacts,
    CodeSymbol,
    extract_code_facts,
)
from harness.ingestion.code_loader import SourceFile, load_source_files
from harness.ingestion.extractor import ExtractedFact, extract_facts, extract_sql_schema_facts
from harness.ingestion.loaders import Document, detect_type, load
from harness.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "Chunk",
    "chunk_documents",
    "chunk_text",
    "CodeFileFacts",
    "CodeSymbol",
    "Document",
    "detect_type",
    "ExtractedFact",
    "extract_code_facts",
    "extract_facts",
    "extract_sql_schema_facts",
    "IngestionPipeline",
    "IngestionResult",
    "load",
    "load_source_files",
    "SourceFile",
]
