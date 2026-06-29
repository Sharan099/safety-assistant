"""Force structure-aware re-chunk for documents (bypasses NO_CHANGE checksum gate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from database.models import Document, Regulation
from parser.pdf_parser import PDFParser
from registry.metadata_extractor import MetadataExtractor
from vectorization.embedder import RegulationEmbedder
from vectorization.incremental_index import prepare_chunks_for_index
from vectorization.indexer import RegulationIndexer
from vectorization.structure_chunker import StructureAwareChunker


def rechunk_document(
    db: Session,
    document: Document,
    *,
    embedder: RegulationEmbedder,
    chunker: StructureAwareChunker | None = None,
) -> dict[str, Any]:
    """Re-parse, structure-chunk, incrementally embed, and replace chunks for one document."""
    file_path = document.file_path
    if not file_path or not Path(file_path).exists():
        return {
            "document_id": document.id,
            "document_name": document.document_name,
            "status": "skipped",
            "reason": f"file not found: {file_path}",
        }

    regulation = document.regulation
    if regulation is None:
        regulation = db.query(Regulation).filter(Regulation.id == document.regulation_id).first()

    metadata = {
        "regulation_code": regulation.regulation_code if regulation else "UNKNOWN",
        "source_type": regulation.source_type if regulation else "INTERNAL",
        "amendment": regulation.amendment if regulation else None,
        "title": regulation.title if regulation else document.document_name,
    }
    if document.amendment_from_text:
        metadata["amendment_from_text"] = document.amendment_from_text
    if document.amendment_from_filename:
        metadata["amendment_from_filename"] = document.amendment_from_filename

    parser = PDFParser(file_path)
    parsed_pages = parser.parse()
    chunker = chunker or StructureAwareChunker()
    chunks_data = chunker.chunk_document(parsed_pages, metadata, document.document_name)

    chunks_data, embed_stats = prepare_chunks_for_index(db, document.id, chunks_data, embedder)
    indexed = RegulationIndexer().index_chunks(db, document.id, chunks_data)

    return {
        "document_id": document.id,
        "document_name": document.document_name,
        "regulation_code": metadata["regulation_code"],
        "status": "success",
        "chunks_indexed": indexed,
        "embed_stats": embed_stats,
    }


def rechunk_all_documents(
    db: Session,
    *,
    embedder: RegulationEmbedder | None = None,
) -> dict[str, Any]:
    """Re-chunk every document with an on-disk PDF."""
    embedder = embedder or RegulationEmbedder()
    chunker = StructureAwareChunker()
    documents = db.query(Document).order_by(Document.id).all()

    totals = {
        "embedded_new": 0,
        "reused": 0,
        "chunks_total": 0,
        "documents_ok": 0,
        "documents_skipped": 0,
        "documents_failed": 0,
    }
    per_doc: list[dict[str, Any]] = []

    for doc in documents:
        try:
            result = rechunk_document(db, doc, embedder=embedder, chunker=chunker)
            per_doc.append(result)
            if result["status"] == "success":
                totals["documents_ok"] += 1
                stats = result.get("embed_stats") or {}
                totals["embedded_new"] += stats.get("embedded_new", 0)
                totals["reused"] += stats.get("reused", 0)
                totals["chunks_total"] += stats.get("chunks_total", 0)
            else:
                totals["documents_skipped"] += 1
        except Exception as exc:
            logger.exception("Re-chunk failed for %s: %s", doc.document_name, exc)
            per_doc.append(
                {
                    "document_id": doc.id,
                    "document_name": doc.document_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            totals["documents_failed"] += 1

    return {"totals": totals, "documents": per_doc}
