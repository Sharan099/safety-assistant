"""Orchestrates one document's ingestion: register -> extract -> quality ->
sections -> chunks -> persist. See TRD.md §13.

Idempotent per `(document, revision_label)`: re-running with the same
`max_pages` returns the existing `DocumentRevision` instead of duplicating
it — revisions are immutable (BACKEND_SCHEMA.md §20), so a genuine re-run
with different extraction logic should bump `EXTRACTOR_VERSION`.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

from sqlalchemy.orm import Session

from packages.domain.knowledge import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentRevision,
    DocumentSection,
    KnowledgeSource,
    SourceSnapshot,
)
from packages.ingestion.chunking import chunk_sections
from packages.ingestion.extract import extract_pages
from packages.ingestion.manifest import get_source
from packages.ingestion.sections import detect_sections

EXTRACTOR = "pymupdf"
EXTRACTOR_VERSION = "pymupdf-pipeline v0.1.0"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_or_create_knowledge_source(session: Session, source_id: str, meta: dict[str, Any]) -> KnowledgeSource:
    row = session.query(KnowledgeSource).filter_by(source_key=source_id).one_or_none()
    if row is None:
        row = KnowledgeSource(
            source_key=source_id,
            source_type=meta["category"],
            category=meta["category"],
            authority_level=meta["authority"],
            publisher=meta.get("publisher"),
            local_path=meta["canonical_path"],
        )
        session.add(row)
        session.flush()
    return row


def _get_or_create_document(
    session: Session, source_id: str, meta: dict[str, Any], knowledge_source: KnowledgeSource
) -> Document:
    row = session.query(Document).filter_by(document_key=source_id).one_or_none()
    if row is None:
        row = Document(
            knowledge_source_id=knowledge_source.id,
            document_key=source_id,
            title=meta["title"],
            document_type=meta["category"],
            publisher=meta.get("publisher"),
        )
        session.add(row)
        session.flush()
    return row


def _get_or_create_snapshot(session: Session, pdf_path: pathlib.Path, sha256: str) -> SourceSnapshot:
    row = session.query(SourceSnapshot).filter_by(sha256=sha256).one_or_none()
    if row is None:
        row = SourceSnapshot(
            storage_uri=pdf_path.as_posix(), filename=pdf_path.name, sha256=sha256, size_bytes=pdf_path.stat().st_size
        )
        session.add(row)
        session.flush()
    return row


def ingest_document(
    session: Session, source_id: str, *, max_pages: int | None = None, repo_root: pathlib.Path = REPO_ROOT
) -> DocumentRevision:
    meta = get_source(source_id)
    pdf_path = repo_root / meta["canonical_path"]
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"canonical copy not found: {pdf_path} (see knowledge/00_registry/source_manifest.yaml)"
        )

    actual_hash = _sha256_file(pdf_path)
    if actual_hash != meta["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {source_id}: manifest={meta['sha256']} actual={actual_hash}")

    knowledge_source = _get_or_create_knowledge_source(session, source_id, meta)
    document = _get_or_create_document(session, source_id, meta, knowledge_source)
    snapshot = _get_or_create_snapshot(session, pdf_path, actual_hash)

    revision_label = EXTRACTOR_VERSION + (f"-first{max_pages}" if max_pages else "-full")
    existing = (
        session.query(DocumentRevision).filter_by(document_id=document.id, revision_label=revision_label).one_or_none()
    )
    if existing is not None:
        return existing

    revision = DocumentRevision(
        document_id=document.id,
        revision_label=revision_label,
        source_snapshot_id=snapshot.id,
        extractor=EXTRACTOR,
        extractor_version=EXTRACTOR_VERSION,
        status="EXTRACTING",
    )
    session.add(revision)
    session.flush()

    pages = extract_pages(str(pdf_path), max_pages=max_pages)
    page_rows: dict[int, DocumentPage] = {}
    for p in pages:
        row = DocumentPage(
            document_revision_id=revision.id,
            page_number=p.page_number,
            text_content=p.text,
            text_quality=p.text_quality,
            layout_quality=0.0 if p.needs_ocr else 1.0,
            ocr_used=False,
        )
        session.add(row)
        page_rows[p.page_number] = row
    session.flush()

    detected = detect_sections(pages)
    chunks_by_section = chunk_sections(detected)

    for i, section in enumerate(detected):
        section_row = DocumentSection(
            document_revision_id=revision.id,
            title=section.title,
            section_number=section.section_number,
            start_page=section.start_page,
            end_page=section.end_page,
            content="\n".join(p for _pn, p in section.paragraphs),
        )
        session.add(section_row)
        session.flush()

        for chunk in chunks_by_section[i]:
            start_page_row = page_rows.get(chunk.start_page)
            session.add(
                DocumentChunk(
                    document_revision_id=revision.id,
                    section_id=section_row.id,
                    page_id=start_page_row.id if start_page_row is not None else None,
                    chunk_type="text",
                    content=chunk.content,
                    token_count=chunk.word_count,  # word-count approximation, not a real tokenizer
                    source_locator={
                        "page_start": chunk.start_page,
                        "page_end": chunk.end_page,
                        "section": section.title,
                    },
                )
            )

    revision.status = "READY"
    session.commit()
    return revision
