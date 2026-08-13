"""Orchestrates one document's ingestion: register -> extract -> quality ->
sections -> chunks -> persist. See TRD.md §13.

Idempotent per `(document, revision_label)`: re-running with the same
`max_pages` returns the existing `DocumentRevision` instead of duplicating
it — revisions are immutable (BACKEND_SCHEMA.md §20), so a genuine re-run
with different extraction logic should bump `EXTRACTOR_VERSION`.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from sqlalchemy.orm import Session

from packages.domain.knowledge import (
    Document,
    DocumentChunk,
    DocumentFigure,
    DocumentPage,
    DocumentRevision,
    DocumentSection,
    DocumentTable,
    KnowledgeSource,
    SourceSnapshot,
)
from packages.ingestion.chunking import chunk_sections
from packages.ingestion.extract import extract_pages
from packages.ingestion.manifest import get_source
from packages.ingestion.qa import ExtractionReport, build_extraction_report
from packages.ingestion.sections import detect_sections
from packages.ingestion.structure import extract_figures, extract_tables

EXTRACTOR = "pymupdf"
EXTRACTOR_VERSION = "pymupdf-pipeline v0.1.0"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "data" / "artifacts"


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


def _write_extraction_report(document_key: str, report: ExtractionReport) -> pathlib.Path:
    """CLAUDE_CODE_LEVEL3_INSTRUCTIONS.md §11: every PDF gets an
    extraction_report.json. Written regardless of status — a FAIL report is
    exactly as important to keep as a PASS one."""
    out_dir = ARTIFACTS_ROOT / "extraction_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{document_key}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return out_path


def _persist_tables(
    session: Session,
    revision: DocumentRevision,
    document_key: str,
    pdf_path: pathlib.Path,
    page_rows: dict[int, DocumentPage],
    *,
    max_pages: int | None,
) -> int:
    tables = extract_tables(str(pdf_path), max_pages=max_pages)
    if not tables:
        return 0
    out_dir = ARTIFACTS_ROOT / "tables" / document_key
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for t in tables:
        page_row = page_rows.get(t.page_number)
        if page_row is None:
            continue  # table detected on a page that failed text extraction — not persisted, not fabricated either
        json_path = out_dir / f"p{t.page_number}_{t.table_index}.json"
        json_path.write_text(json.dumps(t.rows), encoding="utf-8")
        session.add(
            DocumentTable(
                document_revision_id=revision.id,
                page_id=page_row.id,
                table_number=str(t.table_index),
                json_uri=json_path.as_posix(),
                bounding_box={"x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3]},
                extraction_method=t.extraction_method,
                quality_score=t.quality_score,
            )
        )
        count += 1
    return count


def _persist_figures(
    session: Session,
    revision: DocumentRevision,
    document_key: str,
    pdf_path: pathlib.Path,
    page_rows: dict[int, DocumentPage],
    *,
    max_pages: int | None,
) -> int:
    figures = extract_figures(str(pdf_path), max_pages=max_pages)
    if not figures:
        return 0
    out_dir = ARTIFACTS_ROOT / "figures" / document_key
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in figures:
        page_row = page_rows.get(f.page_number)
        if page_row is None:
            continue
        image_path = out_dir / f"p{f.page_number}_{f.figure_index}.{f.image_ext}"
        image_path.write_bytes(f.image_bytes)
        session.add(
            DocumentFigure(
                document_revision_id=revision.id,
                page_id=page_row.id,
                figure_number=str(f.figure_index),
                image_uri=image_path.as_posix(),
                bounding_box=({"x0": f.bbox[0], "y0": f.bbox[1], "x1": f.bbox[2], "y1": f.bbox[3]} if f.bbox else None),
                figure_type="embedded_image",
            )
        )
        count += 1
    return count


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

    # No-silent-loss QA gate (Instructions §11) — build and persist the
    # report *before* committing to ingest anything. A completely unreadable
    # PDF must never silently become an empty "READY" revision.
    report = build_extraction_report(str(pdf_path), actual_hash, max_pages=max_pages)
    _write_extraction_report(source_id, report)
    if report.status == "FAIL":
        raise ValueError(f"extraction QA gate: {source_id} is FAIL ({len(report.failed_pages)} failed page(s))")

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

    _persist_tables(session, revision, source_id, pdf_path, page_rows, max_pages=max_pages)
    _persist_figures(session, revision, source_id, pdf_path, page_rows, max_pages=max_pages)
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
