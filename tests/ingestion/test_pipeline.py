import json

import pytest
from sqlalchemy.orm import Session

from packages.domain.knowledge import (
    Document,
    DocumentChunk,
    DocumentFigure,
    DocumentPage,
    DocumentSection,
    DocumentTable,
)
from packages.ingestion.pipeline import ARTIFACTS_ROOT, ingest_document
from tests.domain.conftest import requires_db


@requires_db
def test_ingest_un_r94_first_pages(session: Session) -> None:
    revision = ingest_document(session, "unece-un-r94", max_pages=5)
    session.commit()

    assert revision.status == "READY"
    assert revision.extractor == "pymupdf"

    document = session.get(Document, revision.document_id)
    assert document is not None
    assert document.document_key == "unece-un-r94"

    pages = session.query(DocumentPage).filter_by(document_revision_id=revision.id).all()
    assert len(pages) == 5

    sections = session.query(DocumentSection).filter_by(document_revision_id=revision.id).all()
    assert len(sections) >= 1

    chunks = session.query(DocumentChunk).filter_by(document_revision_id=revision.id).all()
    assert len(chunks) >= 1
    assert all(c.content.strip() for c in chunks)
    assert all(c.source_locator is not None and "page_start" in c.source_locator for c in chunks)


@requires_db
def test_ingest_writes_extraction_report_and_persists_tables_figures(session: Session) -> None:
    # UN R94's first 30 pages are confirmed (tests/ingestion/test_structure.py)
    # to contain real tables (page 22) and figures (page 1) — used here so
    # this test asserts genuine persisted rows, not just an empty pass.
    revision = ingest_document(session, "unece-un-r94", max_pages=30)
    session.commit()

    report_path = ARTIFACTS_ROOT / "extraction_reports" / "unece-un-r94.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] in ("PASS", "PASS_WITH_WARNINGS", "NEEDS_REVIEW")
    assert report["processed_page_count"] == 30
    assert report["ocr_engine"] == "NOT_AVAILABLE"
    assert report["pages_with_tables"] > 0
    assert report["pages_with_figures"] > 0

    # Every persisted table/figure row must reference a real page belonging
    # to this revision — never a dangling/fabricated page_id.
    page_ids = {p.id for p in session.query(DocumentPage).filter_by(document_revision_id=revision.id).all()}
    tables = session.query(DocumentTable).filter_by(document_revision_id=revision.id).all()
    figures = session.query(DocumentFigure).filter_by(document_revision_id=revision.id).all()
    assert tables, "expected persisted table rows from UN R94's page 22"
    assert figures, "expected persisted figure rows from UN R94's page 1"
    assert all(t.page_id in page_ids for t in tables)
    assert all(f.page_id in page_ids for f in figures)
    assert all(t.extraction_method == "pymupdf_find_tables" for t in tables)


@requires_db
def test_ingest_is_idempotent(session: Session) -> None:
    first = ingest_document(session, "unece-un-r94", max_pages=3)
    session.commit()
    second = ingest_document(session, "unece-un-r94", max_pages=3)
    assert first.id == second.id


@requires_db
def test_unknown_source_id_raises(session: Session) -> None:
    with pytest.raises(KeyError):
        ingest_document(session, "not-a-real-source")
