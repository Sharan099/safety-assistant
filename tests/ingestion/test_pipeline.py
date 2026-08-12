import pytest
from sqlalchemy.orm import Session

from packages.domain.knowledge import Document, DocumentChunk, DocumentPage, DocumentSection
from packages.ingestion.pipeline import ingest_document
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
def test_ingest_is_idempotent(session: Session) -> None:
    first = ingest_document(session, "unece-un-r94", max_pages=3)
    session.commit()
    second = ingest_document(session, "unece-un-r94", max_pages=3)
    assert first.id == second.id


@requires_db
def test_unknown_source_id_raises(session: Session) -> None:
    with pytest.raises(KeyError):
        ingest_document(session, "not-a-real-source")
