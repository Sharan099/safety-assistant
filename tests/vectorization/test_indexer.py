"""Tests for idempotent chunk indexing."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Regulation, Document, Chunk
from vectorization.indexer import RegulationIndexer, delete_chunks_for_document


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    reg = Regulation(
        regulation_code="R95",
        title="Test",
        source_type="UNECE",
        status="ACTIVE",
    )
    session.add(reg)
    session.commit()
    doc = Document(
        regulation_id=reg.id,
        document_name="r95.pdf",
        document_type="PDF",
        file_path="/tmp/r95.pdf",
        hash="abc",
    )
    session.add(doc)
    session.commit()
    yield session, doc.id
    session.close()


def test_index_chunks_replaces_existing(db_session):
    session, doc_id = db_session
    indexer = RegulationIndexer()
    first = [{"chunk_text": "first chunk", "chunk_index": 0, "page_number": 1}]
    second = [
        {"chunk_text": "revised chunk A", "chunk_index": 0, "page_number": 1},
        {"chunk_text": "revised chunk B", "chunk_index": 1, "page_number": 2},
    ]

    assert indexer.index_chunks(session, doc_id, first) == 1
    assert session.query(Chunk).filter(Chunk.document_id == doc_id).count() == 1

    assert indexer.index_chunks(session, doc_id, second) == 2
    chunks = session.query(Chunk).filter(Chunk.document_id == doc_id).all()
    assert len(chunks) == 2
    texts = sorted(c.chunk_text for c in chunks)
    assert texts == ["revised chunk A", "revised chunk B"]


def test_delete_chunks_for_document_idempotent(db_session):
    session, doc_id = db_session
    session.add(
        Chunk(document_id=doc_id, chunk_text="x", chunk_index=0, page_number=1)
    )
    session.commit()
    assert delete_chunks_for_document(session, doc_id) == 1
    assert delete_chunks_for_document(session, doc_id) == 0
    assert session.query(Chunk).filter(Chunk.document_id == doc_id).count() == 0
