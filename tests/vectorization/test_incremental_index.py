"""Tests for incremental embedding reuse."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Chunk, Document, Regulation
from registry.embedding_config import EMBEDDING_DIMENSION
from vectorization.embedder import RegulationEmbedder
from vectorization.incremental_index import embed_chunks_incremental, prepare_chunks_for_index


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    reg = Regulation(regulation_code="UN_R94", title="R94", source_type="UNECE", status="ACTIVE")
    session.add(reg)
    session.commit()
    doc = Document(
        regulation_id=reg.id,
        document_name="r94.pdf",
        document_type="PDF",
        file_path="/tmp/r94.pdf",
        hash="abc",
    )
    session.add(doc)
    session.commit()
    yield session, doc.id
    session.close()


def test_embed_chunks_incremental_reuses_hashes(db_session):
    session, doc_id = db_session
    embedder = RegulationEmbedder()
    existing_vec = embedder.embed_chunks(["existing body"])[0]
    session.add(
        Chunk(
            document_id=doc_id,
            chunk_text="existing body",
            chunk_index=0,
            content_hash="hash-a",
            embedding=existing_vec,
        )
    )
    session.commit()

    chunks = [
        {"chunk_text": "existing body", "content_hash": "hash-a"},
        {"chunk_text": "brand new body", "content_hash": "hash-b"},
    ]
    reusable = {"hash-a": existing_vec}
    vectors, stats = embed_chunks_incremental(embedder, chunks, reusable)

    assert stats["embedded_new"] == 1
    assert stats["reused"] == 1
    assert len(vectors) == 2
    assert vectors[0] == existing_vec
    assert len(vectors[1]) == EMBEDDING_DIMENSION


def test_prepare_chunks_for_index_zero_embed_on_identical_rechunk(db_session):
    session, doc_id = db_session
    embedder = RegulationEmbedder()
    body = "ThCC shall not exceed 42 mm"
    vec = embedder.embed_chunks([body])[0]
    content_hash = "same-hash"
    session.add(
        Chunk(
            document_id=doc_id,
            chunk_text=body,
            chunk_index=0,
            content_hash=content_hash,
            embedding=vec,
        )
    )
    session.commit()

    chunks = [{"chunk_text": body, "content_hash": content_hash, "chunk_index": 0}]
    out, stats = prepare_chunks_for_index(session, doc_id, chunks, embedder)
    assert stats["embedded_new"] == 0
    assert stats["reused"] == 1
    assert out[0]["embedding"] == vec


def test_embedding_dimension_enforced(db_session):
    session, doc_id = db_session
    embedder = RegulationEmbedder()
    chunks = [{"chunk_text": "dimension check", "content_hash": "d1"}]
    out, stats = prepare_chunks_for_index(session, doc_id, chunks, embedder)
    assert stats["embedded_new"] == 1
    assert len(out[0]["embedding"]) == EMBEDDING_DIMENSION
