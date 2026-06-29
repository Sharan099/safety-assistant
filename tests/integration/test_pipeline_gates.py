"""Integration test: pipeline stages run in order and gates block bad input."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Document, IngestLog, Regulation, SourceManifest, Chunk
from parser.pdf_parser import PDFParser
from registry.coverage import build_coverage_report
from registry.pipeline import process_crawled_item
from registry.storage_paths import quarantine_dir, storage_dir
from vectorization.structure_chunker import StructureAwareChunker
from vectorization.embedder import RegulationEmbedder


@pytest.fixture
def pipeline_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, tmp_path
    session.close()


def _make_pdf(path: Path, body: str) -> None:
    doc = fitz.open()
    for _ in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), body + "\n" + ("Padding content for size. " * 80), fontsize=11)
    doc.save(path)
    doc.close()


def test_invalid_pdf_never_reaches_storage(pipeline_db):
    session, root = pipeline_db
    bad = root / "data" / "staging" / "bad.pdf"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a pdf")

    item = {
        "file_path": str(bad),
        "source_url": "https://unece.org/fileadmin/DAM/test.pdf",
        "metadata": {
            "regulation_code": "R94",
            "source_type": "UNECE",
            "title": "Test",
        },
    }

    with patch("registry.pipeline.head_metadata", return_value={"etag": '"x"'}):
        result = process_crawled_item(
            session,
            run_id="test-run",
            item=item,
            full_fetch=True,
            enqueue_ingest=False,
        )

    assert result["outcome"] == "rejected"
    assert not list(storage_dir(root).rglob("*.pdf"))
    assert list(quarantine_dir(root).glob("*"))
    stages = [r.stage for r in session.query(IngestLog).all()]
    assert "validate" in stages
    assert session.query(Document).count() == 0
    assert session.query(SourceManifest).count() == 0


def test_valid_pdf_passes_gates_to_storage(pipeline_db):
    session, root = pipeline_db
    staging = root / "data" / "staging" / "r94.pdf"
    staging.parent.mkdir(parents=True)
    body = "UN REGULATION No. 94\nFrontal collision protection.\n" + ("Spec " * 60)
    _make_pdf(staging, body)

    item = {
        "file_path": str(staging),
        "source_url": "https://unece.org/fileadmin/DAM/r094.pdf",
        "metadata": {
            "regulation_code": "R94",
            "source_type": "UNECE",
            "amendment": "04 Series",
            "title": "R94",
        },
    }

    with patch("registry.pipeline.head_metadata", return_value={"etag": '"new"'}):
        result = process_crawled_item(
            session,
            run_id="test-run-2",
            item=item,
            full_fetch=True,
            enqueue_ingest=False,
        )

    assert result["outcome"] == "new"
    assert Path(result["canonical_path"]).exists()
    assert session.query(SourceManifest).count() == 1
    log_stages = [row.stage for row in session.query(IngestLog).all()]
    assert "validate" in log_stages
    assert "tracker" in log_stages


def test_duplicate_blocked_before_manifest_growth(pipeline_db):
    session, root = pipeline_db
    staging = root / "data" / "staging" / "r95.pdf"
    staging.parent.mkdir(parents=True)
    body = "UN REGULATION No. 95\nLateral protection.\n" + ("Text " * 50)
    _make_pdf(staging, body)
    url = "https://unece.org/fileadmin/DAM/r095.pdf"
    meta = {
        "regulation_code": "R95",
        "source_type": "UNECE",
        "amendment": "05 Series",
        "title": "R95",
    }
    item = {"file_path": str(staging), "source_url": url, "metadata": meta}

    with patch("registry.pipeline.head_metadata", return_value={"etag": '"a"'}):
        first = process_crawled_item(
            session, run_id="r1", item=item, full_fetch=True, enqueue_ingest=False
        )
        second = process_crawled_item(
            session, run_id="r2", item=item, full_fetch=True, enqueue_ingest=False
        )

    assert first["outcome"] == "new"
    assert second["outcome"] == "duplicate"
    assert session.query(SourceManifest).count() == 1


def test_structure_parse_and_chunk_stage(pipeline_db):
  """Phase C: structured parse + chunk metadata without touching live index."""
  session, root = pipeline_db
  pdf = root / "r94_struct.pdf"
  body = (
      "UN REGULATION No. 94\n"
      "5.2.1.4.\n"
      "The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;\n"
  )
  _make_pdf(pdf, body)
  parser = PDFParser(str(pdf))
  pages = parser.parse(extract_tables=False)
  chunks = StructureAwareChunker().chunk_document(
      pages,
      {"regulation_code": "UN_R94", "source_type": "UNECE", "amendment": "04 Series"},
      pdf.name,
  )
  assert any(c.get("chunk_type") == "clause" for c in chunks)
  assert all(c.get("content_hash") for c in chunks)


def test_coverage_gap_report_stage(pipeline_db):
    session, _ = pipeline_db
    session.add(
        Regulation(
            regulation_code="UN_R94",
            title="R94",
            source_type="UNECE",
            status="ACTIVE",
        )
    )
    session.commit()
    report = build_coverage_report(session)
    assert report["summary"]["missing"] > 0
    assert report["summary"]["ingested"] >= 1


def test_incremental_embed_reuses_unchanged_hashes(pipeline_db):
    """Phase D: re-chunk with identical content_hash embeds 0 new vectors."""
    from vectorization.incremental_index import prepare_chunks_for_index

    session, _ = pipeline_db
    reg = Regulation(regulation_code="UN_R95", title="R95", source_type="UNECE", status="ACTIVE")
    session.add(reg)
    session.commit()
    doc = Document(
        regulation_id=reg.id,
        document_name="r95.pdf",
        document_type="PDF",
        file_path="/tmp/r95.pdf",
        hash="deadbeef",
    )
    session.add(doc)
    session.commit()

    embedder = RegulationEmbedder()
    body = "Side impact protection criteria"
    vec = embedder.embed_chunks([body])[0]
    session.add(
        Chunk(
            document_id=doc.id,
            chunk_text=body,
            chunk_index=0,
            content_hash="stable-hash",
            embedding=vec,
        )
    )
    session.commit()

    chunks = [{"chunk_text": body, "content_hash": "stable-hash", "chunk_index": 0}]
    _, stats = prepare_chunks_for_index(session, doc.id, chunks, embedder)
    assert stats["embedded_new"] == 0
    assert stats["reused"] == 1


def test_search_metadata_filter_chunk_type(pipeline_db):
    """Phase D: hybrid search respects chunk_type metadata filter."""
    from registry.search import RegulationSearchEngine

    session, _ = pipeline_db
    reg = Regulation(
        regulation_code="UN_R14",
        title="R14",
        source_type="UNECE",
        amendment="07 Series",
        status="ACTIVE",
    )
    session.add(reg)
    session.commit()
    doc = Document(
        regulation_id=reg.id,
        document_name="UN_R14.pdf",
        document_type="PDF",
        file_path="/tmp/r14.pdf",
        hash="abc",
    )
    session.add(doc)
    session.commit()

    embedder = RegulationEmbedder()
    table_vec = embedder.embed_chunks(["Annex 6 anchorage table M1 N1"])[0]
    clause_vec = embedder.embed_chunks(["General anchorage strength"])[0]
    session.add_all(
        [
            Chunk(
                document_id=doc.id,
                chunk_text="Annex 6 anchorage table M1 N1",
                chunk_index=0,
                chunk_type="table",
                embedding=table_vec,
            ),
            Chunk(
                document_id=doc.id,
                chunk_text="General anchorage strength",
                chunk_index=1,
                chunk_type="clause",
                embedding=clause_vec,
            ),
        ]
    )
    session.commit()

    engine = RegulationSearchEngine()
    result = engine.search(
        session,
        "Annex 6 anchorage table",
        filters={"regulation_code": "UN_R14", "chunk_type": "table"},
        top_k=5,
        rerank=False,
    )
    assert result["sources"]
    assert all(s.get("chunk_type") == "table" for s in result["sources"])
