"""Tests for structure-aware chunking."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from vectorization.structure_chunker import StructureAwareChunker


def _pdf_with_clause_and_table(tmp_path: Path) -> str:
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    body = (
        "UN REGULATION No. 94\n"
        "5.2.1.4.\n"
        "The Thorax Compression Criterion (ThCC) shall not exceed 42 mm;\n"
        "Annex 6\n"
        "Anchorage layout\n"
        "| Position | Load |\n"
        "| --- | --- |\n"
        "| Driver | 1350 daN |\n"
    )
    page.insert_text((72, 72), body, fontsize=10)
    doc.save(path)
    doc.close()
    return str(path)


def test_clause_boundary_detected(tmp_path):
    from parser.pdf_parser import PDFParser

    parser = PDFParser(_pdf_with_clause_and_table(tmp_path))
    pages = parser.parse(extract_tables=False)
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_document(
        pages,
        {"regulation_code": "UN_R94", "source_type": "UNECE", "amendment": "04 Series"},
        "sample.pdf",
    )
    clause_chunks = [c for c in chunks if c.get("chunk_type") == "clause"]
    assert any("42 mm" in c["chunk_text"] for c in clause_chunks)
    assert any(c.get("section") == "5.2.1.4" for c in clause_chunks)


def test_table_kept_whole(tmp_path):
    from parser.pdf_parser import PDFParser

    parser = PDFParser(_pdf_with_clause_and_table(tmp_path))
    pages = parser.parse(extract_tables=False)
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_document(
        pages,
        {"regulation_code": "UN_R14", "source_type": "UNECE"},
        "sample.pdf",
    )
    tables = [c for c in chunks if c.get("chunk_type") == "table"]
    assert tables
    assert "| Driver |" in tables[0]["chunk_text"]
    assert tables[0]["chunk_text"].count("| --- |") >= 1


def test_metadata_fields_populated(tmp_path):
    from parser.pdf_parser import PDFParser

    parser = PDFParser(_pdf_with_clause_and_table(tmp_path))
    pages = parser.parse(extract_tables=False)
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_document(
        pages,
        {"regulation_code": "UN_R94", "source_type": "UNECE", "amendment": "04 Series"},
        "sample.pdf",
    )
    ch = next(c for c in chunks if c.get("chunk_type") == "clause")
    meta = ch["metadata"]
    assert meta["authority"] == "UNECE"
    assert meta["regulation_id"] == "UN_R94"
    assert meta["chunk_type"] == "clause"
    assert meta["content_hash"]
    assert ch["heading_path"]


def test_parent_child_link(tmp_path):
    from parser.pdf_parser import PDFParser

    parser = PDFParser(_pdf_with_clause_and_table(tmp_path))
    pages = parser.parse(extract_tables=False)
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_document(
        pages,
        {"regulation_code": "UN_R94", "source_type": "UNECE"},
        "sample.pdf",
    )
    child = next(c for c in chunks if c.get("parent_chunk_index") is not None)
    parent = chunks[child["parent_chunk_index"]]
    assert parent["chunk_index"] == child["parent_chunk_index"]
    assert parent["chunk_type"] in {"section", "clause"}
