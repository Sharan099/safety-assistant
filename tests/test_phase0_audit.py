"""Corpus audit tests — multilayer scope (corpus_version 4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CORPUS_DIR, CORPUS_VERSION, DATA_DIR  # noqa: E402
from backend.app.core.document_registry import (  # noqa: E402
    INDEXED_LEGAL_CORPUS,
    get_document_meta,
)

# Core belt pilot regs must always remain indexed.
CORE_LEGAL_PDFS = {"UN_R14.pdf", "UN_R16.pdf"}
EXPECTED_PDF_COUNT = 14  # 9 legal + 3 rating + 2 reference (see corpus_manifest.json)


def test_corpus_pdf_count():
    pdfs = list(CORPUS_DIR.rglob("*.pdf"))
    assert len(pdfs) == EXPECTED_PDF_COUNT


def test_core_legal_pdfs_present():
    names = {p.name for p in CORPUS_DIR.rglob("*.pdf")}
    assert CORE_LEGAL_PDFS <= names


def test_iso_not_in_corpus():
    names = {p.name.lower() for p in CORPUS_DIR.rglob("*.pdf")}
    assert not any("iso_26262" in n or "asil" in n for n in names)


def test_iso_archived():
    archived = ROOT / "archive" / "corpus_removed" / "ISO_26262.pdf.pdf"
    assert archived.is_file()


def test_ingestion_code_not_in_data():
    py_names = {p.name for p in DATA_DIR.glob("*.py")}
    assert not py_names & {
        "paddle_ocr_converter.py", "docling_converter.py",
        "hierarchical_chunker.py", "embed_chunks.py",
    }


def test_ingestion_package_exists():
    assert (ROOT / "ingestion" / "hierarchical_chunker.py").is_file()


def test_document_registry_no_iso():
    meta = get_document_meta("ISO")
    assert meta.code == "UNKNOWN"


def test_document_registry_core_entries():
    for code in ("UN_R14", "UN_R16"):
        meta = get_document_meta(code)
        assert meta.doc_type == "legal_regulation"
        assert meta.authority == "UN-ECE"
        assert meta.indexed_revision is not None
        assert meta.has_multiple_revisions
        assert code in INDEXED_LEGAL_CORPUS


def test_corpus_manifest():
    manifest = DATA_DIR / "manifest" / "corpus_manifest.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["corpus_version"] == CORPUS_VERSION
    assert data["total_pdfs"] == EXPECTED_PDF_COUNT
    assert data.get("pilot") is False


def test_audit_script_passes():
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_corpus.py"),
         "--assert-no-iso", "--assert-multilayer", "--assert-layout"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
