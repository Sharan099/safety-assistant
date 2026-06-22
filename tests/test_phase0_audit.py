"""Phase 0 audit tests — corpus layout and ISO removal."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CORPUS_DIR, DATA_DIR  # noqa: E402
from backend.app.core.document_registry import get_document_meta  # noqa: E402


CORE_PDFS = {
    "UN_R14.pdf", "UN_R16.pdf", "UN_R17.pdf", "UN_R94.pdf", "UN_R95.pdf",
    "UN_R135.pdf", "UN_R137.pdf", "FMVSS_208.pdf.pdf",
    "EURO_NCAP_FRONTAL.pdf.pdf", "EURO_NCAP_SIDE.pdf.pdf",
    "EURO_NCAP_REAR.pdf.pdf", "EURO_NCAP_VRU.pdf.pdf",
}


def test_corpus_has_seventeen_pdfs():
    pdfs = list(CORPUS_DIR.rglob("*.pdf"))
    assert len(pdfs) == 17


def test_core_pdfs_present():
    names = {p.name for p in CORPUS_DIR.rglob("*.pdf")}
    assert CORE_PDFS <= names


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


def test_corpus_manifest():
    manifest = DATA_DIR / "manifest" / "corpus_manifest.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["corpus_version"] == 2
    assert data["total_pdfs"] == 17


def test_audit_script_passes():
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_corpus.py"),
         "--assert-no-iso", "--assert-core", "--assert-layout"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
