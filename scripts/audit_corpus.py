#!/usr/bin/env python3
"""
Corpus audit — read-only checks (no deletes).

Usage:
  conda activate rag
  python scripts/audit_corpus.py
  python scripts/audit_corpus.py --assert-no-iso --assert-multilayer --assert-layout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CORPUS_DIR, CORPUS_VERSION, DATA_DIR  # noqa: E402

MANIFEST = ROOT / "output" / "ingest_manifest.json"
CORPUS_MANIFEST = DATA_DIR / "manifest" / "corpus_manifest.json"

CORE_LEGAL_PDFS = {"UN_R14.pdf", "UN_R16.pdf"}
EXPECTED_CORPUS_COUNT = 14  # multilayer v4 — see data/manifest/corpus_manifest.json

INGESTION_PY = {
    "paddle_ocr_converter.py",
    "docling_converter.py",
    "hierarchical_chunker.py",
    "embed_chunks.py",
}


def _pdf_names(paths: list[str]) -> set[str]:
    return {Path(p.replace("\\", "/")).name for p in paths}


def load_indexed() -> set[str]:
    if not MANIFEST.exists():
        return set()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = []
    for key in ("converted", "skipped"):
        for x in data.get(key, []):
            if isinstance(x, str):
                names.append(x)
            elif isinstance(x, dict):
                names.append(x.get("pdf", ""))
    return _pdf_names(names)


def corpus_on_disk() -> set[str]:
    if not CORPUS_DIR.exists():
        return set()
    return {p.name for p in CORPUS_DIR.rglob("*.pdf")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-no-iso", action="store_true")
    ap.add_argument("--assert-pilot", action="store_true", help="Legacy: core UN R14/R16 only")
    ap.add_argument("--assert-multilayer", action="store_true", help="v4 multilayer corpus (14 PDFs)")
    ap.add_argument("--assert-core", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--assert-layout", action="store_true")
    args = ap.parse_args()
    if args.assert_core:
        args.assert_multilayer = True

    indexed = load_indexed()
    on_disk = corpus_on_disk()
    archive = ROOT / "archive" / "corpus_removed"
    archived_iso = list(archive.glob("*26262*")) if archive.exists() else []

    scope = "multilayer" if CORPUS_VERSION >= 4 else "pilot"
    print(f"Corpus PDFs on disk : {len(on_disk)} (expected {EXPECTED_CORPUS_COUNT})")
    print(f"Indexed in manifest : {len(indexed)}")
    print(f"Corpus version      : {CORPUS_VERSION} ({scope})")
    print(f"Ingestion .py in data/: {[p.name for p in DATA_DIR.glob('*.py')]}")
    print(f"Ingestion package     : {(ROOT / 'ingestion').exists()}")

    iso_hits = [n for n in on_disk | indexed if "iso_26262" in n.lower() or "asil" in n.lower()]
    if iso_hits:
        print(f"WARN ISO/FuSa still in corpus/index: {iso_hits}")
    elif archived_iso:
        print(f"OK   ISO archived ({archived_iso[0].name})")
    else:
        print("OK   No ISO_26262 in active corpus")

    missing_core = sorted(CORE_LEGAL_PDFS - on_disk)
    if missing_core:
        print(f"WARN Core legal PDFs missing: {missing_core}")
    else:
        print(f"OK   Core legal PDFs present ({', '.join(sorted(CORE_LEGAL_PDFS))})")

    py_in_data = INGESTION_PY & {p.name for p in DATA_DIR.glob("*.py")}
    if py_in_data:
        print(f"WARN Ingestion code still in data/: {sorted(py_in_data)}")
    else:
        print("OK   Ingestion code moved out of data/")

    failed = False
    if len(on_disk) != EXPECTED_CORPUS_COUNT:
        print(f"WARN Corpus count {len(on_disk)} != {EXPECTED_CORPUS_COUNT}")
    if args.assert_no_iso and iso_hits:
        failed = True
    if args.assert_pilot:
        extra = sorted(on_disk - CORE_LEGAL_PDFS)
        if missing_core or extra:
            failed = True
    if args.assert_multilayer and (len(on_disk) != EXPECTED_CORPUS_COUNT or missing_core):
        failed = True
    if args.assert_layout and py_in_data:
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
