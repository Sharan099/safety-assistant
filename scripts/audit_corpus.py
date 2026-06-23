#!/usr/bin/env python3
"""
Phase 0 pilot corpus audit — read-only checks (no deletes).

Usage:
  conda activate rag
  python scripts/audit_corpus.py
  python scripts/audit_corpus.py --assert-no-iso --assert-pilot --assert-layout
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

PILOT_PDFS = {
    "UN_R14.pdf",
    "UN_R16.pdf",
}

EXPECTED_CORPUS_COUNT = len(PILOT_PDFS)

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
    ap.add_argument("--assert-pilot", action="store_true")
    ap.add_argument("--assert-core", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--assert-layout", action="store_true")
    args = ap.parse_args()
    if args.assert_core:
        args.assert_pilot = True

    indexed = load_indexed()
    on_disk = corpus_on_disk()
    archive = ROOT / "archive" / "corpus_removed"
    archived_iso = list(archive.glob("*26262*")) if archive.exists() else []

    print(f"Corpus PDFs on disk : {len(on_disk)} (expected {EXPECTED_CORPUS_COUNT})")
    print(f"Indexed in manifest : {len(indexed)}")
    print(f"Corpus version      : {CORPUS_VERSION} (pilot)")
    print(f"Ingestion .py in data/: {[p.name for p in DATA_DIR.glob('*.py')]}")
    print(f"Ingestion package     : {(ROOT / 'ingestion').exists()}")

    iso_hits = [n for n in on_disk | indexed if "iso_26262" in n.lower() or "asil" in n.lower()]
    if iso_hits:
        print(f"WARN ISO/FuSa still in corpus/index: {iso_hits}")
    elif archived_iso:
        print(f"OK   ISO archived ({archived_iso[0].name})")
    else:
        print("OK   No ISO_26262 in active corpus")

    extra = sorted(on_disk - PILOT_PDFS)
    missing = sorted(PILOT_PDFS - on_disk)
    if missing:
        print(f"WARN Pilot PDFs missing from corpus: {missing}")
    if extra:
        print(f"WARN Non-pilot PDFs still on disk: {extra}")
    if not missing and not extra:
        print(f"OK   Pilot corpus ({', '.join(sorted(PILOT_PDFS))})")

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
    if args.assert_pilot and (missing or extra):
        failed = True
    if args.assert_layout and py_in_data:
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
