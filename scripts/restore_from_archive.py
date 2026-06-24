#!/usr/bin/env python3
"""Restore selected corpus PDFs from archive/corpus_removed into data/corpus/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ARCHIVE_DIR, CORPUS_DIR, DATA_DIR  # noqa: E402

# (archive_filename, corpus_subdir, dest_filename)
RESTORE_PLAN: list[tuple[str, str, str]] = [
    # Legal regulations (Part B)
    ("UN_R94.pdf", "legal", "UN_R94.pdf"),
    ("UN_R95.pdf", "legal", "UN_R95.pdf"),
    ("UN_R135.pdf", "legal", "UN_R135.pdf"),
    ("UN_R137.pdf", "legal", "UN_R137.pdf"),
    ("UN_R17.pdf", "legal", "UN_R17.pdf"),
    ("R127r3am3e.pdf", "legal", "UN_R127.pdf"),
    ("FMVSS_208.pdf.pdf", "legal", "FMVSS_208.pdf"),
    # Rating protocols (Part C — public NCAP protocols)
    ("euro-ncap-protocol-crash-protection-frontal-impact-v11.pdf", "rating", "EURO_NCAP_FRONTAL.pdf"),
    ("euro-ncap-protocol-crash-protection-side-impact-v11.pdf", "rating", "EURO_NCAP_SIDE.pdf"),
    ("euro-ncap-protocol-overall-assessment-v100.pdf", "rating", "EURO_NCAP_OVERALL.pdf"),
    # Engineering references (licensed internal copies)
    ("CAE-Companion-2025-26.pdf", "reference", "CAE_REFERENCE.pdf"),
    ("SafetyCompanion-2026.pdf", "reference", "SAFETY_REFERENCE.pdf"),
]


def restore(dry_run: bool = False) -> dict:
    restored: list[dict] = []
    missing: list[str] = []
    for archive_name, subdir, dest_name in RESTORE_PLAN:
        src = ARCHIVE_DIR / archive_name
        dest = CORPUS_DIR / subdir / dest_name
        if not src.is_file():
            missing.append(archive_name)
            continue
        if dry_run:
            restored.append({"src": str(src), "dest": str(dest), "dry_run": True})
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        restored.append({
            "archive": archive_name,
            "dest": str(dest.relative_to(ROOT)),
            "subdir": subdir,
            "bytes": dest.stat().st_size,
        })
    return {"restored": restored, "missing": missing}


def write_manifest() -> Path:
    manifest_dir = DATA_DIR / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(CORPUS_DIR.rglob("*.pdf"))
    from backend.app.core.document_registry import get_document_meta

    docs = []
    for p in pdfs:
        reg = __import__(
            "ingestion.hierarchical_chunker", fromlist=["detect_regulation_type"]
        ).detect_regulation_type(p.name)
        meta = get_document_meta(reg)
        docs.append({
            "path": str(p.relative_to(ROOT)),
            "name": p.name,
            "category": p.parent.name,
            "regulation": reg,
            "doc_type": meta.doc_type,
            "authority_tier": meta.authority_tier,
            "impact_mode": meta.impact_mode,
            "license_status": meta.license_status,
        })
    out = {
        "corpus_version": 4,
        "pilot": False,
        "scope": "multilayer UN/ECE + FMVSS + NCAP + references",
        "total_pdfs": len(pdfs),
        "documents": docs,
    }
    path = manifest_dir / "corpus_manifest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Restore corpus PDFs from archive")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = restore(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    if args.dry_run:
        return 0 if not result["missing"] else 1

    manifest = write_manifest()
    print(f"Wrote {manifest}")
    return 0 if not result["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
