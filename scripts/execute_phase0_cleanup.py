#!/usr/bin/env python3
"""Execute Phase 0 pilot corpus scope — UN R14 + UN R16 only."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
CORPUS = DATA / "corpus"
LEGAL = CORPUS / "legal"
ARCHIVE = ROOT / "archive" / "corpus_removed"
REGULATIONS = DATA / "regulations"

# Phase 0 pilot — ingest only these two legal PDFs.
PILOT_LEGAL = [
    "UN_R14.pdf",
    "UN_R16.pdf",
]

PILOT_MARKDOWN_STEMS = {"UN_R14", "UN_R16"}

INGESTION_PY = [
    "paddle_ocr_converter.py",
    "docling_converter.py",
    "hierarchical_chunker.py",
    "embed_chunks.py",
    "__init__.py",
]

OUTPUT = ROOT / "output"
MARKDOWN = OUTPUT / "markdown"
CORPUS_VERSION = 3


def _find_pdf(name: str) -> Path | None:
    for base in (DATA, DATA / "regulations"):
        p = base / name
        if p.is_file():
            return p
    for p in DATA.rglob(name):
        if p.is_file():
            return p
    return None


def _move(src: Path, dest: Path, log: list[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    log.append(f"MOVE {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


def _archive(src: Path, log: list[str]) -> None:
    dest = ARCHIVE / src.name
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    log.append(f"ARCHIVE {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


def scope_corpus(log: list[str]) -> None:
    """Keep only pilot PDFs in data/corpus/legal; archive everything else."""
    LEGAL.mkdir(parents=True, exist_ok=True)

    for name in PILOT_LEGAL:
        src = _find_pdf(name)
        if src and src.resolve() != (LEGAL / name).resolve():
            _move(src, LEGAL / name, log)
        elif (LEGAL / name).is_file():
            log.append(f"OK   pilot PDF already in place: {name}")
        else:
            log.append(f"WARN missing pilot PDF: {name}")

    for pdf in sorted(CORPUS.rglob("*.pdf")):
        if pdf.name in PILOT_LEGAL and pdf.parent == LEGAL:
            continue
        _archive(pdf, log)

    for sub in (CORPUS / "rating", CORPUS / "reference"):
        if sub.exists() and not any(sub.iterdir()):
            sub.rmdir()
            log.append(f"REMOVE empty {sub.relative_to(ROOT)}/")

    if REGULATIONS.exists() and not any(REGULATIONS.iterdir()):
        REGULATIONS.rmdir()
        log.append("REMOVE empty data/regulations/")


def move_ingestion_code(log: list[str]) -> None:
    dest_dir = ROOT / "ingestion"
    dest_dir.mkdir(exist_ok=True)
    for name in INGESTION_PY:
        src = DATA / name
        if not src.is_file():
            continue
        dest = dest_dir / name
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))
        log.append(f"MOVE {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


def clean_artifacts(log: list[str]) -> None:
    for md in MARKDOWN.glob("*.md"):
        if md.stem in PILOT_MARKDOWN_STEMS:
            log.append(f"KEEP {md.relative_to(ROOT)}")
            continue
        md.unlink()
        log.append(f"DELETE {md.relative_to(ROOT)}")

    for fname in (
        "regulation_chunks.json",
        "regulation_embeddings.json",
        "ingest_manifest.json",
        "chunking_diagnostics.txt",
    ):
        p = OUTPUT / fname
        if p.exists():
            p.unlink()
            log.append(f"DELETE {p.relative_to(ROOT)}")


def write_manifest(log: list[str]) -> None:
    manifest_dir = DATA / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(CORPUS.rglob("*.pdf"))
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "pilot": True,
        "scope": "UN_R14 + UN_R16",
        "total_pdfs": len(pdfs),
        "documents": [
            {
                "path": str(p.relative_to(ROOT)),
                "name": p.name,
                "category": p.parent.name,
                "doc_type": "legal",
                "authority": "UN-ECE",
            }
            for p in pdfs
        ],
    }
    out = manifest_dir / "corpus_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.append(f"WRITE {out.relative_to(ROOT)} ({len(pdfs)} PDFs)")


def main() -> int:
    log: list[str] = []
    scope_corpus(log)
    move_ingestion_code(log)
    clean_artifacts(log)
    write_manifest(log)

    report = ROOT / "output" / "phase0_execution.log"
    report.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"\nLog: {report}")
    corpus_count = len(list(CORPUS.rglob("*.pdf")))
    expected = len(PILOT_LEGAL)
    print(f"Corpus PDFs: {corpus_count} (expected {expected})")
    return 0 if corpus_count == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
