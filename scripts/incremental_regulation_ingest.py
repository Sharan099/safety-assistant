#!/usr/bin/env python3
"""Incremental regulation ingest — chunk + classify + append embeddings (Part B)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_FILE, EMBEDDINGS_FILE, MARKDOWN_DIR, OUTPUT_DIR


def _run_fidelity(regulation: str) -> bool:
    md = MARKDOWN_DIR / f"{regulation}.md"
    if not md.is_file():
        print(f"SKIP fidelity: {md} not found")
        return True
    rc = subprocess.call(
        [sys.executable, "scripts/verify_extraction_fidelity.py", "--mode", "quick"],
        cwd=ROOT,
    )
    return rc == 0


def ingest_regulation(reg_code: str, *, skip_fidelity: bool = False) -> dict:
    from ingestion.hierarchical_chunker import chunk_markdown_file, run as chunk_run
    from ingestion.embed_chunks import run_incremental

    md_path = MARKDOWN_DIR / f"{reg_code}.md"
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    if not skip_fidelity and not _run_fidelity(reg_code):
        raise RuntimeError(f"Extraction fidelity gate failed for {reg_code}")

    new_chunks = chunk_markdown_file(md_path)
    if not new_chunks:
        return {"regulation": reg_code, "chunks_added": 0}

    # Merge into existing chunks.json
    existing: dict = {"chunks": []}
    if CHUNKS_FILE.exists():
        existing = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    known_ids = {c["chunk_id"] for c in existing.get("chunks", [])}
    to_add = [c for c in new_chunks if c["chunk_id"] not in known_ids]
    existing["chunks"].extend(to_add)
    CHUNKS_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    embed_result = run_incremental(to_add)
    return {
        "regulation": reg_code,
        "chunks_added": len(to_add),
        "total_vectors": embed_result.get("total_vectors"),
        "embeddings_file": str(EMBEDDINGS_FILE),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental regulation ingest")
    parser.add_argument("regulations", nargs="+", help="e.g. UN_R94 UN_R95")
    parser.add_argument("--skip-fidelity", action="store_true")
    args = parser.parse_args()

    results = []
    for reg in args.regulations:
        code = reg.upper().replace(" ", "_")
        print(f"Ingesting {code}...")
        results.append(ingest_regulation(code, skip_fidelity=args.skip_fidelity))
        print(json.dumps(results[-1], indent=2))

    report_path = OUTPUT_DIR / "incremental_ingest_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
