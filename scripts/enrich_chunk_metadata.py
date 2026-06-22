#!/usr/bin/env python3
"""Enrich existing chunks with metadata fields without re-OCR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_FILE  # noqa: E402
from ingestion.metadata_classifier import classify_chunk  # noqa: E402


def enrich_chunks(chunks: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        "enriched": 0,
        "with_test_type": 0,
        "with_value_type": 0,
        "with_clause": 0,
        "with_clause_topic": 0,
    }
    for chunk in chunks:
        meta = classify_chunk(
            regulation=chunk.get("regulation", ""),
            pdf_name=chunk.get("pdf_name", ""),
            text=chunk.get("text", ""),
            clause_number=chunk.get("clause_number"),
            heading_path=chunk.get("heading_path", ""),
            section_title=chunk.get("section_title", ""),
        )
        chunk.update(meta)
        stats["enriched"] += 1
        if meta.get("test_type") and meta["test_type"] != "general":
            stats["with_test_type"] += 1
        if meta.get("value_type"):
            stats["with_value_type"] += 1
        if meta.get("clause"):
            stats["with_clause"] += 1
        if meta.get("clause_topic") and meta["clause_topic"] != "general":
            stats["with_clause_topic"] += 1
    return chunks, stats


def main() -> int:
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found")
        return 1
    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    chunks, stats = enrich_chunks(chunks)
    data["chunks"] = chunks
    data["metadata_schema_version"] = 3
    CHUNKS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Enriched {stats['enriched']} chunks")
    for k, v in stats.items():
        if k != "enriched":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
