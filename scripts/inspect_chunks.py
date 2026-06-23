#!/usr/bin/env python3
"""Inspect chunk quality for a document or the full corpus."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_FILE  # noqa: E402

NUMERIC_LIMIT_RE = re.compile(
    r"(chest\s+deflection|head\s+injury|hic|deflection|acceleration|force|load)"
    r"\s*[:\s]*\d",
    re.I,
)
UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|g|kN|daN|m/s|ms)", re.I)


def inspect(chunks: list[dict], doc_filter: str | None = None) -> dict:
    if doc_filter:
        flt = doc_filter.lower()
        chunks = [
            c for c in chunks
            if flt in (c.get("pdf_name") or "").lower()
            or flt in (c.get("regulation") or "").lower()
            or flt in (c.get("doc_id") or "").lower()
        ]
    if not chunks:
        return {"error": "no chunks match filter"}

    words = [c.get("word_count", 0) for c in chunks]
    with_limit = sum(
        1 for c in chunks
        if NUMERIC_LIMIT_RE.search(c.get("text", "")) and UNIT_RE.search(c.get("text", ""))
    )
    with_parent = sum(1 for c in chunks if c.get("parent_id"))
    with_clause = sum(1 for c in chunks if c.get("clause") or c.get("clause_number"))
    with_meta = sum(1 for c in chunks if c.get("test_type"))

    samples = random.sample(chunks, min(5, len(chunks)))
    return {
        "chunk_count": len(chunks),
        "avg_words": round(sum(words) / len(words), 1),
        "max_words": max(words),
        "pct_numeric_limit": round(100 * with_limit / len(chunks), 1),
        "pct_with_parent": round(100 * with_parent / len(chunks), 1),
        "pct_with_clause": round(100 * with_clause / len(chunks), 1),
        "pct_with_test_type": round(100 * with_meta / len(chunks), 1),
        "samples": [
            {
                "chunk_id": s.get("chunk_id"),
                "test_type": s.get("test_type"),
                "value_type": s.get("value_type"),
                "clause": s.get("clause") or s.get("clause_number"),
                "text_preview": (s.get("text") or "")[:200],
            }
            for s in samples
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="Filter by pdf_name, regulation, or doc_id substring")
    args = ap.parse_args()

    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found")
        return 1

    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    result = inspect(data.get("chunks", []), args.doc)
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
