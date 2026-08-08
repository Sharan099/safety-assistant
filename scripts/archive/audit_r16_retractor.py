"""Audit R16 retractor / ELR indexing without opening the live Qdrant lock.

Reads ``data/qdrant/collection/regulations/storage.sqlite`` in read-only mode
(pickle PointStruct payloads) and reports whether Emergency Locking Retractor
content is present, correctly section-tagged, or buried in mega-chunks.

Usage::

    python scripts/archive/audit_r16_retractor.py

Exit code 1 when ELR requirements (Â§6.2.5.3) lack a dedicated section_number.
"""

from __future__ import annotations

import pickle
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "qdrant" / "collection" / "regulations" / "storage.sqlite"


def load_payloads(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    out: list[dict] = []
    try:
        for _pid, blob in conn.execute("SELECT id, point FROM points"):
            pt = pickle.loads(blob)
            out.append(dict(pt.payload or {}))
    finally:
        conn.close()
    return out


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not db.is_file():
        print(f"MISSING db: {db}", file=sys.stderr)
        return 2
    chunks = load_payloads(db)
    r16 = [c for c in chunks if c.get("regulation_id") == "UN-ECE-R16"]
    print(f"total_chunks={len(chunks)} R16={len(r16)}")

    retr = [
        c
        for c in r16
        if "retractor" in ((c.get("text") or "") + " " + (c.get("enriched_text") or "")).lower()
    ]
    print(f"R16_retractor_keyword_hits={len(retr)}")

    by_sec = Counter((c.get("section_number") or "?") for c in retr)
    print("retractor_by_section:", by_sec.most_common(20))

    dedicated_625 = [
        c
        for c in r16
        if str(c.get("section_number") or "").startswith("6.2.5")
    ]
    print(f"chunks_with_section_number_6.2.5*={len(dedicated_625)}")

    buried = []
    for c in r16:
        text = c.get("text") or ""
        sec = str(c.get("section_number") or "")
        if re.search(r"6\.2\.5\.3", text) and not sec.startswith("6.2.5"):
            buried.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "section_number": sec,
                    "section_title": c.get("section_title"),
                    "chars": len(text),
                    "preview": " ".join(text.split())[:160],
                }
            )
    print(f"ELR_6.2.5.3_buried_under_wrong_section={len(buried)}")
    for row in buried[:8]:
        print(" BURIED", row)

    defs = [c for c in r16 if str(c.get("section_number") or "").startswith("2.14")]
    print(f"definition_2.14*_chunks={len(defs)}")
    for c in defs:
        if "emergency" in (c.get("text") or "").lower():
            print(
                " DEF",
                c.get("section_number"),
                c.get("chunk_id"),
                (c.get("section_title") or "")[:60],
            )

    if not dedicated_625 and buried:
        print(
            "\nVERDICT: ELR requirements exist in the index but are MIS-CHUNKED "
            "(buried under another section_number, typically a ~20k Â§6.2.2 mega-chunk). "
            "Routing alone cannot fix this â€” re-ingest R16 after the multi-clause "
            "TextItem split in ingestion/chunk.py (iter_inline_clause_segments)."
        )
        return 1
    if dedicated_625:
        print("\nVERDICT: dedicated Â§6.2.5* chunks present.")
        return 0
    if not retr:
        print("\nVERDICT: no R16 retractor text indexed at all.")
        return 1
    print("\nVERDICT: retractor text present; check retrieval filters for R95 leak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
