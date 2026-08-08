"""Post-fix verification helpers for section_number quality."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client


def main() -> int:
    load_dotenv(ROOT / ".env")
    client = get_qdrant_client()
    pts, _ = client.scroll(
        collection_name=DEFAULT_COLLECTION,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    nulls = [
        p
        for p in pts
        if not str((p.payload or {}).get("section_number") or "").strip()
    ]
    print(f"total={len(pts)} null_section={len(nulls)} ({100*len(nulls)/max(len(pts),1):.1f}%)")
    if nulls:
        print("sample null titles:")
        for p in nulls[:10]:
            pl = p.payload or {}
            print(" -", pl.get("content_type"), pl.get("section_title"), "p", pl.get("page_number"))

    for needle in (
        "Femur force",
        "Frequency response",
        "Annex 11",
        "Test procedures for the vehicles",
    ):
        print(f"\n=== {needle} ===")
        for p in pts:
            pl = p.payload or {}
            blob = f"{pl.get('section_title') or ''} {(pl.get('text') or '')[:240]}"
            if needle.lower() not in blob.lower():
                continue
            print(
                f"  section_number={pl.get('section_number')!r} "
                f"title={pl.get('section_title')!r} "
                f"type={pl.get('content_type')} page={pl.get('page_number')} "
                f"ingested_at={pl.get('ingested_at')}"
            )
    return 0 if not nulls else 1


if __name__ == "__main__":
    raise SystemExit(main())
