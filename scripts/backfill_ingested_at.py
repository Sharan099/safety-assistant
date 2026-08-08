"""Best-effort backfill of ``ingested_at`` on existing Qdrant points (read+set_payload).

Existing R94 chunks predate the field. Prefer PDF mtime as ISO UTC; else \"unknown\".

Usage::

    uv run python scripts/backfill_ingested_at.py
    uv run python scripts/backfill_ingested_at.py --value unknown
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client


def _pdf_mtime_iso(regulation_id: str) -> str | None:
    pdf_dir = Path(ROOT / "data" / "pdfs")
    # Heuristic filename map
    candidates = [
        pdf_dir / f"{regulation_id}.pdf",
        pdf_dir / f"{regulation_id.replace('UN-ECE-', 'UN_')}.pdf",
        pdf_dir / "UN_R94.pdf" if "R94" in regulation_id else None,
        pdf_dir / "UN_R95.pdf" if "R95" in regulation_id else None,
        pdf_dir / "UN_R16.pdf" if "R16" in regulation_id else None,
        pdf_dir / "UN_R129.pdf" if "R129" in regulation_id else None,
    ]
    for path in candidates:
        if path and path.is_file():
            ts = path.stat().st_mtime
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Backfill ingested_at on indexed chunks")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument(
        "--value",
        default=None,
        help='Force a single value for all points (e.g. "unknown"). Default: PDF mtime per regulation.',
    )
    args = p.parse_args(argv)

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        print(f"FAIL: collection {args.collection!r} missing")
        return 1

    updated = 0
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            payload = pt.payload or {}
            if payload.get("ingested_at") and args.value is None:
                skipped += 1
                continue
            rid = str(payload.get("regulation_id") or "")
            if args.value is not None:
                value = args.value
            else:
                value = _pdf_mtime_iso(rid) or "unknown"
            client.set_payload(
                collection_name=args.collection,
                payload={"ingested_at": value},
                points=[pt.id],
            )
            updated += 1
        if offset is None:
            break

    print(f"Backfill complete: updated={updated} skipped_existing={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
