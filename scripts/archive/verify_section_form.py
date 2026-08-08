"""One-off Qdrant scroll: report ``section_number`` null rate across the index.

Historical diagnostic — quantified missing clause metadata on indexed chunks
(the R94 audit that surfaced ~24.8% null section numbers). Prefer
``scripts/audit_index.py`` for ongoing read-only audits.

Usage::

    python scripts/archive/verify_section_form.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from ingestion.embed_upsert import DEFAULT_COLLECTION, get_qdrant_client


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    client = get_qdrant_client()
    pts: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=DEFAULT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        pts.extend(batch)
        if offset is None:
            break

    def _missing(val: object) -> bool:
        if val is None:
            return True
        return not str(val).strip()

    nulls = [p for p in pts if _missing((p.payload or {}).get("section_number"))]
    print(f"total={len(pts)} null_section={len(nulls)} ({100 * len(nulls) / max(len(pts), 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
