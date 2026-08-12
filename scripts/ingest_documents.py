"""Run the ingestion pipeline over a bounded set of sources.

Per IMPLEMENTATION_PLAN.md Phase 4: "First test on UN_R94. Then LS-DYNA
Theory R17. Then LS-DYNA Keyword Vol I R17. Do not process all documents
simultaneously on the 8 GB machine." UN_R94 (the smallest regulation) is
ingested in full; the large LS-DYNA manuals are bounded by `max_pages` here
as a smoke run — see docs/ADR/0006 for why full-corpus ingestion is a
separate, later batch job, not attempted synchronously in this script.

Usage:
    uv run python scripts/ingest_documents.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from packages.domain.db import get_engine  # noqa: E402
from packages.ingestion.pipeline import ingest_document  # noqa: E402

# (source_id, max_pages) — None means the whole document.
PLAN: list[tuple[str, int | None]] = [
    ("unece-un-r94", None),
    ("lsdyna-r17-theory", 60),
    ("lsdyna-r17-vol-i", 60),
]


def main() -> None:
    engine = get_engine()
    with Session(engine) as session:
        for source_id, max_pages in PLAN:
            revision = ingest_document(session, source_id, max_pages=max_pages)
            print(f"{source_id}: revision={revision.id} status={revision.status} label={revision.revision_label!r}")


if __name__ == "__main__":
    main()
