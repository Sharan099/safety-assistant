"""Bounded ingestion of every registered Level-3 PDF — TRD_LEVEL3.md §43/
Instructions §36: "Only after smoke tests pass... Use conservative
concurrency." Not a claim of unlimited-depth full-corpus ingestion — every
PDF is bounded to MAX_PAGES, consistent with the existing V1 precedent
(docs/ADR/0006) for the 8 GB RAM dev machine. Revisit the bound once a real
corpus-scale run is actually needed, not before.

Usage:
    uv run python scripts/ingest_level3_pdfs.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from packages.domain.db import get_engine  # noqa: E402
from packages.ingestion.manifest import load_manifest  # noqa: E402
from packages.ingestion.pipeline import ingest_document  # noqa: E402
from packages.retrieval.index import index_chunks  # noqa: E402

MAX_PAGES = 20


def main() -> None:
    manifest = load_manifest()
    pdf_source_ids = [s["source_id"] for s in manifest["sources"] if s.get("canonical_path")]

    with Session(get_engine()) as session:
        for source_id in pdf_source_ids:
            try:
                revision = ingest_document(session, source_id, max_pages=MAX_PAGES)
                session.commit()
                print(f"[{revision.status}] {source_id} (revision {revision.id})")
            except Exception as exc:  # noqa: BLE001 — one bad PDF must not abort the batch
                session.rollback()
                print(f"[FAILED] {source_id}: {exc}")

        embedded = index_chunks(session)
        print(f"Embedded {embedded} new chunk(s).")


if __name__ == "__main__":
    main()
