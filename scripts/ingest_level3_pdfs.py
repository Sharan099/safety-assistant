"""Full ingestion of every registered Level-3 PDF —
PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §7/§18.

No page-count cap: `docs/ADR/0016` explains why the earlier 20-page bound
was an artificial limit in this script, not an architectural one —
`ingest_document()`'s own default is already unbounded. Per §18's
no-silent-loss rule, verifies `original page count == accounted page
count` (every extracted page becomes a `DocumentPage` row, whatever its
status) for every PDF it ingests.

Usage:
    uv run python scripts/ingest_level3_pdfs.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from packages.domain.db import get_engine  # noqa: E402
from packages.domain.knowledge import DocumentPage  # noqa: E402
from packages.ingestion.manifest import get_source, load_manifest  # noqa: E402
from packages.ingestion.pipeline import REPO_ROOT, ingest_document  # noqa: E402
from packages.retrieval.index import index_chunks  # noqa: E402


def _safe_print(text: str) -> None:
    """The Windows console's default codepage (cp1252) can't encode every
    Unicode character a real PDF's extracted text (or an exception message
    quoting it) might contain — a crash here would violate "one bad PDF
    must not abort the batch" just as much as a crash during ingestion
    itself. Encode-with-replace rather than let print() raise."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def main() -> None:
    manifest = load_manifest()
    pdf_source_ids = [s["source_id"] for s in manifest["sources"] if s.get("canonical_path")]

    with Session(get_engine()) as session:
        for source_id in pdf_source_ids:
            try:
                revision = ingest_document(session, source_id)  # max_pages=None: the whole document
                session.commit()

                meta = get_source(source_id)
                pdf_path = REPO_ROOT / meta["canonical_path"]
                with pymupdf.open(str(pdf_path)) as doc:  # type: ignore[no-untyped-call]
                    original_page_count = len(doc)
                accounted = session.query(DocumentPage).filter_by(document_revision_id=revision.id).count()
                loss_note = (
                    ""
                    if accounted == original_page_count
                    else f" **PAGE ACCOUNTING MISMATCH** ({accounted}/{original_page_count})"
                )

                _safe_print(
                    f"[{revision.status}] {source_id}: {accounted}/{original_page_count} pages accounted{loss_note}"
                )
            except Exception as exc:  # noqa: BLE001 — one bad PDF must not abort the batch
                session.rollback()
                _safe_print(f"[FAILED] {source_id}: {str(exc)[:300]}")  # a full SQL-statement dump isn't useful here

        embedded = index_chunks(session)
        _safe_print(f"\nEmbedded {embedded} new chunk(s).")


if __name__ == "__main__":
    main()
