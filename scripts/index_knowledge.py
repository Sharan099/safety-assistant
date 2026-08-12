"""Embed every ingested chunk that doesn't have an embedding yet.

Run after scripts/ingest_documents.py. Idempotent.

Usage:
    uv run python scripts/index_knowledge.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from packages.domain.db import get_engine  # noqa: E402
from packages.retrieval.index import index_chunks  # noqa: E402


def main() -> None:
    with Session(get_engine()) as session:
        count = index_chunks(session)
    print(f"Embedded {count} new chunk(s).")


if __name__ == "__main__":
    main()
