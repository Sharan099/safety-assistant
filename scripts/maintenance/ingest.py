"""Ingest registered sources through the lifecycle.

    uv run python scripts/maintenance/ingest.py                 # every registry entry
    uv run python scripts/maintenance/ingest.py unece-un-r94    # one source
    uv run python scripts/maintenance/ingest.py --force ...     # re-run even if unchanged
    uv run python scripts/maintenance/ingest.py --max-pages 20  # bounded smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from sqlalchemy.orm import Session

from safety_assistant.ingestion.sources import get_registry
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence import get_engine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_keys", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--no-activate", action="store_true")
    args = ap.parse_args(argv)

    registry = get_registry()
    keys = args.source_keys or registry.keys()
    failures = 0
    for key in keys:
        t0 = time.perf_counter()
        with Session(get_engine(), expire_on_commit=False) as session:
            out = ingest_source(
                session, key, registry=registry, force=args.force, activate=not args.no_activate, max_pages=args.max_pages
            )
        line = {
            "source_key": key,
            "run_status": out.status,
            "version_status": out.final_version_status,
            "seconds": round(time.perf_counter() - t0, 1),
            **{k: v for k, v in out.stats.items() if k in ("pages", "sections", "chunks", "embed_new", "embed_reused")},
        }
        if out.error:
            line["error"] = out.error.splitlines()[0][:200]
            failures += 1
        print(json.dumps(line), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
