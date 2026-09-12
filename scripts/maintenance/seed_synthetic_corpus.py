"""Seed the configured database with the synthetic two-version UN-R999 regulation (the test fixture),
so end-to-end flows can run where the licensed corpus is not available (CI).

    uv run python scripts/maintenance/seed_synthetic_corpus.py

Uses the configured embedding provider (fastembed by default) and the configured artifact store.
Idempotent: re-running is a no-op for unchanged sources.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import tempfile
from typing import Any

from sqlalchemy.orm import Session

from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.persistence import get_engine

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
# The synthetic regulation lives with the test fixtures; loaded dynamically so this script has no
# hard dependency on the test package layout.
registry_for: Any = importlib.import_module("tests.support.minireg").registry_for


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="synthetic-corpus-"))
    registry = registry_for(tmp, revisions=(1, 2), include_r998=True)
    with Session(get_engine(), expire_on_commit=False) as session:
        for entry in registry.sources:
            out = ingest_source(session, entry.source_key, registry=registry, repo_root=tmp)
            print(f"{entry.source_key}: {out.status} → {out.final_version_status}")
            if out.status not in ("SUCCEEDED", "SKIPPED_UNCHANGED"):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
