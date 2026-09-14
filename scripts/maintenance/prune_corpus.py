"""Remove registry documents that are no longer in the source registry.

    uv run python scripts/maintenance/prune_corpus.py            # dry run: lists what would go
    uv run python scripts/maintenance/prune_corpus.py --apply    # deletes

Scope: regulations whose `regulation_key` is absent from knowledge/00_registry/sources.yaml and
that are not user uploads (kind PROJECT_DOCUMENT), plus — with --superseded — the non-ACTIVE
versions of registry regulations, so only the version in force remains. Versions cascade to sections, chunks,
embeddings, summaries and jobs; message citations are set to NULL by the schema; ingestion runs
and events of the removed versions are deleted explicitly (no cascade there, by design).
Artifacts in the object store are left in place (content-addressed, harmless).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from safety_assistant.ingestion.sources import load_registry
from safety_assistant.persistence import get_engine
from safety_assistant.persistence.models import IngestionEvent, IngestionRun, Regulation, RegulationVersion
from safety_assistant.retrieval.sparse import invalidate_cache


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--superseded", action="store_true", help="also drop non-ACTIVE versions of registry documents")
    args = ap.parse_args(argv)
    keep = {e.regulation_key for e in load_registry().sources}
    with Session(get_engine()) as s:
        gone = s.scalars(
            select(Regulation).where(Regulation.regulation_key.not_in(keep), Regulation.kind != "PROJECT_DOCUMENT")
        ).all()
        for r in gone:
            versions = s.scalars(select(RegulationVersion).where(RegulationVersion.regulation_id == r.id)).all()
            verb = "DELETE" if args.apply else "would delete"
            print(f"{verb} {r.regulation_key}: {[v.version_label for v in versions]}")
            if not args.apply:
                continue
            vids = [v.id for v in versions]
            run_ids = s.scalars(select(IngestionRun.id).where(IngestionRun.version_id.in_(vids))).all()
            s.execute(delete(IngestionEvent).where(IngestionEvent.run_id.in_(run_ids)))
            s.execute(delete(IngestionRun).where(IngestionRun.id.in_(run_ids)))
            s.execute(delete(RegulationVersion).where(RegulationVersion.id.in_(vids)))
            s.delete(r)
        verb = "DELETE" if args.apply else "would delete"
        old: Sequence[RegulationVersion] = []
        if args.superseded:
            old = s.scalars(
                select(RegulationVersion)
                .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
                .where(Regulation.regulation_key.in_(keep), RegulationVersion.status != "ACTIVE")
            ).all()
            for v in old:
                print(f"{verb} version {v.version_label} ({v.status}) of {v.regulation_id}")
            if args.apply and old:
                vids = [v.id for v in old]
                run_ids = s.scalars(select(IngestionRun.id).where(IngestionRun.version_id.in_(vids))).all()
                s.execute(delete(IngestionEvent).where(IngestionEvent.run_id.in_(run_ids)))
                s.execute(delete(IngestionRun).where(IngestionRun.id.in_(run_ids)))
                s.execute(delete(RegulationVersion).where(RegulationVersion.id.in_(vids)))
        if args.apply:
            s.commit()
            invalidate_cache()
    verb = "removed" if args.apply else "found"
    print(f"{len(gone)} document(s) not in registry, {len(old)} old version(s) {verb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
