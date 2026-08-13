"""Recursive ingestion of real LS-DYNA decks into cae_* —
PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §8: moved from "main deck + direct
includes only" to full transitive resolution (packages/cae/lsdyna/resolve.py),
docs/ADR/0016.

Individual component files over 20 MB are recorded as reached but not
opened (`SKIPPED_TOO_LARGE`, resolve.py's own safety bound) — real memory
safety on an 8 GB machine, not silently pretending they don't exist.

Usage:
    uv run python scripts/ingest_level3_cae_decks.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from packages.cae.lsdyna.persistence import FileMeta, persist_deck  # noqa: E402
from packages.cae.lsdyna.resolve import resolve_recursive  # noqa: E402
from packages.domain.cae import (  # noqa: E402
    CaeContact,
    CaeControl,
    CaeDatabase,
    CaeDeck,
    CaeFile,
    CaeInclude,
    CaeKeyword,
    CaeMaterial,
    CaePart,
    CaeSection,
)
from packages.domain.db import get_engine  # noqa: E402
from packages.domain.knowledge import KnowledgeSource  # noqa: E402
from packages.ingestion.archives import inspect_archive  # noqa: E402
from packages.ingestion.manifest import get_source  # noqa: E402


def _clear_existing_deck(session: Session, deck_key: str) -> None:
    """persist_deck() is idempotent per deck_key (returns the existing row
    unchanged) — the right behavior for a genuine re-run with the same
    parser, wrong for re-running with a *deeper* resolution (this script's
    move from direct-includes-only to recursive). Clears the old, shallower
    deck first, matching the regenerate-by-clearing precedent in
    scripts/generate_synthetic_dataset.py (docs/ADR/0008)."""
    deck = session.query(CaeDeck).filter_by(deck_key=deck_key).one_or_none()
    if deck is None:
        return
    for model in (
        CaePart,
        CaeMaterial,
        CaeSection,
        CaeContact,
        CaeControl,
        CaeDatabase,
        CaeInclude,
        CaeKeyword,
        CaeFile,
    ):
        session.query(model).filter_by(deck_id=deck.id).delete()
    session.delete(deck)
    session.commit()


# (source_id, main deck member path within its archive)
TARGETS = [
    (
        "nhtsa-honda-accord-2014-oblique-fe-model",
        "01_OBLIQUE/01_LEFT_SIDE_IMPACT/25_50_NHTSA_OBLIQUE_L-ACCORD_Full_model.key",
    ),
    (
        "nhtsa-silverado-2007-fe-model",
        "SILVERADO-2017/BASELINE/SAMPLE_NCAP_FRONTAL_SETUP/BASELINE/BASE_NCAP_FRONTAL.key",
    ),
    ("nhtsa-toyota-yaris-2010-fe-model", "Yaris/Combine.key"),
    ("nhtsa-dodge-neon-1996-fe-model", "neon-0.7/Combine.k"),
]


def _get_or_create_knowledge_source(session: Session, meta: dict[str, object]) -> KnowledgeSource:
    ks = session.query(KnowledgeSource).filter_by(source_key=meta["source_id"]).one_or_none()
    if ks is None:
        ks = KnowledgeSource(
            source_key=meta["source_id"],
            source_type=meta["category"],
            authority_level=meta["authority"],
            local_path=meta["original_path"],
        )
        session.add(ks)
        session.flush()
    return ks


def main() -> None:
    with Session(get_engine()) as session:
        for source_id, main_member in TARGETS:
            meta = get_source(source_id)
            archive_path = ROOT / meta["original_path"]
            manifest = inspect_archive(archive_path)

            result = resolve_recursive(archive_path, main_member, manifest.members)

            member_by_path = {m.path: m for m in manifest.members}
            file_meta = {
                relpath: FileMeta(
                    sha256=member_by_path[relpath].sha256 or "",
                    size_bytes=member_by_path[relpath].size_bytes,
                    archive_member_path=relpath,
                )
                for relpath in result.decks
            }

            ks = _get_or_create_knowledge_source(session, meta)
            deck_key = f"{source_id}::{main_member}"
            _clear_existing_deck(session, deck_key)
            deck = persist_deck(
                session,
                knowledge_source_id=ks.id,
                deck_key=deck_key,
                main_file_relpath=main_member,
                decks=result.decks,
                graph=result.graph,
                file_meta=file_meta,
            )
            skipped_note = f", {len(result.skipped_too_large)} skipped (too large)" if result.skipped_too_large else ""
            truncated_note = " [TRUNCATED at max_files]" if result.truncated else ""
            print(
                f"[{deck.include_status}] {source_id}: {len(result.decks)} deck file(s) parsed"
                f"{skipped_note}{truncated_note}"
            )


if __name__ == "__main__":
    main()
