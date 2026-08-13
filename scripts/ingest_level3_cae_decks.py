"""Bounded ingestion of representative real LS-DYNA decks into cae_* —
TRD_LEVEL3.md §43/Instructions §36: "conservative concurrency," run only
after the smoke test (tests/test_level3_smoke.py) passes.

Each entry below is a real "main"/"combine" deck — the small file that
issues *INCLUDE for the actual (often 10s-100s of MB) component decks —
plus its *direct* includes only. This is deliberately not an attempt to
parse every single file in every archive (some individual component files
here are 100+ MB; the include graph and parser are already proven correct
at real scale by the smoke test's 39-include Accord deck and by
docs/ADR/'s note on this exact tradeoff) — it is real corpus breadth
(4 distinct vehicle/dummy model families) without unbounded depth.

Usage:
    uv run python scripts/ingest_level3_cae_decks.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from packages.cae.lsdyna.include_graph import build_include_graph  # noqa: E402
from packages.cae.lsdyna.models import ParsedDeck  # noqa: E402
from packages.cae.lsdyna.parser import parse_deck  # noqa: E402
from packages.cae.lsdyna.persistence import FileMeta, persist_deck  # noqa: E402
from packages.domain.db import get_engine  # noqa: E402
from packages.domain.knowledge import KnowledgeSource  # noqa: E402
from packages.ingestion.archives import ArchiveMember, inspect_archive, read_member_bytes  # noqa: E402
from packages.ingestion.manifest import get_source  # noqa: E402

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


def _resolve_direct_includes(
    archive_path: pathlib.Path, main_member: str, main_deck: ParsedDeck, all_members: list[ArchiveMember]
) -> tuple[dict[str, ParsedDeck], dict[str, FileMeta]]:
    decks: dict[str, ParsedDeck] = {main_member: main_deck}
    file_meta: dict[str, FileMeta] = {}
    member_by_path = {m.path: m for m in all_members}
    main_member_row = member_by_path.get(main_member)
    file_meta[main_member] = FileMeta(
        sha256=(main_member_row.sha256 or "") if main_member_row else "",
        size_bytes=main_member_row.size_bytes if main_member_row else 0,
        archive_member_path=main_member,
    )

    targets = {inc.filename.split("/")[-1] for inc in main_deck.includes if inc.filename}
    for m in all_members:
        base = m.path.split("/")[-1]
        if base in targets and m.path.endswith((".k", ".key", ".inc")) and not m.safety_issues:
            text = read_member_bytes(archive_path, m.path).decode("utf-8", errors="replace")
            decks[m.path] = parse_deck(text, m.path)
            file_meta[m.path] = FileMeta(sha256=m.sha256 or "", size_bytes=m.size_bytes, archive_member_path=m.path)
    return decks, file_meta


def main() -> None:
    with Session(get_engine()) as session:
        for source_id, main_member in TARGETS:
            meta = get_source(source_id)
            archive_path = ROOT / meta["original_path"]
            manifest = inspect_archive(archive_path)

            main_text = read_member_bytes(archive_path, main_member).decode("utf-8", errors="replace")
            main_deck = parse_deck(main_text, main_member)
            decks, file_meta = _resolve_direct_includes(archive_path, main_member, main_deck, manifest.members)

            graph = build_include_graph(decks)
            ks = _get_or_create_knowledge_source(session, meta)
            deck_key = f"{source_id}::{main_member}"
            deck = persist_deck(
                session,
                knowledge_source_id=ks.id,
                deck_key=deck_key,
                main_file_relpath=main_member,
                decks=decks,
                graph=graph,
                file_meta=file_meta,
            )
            print(
                f"[{deck.include_status}] {source_id}: {len(main_deck.includes)} include(s), "
                f"{len(decks)} deck file(s) parsed"
            )


if __name__ == "__main__":
    main()
