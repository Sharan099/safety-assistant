"""Level 3 real-corpus smoke test — Instructions §35, TRD_LEVEL3.md §42.

"Select representative actual files... Run: discover -> hash -> profile ->
extract -> parse -> validate -> index -> retrieve." Every step here touches
a real file under "Knowledge source/", not a fixture — this is the
end-to-end acceptance evidence for Level 3, run before attempting the full
1.3 GB corpus (TRD_LEVEL3.md §43).

Representative files (Instructions §35's list, matched against what the
real corpus actually contains):
  - 1 regulation PDF          -> UN_R94.pdf
  - 1 technical report        -> NHTSA structural countermeasure report
  - 1 ZIP with .k files       -> Oblique-Accord-Updated-PAB.zip (39-include deck)
  - 1 standalone .k           -> neon-0.7.tar's Combine.k (364 bytes)
  - 1 .key                    -> Yaris.tar.gz's Combine.key
  - 1 nested include deck     -> the same 39-include Accord assembly deck
  - 1 TAR                     -> neon-0.7.tar
  - 1 TAR.GZ                  -> Yaris.tar.gz
No scanned/mixed PDF is known to exist in this corpus (all PDFs found so
far have native text layers) — not assumed, per PRD_LEVEL3.md §4's own
rule against assuming a fixed corpus shape.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from packages.cae.lsdyna.include_graph import build_include_graph
from packages.cae.lsdyna.parser import parse_deck
from packages.cae.lsdyna.persistence import FileMeta, persist_deck
from packages.domain.knowledge import KnowledgeSource
from packages.ingestion.archives import inspect_archive, read_member_bytes, sha256_file
from packages.ingestion.manifest import get_source
from packages.ingestion.pipeline import REPO_ROOT, ingest_document
from packages.retrieval.index import index_chunks
from packages.retrieval.search import retrieve
from packages.retrieval.structured import get_includes_for_deck
from tests.legacy.conftest import requires_db

ACCORD_MAIN_MEMBER = "01_OBLIQUE/01_LEFT_SIDE_IMPACT/25_50_NHTSA_OBLIQUE_L-ACCORD_Full_model.key"


@requires_db
def test_smoke_regulation_pdf_discover_hash_extract_index_retrieve(session: Session) -> None:
    meta = get_source("unece-un-r94")
    pdf_path = REPO_ROOT / meta["canonical_path"]
    assert pdf_path.is_file()  # discover
    assert sha256_file(pdf_path) == meta["sha256"]  # hash

    revision = ingest_document(session, "unece-un-r94", max_pages=10)  # extract
    session.commit()
    assert revision.status == "READY"

    index_chunks(session)  # index
    results = retrieve(session, "frontal collision occupant protection", limit=5)  # retrieve
    assert results
    assert any(r.document_key == "unece-un-r94" for r in results)


@requires_db
def test_smoke_technical_report_pdf_discover_hash_extract(session: Session) -> None:
    meta = get_source("nhtsa-structural-countermeasure-research-report")
    pdf_path = REPO_ROOT / meta["canonical_path"]
    assert pdf_path.is_file()
    assert sha256_file(pdf_path) == meta["sha256"]

    revision = ingest_document(session, "nhtsa-structural-countermeasure-research-report", max_pages=10)
    session.commit()
    assert revision.status == "READY"


@requires_db
def test_smoke_zip_with_nested_include_deck_discover_hash_profile_parse_validate_index(session: Session) -> None:
    meta = get_source("nhtsa-honda-accord-2014-oblique-fe-model")
    archive_path = REPO_ROOT / meta["original_path"]
    assert archive_path.is_file()  # discover
    assert sha256_file(archive_path) == meta["sha256"]  # hash

    manifest = inspect_archive(archive_path)  # profile
    member_by_path = {m.path: m for m in manifest.members}
    assert ACCORD_MAIN_MEMBER in member_by_path

    main_text = read_member_bytes(archive_path, ACCORD_MAIN_MEMBER).decode("utf-8", errors="replace")
    main_deck = parse_deck(main_text, ACCORD_MAIN_MEMBER)  # parse
    assert len(main_deck.includes) == 39  # the real nested-include deck found during Phase 3-5's smoke testing

    decks = {ACCORD_MAIN_MEMBER: main_deck}
    file_meta = {
        ACCORD_MAIN_MEMBER: FileMeta(
            sha256=member_by_path[ACCORD_MAIN_MEMBER].sha256 or "",
            size_bytes=member_by_path[ACCORD_MAIN_MEMBER].size_bytes,
            archive_member_path=ACCORD_MAIN_MEMBER,
        )
    }
    targets = {inc.filename.split("/")[-1] for inc in main_deck.includes if inc.filename}
    for m in manifest.members:
        base = m.path.split("/")[-1]
        if base in targets and m.path.endswith((".k", ".key", ".inc")) and not m.safety_issues:
            text = read_member_bytes(archive_path, m.path).decode("utf-8", errors="replace")
            decks[m.path] = parse_deck(text, m.path)
            file_meta[m.path] = FileMeta(sha256=m.sha256 or "", size_bytes=m.size_bytes, archive_member_path=m.path)

    graph = build_include_graph(decks)  # validate
    assert graph.deck_status[ACCORD_MAIN_MEMBER] == "COMPLETE"
    assert all(e.status == "RESOLVED" for e in graph.edges)

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

    deck_key = f"{meta['source_id']}::{ACCORD_MAIN_MEMBER}"
    deck = persist_deck(  # index (structured)
        session,
        knowledge_source_id=ks.id,
        deck_key=deck_key,
        main_file_relpath=ACCORD_MAIN_MEMBER,
        decks=decks,
        graph=graph,
        file_meta=file_meta,
    )
    assert deck.include_status == "COMPLETE"

    includes = get_includes_for_deck(session, deck.id)  # retrieve (structured)
    assert len(includes) == 39
    assert all(i.status == "RESOLVED" for i in includes)


@requires_db
def test_smoke_tar_standalone_k_discover_hash_parse() -> None:
    meta = get_source("nhtsa-dodge-neon-1996-fe-model")
    archive_path = REPO_ROOT / meta["original_path"]
    assert archive_path.is_file()
    assert sha256_file(archive_path) == meta["sha256"]

    manifest = inspect_archive(archive_path)
    member = next(m for m in manifest.members if m.path == "neon-0.7/Combine.k")
    assert not member.safety_issues

    text = read_member_bytes(archive_path, member.path).decode("utf-8", errors="replace")
    deck = parse_deck(text, member.path)
    assert deck.all_cards  # genuinely parsed real content
    # Turns out Combine.k is itself a small include-issuing file (not the
    # includeless leaf originally assumed when this test was written) —
    # a real, useful finding: it exercises *INCLUDE_TRANSFORM, a variant
    # distinct from plain *INCLUDE that this corpus actually contains.
    assert {i.filename for i in deck.includes} == {"Neon.k", "US-NCAP/loadcellwall.k"}
    transform_include = next(i for i in deck.includes if i.filename == "US-NCAP/loadcellwall.k")
    assert transform_include.raw.keyword == "INCLUDE_TRANSFORM"


@requires_db
def test_smoke_targz_key_discover_hash_parse() -> None:
    meta = get_source("nhtsa-toyota-yaris-2010-fe-model")
    archive_path = REPO_ROOT / meta["original_path"]
    assert archive_path.is_file()
    assert sha256_file(archive_path) == meta["sha256"]

    manifest = inspect_archive(archive_path)
    member = next(m for m in manifest.members if m.path == "Yaris/Combine.key")
    assert not member.safety_issues

    text = read_member_bytes(archive_path, member.path).decode("utf-8", errors="replace")
    deck = parse_deck(text, member.path)
    assert deck.all_cards
