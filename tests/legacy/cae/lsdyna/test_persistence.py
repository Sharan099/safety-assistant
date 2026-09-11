"""Persisting a parsed deck + include graph into cae_* — TRD_LEVEL3.md §19."""

import uuid

from sqlalchemy.orm import Session

from packages.cae.lsdyna.include_graph import build_include_graph
from packages.cae.lsdyna.parser import parse_deck
from packages.cae.lsdyna.persistence import FileMeta, persist_deck
from packages.domain.cae import CaeFile, CaeInclude, CaeKeyword, CaeMaterial, CaePart
from packages.domain.knowledge import KnowledgeSource
from tests.legacy.conftest import requires_db


def _make_knowledge_source(session: Session, key: str) -> KnowledgeSource:
    ks = KnowledgeSource(
        source_key=key,
        source_type="NHTSA_VEHICLE_MODEL",
        authority_level="PUBLIC_RESEARCH",
        local_path=f"Knowledge source/{key}.zip",
    )
    session.add(ks)
    session.flush()
    return ks


@requires_db
def test_persist_deck_writes_files_keywords_and_structured_entities(session: Session) -> None:
    ks = _make_knowledge_source(session, f"test-persist-{uuid.uuid4().hex[:8]}")

    main_text = "*KEYWORD\n*INCLUDE\nvehicle.k\n*END\n"
    vehicle_text = "*KEYWORD\n*PART\nBody\n1,2,3\n*MAT_ELASTIC\n3,7.85e-9,2.1e5,0.3\n*END\n"
    decks = {
        "main.key": parse_deck(main_text, "main.key"),
        "vehicle.k": parse_deck(vehicle_text, "vehicle.k"),
    }
    graph = build_include_graph(decks)

    deck_key = f"{ks.source_key}::main.key"
    deck = persist_deck(
        session,
        knowledge_source_id=ks.id,
        deck_key=deck_key,
        main_file_relpath="main.key",
        decks=decks,
        graph=graph,
        file_meta={
            "main.key": FileMeta(sha256="a" * 64, size_bytes=100),
            "vehicle.k": FileMeta(sha256="b" * 64, size_bytes=200, archive_member_path="vehicle.k"),
        },
    )

    assert deck.include_status == "COMPLETE"

    files = session.query(CaeFile).filter_by(deck_id=deck.id).all()
    assert {f.relpath for f in files} == {"main.key", "vehicle.k"}

    keywords = session.query(CaeKeyword).filter_by(deck_id=deck.id).all()
    assert {k.keyword for k in keywords} >= {"KEYWORD", "INCLUDE", "END", "PART", "MAT_ELASTIC"}

    parts = session.query(CaePart).filter_by(deck_id=deck.id).all()
    assert len(parts) == 1
    assert parts[0].part_id == 1
    assert parts[0].material_id == 3

    materials = session.query(CaeMaterial).filter_by(deck_id=deck.id).all()
    assert len(materials) == 1
    assert materials[0].material_id == 3

    includes = session.query(CaeInclude).filter_by(deck_id=deck.id).all()
    assert len(includes) == 1
    assert includes[0].status == "RESOLVED"
    assert includes[0].resolved_file_id is not None


@requires_db
def test_persist_deck_is_idempotent_per_deck_key(session: Session) -> None:
    ks = _make_knowledge_source(session, f"test-idempotent-{uuid.uuid4().hex[:8]}")
    decks = {"a.k": parse_deck("*KEYWORD\n*PART\ntitle\n1,1,1\n*END\n", "a.k")}
    graph = build_include_graph(decks)
    deck_key = f"{ks.source_key}::a.k"

    first = persist_deck(
        session, knowledge_source_id=ks.id, deck_key=deck_key, main_file_relpath="a.k", decks=decks, graph=graph
    )
    second = persist_deck(
        session, knowledge_source_id=ks.id, deck_key=deck_key, main_file_relpath="a.k", decks=decks, graph=graph
    )

    assert first.id == second.id
    assert session.query(CaeFile).filter_by(deck_id=first.id).count() == 1
