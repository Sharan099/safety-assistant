"""Structured CAE retrieval — TRD_LEVEL3.md §18/§23."""

import uuid

from sqlalchemy.orm import Session

from packages.cae.lsdyna.include_graph import build_include_graph
from packages.cae.lsdyna.parser import parse_deck
from packages.cae.lsdyna.persistence import persist_deck
from packages.domain.knowledge import KnowledgeSource
from packages.retrieval.structured import (
    find_deck_by_key,
    get_contacts_in_deck,
    get_control_cards_in_deck,
    get_database_outputs_in_deck,
    get_includes_for_deck,
    get_materials_for_part,
    get_parts_using_material,
    get_section_for_part,
)
from tests.conftest import requires_db

DECK_TEXT = """\
*KEYWORD
*PART
Body panel
1042,2,3
*SECTION_SHELL
2,16,0.0
*MAT_024
3,7.85e-9,2.1e5,0.3
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
10,11
*CONTROL_TERMINATION
100.0
*DATABASE_BINARY_D3PLOT
1.0
*INCLUDE
sibling.k
*END
"""


def _seed(session: Session) -> uuid.UUID:
    ks = KnowledgeSource(
        source_key=f"test-structured-{uuid.uuid4().hex[:8]}",
        source_type="NHTSA_VEHICLE_MODEL",
        authority_level="PUBLIC_RESEARCH",
        local_path="Knowledge source/x.zip",
    )
    session.add(ks)
    session.flush()

    decks = {"main.k": parse_deck(DECK_TEXT, "main.k"), "sibling.k": parse_deck("*KEYWORD\n*END\n", "sibling.k")}
    graph = build_include_graph(decks)
    deck = persist_deck(
        session,
        knowledge_source_id=ks.id,
        deck_key=f"{ks.source_key}::main.k",
        main_file_relpath="main.k",
        decks=decks,
        graph=graph,
    )
    return deck.id


@requires_db
def test_get_materials_for_part_answers_the_prd_example_question(session: Session) -> None:
    deck_id = _seed(session)
    materials = get_materials_for_part(session, deck_id, 1042)
    assert len(materials) == 1
    assert materials[0].material_id == 3
    assert materials[0].mat_type == "024"


@requires_db
def test_get_materials_for_part_returns_empty_for_unknown_part_not_fabricated(session: Session) -> None:
    deck_id = _seed(session)
    assert get_materials_for_part(session, deck_id, 999999) == []


@requires_db
def test_get_section_for_part(session: Session) -> None:
    deck_id = _seed(session)
    section = get_section_for_part(session, deck_id, 1042)
    assert section is not None
    assert section.section_id == 2
    assert section.section_type == "SHELL"


@requires_db
def test_get_parts_using_material(session: Session) -> None:
    deck_id = _seed(session)
    parts = get_parts_using_material(session, deck_id, 3)
    assert len(parts) == 1
    assert parts[0].part_id == 1042


@requires_db
def test_get_contacts_in_deck(session: Session) -> None:
    deck_id = _seed(session)
    contacts = get_contacts_in_deck(session, deck_id)
    assert len(contacts) == 1
    assert contacts[0].contact_type == "AUTOMATIC_SURFACE_TO_SURFACE"
    assert contacts[0].ssid == 10
    assert contacts[0].msid == 11


@requires_db
def test_get_control_cards_in_deck(session: Session) -> None:
    deck_id = _seed(session)
    controls = get_control_cards_in_deck(session, deck_id)
    assert len(controls) == 1
    assert controls[0].control_type == "TERMINATION"


@requires_db
def test_get_database_outputs_in_deck(session: Session) -> None:
    deck_id = _seed(session)
    outputs = get_database_outputs_in_deck(session, deck_id)
    assert len(outputs) == 1
    assert outputs[0].database_type == "BINARY_D3PLOT"
    assert outputs[0].dt == 1.0


@requires_db
def test_get_includes_for_deck(session: Session) -> None:
    deck_id = _seed(session)
    includes = get_includes_for_deck(session, deck_id)
    assert len(includes) == 1
    assert includes[0].target_as_written == "sibling.k"
    assert includes[0].status == "RESOLVED"


@requires_db
def test_find_deck_by_key(session: Session) -> None:
    deck_id = _seed(session)
    # Re-find via a fresh query rather than assuming the seeded object stays around.
    from packages.domain.cae import CaeDeck

    row = session.get(CaeDeck, deck_id)
    assert row is not None
    found = find_deck_by_key(session, row.deck_key)
    assert found is not None
    assert found.id == deck_id
