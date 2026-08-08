"""Fix 5 acronyms config + DESIGN_IMPLICATION component→concept map."""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.acronyms import (
    DEFAULT_CONFIG_PATH,
    expand_acronyms,
    get_acronyms,
    load_acronym_table,
)
from retrieval.design_implication import (
    expand_design_query,
    load_design_components,
    match_component,
)


def test_acronyms_config_loads_and_expands():
    table, phrases, meta = load_acronym_table(force=True)
    assert DEFAULT_CONFIG_PATH.exists()
    assert meta.get("review_status") == "agent_reviewed"
    assert meta.get("production_approved") is False
    assert table["HPC"] == "Head Performance Criterion"
    assert table["RDC"] == "Rib Deflection Criterion"
    assert table["ELR"] == "Emergency Locking Retractor"
    assert table["MDB"] == "mobile deformable barrier"
    assert "APF" in table
    assert expand_acronyms("What is the VC limit?").startswith("What is the VC (Viscous Criterion)")
    assert "RDC" in expand_acronyms("Rib Deflection of 45 mm")
    assert get_acronyms()["ThCC"] == "Thorax Compression Criterion"
    assert phrases  # phrase synonyms compiled


def test_design_components_agent_reviewed_meta():
    path = Path("config/design_components.json")
    meta = json.loads(path.read_text(encoding="utf-8"))["_meta"]
    assert meta["review_status"] == "agent_reviewed"
    assert meta["production_approved"] is False
    assert "B-pillar" in " ".join(meta.get("review_decisions") or []) or any(
        "B-pillar" in d for d in meta.get("review_decisions") or []
    )
    specs = load_design_components(force=True)
    ids = {s.id for s in specs}
    assert {"b_pillar", "seat_belt_geometry", "drivers_seat", "side_door", "reess_mounting"} <= ids


def test_b_pillar_maps_to_r95_side_concepts():
    exp = expand_design_query("What requirements affect B-Pillar design?")
    assert exp.component and exp.component.id == "b_pillar"
    assert exp.likely_regulations[0] == "UN-ECE-R95"
    blob = " ".join(exp.concepts).lower()
    assert "intrusion" in blob
    assert "passenger compartment" in blob
    assert "side impact" in blob or "side" in blob


def test_seat_belt_geometry_maps_to_r16_and_impact_regs():
    for q in (
        "What requirements affect seat belt geometry?",
        "How does safety-belt routing affect design?",
        "What requirements affect the seat belt design?",
    ):
        spec = match_component(q)
        assert spec is not None, q
        assert spec.id == "seat_belt_geometry", (q, spec.id)
    exp = expand_design_query("What requirements affect seat belt geometry?")
    assert exp.likely_regulations[0] == "UN-ECE-R16"
    assert "UN-ECE-R94" in exp.likely_regulations
    assert "UN-ECE-R95" in exp.likely_regulations
    assert any("anchorage" in c.lower() or "geometry" in c.lower() for c in exp.concepts)
