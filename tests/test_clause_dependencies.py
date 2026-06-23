"""Part B: clause dependency map and §6.4 denormalization."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.clause_dependencies import (
    build_denormalized_block,
    enrich_clause_dependency_meta,
    infer_test_configuration_from_query,
    resolve_test_configuration,
)


def test_rear_centre_no_retractor_maps_to_642():
    clause = infer_test_configuration_from_query(
        "UN R14 M1 rear centre seat belt anchorage load without retractor"
    )
    assert clause == "6.4.2"
    cfg = resolve_test_configuration("6.4.2.1")
    assert cfg["seat_position"] == "rear_centre"
    assert cfg["retractor_required"] == "not_required"


def test_denormalized_block_inlines_angle_and_duration():
    block = build_denormalized_block("6.4.2.1", "UN_R14")
    assert "6.3.2" in block or "10 degrees" in block
    assert "0.2 second" in block
    assert "6.3.5" in block or "without retractor" in block.lower()


def test_clause_meta_includes_general_conditions():
    meta = enrich_clause_dependency_meta(regulation="UN_R14", clause_number="6.4.2.1")
    assert "6.3.2" in meta["applicable_general_conditions"]
    assert "6.3.3" in meta["applicable_general_conditions"]
    assert meta["applicable_test_configuration"] == "rear_centre_no_retractor"
