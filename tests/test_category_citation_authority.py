"""Category citation authority and eval fixture validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.citations import (  # noqa: E402
    chunk_authority_for_categories,
    validate_category_citation_authority,
)


def test_not_m1_n1_clause_rejected_for_m1_n1_citation():
    """Q15 regression — contrast mention must not confer authority."""
    doc = {
        "applies_to_category": ["NOT_M1_N1", "M3_N3"],
        "text": (
            "In the case of vehicles of categories other than M1 and N1, "
            "the test load shall be 675 daN; for M1 and N1 vehicles 1,350 daN applies."
        ),
    }
    cite = {"marker": "S1", "document": "UN R14"}
    flags = validate_category_citation_authority(
        "What is the M1/N1 lower anchorage test load under UN R14?",
        [cite],
        [doc],
    )
    assert flags
    assert flags[0]["type"] == "applicability_mismatch"


def test_m1_n1_clause_accepted_for_m1_n1_citation():
    doc = {"applies_to_category": ["M1_N1"], "text": "1,350 daN ± 20 daN"}
    assert chunk_authority_for_categories(doc, {"M1_N1"}) is True
    flags = validate_category_citation_authority(
        "M1/N1 anchorage load UN R14?",
        [{"marker": "S1", "document": "UN R14"}],
        [doc],
    )
    assert not flags


@pytest.mark.parametrize(
    "fixture",
    [
        "test_cases_arithmetic.json",
        "test_cases_definitions.json",
    ],
)
def test_eval_fixtures_valid(fixture: str):
    path = ROOT / "tests" / fixture
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(cases, list) and cases
    for case in cases:
        assert case.get("id")
        assert case.get("question")
        assert case.get("mode")
