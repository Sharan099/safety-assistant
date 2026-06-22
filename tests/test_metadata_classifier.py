"""Phase 1 — metadata classifier tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.metadata_classifier import classify_chunk  # noqa: E402


@pytest.mark.parametrize(
    "text,regulation,expected_test,expected_value",
    [
        ("frontal impact chest deflection shall not exceed 34 mm", "UN_R94", "frontal", "legal_limit"),
        ("lateral collision WorldSID thorax deflection 42 mm", "UN_R95", "side", "legal_limit"),
        ("pole side impact at 32 km/h", "UN_R135", "pole_side", None),
        ("Euro NCAP adult occupant frontal score 0 points at 55 mm", "EURO_NCAP", "frontal", "rating_threshold"),
        ("internal design target chest deflection 30 mm", "SAFETY_REFERENCE", "frontal", "target"),
        ("safety belt anchorage test load 1350 daN", "UN_R14", "belt", "legal_limit"),
    ],
)
def test_classify_test_type_and_value(text, regulation, expected_test, expected_value):
    meta = classify_chunk(regulation=regulation, pdf_name="test.pdf", text=text)
    assert meta["test_type"] == expected_test
    assert meta["value_type"] == expected_value


def test_classify_doc_type_legal():
    meta = classify_chunk(regulation="UN_R94", pdf_name="UN_R94.pdf", text="frontal impact")
    assert meta["doc_type"] == "legal"
    assert meta["authority"] == "UNECE"


def test_classify_reference_not_legal():
    meta = classify_chunk(
        regulation="SAFETY_REFERENCE",
        pdf_name="SAFETY_COMPANION.pdf.pdf",
        text="frontal chest deflection target 30 mm",
    )
    assert meta["doc_type"] == "reference"
