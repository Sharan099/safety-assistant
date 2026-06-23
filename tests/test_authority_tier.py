"""Authority tier architecture — Part A regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.authority_tier import (
    ENGINEERING_REF,
    LEGAL_BINDING,
    RATING_PROTOCOL,
    SYNTHETIC,
    badge_for_tier,
    resolve_authority_tier,
)
from backend.app.core.document_registry import get_document_meta
from backend.app.retrieval.authority_filter import is_compliance_determination_query
from backend.app.retrieval.citations import (
    build_citation,
    detect_authority_blur_flags,
    score_authority_correctness,
)
from backend.app.retrieval.query_intent import detect_query_intent


def test_registry_has_authority_tier():
    meta = get_document_meta("UN_R14")
    assert meta.authority_tier == LEGAL_BINDING
    assert get_document_meta("EURO_NCAP").authority_tier == RATING_PROTOCOL
    assert get_document_meta("PROG_X_FT_001").authority_tier == SYNTHETIC


def test_compliance_query_binding_only_intent():
    intent = detect_query_intent("Is this anchorage design compliant with UN R14?")
    assert intent.compliance_determination
    assert intent.binding_authority_only


def test_citation_includes_tier_badge():
    cite = build_citation({
        "regulation": "UN_R14",
        "text": "test load 1,350 daN",
        "heading_path": "UN_R14 > 6.4.1",
    }, 1)
    assert cite["authority_tier_badge"] == "LEGAL"
    assert cite["is_legal"]


def test_authority_blur_flagged_when_advisory_cited_as_binding():
    citations = [{
        "marker": "S1",
        "authority_tier": ENGINEERING_REF,
        "authority_tier_badge": badge_for_tier(ENGINEERING_REF),
    }]
    flags = detect_authority_blur_flags(
        "This anchorage load is required by regulation at 1,350 daN [S1].",
        citations,
    )
    assert any(f["type"] == "authority_blur" for f in flags)


def test_authority_correctness_passes_with_advisory_framing():
    citations = [{
        "marker": "S1",
        "authority_tier": RATING_PROTOCOL,
        "authority_tier_badge": "RATING",
    }]
    score = score_authority_correctness(
        "Euro NCAP recommends a higher score target; this is not legally binding [S1].",
        citations,
    )
    assert score["passed"]
