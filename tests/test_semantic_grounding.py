"""Semantic groundedness: negative-claim filter + claim/chunk support."""

from __future__ import annotations

from generation.answer import AnswerSegment, StructuredAnswer, parse_structured_answer
from generation.semantic_grounding import (
    filter_unsupported_negative_segments,
    heuristic_claim_supported,
    is_negative_or_absence_claim,
)
from retrieval.retrieve import RetrievedChunk


def test_detects_confident_negative_claims():
    assert is_negative_or_absence_claim(
        "There is no direct relationship between UN R16 and UN R94."
    )
    assert is_negative_or_absence_claim("This is not covered in the regulation.")
    assert not is_negative_or_absence_claim(
        "UN R94 limits ThCC to 42 mm for the 50th percentile male dummy."
    )


def test_negative_claim_rejected_without_explicit_absence_in_chunk():
    ok, reason = heuristic_claim_supported(
        "There is no direct relationship between UN R16 and UN R94.",
        "Annex 2 Arrangements of the approval mark Model A (See paragraph 4.5.)",
    )
    assert not ok
    assert "without_explicit_absence" in reason


def test_negative_claim_allowed_when_chunk_states_non_applicability():
    ok, _ = heuristic_claim_supported(
        "This requirement does not apply to category N1 vehicles.",
        "This regulation does not apply to vehicles of category N1.",
    )
    assert ok


def test_filter_drops_unsupported_negative_segment():
    segs = [
        AnswerSegment(
            text="There is no direct relationship between UN R16 and UN R94.",
            citation_chunk_id="annex2",
        ),
        AnswerSegment(
            text="Safety-belts shall meet the strength requirements.",
            citation_chunk_id="r16_6",
        ),
    ]
    by_id = {
        "annex2": RetrievedChunk(
            chunk_id="annex2",
            text="Annex 2 Arrangements of the approval mark Model A",
            regulation_id="UN-ECE-R16",
            section_number="Annex 2",
            page_number=46,
        ),
        "r16_6": RetrievedChunk(
            chunk_id="r16_6",
            text="Safety-belts shall meet the strength requirements of paragraph 6.",
            regulation_id="UN-ECE-R16",
            section_number="6",
        ),
    }
    kept, dropped = filter_unsupported_negative_segments(segs, by_id)
    assert len(dropped) == 1
    assert "no direct relationship" in dropped[0].lower()
    assert len(kept) == 1
    assert kept[0].citation_chunk_id == "r16_6"


def test_structured_empty_after_filter_means_not_found_path():
    """All-negative unsupported → empty segments (backend not_found)."""
    segs = [
        AnswerSegment(
            text="UN R16 and UN R94 have no relationship.",
            citation_chunk_id="x",
        )
    ]
    by_id = {
        "x": RetrievedChunk(
            chunk_id="x",
            text="Approval mark affixed to a safety-belt",
            regulation_id="UN-ECE-R16",
            section_number="Annex 2",
        )
    }
    kept, dropped = filter_unsupported_negative_segments(segs, by_id)
    assert not kept and dropped
    structured = StructuredAnswer(answer_segments=list(kept))
    assert structured.answer_segments == []


def test_llm_claim_supported_fail_closed_on_unparseable(monkeypatch):
    from generation import semantic_grounding as sg

    class _FakeLLM:
        def complete(self, **kwargs):
            class R:
                text = "MAYBE???"

            return R()

    res = sg.llm_claim_supported(
        "HPC shall not exceed 1000",
        "Some passage",
        llm=_FakeLLM(),
        question="What is HPC?",
    )
    assert res.supported is False
    assert "fail_closed" in res.reason


def test_drop_unsupported_audit_segments():
    from generation.semantic_grounding import (
        SemanticGroundednessReport,
        SegmentSupportResult,
        drop_unsupported_audit_segments,
    )

    segs = [
        AnswerSegment(text="claim a", citation_chunk_id="a"),
        AnswerSegment(text="claim b", citation_chunk_id="b"),
    ]
    report = SemanticGroundednessReport(
        question="q",
        results=[
            SegmentSupportResult(
                claim="claim a", chunk_id="a", supported=False, method="llm", reason="no"
            ),
            SegmentSupportResult(
                claim="claim b", chunk_id="b", supported=True, method="llm", reason="yes"
            ),
        ],
    )
    kept, dropped = drop_unsupported_audit_segments(segs, report)
    assert len(kept) == 1
    assert kept[0].citation_chunk_id == "b"
    assert dropped
