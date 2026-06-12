"""Tests for answer-scoped, deduplicated revision flags (citations.py)."""

from backend.app.retrieval.citations import (
    derive_answer_flags,
    extract_cited_markers,
)


def _cite(marker: str, regulation: str, doc_type: str) -> dict:
    return {
        "marker": marker,
        "regulation": regulation,
        "doc_type": doc_type,
        "doc_type_label": doc_type,
    }


def test_extract_cited_markers_parses_inline_markers():
    answer = "Load is 13.5 kN [S1] applied for 0.2 s [S3]. See also [S1]."
    assert extract_cited_markers(answer) == {"S1", "S3"}


def test_extract_cited_markers_empty():
    assert extract_cited_markers("") == set()
    assert extract_cited_markers("no markers here") == set()


def test_abstain_returns_no_flags():
    cites = [_cite("S1", "UN_R94", "legal_regulation")]
    assert derive_answer_flags(cites, answer_text="anything [S1]", should_abstain=True) == []


def test_flags_scoped_to_cited_markers():
    # Both a legal regulation and a rating protocol are retrieved, but only the
    # legal one is actually cited in the answer -> no mixed-doc-types flag.
    cites = [
        _cite("S1", "UN_R94", "legal_regulation"),
        _cite("S2", "EuroNCAP_Frontal", "rating_protocol"),
    ]
    flags_only_legal = derive_answer_flags(cites, answer_text="value [S1]")
    assert not any(f["type"] == "mixed_doc_types" for f in flags_only_legal)

    # When the answer cites both, the mixed-doc-types flag fires once.
    flags_both = derive_answer_flags(cites, answer_text="legal [S1] vs rating [S2]")
    mixed = [f for f in flags_both if f["type"] == "mixed_doc_types"]
    assert len(mixed) == 1


def test_flags_deduplicated_per_regulation():
    # The same regulation cited via two markers must not duplicate its flag.
    cites = [
        _cite("S1", "UN_R94", "legal_regulation"),
        _cite("S2", "UN_R94", "legal_regulation"),
    ]
    flags = derive_answer_flags(cites, answer_text="claim a [S1]; claim b [S2]")
    keys = [(f["type"], f.get("regulation")) for f in flags]
    assert len(keys) == len(set(keys))


def test_no_answer_text_considers_all_citations():
    # Backward-compatible path: without answer_text, all citations are used.
    cites = [
        _cite("S1", "UN_R94", "legal_regulation"),
        _cite("S2", "EuroNCAP_Frontal", "rating_protocol"),
    ]
    flags = derive_answer_flags(cites)
    assert any(f["type"] == "mixed_doc_types" for f in flags)
