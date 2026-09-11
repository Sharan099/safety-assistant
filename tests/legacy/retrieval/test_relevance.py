"""Direct regression test for CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 8's
"deliberately irrelevant retrieval result... must be rejected" — using the
exact category of false positive the PRD addendum reported (an unrelated
AES/Rijndael passage surfacing as evidence for an engineering question).
"""

from packages.retrieval.relevance import has_known_authority, is_relevant, shared_term_count

IRRELEVANT_PASSAGE = (
    "The Advanced Encryption Standard (AES), also known as Rijndael, is a specification "
    "for the encryption of electronic data. It supersedes the Data Encryption Standard "
    "(DES) and uses a substitution-permutation network with key sizes of 128, 192, or 256 bits."
)

RELEVANT_PASSAGE = (
    "The belt force-limiter level was reduced from 4000 N to 3500 N between Run A and "
    "Run B; the webbing revision was also updated as part of the same restraint change."
)


def test_irrelevant_passage_rejected_for_engineering_query() -> None:
    query = "why did chest deflection increase due to belt force limiter and webbing revision"
    assert not is_relevant(query, IRRELEVANT_PASSAGE)


def test_relevant_passage_accepted_for_engineering_query() -> None:
    query = "why did chest deflection increase due to belt force limiter and webbing revision"
    assert is_relevant(query, RELEVANT_PASSAGE)


def test_shared_term_count_is_literal_not_semantic() -> None:
    # "encryption" and "restraint" share zero literal vocabulary.
    assert shared_term_count("restraint belt force limiter", IRRELEVANT_PASSAGE) == 0


def test_empty_query_does_not_reject_everything() -> None:
    assert is_relevant("", IRRELEVANT_PASSAGE)


def test_known_authority_levels() -> None:
    assert has_known_authority("REGULATION") is False  # this is a source_type, not an authority_level
    assert has_known_authority("AUTHORITATIVE")
    assert has_known_authority("OFFICIAL_DOCUMENTATION")
    assert has_known_authority("NOT_A_REAL_LEVEL") is False
