"""Reranker — TRD_LEVEL3.md §21/§25/§31/§37."""

import uuid

from packages.retrieval.rerank import CrossEncoderReranker, LexicalAuthorityReranker, RerankCandidate, rerank


def _candidate(content: str, authority: str = "REFERENCE", fused_score: float = 0.1) -> RerankCandidate:
    return RerankCandidate(id=uuid.uuid4(), content=content, authority_level=authority, fused_score=fused_score)


def test_reranker_boosts_exact_term_overlap() -> None:
    reranker = LexicalAuthorityReranker()
    a = _candidate("contact automatic surface to surface definition in LS-DYNA")
    b = _candidate("completely unrelated content about seat belts")
    observation = rerank(reranker, "contact automatic surface definition", [a, b])
    assert observation.results[0].id == a.id


def test_reranker_boosts_higher_authority_when_otherwise_close() -> None:
    reranker = LexicalAuthorityReranker()
    reference = _candidate("frontal collision protection requirements", authority="REFERENCE", fused_score=0.1)
    authoritative = _candidate("frontal collision protection requirements", authority="AUTHORITATIVE", fused_score=0.1)
    observation = rerank(reranker, "frontal collision protection", [reference, authoritative])
    assert observation.results[0].id == authoritative.id


def test_reranker_never_lets_authority_override_a_much_stronger_fused_score() -> None:
    reranker = LexicalAuthorityReranker()
    weak_but_authoritative = _candidate("x", authority="AUTHORITATIVE", fused_score=0.001)
    strong_but_reference = _candidate("x", authority="REFERENCE", fused_score=0.9)
    observation = rerank(reranker, "x", [weak_but_authoritative, strong_but_reference])
    assert observation.results[0].id == strong_but_reference.id


def test_rerank_observation_records_model_identity_and_latency() -> None:
    reranker = LexicalAuthorityReranker()
    observation = rerank(reranker, "query", [_candidate("content")])
    assert observation.model_name == "lexical-authority-heuristic"
    assert observation.model_version == "v1"
    assert observation.latency_ms >= 0.0


def test_rerank_results_are_sorted_descending() -> None:
    reranker = LexicalAuthorityReranker()
    candidates = [_candidate("x", fused_score=0.1), _candidate("x", fused_score=0.9), _candidate("x", fused_score=0.5)]
    observation = rerank(reranker, "irrelevant query", candidates)
    scores = [r.rerank_score for r in observation.results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_candidates_returns_empty() -> None:
    reranker = LexicalAuthorityReranker()
    observation = rerank(reranker, "query", [])
    assert observation.results == []


# CrossEncoderReranker — docs/ADR/0015. No @requires_db: exercises the
# model directly, not the database. Downloads the small ONNX model to the
# local fastembed cache on first run (~80 MB) if not already cached.
def test_cross_encoder_ranks_the_genuinely_relevant_candidate_first() -> None:
    reranker = CrossEncoderReranker()
    relevant = _candidate("Frontal collision protection requirements for occupant restraint systems.")
    irrelevant = _candidate("The recipe calls for two cups of flour and a pinch of salt.")
    observation = rerank(reranker, "frontal collision occupant protection", [irrelevant, relevant])
    assert observation.results[0].id == relevant.id


def test_cross_encoder_score_is_the_raw_model_logit_not_summed_with_fused_score() -> None:
    """Unlike LexicalAuthorityReranker's additive adjustment, a real
    cross-encoder fully re-scores — a candidate with a much higher
    fused_score but irrelevant content must still lose to a genuinely
    relevant one with a low fused_score."""
    reranker = CrossEncoderReranker()
    relevant_but_low_fused = _candidate(
        "Frontal collision protection requirements for occupant restraint systems.", fused_score=0.001
    )
    irrelevant_but_high_fused = _candidate(
        "The recipe calls for two cups of flour and a pinch of salt.", fused_score=100.0
    )
    observation = rerank(
        reranker, "frontal collision occupant protection", [irrelevant_but_high_fused, relevant_but_low_fused]
    )
    assert observation.results[0].id == relevant_but_low_fused.id


def test_cross_encoder_model_identity_is_reported() -> None:
    reranker = CrossEncoderReranker()
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert reranker.model_version == "v1"


def test_cross_encoder_empty_candidates_returns_empty() -> None:
    reranker = CrossEncoderReranker()
    observation = rerank(reranker, "query", [])
    assert observation.results == []
