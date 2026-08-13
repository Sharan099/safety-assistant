"""Reranker — TRD_LEVEL3.md §21/§25/§31/§37."""

import uuid

from packages.retrieval.rerank import LexicalAuthorityReranker, RerankCandidate, rerank


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
