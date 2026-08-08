"""Unit tests for RERANK_MIN_SCORE relevance floor."""

from __future__ import annotations

from retrieval.rerank import Reranker, rerank
from retrieval.retrieve import RetrievedChunk


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        enriched_text=text,
        regulation_id="UN-ECE-R94",
        revision="Rev.3",
        section_number="5.2.1",
        section_title="",
        section_id=f"UN-ECE-R94::{cid}",
        score=0.0,
    )


def test_rerank_min_score_drops_low_relevance(monkeypatch):
    chunks = [
        _chunk("a", "highly relevant femur force criterion text"),
        _chunk("b", "somewhat related injury criteria"),
        _chunk("c", "unrelated administrative approval form"),
        _chunk("d", "noise page header"),
        _chunk("e", "more noise"),
    ]
    # Deterministic scores in chunk order (rerank will sort desc).
    scores = [0.9, 0.4, 0.05, -0.2, -0.5]

    class Fake(Reranker):
        def __init__(self):
            self.provider = "none"
            self.model_name = "fake"

        def score(self, query, texts):
            assert len(texts) == len(scores)
            return list(scores)

    monkeypatch.delenv("RERANK_MIN_SCORE", raising=False)
    kept_all = rerank("femur", chunks, top_n=5, min_score=None, reranker=Fake())
    assert [c.chunk_id for c in kept_all] == ["a", "b", "c", "d", "e"]

    kept = rerank("femur", chunks, top_n=5, min_score=0.3, reranker=Fake())
    assert [c.chunk_id for c in kept] == ["a", "b"]
    assert all(float(c.score) >= 0.3 for c in kept)


def test_rerank_min_score_keeps_best_when_all_below(monkeypatch):
    chunks = [_chunk("x", "weak"), _chunk("y", "weaker")]

    class Fake(Reranker):
        def __init__(self):
            self.provider = "none"
            self.model_name = "fake"

        def score(self, query, texts):
            return [0.1, 0.05]

    kept = rerank("q", chunks, top_n=5, min_score=0.5, reranker=Fake())
    assert len(kept) == 1
    assert kept[0].chunk_id == "x"


def test_rerank_score_margin_keeps_near_best(monkeypatch):
    chunks = [
        _chunk("a", "best"),
        _chunk("b", "close"),
        _chunk("c", "far"),
    ]

    class Fake(Reranker):
        def __init__(self):
            self.provider = "none"
            self.model_name = "fake"

        def score(self, query, texts):
            return [30.0, 29.0, 20.0]

    kept = rerank("q", chunks, top_n=5, min_score=None, score_margin=1.5, reranker=Fake())
    assert [c.chunk_id for c in kept] == ["a", "b"]
