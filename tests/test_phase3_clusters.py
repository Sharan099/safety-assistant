"""Phase 3 — corpus lock + requirement clusters (Q05)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.document_registry import (
    GHOST_REGULATIONS,
    INDEXED_LEGAL_CORPUS,
    cluster_member_codes,
    is_indexed_regulation,
    match_requirement_cluster,
)
from tests.eval_harness.scoring import retrieval_recall

Q05_QUESTION = (
    "Which regulations govern seat belt and restraint system design for a new vehicle?"
)


class TestCorpusLock:
    def test_indexed_corpus_defined(self):
        assert INDEXED_LEGAL_CORPUS == frozenset({
            "UN_R14", "UN_R16", "UN_R17", "UN_R94", "UN_R95",
            "UN_R127", "UN_R135", "UN_R137", "FMVSS",
        })
        assert "FMVSS_210" in GHOST_REGULATIONS
        assert not is_indexed_regulation("FMVSS_210")
        assert is_indexed_regulation("UN_R94")


class TestRequirementClusters:
    def test_q05_maps_to_belt_cluster(self):
        cluster = match_requirement_cluster(Q05_QUESTION)
        assert cluster == "belt_restraint_design"
        members = cluster_member_codes(cluster)
        assert "UN_R14" in members
        assert "UN_R16" in members
        assert "UN_R17" in members


@pytest.fixture(scope="module")
def retriever():
    from backend.app.retrieval.hybrid import HybridRetriever
    return HybridRetriever()


@pytest.fixture(scope="module")
def reranker():
    from backend.app.retrieval.reranker import CrossEncoderReranker
    return CrossEncoderReranker()


class TestQ05Retrieval:
    def test_q05_retrieves_r14_r16(self, retriever, reranker):
        docs = retriever.retrieve(Q05_QUESTION)["documents"]
        docs = reranker.rerank(Q05_QUESTION, docs)["documents"]
        status, detail = retrieval_recall(
            ["UN_R14", "UN_R16"],
            docs,
            retriever._chunk_by_id,
        )
        assert status == "PASS", detail
