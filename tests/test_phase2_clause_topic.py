"""Phase 2 — clause_topic tagging and hard topic filter (Q03)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.clause_topic import (
    detect_allowed_clause_topics,
    detect_clause_topic,
)
from backend.app.retrieval.query_intent import detect_query_intent
from tests.eval_harness.scoring import must_not_retrieve

Q03_QUESTION = "Which dummy and injury criteria are used in UN R95 side impact?"
Q02_QUESTION = "Which dummy is used in the UN R94 frontal impact test?"


class TestClauseTopicDetection:
    def test_evacuation_chunk_tagged(self):
        topic = detect_clause_topic(
            "The 50th percentile manikin can be evacuated through the door opening."
        )
        assert topic == "evacuation"

    def test_dummy_spec_tagged(self):
        topic = detect_clause_topic("The ES-2 side impact dummy shall be used.")
        assert topic == "dummy_spec"


class TestTopicIntent:
    def test_q03_allows_dummy_and_injury_only(self):
        allowed = detect_allowed_clause_topics(Q03_QUESTION)
        assert allowed == frozenset({"dummy_spec", "injury_criteria"})

    def test_performance_criteria_includes_barrier_topics(self):
        allowed = detect_allowed_clause_topics(
            "UN R135 pole side impact performance criteria"
        )
        assert "barrier" in allowed
        assert "injury_criteria" in allowed

    def test_heading_injury_overrides_pole_barrier(self):
        topic = detect_clause_topic(
            "CONTEXT:\n  Covers: pole side test\n---\nPole geometry details",
            heading_path="UN_R135 > 5.3.2",
            section_title="Head Injury Criteria",
        )
        assert topic == "injury_criteria"


@pytest.fixture(scope="module")
def retriever():
    from backend.app.retrieval.hybrid import HybridRetriever
    return HybridRetriever()


class TestQ03RetrievalFilter:
    def test_evacuation_excluded_from_primary(self, retriever):
        result = retriever.retrieve(Q03_QUESTION)
        docs = result["documents"][:5]
        status, _ = must_not_retrieve(
            ["evacuation", "50th percentile manikin can be evacuated"],
            docs,
            retriever._chunk_by_id,
        )
        assert status == "PASS"

    def test_primary_topics_allowed(self, retriever):
        intent = detect_query_intent(Q03_QUESTION)
        assert intent.allowed_clause_topics is not None
        from backend.app.retrieval.clause_topic import chunk_passes_topic_filter

        for d in retriever.retrieve(Q03_QUESTION)["documents"][:5]:
            chunk = retriever._chunk_by_id.get(d["id"], {})
            assert chunk_passes_topic_filter(chunk, intent.allowed_clause_topics)


class TestQ02StillRetrieves:
    def test_r94_dummy_query_has_results(self, retriever):
        docs = retriever.retrieve(Q02_QUESTION)["documents"]
        assert len(docs) >= 1
        regs = {retriever._chunk_by_id.get(d["id"], {}).get("regulation") for d in docs[:5]}
        assert "UN_R94" in regs
