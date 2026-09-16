"""Bounded agent routing (ENGINEERING.md §9): comparison, change analysis, budgets."""

from __future__ import annotations

import pathlib

import pytest

from safety_assistant.agents import Budget, RegulatoryAgent
from safety_assistant.ingestion.fetch import FilesystemBlobStore
from safety_assistant.ingestion.workflows import ingest_source
from safety_assistant.providers.embeddings.hashing import HashingEmbeddingProvider
from safety_assistant.retrieval import RetrievalService
from safety_assistant.retrieval.service import RetrievalConfig
from tests.conftest import requires_db
from tests.support.minireg import registry_for

pytestmark = requires_db


@pytest.fixture
def agent(clean_db, db_session, tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    reg = registry_for(tmp_path, include_r998=True)
    emb = HashingEmbeddingProvider(384)
    store = FilesystemBlobStore(tmp_path / "b")
    for key in ("test-un-r999-rev1", "test-un-r999-rev2", "test-un-r998-rev1"):
        assert (
            ingest_source(db_session, key, registry=reg, blob_store=store, embedder=emb, repo_root=tmp_path).status
            == "SUCCEEDED"
        )
    retrieval = RetrievalService(embedder=emb, config=RetrievalConfig(min_shared_terms=1))
    return RegulatoryAgent(db_session, retrieval, llm=None)


def test_comparison_route_retrieves_each_regulation_separately(agent) -> None:  # type: ignore[no-untyped-def]
    s = agent.run("Compare the head performance criterion requirement in R999 and R998", k=6)
    assert s["route"] == "comparison" and s["intent"] == "comparison"
    regs = {e.regulation_key for e in s["evidence"]}
    assert regs == {"UN-R999", "UN-R998"}
    assert [e.evidence_id for e in s["evidence"]] == [f"E{i}" for i in range(1, len(s["evidence"]) + 1)]
    assert s["retrieval_attempts"] == 2 and len(s["sub_results"]) == 2
    assert s["mode"] == "EVIDENCE_ONLY"  # no LLM configured


def test_change_analysis_route_uses_the_diff_tool(agent) -> None:  # type: ignore[no-untyped-def]
    s = agent.run("What changed in R999 between the 01 and 02 series of amendments?", k=6)
    assert s["route"] == "change_analysis"
    assert s["extra_context"] and "changed: 3.2.2" in s["extra_context"] and "added: 3.2.4" in s["extra_context"]
    assert "-The thorax compression criterion (ThCC) shall not exceed 42 mm." in s["extra_context"]
    assert all(e.version_label.startswith("Rev.2") for e in s["evidence"])  # focused on the newer text
    assert s["tool_calls"] == 2  # diff + one retrieval


def test_change_analysis_with_single_version_warns_instead_of_inventing(agent) -> None:  # type: ignore[no-untyped-def]
    s = agent.run("What changed in R998 in the latest amendment?", k=4)
    assert s["route"] == "change_analysis" and s["extra_context"] is None
    assert any("only one verified version" in w for w in s["warnings"])


def test_standard_route_retries_once_then_stops(agent) -> None:  # type: ignore[no-untyped-def]
    s = agent.run("HPC R999", k=4)  # acronym → one rewrite is allowed at most
    assert s["route"] == "standard" and s["retrieval_attempts"] <= 2


def test_budget_exhaustion_abstains_safely(agent) -> None:  # type: ignore[no-untyped-def]
    agent.budget = Budget(max_tool_calls=1)
    s = agent.run("Compare the head performance criterion requirement in R999 and R998", k=6)
    assert s["mode"] == "ABSTAINED" and s["abstain_reason"] == "generation_unavailable"
    assert "budget" in (s["message"] or "")
    agent.budget = Budget(timeout_seconds=0.0)
    s = agent.run("thorax compression criterion limit R999", k=4)
    assert s["mode"] == "ABSTAINED" and "timeout" in (s["message"] or "")


def test_uncleared_data_class_is_withheld_but_the_cleared_part_is_answered(agent, db_session) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import update

    from safety_assistant.persistence.models import Regulation
    from safety_assistant.retrieval import ScopeFilter
    from safety_assistant.retrieval.sparse import invalidate_cache
    from tests.integration.conftest import _EvidenceAwareMock

    db_session.execute(
        update(Regulation).where(Regulation.regulation_key == "UN-R998").values(data_class="CONFIDENTIAL")
    )
    db_session.commit()
    invalidate_cache()
    agent.llm = _EvidenceAwareMock()
    scope = ScopeFilter(data_classes=("PUBLIC", "CONFIDENTIAL"))
    s = agent.run("Compare the head performance criterion requirement in R999 and R998", k=6, scope=scope)
    assert {e.regulation_key for e in s["evidence"]} == {"UN-R999", "UN-R998"}
    assert s["mode"] == "GENERATED"  # the public regulation is answered
    assert all(
        e.regulation_key == "UN-R999"
        for c in s["draft"].claims
        for e in s["evidence"]
        if e.evidence_id in c.evidence_ids
    )
    assert any(w.startswith("not sent to the answer model") for w in s["warnings"])
