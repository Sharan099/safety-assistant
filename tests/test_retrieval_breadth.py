"""Retrieval breadth, category attribution at scale, and knowledge-boundary regression."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.citations import (
    chunk_authority_for_categories,
    detect_category_value_misattribution,
    detect_knowledge_boundary_flags,
    detect_scope_overclaim_flags,
)
from backend.app.retrieval.query_breadth import assess_query_breadth, effective_retrieval_k
from tests.eval_harness.retrieval_breadth_stats import score_retrieval_breadth


BROAD_CASES = ROOT / "tests" / "test_cases_broad_synthesis.json"
BOUNDARY_CASES = ROOT / "tests" / "test_cases_knowledge_boundary.json"


def test_narrow_query_not_broad():
    b = assess_query_breadth("What is the M1/N1 lower anchorage test load under UN R14?")
    assert not b.is_broad


def test_q20_style_query_detected_broad():
    q = (
        "Give a complete summary of all UN R14 and UN R16 belt anchorage requirements "
        "that apply to M1 vehicles, including test loads, hold times, angles, and "
        "geometric constraints."
    )
    b = assess_query_breadth(q)
    assert b.is_broad
    assert b.retrieval_k >= 15


def test_broad_query_increases_k_not_narrow():
    narrow_k = effective_retrieval_k(
        8, assess_query_breadth("M1 anchorage load UN R14?"), default_pool=12
    )
    broad_k = effective_retrieval_k(
        8,
        assess_query_breadth(
            "Complete summary of all M1 anchorage requirements including loads and angles"
        ),
        default_pool=12,
    )
    assert narrow_k == 8
    assert broad_k >= 15


@pytest.fixture(scope="module")
def retriever():
    from backend.app.retrieval.hybrid import HybridRetriever

    return HybridRetriever()


def test_broad_m1_summary_retrieval_coverage(retriever):
    cases = json.loads(BROAD_CASES.read_text(encoding="utf-8"))
    case = cases[0]
    docs = retriever.retrieve(case["question"], mode=case["mode"])["documents"]
    row = score_retrieval_breadth(case, docs, retriever._chunk_by_id)
    assert row["chunk_count"] >= case["min_retrieved_chunks"], row
    assert row["signal_hits"].get("load"), row
    assert row["signal_hits"].get("duration"), row
    assert row["pass"], row["failures"]


def test_mixed_applicability_pool_no_m1_from_excluding_chunk():
    """15+ chunk context: NOT_M1_N1 chunk must not authorize M1 values."""
    docs = []
    for i in range(16):
        if i % 2 == 0:
            docs.append({
                "applies_to_category": ["NOT_M1_N1", "M3_N3"],
                "text": "For categories other than M1 and N1, test load 675 daN. M1 is 1,350.",
            })
        else:
            docs.append({
                "applies_to_category": ["M1_N1"],
                "text": "For M1 and N1 vehicles, test load 1,350 daN ± 20.",
            })
    citations = [{"marker": f"S{j+1}", "document": "UN R14"} for j in range(16)]
    answer_bad = "For M1 vehicles the lower anchorage load is 675 daN [S1]."
    flags = detect_category_value_misattribution(
        "Summarize M1 anchorage requirements under UN R14",
        answer_bad,
        citations,
        docs,
    )
    assert flags
    assert any(f["type"] == "category_value_misattribution" for f in flags)

    for d in docs:
        if d["applies_to_category"] == ["M1_N1"]:
            assert chunk_authority_for_categories(d, {"M1_N1"})
        else:
            assert not chunk_authority_for_categories(d, {"M1_N1"})


def test_outside_knowledge_label_required_for_classification():
    cases = json.loads(BOUNDARY_CASES.read_text(encoding="utf-8"))
    kb01 = cases[0]
    undisclosed = detect_knowledge_boundary_flags(
        kb01["question"], kb01["sample_answer_without_label"]
    )
    disclosed = detect_knowledge_boundary_flags(
        kb01["question"], kb01["sample_answer_with_label"]
    )
    assert any(f["type"] == "outside_knowledge_undisclosed" for f in undisclosed)
    assert any(f["type"] == "outside_knowledge_disclosed" for f in disclosed)


def test_abstain_on_absent_regulation_no_outside_knowledge_flag():
    cases = json.loads(BOUNDARY_CASES.read_text(encoding="utf-8"))
    kb02 = cases[1]
    flags = detect_knowledge_boundary_flags(
        kb02["question"], kb02["sample_abstain_answer"], should_abstain=True
    )
    assert not flags


def test_scope_overclaim_flags_broad_headline():
    doc = {
        "applies_to_category": ["M3_N3"],
        "text": "For M3 vehicles with side-facing seats, structural requirements apply.",
    }
    cite = {"marker": "S1", "document": "UN R14"}
    answer = "UN R14 addresses seat structure crash loads for vehicles.\n\nDetails for M3 [S1]."
    flags = detect_scope_overclaim_flags(answer, [cite], [doc])
    assert flags
    assert flags[0]["type"] == "scope_overclaim"


def test_narrow_query_retrieval_k_unchanged(retriever):
    q = "What lower anchorage test load applies to M3/N3 vehicles under UN R14?"
    result = retriever.retrieve(q, mode="regulation_lookup")
    breadth = result.get("query_breadth") or {}
    assert not breadth.get("is_broad")
    assert len(result["documents"]) <= 10
