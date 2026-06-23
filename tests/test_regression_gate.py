"""CI regression gate — golden set retrieval baseline (v2 schema)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.document_registry import (  # noqa: E402
    INDEXED_LEGAL_CORPUS,
    regulation_matches_corpus,
)

GOLDEN = ROOT / "tests" / "golden_set.json"
BASELINE = ROOT / "tests" / "regression_baseline_pilot.json"
FALLBACK_BASELINE = ROOT / "tests" / "regression_baseline.json"

DEFAULT_RECALL_AT_10 = 0.50


def _load_items():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("items", [])


def _pilot_applicable(case: dict) -> bool:
    """Include abstention cases and cases whose expected docs are in the pilot corpus."""
    if case.get("expected_behavior") == "abstain":
        return True
    expected_docs = case.get("expected_source_docs") or []
    if not expected_docs:
        return False
    return all(
        any(regulation_matches_corpus(exp, code) for code in INDEXED_LEGAL_CORPUS)
        for exp in expected_docs
    )


@pytest.fixture(scope="module")
def retriever():
    from backend.app.retrieval.hybrid import HybridRetriever
    return HybridRetriever()


def _doc_match(chunk: dict, expected_doc: str) -> bool:
    reg = chunk.get("regulation", "") or chunk.get("doc_id", "")
    pdf = chunk.get("pdf_name", "")
    norm = expected_doc.upper().replace(" ", "_")
    return (
        norm in reg.upper().replace(" ", "_")
        or expected_doc.lower() in pdf.lower()
    )


def test_golden_recall_at_10(retriever):
    cases = [
        c for c in _load_items()
        if _pilot_applicable(c)
        and c.get("expected_source_docs")
        and c.get("expected_behavior") != "abstain"
    ]
    assert cases, "No pilot-applicable golden cases found"
    hits = 0
    for case in cases:
        result = retriever.retrieve(case["question"])
        expected_docs = case["expected_source_docs"]
        found = any(
            any(
                _doc_match(retriever._chunk_by_id.get(d["id"], {}), exp)
                for exp in expected_docs
            )
            for d in result["documents"][:10]
        )
        hits += int(found)
    recall = hits / max(len(cases), 1)
    baseline = DEFAULT_RECALL_AT_10
    baseline_path = BASELINE if BASELINE.exists() else FALLBACK_BASELINE
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text()).get("recall_at_10", baseline)
    assert recall >= baseline, f"recall@10 {recall:.3f} < baseline {baseline} (pilot cases={len(cases)})"


def test_abstention_cases_not_in_corpus(retriever):
    cases = [c for c in _load_items() if c.get("expected_behavior") == "abstain"]
    for case in cases:
        result = retriever.retrieve(case["question"])
        for d in result["documents"][:3]:
            chunk = retriever._chunk_by_id.get(d["id"], {})
            text = (chunk.get("text") or "").lower()
            if "iso 26262" in text or "asil" in text:
                pytest.fail(f"ISO content retrieved for abstention case: {case['id']}")


def test_reliability_suite_importable():
    """Ensure deployment-readiness reliability tests are collected with the gate."""
    import tests.test_reliability as rel

    assert rel.test_groq_429_failover_to_haiku is not None
    assert rel.test_orphan_applicability_chunk_inherits_parent_attribution is not None
