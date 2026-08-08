"""Tests for golden set loading + scorecard metrics (no Qdrant/RAGAS)."""

from __future__ import annotations

from pathlib import Path

from eval.generation_eval import estimate_groq_calls
from eval.gold import chunk_match_keys, gold_keys, load_golden_set
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from eval.scorecard import build_scorecard, print_scorecard
from retrieval.retrieve import RetrievedChunk


def test_golden_set_has_structured_cases():
    cases = load_golden_set()
    assert len(cases) >= 30
    allowed = {
        "factual_lookup",
        "compliance_check",
        "numeric_safety",
        "multi_hop",
        "enumerative",
        "cross_regulation",
        "design_implication",
        "out_of_scope",
        "hallucination_probe",
        "guardrail",
        "prompt_injection",
    }
    for c in cases:
        assert c["question"]
        assert c.get("category") in allowed
        assert c.get("severity") in {"CRITICAL", "HIGH", "MEDIUM"}
        assert isinstance(c.get("expected_chunk_ids"), list)
        assert isinstance(c.get("expected_answer_contains"), list)
        assert isinstance(c.get("must_not_contain"), list)
        assert c.get("expected_behavior")


def test_missing_golden_set_refuses_legacy_gold_json_fallback(tmp_path: Path, monkeypatch):
    """Missing golden_set.jsonl must not silently load eval/archive/gold.json."""
    import eval.gold as gold_mod

    missing = tmp_path / "golden_set.jsonl"
    assert not missing.is_file()
    # Legacy fixture still exists on disk (archived) — must not be used.
    assert (Path(__file__).resolve().parents[1] / "eval" / "archive" / "gold.json").is_file()

    monkeypatch.setattr(gold_mod, "DEFAULT_GOLDEN", missing)
    monkeypatch.setattr(gold_mod, "DEFAULT_GOLD", missing)
    # Even if a gold.json were dropped next to the expected JSONL, refuse it.
    (tmp_path / "gold.json").write_text("[]", encoding="utf-8")

    try:
        load_golden_set()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "golden_set.jsonl not found" in msg
        assert "refusing to silently fall back to the 6-case legacy set" in msg


def test_archived_legacy_gold_json_is_explicit_only():
    """Archived 6-case gold.json remains readable as a historical fixture, not via default load."""
    import json

    archive = Path(__file__).resolve().parents[1] / "eval" / "archive" / "gold.json"
    cases = json.loads(archive.read_text(encoding="utf-8"))
    assert len(cases) == 6
    assert all("query" in c or "question" in c for c in cases)
    legacy_ids = {c["id"] for c in cases}
    live = load_golden_set()
    live_ids = {c["id"] for c in live}
    assert len(live) >= 30
    assert legacy_ids.isdisjoint(live_ids)


def test_archived_evaluation_cases_path_is_explicit():
    """Orphan evaluation package lives under archive/; cases are not a live gold path."""
    import json

    root = Path(__file__).resolve().parents[1]
    assert not (root / "evaluation").exists()
    cases_path = root / "archive" / "evaluation" / "archive" / "cases.json"
    assert cases_path.is_file()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(cases) == 10


def test_scorecard_retrieval_keys():
    retrieved = ["5.2.1.4", "2.1", "1", "x"]
    gold = ["5.2.1.4", "5.2.1"]
    assert recall_at_k(retrieved, gold, 5) == 0.5
    assert precision_at_k(retrieved, gold, 5) == 0.25
    assert mrr(retrieved, gold) == 1.0
    assert ndcg_at_k(retrieved, gold, 10) > 0


def test_expanded_section_matches_gold():
    case = {"expected_section_ids": ["UN-ECE-R94::5.2.1"], "expected_sections": ["5.2.1"]}
    keys = gold_keys(case)
    chunk = RetrievedChunk(
        chunk_id="expanded::UN-ECE-R94::5.2.1",
        text="ThCC 42 mm",
        section_id="UN-ECE-R94::5.2.1",
        section_number="5.2.1",
    )
    assert chunk_match_keys(chunk) & keys


def test_estimate_full_eval_calls():
    est = estimate_groq_calls(40, provider="groq", rewrite_enabled=True)
    assert est["total"] > 40
    assert estimate_groq_calls(40, provider="mock")["total"] == 0


def test_build_scorecard_shape(capsys):
    report = build_scorecard(
        tag="baseline",
        retrieval={"averages": {"recall@5": 0.5, "precision@5": 0.2, "mrr": 0.4, "ndcg@10": 0.3}},
        ragas={"averages": {"faithfulness": 0.7, "answer_relevancy": 0.6, "context_precision": 0.5, "context_recall": 0.55}},
    )
    assert report["scorecard"]["recall@5"] == 0.5
    assert report["scorecard"]["ragas_faithfulness"] == 0.7
    print_scorecard(report)
    out = capsys.readouterr().out
    assert "SCORECARD" in out
