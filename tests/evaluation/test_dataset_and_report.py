"""The gold dataset is a product artefact: it must load, be internally consistent,
and every reviewed answerable case must carry retrieval truth."""

from __future__ import annotations

import json
import pathlib

from safety_assistant.evaluation import load_dataset

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASET = ROOT / "evals" / "datasets" / "regulatory_v1.yaml"
REQUIRED_SLICES = {
    "exact_clause_lookup", "paraphrase", "regulation_clause_identifier", "definition", "numeric_threshold",
    "units_operators", "table_annex", "exception_condition", "multi_clause_reasoning", "cross_reference",
    "cross_regulation_comparison", "historical_as_of", "amendment_change_analysis", "ambiguous", "unanswerable",
    "adversarial_injection",
}  # fmt: skip


def test_dataset_loads_and_covers_required_slices() -> None:
    ds = load_dataset(DATASET)
    assert ds.dataset_version == "regulatory_v1" and len(ds.cases) >= 40
    assert REQUIRED_SLICES <= set(ds.slices())


def test_reviewed_answerable_cases_have_truth_and_refusal_cases_have_none() -> None:
    ds = load_dataset(DATASET)
    for c in ds.cases:
        if c.answerability == "answerable" and c.review_status == "REVIEWED":
            assert c.regulation_keys, c.case_id
        if c.answerability in ("unanswerable_not_in_corpus", "ambiguous"):
            assert not c.expected_section_paths and not c.key_facts, c.case_id
        if c.as_of_date is not None:
            assert c.query_type == "historical_as_of", c.case_id


def test_latest_result_records_provenance() -> None:
    latest = ROOT / "evals" / "results" / "retrieval_regulatory_v1_latest.json"
    if not latest.exists():
        return  # results are produced by `safety-assistant eval-retrieval` against a live corpus
    r = json.loads(latest.read_text(encoding="utf-8"))
    assert r["git_sha"] and r["timestamp"] and r["dataset_version"] == "regulatory_v1"
    assert r["corpus"]["chunks"] > 0 and r["versions"].get("embedding_model")
    legs = {leg["leg"] for leg in r["legs"]}
    assert {"dense", "sparse", "hybrid_rrf", "full"} <= legs
    for leg in r["legs"]:
        assert set(leg["aggregate"]) >= {"recall@5", "recall@10", "mrr", "ndcg@10", "latency_p50_ms"}
