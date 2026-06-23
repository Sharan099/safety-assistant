"""Regression tests for UN R14 anchorage applicability enrichment and retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.applicability_enrichment import (  # noqa: E402
    bond_load_duration_sentences,
    build_applicability_meta,
    enrich_section_body,
    is_anchorage_clause_family,
    resolve_anchorage_test_type,
)
from ingestion.hierarchical_chunker import chunk_markdown_file, _split_words  # noqa: E402
from tests.eval_harness.scoring import (  # noqa: E402
    aggregate_pass_rates,
    score_item,
)


R14_MD = ROOT / "output" / "markdown" / "UN_R14.md"
REG_CASES = ROOT / "tests" / "test_cases_regulation.json"


class TestApplicabilityParsing:
    def test_m3_n3_lower_anchorage_meta(self):
        body = (
            "At the same time a tractive force of 1,350 daN ± 20 daN shall be applied. "
            "In the case of vehicles of categories other than M1 and N1, the test load "
            "shall be 675 ± 20 daN, except that for M3 and N3 vehicles the test load "
            "shall be 450 ± 20 daN."
        )
        meta = build_applicability_meta(
            regulation="UN_R14",
            clause_number="6.4.1.3",
            section_title="Lower anchorages",
            body=body,
            duration_snippet="Duration requirement (§6.3.3): not less than 0.2 second.",
        )
        assert meta["anchorage_test_type"] == "lower_anchorage_traction"
        assert "M3_N3" in meta["applies_to_category"]
        assert meta.get("has_duration_link") is True

    def test_enrich_prepends_applicability_header(self):
        body = (
            "A test load of 1,350 daN ± 20 daN shall be applied. "
            "In the case of vehicles of categories other than M1 and N1, the test load "
            "shall be 675 ± 20 daN, except that for M3 and N3 vehicles the test load "
            "shall be 450 ± 20 daN."
        )
        enriched, meta = enrich_section_body(
            regulation="UN_R14",
            clause_number="6.4.1.2",
            section_title="Upper torso strap",
            body=body,
            duration_snippet="Duration requirement (§6.3.3): not less than 0.2 second.",
        )
        assert enriched.startswith("APPLICABILITY:")
        assert "450" in enriched
        assert "0.2 second" in enriched
        assert meta["anchorage_test_type"] == "upper_torso_strap_anchorage"

    def test_clause_family_detection(self):
        assert is_anchorage_clause_family("6.4.1.3", "UN_R14")
        assert is_anchorage_clause_family("6.3.3", "UN_R14")
        assert not is_anchorage_clause_family("6.4.1.3", "UN_R94")

    def test_resolve_special_type_belt(self):
        assert resolve_anchorage_test_type("6.4.5.3", "") == "special_type_belt_reduced_load"


class TestLoadDurationBinding:
    def test_bond_load_duration_sentences(self):
        sents = [
            "A test load of 450 daN shall be applied.",
            "The belt anchorages must withstand the specified load for not less than 0.2 second.",
            "Another unrelated sentence.",
        ]
        bonded = bond_load_duration_sentences(sents)
        assert len(bonded) == 2
        assert "0.2 second" in bonded[0]
        assert "450" in bonded[0]

    def test_m3_n3_chunk_has_load_and_duration(self):
        if not R14_MD.is_file():
            pytest.skip("UN_R14 markdown not available")
        chunks = chunk_markdown_file(R14_MD)
        m3_chunks = [
            c for c in chunks
            if c.get("regulation") == "UN_R14"
            and "450" in (c.get("text") or "")
            and (
                "0.2" in (c.get("text") or "")
                or c.get("has_duration_link")
                or c.get("has_duration_requirement")
            )
            and (
                c.get("anchorage_test_type") == "lower_anchorage_traction"
                or (c.get("clause_number") or "").startswith("6.4.1.3")
            )
        ]
        assert m3_chunks, "expected §6.4.1.3 chunk with 450 daN"
        hit = m3_chunks[0]
        text = hit.get("text") or ""
        assert "APPLICABILITY:" in text
        assert "450" in text
        assert "0.2" in text or hit.get("has_duration_link")
        assert hit.get("anchorage_test_type") == "lower_anchorage_traction"


class TestNearDuplicateSuppression:
    @pytest.fixture(scope="class")
    def retriever(self):
        from backend.app.retrieval.hybrid import HybridRetriever

        return HybridRetriever()

    def test_m3_n3_query_retrieves_duration_signal(self, retriever):
        q = (
            "For M3/N3 vehicles under UN R14, what lower anchorage test load "
            "and minimum hold time apply?"
        )
        docs = retriever.retrieve(q, mode="regulation_lookup")["documents"][:5]
        blob = " ".join(
            (retriever._chunk_by_id.get(d["id"], d).get("text") or d.get("text") or "")
            for d in docs
        ).lower()
        assert "450" in blob
        assert "0.2" in blob


class TestRegulationCasesRetrieval:
    @pytest.fixture(scope="class")
    def retriever(self):
        from backend.app.retrieval.hybrid import HybridRetriever

        return HybridRetriever()

    def test_regulation_retrieval_pass_rate(self, retriever):
        cases = json.loads(REG_CASES.read_text(encoding="utf-8"))
        rows = []
        for case in cases:
            docs = retriever.retrieve(case["question"], mode=case.get("mode"))["documents"]
            row = score_item(case, "", docs, retriever._chunk_by_id, require_answer=False)
            # Retrieval-only pass: recall + must_not + retrieval_contains
            retr_pass = all(
                row.get(k) in ("PASS", "skip")
                for k in ("recall", "must_not", "retrieval_contains")
            )
            row["pass"] = retr_pass
            rows.append(row)
        agg = aggregate_pass_rates(rows)
        assert agg["overall"]["rate"] >= 0.8, (
            f"retrieval regression {agg['overall']['pass']}/{agg['overall']['total']}: "
            f"{[r for r in rows if not r.get('pass')]}"
        )
