"""Gateway tier routing, capability escalation, and regression metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.gateway import config as cfg
from backend.app.gateway.classifier import classify
from backend.app.gateway.fallback_safeguards import is_fast_groq_model
from backend.app.gateway.query_capability import (
    apply_capability_escalation,
    assess_query_capability,
)
from backend.app.gateway.types import RoutingContext
from backend.app.retrieval.citations import (
    chunk_authority_for_categories,
    enrich_doc_provenance,
)
from ingestion.annex_table_enrichment import enrich_annex_chunks, extract_annex6_body


def test_primary_tier_defaults_to_70b_not_8b():
    assert cfg.GROQ_TIER_MODEL == cfg.GROQ_MODEL or not is_fast_groq_model(cfg.GROQ_TIER_MODEL)
    assert is_fast_groq_model(cfg.GROQ_TIER_MODEL_FAST)


def test_regulation_lookup_routes_tier2_power():
    ctx = RoutingContext(
        query="What anchorage strength test load does UN R14 require for M1/N1?",
        prompt="x" * 200,
        grounding={"confidence": 0.8},
        mode="regulation_lookup",
        llm_tier_floor=2,
    )
    decision = classify(ctx)
    assert decision.tier >= 2
    assert not is_fast_groq_model(decision.model)


def test_arithmetic_query_escalates_to_tier3():
    cap = assess_query_capability(
        "Convert 1350 daN to kN for UN R14 M1 anchorage load",
        prompt="context",
    )
    assert cap.requires_arithmetic
    base = classify(RoutingContext(query="convert daN to kN", prompt="ctx"))
    escalated, reasons = apply_capability_escalation(base, cap)
    assert escalated.tier >= 3
    assert "anthropic" in escalated.provider
    assert reasons


def test_definition_distinction_escalates_to_tier3():
    cap = assess_query_capability(
        "How does anchorage differ from belt anchorage under UN R14?",
        prompt="context",
    )
    assert cap.requires_definition_distinction
    base = classify(RoutingContext(query="compare anchorage vs belt anchorage", prompt="ctx"))
    escalated, reasons = apply_capability_escalation(base, cap)
    assert escalated.tier >= 3
    assert escalated.model == cfg.CLAUDE_SONNET_MODEL
    assert reasons


def test_simple_lookup_stays_below_tier3_without_capability_escalation():
    cap = assess_query_capability(
        "What anchorage strength test load does UN R14 require for M1/N1?",
        prompt="short context",
    )
    assert not cap.unsafe_for_fast_tier
    ctx = RoutingContext(
        query="What anchorage strength test load does UN R14 require for M1/N1?",
        prompt="context",
        grounding={"confidence": 0.85},
        mode="regulation_lookup",
        llm_tier_floor=2,
    )
    decision, _ = apply_capability_escalation(classify(ctx), cap)
    assert decision.tier == 2
    assert decision.model == cfg.GROQ_TIER_MODEL_POWER


def test_failover_chain_does_not_downgrade_to_8b_before_anthropic():
    chain = cfg.FALLBACK_CHAINS["groq_power"]
    groq_fast_idx = chain.index("groq_fast")
    haiku_idx = chain.index("anthropic_haiku")
    assert haiku_idx < groq_fast_idx


def test_not_m1_n1_chunk_not_authority_for_m1_n1_query():
    doc = {
        "applies_to_category": ["NOT_M1_N1", "M3_N3"],
        "text": "In the case of vehicles of categories other than M1 and N1, "
        "the test load shall be 675 daN; M1/N1 is 1,350 daN.",
    }
    assert chunk_authority_for_categories(doc, {"M1_N1"}) is False
    assert chunk_authority_for_categories(doc, {"M3_N3"}) is True


def test_annex6_table_extracted_from_markdown():
    md = (ROOT / "output" / "markdown" / "UN_R14.md").read_text(encoding="utf-8")
    body = extract_annex6_body(md)
    assert body is not None
    assert "M1" in body and "anchorage" in body.lower()


def test_annex6_chunk_enrichment():
    md_path = ROOT / "output" / "markdown" / "UN_R14.md"
    sparse = [{
        "clause": "Annex 6",
        "section_title": "Annex 6",
        "text": "# UN_R14 > Annex 6\n\n32",
        "word_count": 6,
    }]
    enriched = enrich_annex_chunks(sparse, md_path)
    assert len(enriched[0]["text"].split()) > 40
    assert enriched[0].get("annex_table_enriched")


def test_fast_tier_usage_rate_regression_gate():
    """Simulated distribution — primary 70B must be majority under normal routing."""
    samples = []
    for q in (
        "UN R14 M1/N1 anchorage load?",
        "Compare anchorage vs belt anchorage UN R14",
        "Convert 450 daN to kN UN R14 M3",
    ):
        ctx = RoutingContext(
            query=q,
            prompt="ctx",
            grounding={"confidence": 0.75},
            mode="regulation_lookup",
            llm_tier_floor=2,
        )
        cap = assess_query_capability(q, "ctx")
        d, _ = apply_capability_escalation(classify(ctx), cap)
        samples.append(is_fast_groq_model(d.model))
    fast_rate = sum(samples) / len(samples)
    assert fast_rate < 0.5, f"fast-tier routing rate {fast_rate:.0%} too high"
