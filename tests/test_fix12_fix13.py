"""Fix 12 (temp=0 non-creative) + Fix 13 (standard 5/3k hard cap)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from generation.llm_client import (
    DETERMINISTIC_SEED,
    JUDGE_TEMPERATURE,
    LLMClient,
    LLMResult,
    LLMRole,
    REWRITE_TEMPERATURE,
)
from retrieval.context_budget import (
    STANDARD_MAX_CHUNKS,
    STANDARD_MAX_TOKENS,
    apply_context_budget,
    context_budgets,
)
from retrieval.retrieve import RetrievedChunk
from retrieval.router import QueryIntent, budgets_for_intent, pipeline_for


def _c(cid: str, text: str = "word " * 40) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        regulation_id="UN-ECE-R94",
        section_number="5.2.1",
        section_id=f"UN-ECE-R94::{cid}",
        page_number=1,
        bounding_box=[0.0, 0.0, 1.0, 1.0],
        score=0.5,
    )


def test_fix12_non_creative_temps_are_zero():
    assert REWRITE_TEMPERATURE == 0.0
    assert JUDGE_TEMPERATURE == 0.0
    assert DETERMINISTIC_SEED == 42


def test_fix12_complete_clamps_non_answer_temperature():
    """Non-ANSWER roles always send temperature=0 and seed=42 to the provider."""
    client = LLMClient(provider="mock")
    client.provider = "groq"  # route through _portkey_complete without live init
    sent: dict = {}

    def fake_portkey(**kwargs):
        sent.update(kwargs)
        return LLMResult(
            text='{"ok":true}',
            model=kwargs.get("model") or "mock",
            provider="mock",
            role="rewrite",
        )

    with patch.object(client, "_portkey_complete", side_effect=fake_portkey):
        client.complete(
            messages=[{"role": "user", "content": "classify me"}],
            role=LLMRole.REWRITE,
            temperature=0.8,
            seed=123,
        )
    assert sent["temperature"] == 0.0
    assert sent["seed"] == DETERMINISTIC_SEED

    sent.clear()
    with patch.object(client, "_portkey_complete", side_effect=fake_portkey):
        client.complete(
            messages=[{"role": "user", "content": "judge"}],
            role=LLMRole.JUDGE,
            temperature=1.0,
            seed=7,
        )
    assert sent["temperature"] == 0.0
    assert sent["seed"] == DETERMINISTIC_SEED


def test_fix12_rewrite_and_condense_pass_zero(monkeypatch: pytest.MonkeyPatch):
    """Production rewrite/condense call sites request temperature=0."""
    from api.conversations import Turn
    from retrieval.rewrite import condense_followup

    client = LLMClient(provider="mock")
    seen: list[dict] = []

    def spy_complete(**kwargs):
        seen.append(dict(kwargs))
        return LLMResult(
            text="standalone question about HPC in R94",
            model="mock",
            provider="mock",
            role="rewrite",
        )

    monkeypatch.setattr(client, "complete", spy_complete)
    condense_followup(
        "and the HPC limit?",
        history=[Turn(question="What about UN R94?", answer="Frontal impact regulation.")],
        llm=client,
        use_llm=True,
    )
    assert seen, "condense should call complete"
    assert seen[0].get("temperature") == 0.0
    assert seen[0].get("seed") == 42


def test_fix13_standard_hard_cap_ignores_inflated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_MAX_CHUNKS", "50")
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "20000")
    chunks, tokens, mode = context_budgets("What is the HIC15 limit in UN R94?")
    assert mode == "standard"
    assert chunks == STANDARD_MAX_CHUNKS
    assert tokens == STANDARD_MAX_TOKENS

    many = [_c(f"c{i}") for i in range(20)]
    trimmed, stats = apply_context_budget(
        many, question="What is the HIC15 limit in UN R94?"
    )
    assert len(trimmed) <= STANDARD_MAX_CHUNKS
    assert stats["mode"] == "standard"


def test_fix13_factual_compliance_clamped_despite_intent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INTENT_FACTUAL_LOOKUP_MAX_CHUNKS", "40")
    monkeypatch.setenv("INTENT_FACTUAL_LOOKUP_MAX_TOKENS", "12000")
    monkeypatch.setenv("INTENT_COMPLIANCE_CHECK_MAX_CHUNKS", "40")
    monkeypatch.setenv("INTENT_COMPLIANCE_CHECK_MAX_TOKENS", "12000")
    c, t, mode = budgets_for_intent(QueryIntent.FACTUAL_LOOKUP)
    assert mode == "factual_lookup"
    assert c == STANDARD_MAX_CHUNKS
    assert t == STANDARD_MAX_TOKENS
    c2, t2, mode2 = budgets_for_intent(QueryIntent.COMPLIANCE_CHECK)
    assert mode2 == "compliance_check"
    assert c2 == STANDARD_MAX_CHUNKS
    assert t2 == STANDARD_MAX_TOKENS


def test_fix13_wider_budget_for_enumerative_and_layer45_intents():
    c, t, mode = context_budgets(
        "Generate a checklist for preparing a vehicle for UN R94 testing"
    )
    assert mode == "enumerative"
    assert c > STANDARD_MAX_CHUNKS
    assert t > STANDARD_MAX_TOKENS

    for intent in (
        QueryIntent.DESIGN_IMPLICATION,
        QueryIntent.CHECKLIST_GEN,
        QueryIntent.SCOPE_SUMMARY,
        QueryIntent.APPLICABILITY,
        QueryIntent.RETEST_SCOPE,
    ):
        pipe = pipeline_for(intent)
        assert pipe.max_chunks > STANDARD_MAX_CHUNKS, intent
        bc, bt, _ = budgets_for_intent(intent)
        assert bc > STANDARD_MAX_CHUNKS, intent
        assert bt > STANDARD_MAX_TOKENS, intent
