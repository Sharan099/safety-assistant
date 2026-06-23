"""Deployment-readiness reliability: gateway failover, fallback safeguards, citations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.gateway import config as gw_cfg
from backend.app.gateway.errors import (
    GENERATION_UNAVAILABLE_MESSAGE,
    is_raw_provider_error,
)
from backend.app.gateway.fallback_safeguards import (
    detect_repetition_loop,
    effective_max_tokens,
    sanitize_fast_model_output,
    truncate_repetition,
)
from backend.app.gateway.gateway import LLMGateway
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.router import Router
from backend.app.gateway.types import ProviderResponse
from backend.app.graph.prompt_budget import estimate_tokens, strip_chunk_boilerplate
from backend.app.graph.workflow import _build_grounded_context, _build_prompt
from backend.app.retrieval.citations import (
    build_citation,
    build_citations,
    enrich_doc_provenance,
    validate_citation_attribution,
)

CASES = json.loads((ROOT / "tests" / "test_cases_reliability.json").read_text(encoding="utf-8"))


class RateLimitProvider(Provider):
    def __init__(self, key: str, *, behaviour: str = "429"):
        self.key = key
        self._behaviour = behaviour
        self.calls = 0

    @property
    def provider_name(self):
        return self.key

    @property
    def model(self):
        return f"{self.key}-model"

    def available(self):
        return True

    def chat(self, messages, *, temperature, max_tokens, timeout_s):
        self.calls += 1
        if self._behaviour == "429":
            raise ProviderError("Error 429: rate limit exceeded", retryable=True)
        return ProviderResponse(
            answer=f"answer from {self.key}",
            model=self.model,
            provider=self.key,
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=5,
        )


@pytest.fixture(autouse=True)
def _fast_router(monkeypatch):
    monkeypatch.setattr(gw_cfg, "MAX_RETRIES", 0)
    monkeypatch.setattr(gw_cfg, "BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(gw_cfg, "BACKOFF_MAX_S", 0.0)


def _msgs():
    return [{"role": "user", "content": "hi"}]


def test_groq_429_failover_to_haiku():
    """REL01 — Groq 429 must engage failover chain, not return raw error text."""
    providers = {
        "groq": RateLimitProvider("groq"),
        "groq_power": RateLimitProvider("groq_power"),
        "anthropic_haiku": RateLimitProvider("anthropic_haiku", behaviour="ok"),
        "anthropic_sonnet": RateLimitProvider("anthropic_sonnet", behaviour="ok"),
    }
    outcome = Router(providers).route("groq", _msgs(), temperature=0, max_tokens=64)
    assert outcome.provider_key == "anthropic_haiku"
    assert outcome.fallback_used is True
    assert "429" not in outcome.response.answer.lower()


def test_all_providers_failed_returns_clean_message():
    providers = {
        "groq": RateLimitProvider("groq"),
        "groq_power": RateLimitProvider("groq_power"),
        "anthropic_haiku": RateLimitProvider("anthropic_haiku"),
        "anthropic_sonnet": RateLimitProvider("anthropic_sonnet"),
    }
    gw = LLMGateway(providers=providers, cache=None)
    with patch("backend.app.gateway.gateway.classifier.classify") as mock_cls:
        mock_cls.return_value = MagicMock(
            tier=1, provider="groq", model="llama-test", score=1.0, reasons=[]
        )
        result = gw.complete("What is UN R14 load?")
    assert result.generation_failed is True
    assert result.answer == GENERATION_UNAVAILABLE_MESSAGE
    assert not is_raw_provider_error(result.answer)


def test_fast_fallback_repetition_truncation():
    """REL02 — repetitive fast-tier output is truncated before returning."""
    sentence = (
        "The design shall meet anchorage requirements per clause 6.4.1 [S1]. "
    )
    looped = sentence * 5
    assert detect_repetition_loop(looped)
    cleaned, truncated = truncate_repetition(looped)
    assert truncated
    out, meta = sanitize_fast_model_output(
        looped,
        provider_key="groq",
        model="llama-3.1-8b-instant",
    )
    assert meta.get("repetition_truncated") or meta.get("fast_fallback_safeguard")
    assert detect_repetition_loop(out) is False


def test_design_review_prompt_uses_lower_fast_token_cap():
    q9_style = (
        "List each UN R14 anchorage test requirement for M1/N1 vs M3/N3, "
        "compare clause loads, and cite each clause."
    )
    cap = effective_max_tokens(
        2000,
        provider_key="groq",
        model="llama-3.1-8b-instant",
        prompt=q9_style,
        fallback_used=True,
    )
    assert cap <= gw_cfg.FAST_FALLBACK_MAX_TOKENS_LIST


def test_orphan_applicability_chunk_inherits_parent_attribution():
    """REL03 — related-clause stub without regulation inherits parent provenance."""
    parent = {
        "chunk_id": "UN_R14-H001-SEC",
        "regulation": "UN_R14",
        "revision": "Revision 7 (09 series of amendments)",
        "doc_type": "legal",
        "heading_path": "UN_R14 > 6.4.1 3.",
        "section_title": "6.4.1.3",
        "clause_number": "6.4.1.3",
        "text": "Parent section body",
    }
    orphan = {
        "id": "UN_R14-H001-C001",
        "text": (
            "[UN_R14 | Revision 7 | 6.4.1.3]\n"
            "APPLICABILITY:\n  Applies to: M3 and N3\n---\n"
            "Duration requirement (§6.3.3): not less than 0.2 second."
        ),
        "related_to": parent["chunk_id"],
    }
    lookup = {parent["chunk_id"]: parent}
    enriched = enrich_doc_provenance(orphan, lookup)
    cite = build_citation(enriched, 1)
    failures = validate_citation_attribution(cite)
    assert failures == [], failures
    assert cite["document"] == "UN R14"
    assert cite["is_legal"] is True
    assert "Unknown" not in cite["label"]
    assert "revision unverified" not in cite["label"].lower()


def test_regulation_lookup_citations_attribution_integration():
    from backend.app.retrieval.hybrid import HybridRetriever

    case = next(c for c in CASES if c["id"] == "REL03_citation_attribution")
    retriever = HybridRetriever()
    docs = retriever.retrieve(case["question"], mode=case["mode"])["documents"][:5]
    lookup = retriever._chunk_by_id
    enriched = [enrich_doc_provenance(d, lookup) for d in docs]
    citations = build_citations(enriched, lookup)
    assert citations, "expected retrieved citations"
    for cite in citations:
        failures = validate_citation_attribution(cite)
        assert failures == [], f"{cite.get('label')}: {failures}"
        label_low = cite["label"].lower()
        for pat in case.get("forbidden_citation_patterns", []):
            assert pat.lower() not in label_low


def test_prompt_boilerplate_strips_reduce_tokens():
    raw = (
        "[UN_R14 | Revision 7 | 6.4.1.3]\n"
        "[UN_R14 > 6.4.1 3.]\n\n"
        "APPLICABILITY:\n  Applies to: M3 and N3\n---\n"
        "At the same time a tractive force of 450 daN shall be applied."
    )
    stripped = strip_chunk_boilerplate(raw)
    before = estimate_tokens(raw)
    after = estimate_tokens(stripped)
    assert after < before
    assert "450" in stripped
    assert "APPLICABILITY" in stripped

    doc = {"text": raw, "regulation": "UN_R14"}
    cite = build_citation(doc, 1)
    ctx = _build_grounded_context([doc], [cite])
    prompt = _build_prompt("M3/N3 anchorage load?", ctx, mode_name="regulation_lookup")
    opt_tokens = estimate_tokens(prompt)
    legacy = estimate_tokens(
        _build_prompt(
            "M3/N3 anchorage load?",
            f"[S1] {cite['label']}\n(type: {cite['doc_type_label']})\n{raw[:700]}",
            mode_name="regulation_lookup",
        )
    )
    assert opt_tokens <= legacy
