"""Gateway routing observability and forced-path assertions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.gateway.error_policy import ErrorKind
from backend.app.gateway.gateway import LLMGateway, reset_gateway_state_for_tests
from backend.app.gateway.providers.base import ProviderError


@pytest.fixture(autouse=True)
def _clean_gateway():
    reset_gateway_state_for_tests()
    yield
    reset_gateway_state_for_tests()


def _messages():
    return [
        {"role": "system", "content": "You are a safety engineer."},
        {
            "role": "user",
            "content": "RETRIEVED CONTEXT\n[S1] UN R94 clause 5.2.1.4 ThCC 42 mm\n\nQUESTION: chest deflection?",
        },
    ]


def _chunks():
    return [
        {
            "regulation_code": "UN_R94",
            "document_name": "UN_R94.pdf",
            "page_number": 12,
            "chunk_text": "ThCC must not exceed 42 mm.",
        }
    ]


def _gw(**kwargs):
    defaults = {"anthropic": MagicMock(), "openrouter": MagicMock(), "use_cache": False}
    defaults.update(kwargs)
    return LLMGateway(**defaults)


def test_normal_query_records_serving_model_and_tokens():
    groq = MagicMock()
    groq.complete.return_value = {
        "text": "ThCC limit is 42 mm [1].",
        "prompt_tokens": 120,
        "completion_tokens": 40,
    }
    result = _gw(groq=groq).complete(_messages(), context_chunks=_chunks())

    assert result.model_key == "groq"
    assert result.model_id
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 40
    assert result.evidence_only is False
    assert result.latency_ms >= 0
    assert any(s.outcome == "success" for s in result.steps)


def test_search_response_includes_routing_metadata(db_session):
    """Integration: search() metadata.routing exposes model + latency + tokens."""
    from registry.search import RegulationSearchEngine

    routing_dict = {
        "model_key": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "evidence_only": False,
        "cache_hit": False,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "latency_ms": 12.5,
        "steps": [],
    }
    fake_chunk = {
        "chunk_id": 1,
        "chunk_text": "ThCC 42 mm",
        "page_number": 1,
        "section": None,
        "paragraph": None,
        "document_id": 1,
        "document_name": "UN_R94.pdf",
        "regulation_code": "UN_R94",
        "amendment": "04 Series",
        "score": 0.9,
    }
    engine = RegulationSearchEngine()
    with patch.object(engine, "_generate_grounded_answer", return_value=("answer text", routing_dict)):
        with patch.object(engine, "_dense_search_sqlite", return_value=[fake_chunk]):
            with patch.object(engine, "_sparse_search_sqlite", return_value=[]):
                with patch.object(engine.embedder, "embed_query", return_value=[0.0] * 768):
                    out = engine.search(db_session, "UN R94 chest deflection", top_k=3, rerank=False)

    routing = out["metadata"]["routing"]
    assert routing["model_key"] == "groq"
    assert routing["prompt_tokens"] == 50
    assert routing["completion_tokens"] == 20
    assert "latency_ms" in routing


def test_429_failover_to_groq_fast():
    groq = MagicMock()
    groq.complete.side_effect = [
        ProviderError("rate limit 429", kind=ErrorKind.RATE_LIMIT.value, status_code=429),
        {
            "text": "Failover answer.",
            "prompt_tokens": 80,
            "completion_tokens": 15,
        },
    ]
    result = _gw(groq=groq).complete(_messages(), context_chunks=_chunks())

    assert result.model_key == "groq_fast"
    assert result.text == "Failover answer."
    assert any(s.outcome == "rate_limit" for s in result.steps)
    assert groq.complete.call_count == 2


def test_openrouter_serves_when_both_groq_tiers_rate_limited():
    groq = MagicMock()
    groq.complete.side_effect = [
        ProviderError("rate limit 429", kind=ErrorKind.RATE_LIMIT.value, status_code=429),
        ProviderError("rate limit 429", kind=ErrorKind.RATE_LIMIT.value, status_code=429),
    ]
    openrouter = MagicMock()
    openrouter.complete.return_value = {
        "text": "OpenRouter failover answer.",
        "prompt_tokens": 90,
        "completion_tokens": 22,
    }
    result = _gw(groq=groq, openrouter=openrouter).complete(_messages(), context_chunks=_chunks())

    assert result.model_key == "openrouter_llama"
    assert result.provider == "openrouter"
    assert result.text == "OpenRouter failover answer."
    assert openrouter.complete.call_count == 1
    assert any(s.outcome == "rate_limit" for s in result.steps)


def test_openrouter_429_falls_through_to_evidence_only():
    groq = MagicMock()
    groq.complete.side_effect = ProviderError("rate limit", kind=ErrorKind.RATE_LIMIT.value)
    openrouter = MagicMock()
    openrouter.complete.side_effect = ProviderError("rate limit 429", kind=ErrorKind.RATE_LIMIT.value)
    result = _gw(groq=groq, openrouter=openrouter).complete(_messages(), context_chunks=_chunks())

    assert result.evidence_only is True
    assert result.model_key == "evidence_only"
    assert openrouter.complete.call_count == 2


def test_connection_failover_reaches_openrouter_after_groq_tiers():
    groq = MagicMock()
    groq.complete.side_effect = [
        ProviderError("rate limit 429", kind=ErrorKind.RATE_LIMIT.value, status_code=429),
        ProviderError("rate limit 429", kind=ErrorKind.RATE_LIMIT.value, status_code=429),
    ]
    openrouter = MagicMock()
    openrouter.complete.side_effect = ProviderError(
        "[WinError 10060] connection attempt failed",
        kind=ErrorKind.CONNECTION.value,
    )
    result = _gw(groq=groq, openrouter=openrouter).complete(_messages(), context_chunks=_chunks())

    assert openrouter.complete.call_count == 2
    assert any(s.outcome == "connection" for s in result.steps)
    assert result.evidence_only is True


def test_circuit_breaker_skips_openrouter_after_repeated_connection_failures():
    from backend.app.gateway.error_policy import is_disabled

    groq = MagicMock()
    groq.complete.side_effect = ProviderError("rate limit", kind=ErrorKind.RATE_LIMIT.value)
    openrouter = MagicMock()
    openrouter.complete.side_effect = ProviderError(
        "connect timeout", kind=ErrorKind.CONNECTION.value
    )
    gw = _gw(groq=groq, openrouter=openrouter)

    gw.complete(_messages(), context_chunks=_chunks())
    gw.complete(_messages(), context_chunks=_chunks())
    assert is_disabled("openrouter_llama")
    calls = openrouter.complete.call_count
    gw.complete(_messages(), context_chunks=_chunks())
    assert openrouter.complete.call_count == calls


def test_413_compress_and_retry_same_model():
    groq = MagicMock()
    groq.complete.side_effect = [
        ProviderError("request too large 413", kind=ErrorKind.TOO_LARGE.value, status_code=413),
        {"text": "Compressed answer.", "prompt_tokens": 60, "completion_tokens": 10},
    ]
    result = _gw(groq=groq).complete(_messages(), context_chunks=_chunks())

    assert result.model_key == "groq"
    assert result.text == "Compressed answer."
    assert groq.complete.call_count == 2
    assert any(s.outcome == "too_large" for s in result.steps)


def test_all_fail_evidence_only():
    groq = MagicMock()
    groq.complete.side_effect = ProviderError("fatal", kind=ErrorKind.FATAL.value)
    openrouter = MagicMock()
    openrouter.complete.side_effect = ProviderError("fatal", kind=ErrorKind.FATAL.value)
    result = _gw(groq=groq, openrouter=openrouter, primary="groq_fast").complete(
        _messages(), context_chunks=_chunks()
    )

    assert result.evidence_only is True
    assert result.model_key == "evidence_only"
    assert "[evidence-only]" in result.text
    assert "ThCC" in result.text or "42" in result.text


def test_repeat_question_cache_hit():
    groq = MagicMock()
    groq.complete.return_value = {
        "text": "Cached path answer.",
        "prompt_tokens": 30,
        "completion_tokens": 8,
    }
    gw = LLMGateway(groq=groq, anthropic=MagicMock(), openrouter=MagicMock(), use_cache=True)
    first = gw.complete(_messages(), context_chunks=_chunks())
    second = gw.complete(_messages(), context_chunks=_chunks())

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == "Cached path answer."
    assert groq.complete.call_count == 1


def test_routing_step_records_openrouter_provider():
    groq = MagicMock()
    groq.complete.side_effect = [
        ProviderError("rate limit", kind=ErrorKind.RATE_LIMIT.value),
        ProviderError("rate limit", kind=ErrorKind.RATE_LIMIT.value),
    ]
    openrouter = MagicMock()
    openrouter.complete.return_value = {
        "text": "ok",
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    result = _gw(groq=groq, openrouter=openrouter).complete(_messages(), context_chunks=_chunks())

    success = next(s for s in result.steps if s.outcome == "success")
    assert success.provider == "openrouter"
    assert success.model_key == "openrouter_llama"
