"""Provider breakdown + trace llm_calls cost accounting."""

from __future__ import annotations

import pytest

from api.routes_metrics import metrics_get
from observability.trace import _provider_breakdown, new_trace, save_trace


def test_add_llm_records_served_provider_and_zeros_cache_hit():
    tr = new_trace("What is HIC?")
    tr.add_llm(
        role="answer",
        model="llama-3.3-70b-versatile",
        input_tokens=100,
        output_tokens=20,
        provider="groq",
        cache_status="HIT",
        cost_usd=0.01,
    )
    assert tr.answer_provider == "groq"
    assert tr.input_tokens == 0
    assert tr.output_tokens == 0
    assert tr.llm_calls[0]["cost_usd"] == 0.0
    assert tr.llm_calls[0]["cache_status"] == "HIT"


def test_provider_breakdown_share():
    traces = [
        {
            "llm_calls": [
                {"provider": "groq", "cost_usd": 0.01, "cache_status": "MISS"},
                {"provider": "nvidia_nim", "cost_usd": 0.0, "cache_status": "MISS"},
            ]
        },
        {
            "llm_calls": [
                {"provider": "groq", "cost_usd": 0.02, "cached": True, "cache_status": "HIT"},
            ]
        },
    ]
    by = _provider_breakdown(traces)
    assert by["groq"]["n"] == 2
    assert by["nvidia_nim"]["n"] == 1
    assert by["groq"]["n_cache_hit"] == 1
    assert by["groq"]["share"] == pytest.approx(2 / 3, rel=1e-3)


def test_provider_breakdown_includes_latency():
    traces = [
        {
            "llm_calls": [
                {
                    "provider": "nvidia_nim",
                    "cost_usd": 0.0,
                    "cache_status": "MISS",
                    "latency_ms": 1200,
                },
                {
                    "provider": "nvidia_nim",
                    "cost_usd": 0.0,
                    "cache_status": "MISS",
                    "latency_ms": 18000,
                },
                {"provider": "groq", "cost_usd": 0.01, "latency_ms": 400},
            ]
        }
    ]
    by = _provider_breakdown(traces)
    assert by["nvidia_nim"]["avg_latency_ms"] == pytest.approx(9600.0)
    assert by["nvidia_nim"]["p95_latency_ms"] >= 1200
    assert by["nvidia_nim"]["slow"] is True
    assert by["groq"]["slow"] is False


def test_metrics_get_exposes_served_provider_and_cache_status(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    tr = new_trace("What is ThCC?")
    tr.add_llm(
        role="rewrite",
        model="llama-3.1-8b-instant",
        input_tokens=10,
        output_tokens=5,
        provider="groq",
        cache_status="MISS",
        latency_ms=12.0,
    )
    tr.add_llm(
        role="answer",
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        input_tokens=200,
        output_tokens=40,
        provider="nvidia_nim",
        cache_status="MISS",
        target_index=1,
        latency_ms=80.0,
        cost_usd=0.0,
    )
    save_trace(tr.finalize())

    payload = metrics_get(tr.trace_id)
    assert payload["answer_provider"] == "nvidia_nim"
    assert payload["target_index"] == 1
    assert payload["cache_status"] == "MISS"
    assert payload["input_tokens"] == 210
    assert any(c["provider"] == "nvidia_nim" for c in payload["llm_calls"])


def test_metrics_get_portkey_cache_hit_shows_hit(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    tr = new_trace("cached Q")
    tr.add_llm(
        role="answer",
        model="llama-3.3-70b-versatile",
        input_tokens=500,
        output_tokens=50,
        provider="groq",
        cache_status="HIT",
    )
    save_trace(tr.finalize())

    payload = metrics_get(tr.trace_id)
    assert payload["cache_status"] == "HIT"
    assert payload["input_tokens"] == 0
    assert payload["cost_usd"] == 0.0
    assert payload["answer_provider"] == "groq"
