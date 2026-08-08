"""LLM client: mock provider, Portkey configs, cache, routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation.llm_client import (
    LLMClient,
    LLMError,
    LLMRole,
    RateLimitError,
    _cache_key,
    _normalize_target,
    load_portkey_config,
)


@pytest.fixture()
def tmp_client(tmp_path: Path) -> LLMClient:
    return LLMClient(
        provider="mock",
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "logs" / "llm_calls.jsonl",
    )


def test_default_provider_is_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = LLMClient(cache_dir=tmp_path / "c", log_path=tmp_path / "l.jsonl")
    assert client.provider == "mock"


def test_model_routing_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("GROQ_SMALL_MODEL", "small-test")
    monkeypatch.setenv("GROQ_LARGE_MODEL", "large-test")
    client = LLMClient(cache_dir=tmp_path / "c", log_path=tmp_path / "l.jsonl")
    assert client.model_for(LLMRole.REWRITE) == "small-test"
    assert client.model_for(LLMRole.ANSWER) == "large-test"


def test_mock_answer_is_deterministic_and_cites_chunks(tmp_client: LLMClient):
    r1 = tmp_client.answer(
        "What is HIC?",
        context="[chunk a] HIC <= 1000",
        chunk_ids=["UN-ECE-R94::5.2.1", "UN-ECE-R94::5.2.2"],
    )
    r2 = tmp_client.answer(
        "What is HIC?",
        context="[chunk a] HIC <= 1000",
        chunk_ids=["UN-ECE-R94::5.2.1", "UN-ECE-R94::5.2.2"],
    )
    assert r1.provider == "mock"
    assert r1.text.startswith("[MOCK]")
    assert "UN-ECE-R94::5.2.1" in r1.text
    assert r1.text == r2.text
    assert r2.cached is True


def test_mock_structured_json_segments(tmp_client: LLMClient):
    r = tmp_client.complete(
        messages=[{"role": "user", "content": "q"}],
        role=LLMRole.ANSWER,
        question="Define H-point",
        chunk_ids=["efaa3fe01ca6c6fd"],
        response_format={"type": "json_object"},
    )
    data = json.loads(r.text)
    assert data["answer_segments"][0]["citation_chunk_id"] == "efaa3fe01ca6c6fd"
    assert "Define H-point" in data["answer_segments"][0]["text"]


def test_rewrite_uses_small_model(tmp_client: LLMClient):
    r = tmp_client.rewrite("HIC limit in R94?")
    assert r.role == "rewrite"
    assert r.model == tmp_client.small_model


def test_cache_key_stable_on_chunk_order():
    a = _cache_key(question="q", chunk_ids=["b", "a"], model="m", role="answer")
    b = _cache_key(question="q", chunk_ids=["a", "b"], model="m", role="answer")
    assert a == b


def test_jsonl_log_written(tmp_client: LLMClient):
    tmp_client.complete(
        messages=[{"role": "user", "content": "hello"}],
        role=LLMRole.ANSWER,
        question="hello",
        chunk_ids=["c1"],
    )
    lines = tmp_client.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for key in ("input_tokens", "output_tokens", "model", "latency_ms", "provider", "role"):
        assert key in rec


def test_live_requires_provider_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key in (
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("PORTKEY_GATEWAY_URL", "http://localhost:8787/v1")
    with pytest.raises(LLMError, match="no provider API keys"):
        LLMClient(
            provider="groq",
            api_key="",
            cache_dir=tmp_path / "c",
            log_path=tmp_path / "l.jsonl",
        )


def test_rate_limit_error_message():
    err = RateLimitError("boom", retry_after=2.5)
    assert err.retry_after == 2.5
    assert "boom" in str(err)


def test_normalize_nvidia_nim_maps_to_openai_custom_host():
    out = _normalize_target(
        {
            "provider": "nvidia_nim",
            "api_key": "nv-test",
            "override_params": {"model": "meta/llama-3.1-8b-instruct"},
        }
    )
    assert out is not None
    assert out["provider"] == "openai"
    assert out["custom_host"] == "https://integrate.api.nvidia.com/v1"
    assert out["override_params"]["model"] == "meta/llama-3.1-8b-instruct"


def test_load_portkey_config_drops_empty_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-only")
    cfg_path = tmp_path / "query_rewrite.json"
    cfg_path.write_text(
        json.dumps(
            {
                "strategy": {"mode": "fallback"},
                "targets": [
                    {
                        "provider": "groq",
                        "api_key": "${GROQ_API_KEY}",
                        "override_params": {"model": "llama-3.1-8b-instant"},
                    },
                    {
                        "provider": "nvidia_nim",
                        "api_key": "${NVIDIA_API_KEY}",
                        "override_params": {"model": "meta/llama-3.1-8b-instruct"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_portkey_config("query_rewrite", path=cfg_path)
    assert len(cfg["targets"]) == 1
    assert cfg["targets"][0]["provider"] == "openai"
    assert cfg["targets"][0]["api_key"] == "nv-only"


def test_portkey_configs_exist_on_disk():
    root = Path(__file__).resolve().parents[1] / "config" / "portkey"
    for name in (
        "query_rewrite.json",
        "final_answer.json",
        "judge.json",
        "eval_judge_pinned.json",
        "eval_judge_overflow.json",  # legacy FreeLLMAPI allowlist artifact
    ):
        assert (root / name).is_file()


def test_eval_judge_pinned_config_shape():
    root = Path(__file__).resolve().parents[1] / "config" / "portkey"
    raw = json.loads((root / "eval_judge_pinned.json").read_text(encoding="utf-8"))
    assert (raw.get("strategy") or {}).get("mode") == "fallback"
    assert len(raw["targets"]) == 1
    assert raw["targets"][0]["provider"] == "google"
    assert raw["targets"][0]["override_params"]["model"] == "gemini-2.5-flash"
    # Pinned judge must not embed FreeLLMAPI / multi-provider ladders.
    text = (root / "eval_judge_pinned.json").read_text(encoding="utf-8")
    assert "3001" not in text
    assert "FREELLMAPI" not in text


def test_eval_judge_overflow_config_shape():
    """Legacy overflow JSON remains the sole FreeLLMAPI Portkey home (unused by scoring)."""
    root = Path(__file__).resolve().parents[1] / "config" / "portkey"
    raw = json.loads((root / "eval_judge_overflow.json").read_text(encoding="utf-8"))
    assert (raw.get("strategy") or {}).get("mode") == "fallback"
    providers = [t.get("provider") for t in raw["targets"]]
    assert providers[0] == "groq"
    assert providers[1] == "openai"  # openai_compatible → openai + custom_host
    assert providers[2] == "google"
    freellm = raw["targets"][1]
    assert freellm["override_params"]["model"] == "auto"
    assert "3001" in freellm["custom_host"]
    assert freellm["api_key"] == "${FREELLMAPI_UNIFIED_KEY}"
    assert freellm["metadata"]["logical_provider"] == ("free" + "llmapi")
    # Production configs must not mention FreeLLMAPI
    for name in ("query_rewrite.json", "final_answer.json", "eval_judge_pinned.json"):
        text = (root / name).read_text(encoding="utf-8")
        assert "3001" not in text
        assert "FREELLMAPI" not in text
        assert "freellmapi" not in text.lower()


def test_install_eval_judge_overflow_only_affects_judge(
    monkeypatch: pytest.MonkeyPatch,
):
    from eval.eval_judge_overflow import install_eval_judge_overflow
    from generation.llm_client import LLMClient, LLMRole

    monkeypatch.setenv("GOOGLE_API_KEY", "AIza_test")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "google")
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    client = LLMClient(provider="mock")
    install_eval_judge_overflow(client)
    judge_cfg = client.portkey_config_for(LLMRole.JUDGE)
    assert len(judge_cfg["targets"]) == 1
    assert judge_cfg["targets"][0]["override_params"]["model"] == "gemini-2.5-flash"
    assert judge_cfg["targets"][0]["provider"] == "google"
    # Answer path still loads final_answer (not pinned eval judge)
    answer_cfg = client.portkey_config_for(LLMRole.ANSWER)
    assert not any(
        (t.get("override_params") or {}).get("model") == "gemini-2.5-flash"
        and str(t.get("provider") or "") == "google"
        and len(answer_cfg.get("targets") or []) == 1
        for t in answer_cfg.get("targets") or []
    )
    assert not any(
        "3001" in str(t.get("custom_host") or "") for t in answer_cfg.get("targets") or []
    )


def test_rewrite_and_answer_configs_enable_simple_cache():
    root = Path(__file__).resolve().parents[1] / "config" / "portkey"
    for name in ("query_rewrite.json", "final_answer.json"):
        cfg = json.loads((root / name).read_text(encoding="utf-8"))
        assert cfg.get("cache", {}).get("mode") == "simple"
        assert int(cfg["cache"]["max_age"]) == 86400
    judge = json.loads((root / "judge.json").read_text(encoding="utf-8"))
    assert "cache" not in judge


def test_portkey_cache_namespace_tracks_cache_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from generation.llm_client import portkey_cache_namespace

    ver = tmp_path / "cache_version.json"
    monkeypatch.setattr("api.cache_version.VERSION_PATH", ver)
    ver.write_text(json.dumps({"version": 7}), encoding="utf-8")
    assert portkey_cache_namespace(LLMRole.REWRITE) == "passive-safety-rag:v7:rewrite"
    assert portkey_cache_namespace(LLMRole.ANSWER) == "passive-safety-rag:v7:answer"
    assert portkey_cache_namespace(LLMRole.JUDGE) is None


def test_resolve_served_target_maps_nvidia_and_index():
    from generation.llm_client import resolve_served_target

    cfg = {
        "targets": [
            {"provider": "groq", "override_params": {"model": "llama-3.3-70b-versatile"}},
            {
                "provider": "openai",
                "custom_host": "https://integrate.api.nvidia.com/v1",
                "override_params": {"model": "nvidia/llama-3.3-nemotron-super-49b-v1.5"},
                "metadata": {"logical_provider": "nvidia_nim"},
            },
        ]
    }
    hit = resolve_served_target(cfg, option_index="config.targets[1]")
    assert hit["provider"] == "nvidia_nim"
    assert hit["target_index"] == 1
    assert "nemotron" in hit["model"]


def test_format_answering_model_and_fallback():
    from generation.llm_client import (
        answering_attribution,
        answering_provider_was_fallback,
        format_answering_model,
    )

    assert format_answering_model("groq", "llama-3.3-70b-versatile") == (
        "groq/llama-3.3-70b-versatile"
    )
    assert format_answering_model(
        "nvidia_nim", "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    ) == ("nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5")
    assert format_answering_model("groq", "groq/llama-3.3-70b-versatile") == (
        "groq/llama-3.3-70b-versatile"
    )
    assert answering_provider_was_fallback(0) is False
    assert answering_provider_was_fallback(1) is True
    assert answering_provider_was_fallback(None) is False
    attr = answering_attribution(
        provider="groq", model="llama-3.3-70b-versatile", target_index=1
    )
    assert attr["answering_model"] == "groq/llama-3.3-70b-versatile"
    assert attr["answering_provider_was_fallback"] is True
    assert attr["sut_provider"] == "groq"
    assert attr["sut_model"] == "llama-3.3-70b-versatile"


def test_jsonl_includes_served_provider_and_cost(tmp_client: LLMClient):
    tmp_client.complete(
        messages=[{"role": "user", "content": "hello"}],
        role=LLMRole.ANSWER,
        question="hello",
        chunk_ids=["c1"],
    )
    rec = json.loads(tmp_client.log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["served_provider"] == "mock"
    assert "cost_usd" in rec
    assert "cache_status" in rec
    assert rec.get("was_fallback") is False
    assert rec.get("target_index") == 0
