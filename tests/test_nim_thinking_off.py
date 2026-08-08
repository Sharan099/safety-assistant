"""NIM / Gemini reasoning-off helpers (disable extended CoT that inflates latency)."""

from __future__ import annotations

import json
from pathlib import Path

from generation.llm_client import (
    apply_no_think_system_prefix,
    is_gemini_thinking_model,
    is_reasoning_nim_model,
    load_portkey_config,
    strip_thinking_tokens,
)


def test_detect_reasoning_nim_models():
    assert is_reasoning_nim_model("nvidia/llama-3.3-nemotron-super-49b-v1.5")
    assert not is_reasoning_nim_model("meta/llama-3.3-70b-instruct")
    assert not is_reasoning_nim_model("llama-3.3-70b-versatile")


def test_detect_gemini_thinking_models():
    assert is_gemini_thinking_model("gemini-2.5-flash")
    assert is_gemini_thinking_model("gemini-2.5-flash-lite")
    assert not is_gemini_thinking_model("gemini-2.0-flash")
    assert not is_gemini_thinking_model("llama-3.3-70b-versatile")


def test_strip_thinking_blocks():
    raw = "<think>long chain of thought</think>\n{\"answer_segments\": []}"
    assert strip_thinking_tokens(raw).startswith("{")


def test_no_think_prefix():
    msgs = apply_no_think_system_prefix(
        [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}],
        enabled=True,
    )
    assert msgs[0]["content"].startswith("/no_think")


def test_final_answer_config_disables_thinking_and_has_timeout():
    cfg = load_portkey_config("final_answer")
    assert cfg.get("request_timeout") == 18000
    strategy = cfg.get("strategy") or {}
    assert "408" in str(strategy.get("on_status_codes") or []) or 408 in (
        strategy.get("on_status_codes") or []
    )
    nim = next(
        (
            t
            for t in (cfg.get("targets") or [])
            if "nemotron" in str((t.get("override_params") or {}).get("model") or "").lower()
            or str(t.get("metadata", {}).get("logical_provider") or "") == "nvidia_nim"
            or (
                t.get("provider") == "openai"
                and "nvidia" in str(t.get("custom_host") or "").lower()
            )
        ),
        None,
    )
    # When NVIDIA_API_KEY is unset the NIM target is dropped — still assert file intent
    # via raw JSON if needed.
    if nim is None:
        raw = json.loads(
            (Path(__file__).resolve().parents[1] / "config/portkey/final_answer.json").read_text(
                encoding="utf-8"
            )
        )
        nim_raw = next(t for t in raw["targets"] if t.get("provider") == "nvidia_nim")
        assert nim_raw["request_timeout"] == 18000
        assert nim_raw["override_params"]["chat_template_kwargs"]["enable_thinking"] is False
    else:
        params = nim.get("override_params") or {}
        assert params.get("chat_template_kwargs", {}).get("enable_thinking") is False
        assert int(nim.get("request_timeout") or 0) == 18000

    google = next(
        (t for t in (cfg.get("targets") or []) if t.get("provider") == "google"),
        None,
    )
    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "config/portkey/final_answer.json").read_text(
            encoding="utf-8"
        )
    )
    google_raw = next(t for t in raw["targets"] if t.get("provider") == "google")
    assert int(google_raw["request_timeout"]) == 10000
    thinking = (google_raw.get("override_params") or {}).get("thinking") or {}
    assert thinking.get("budget_tokens") == 0
    assert thinking.get("type") == "disabled"
    if google is not None:
        assert int(google.get("request_timeout") or 0) == 10000
        g_think = (google.get("override_params") or {}).get("thinking") or {}
        assert g_think.get("budget_tokens") == 0
