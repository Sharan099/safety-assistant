"""Fix 17/28 — thinking off + per-provider timeout caps."""

from __future__ import annotations

import json
from pathlib import Path

from generation.llm_client import (
    _clamp_target_timeout_ms,
    _normalize_target,
    load_portkey_config,
    portkey_client_timeout_s,
)


ROOT = Path(__file__).resolve().parents[1]


def test_timeout_caps_by_provider():
    assert _clamp_target_timeout_ms(60_000, logical="google") == 10_000
    assert _clamp_target_timeout_ms(None, logical="google") == 10_000
    assert _clamp_target_timeout_ms(60_000, logical="nvidia_nim") == 18_000
    assert _clamp_target_timeout_ms(5_000, logical="groq") == 5_000
    assert portkey_client_timeout_s() >= 10.0


def test_normalize_forces_thinking_off_even_if_misconfigured():
    nim = _normalize_target(
        {
            "provider": "nvidia_nim",
            "api_key": "fake-nim-key",
            "request_timeout": 120000,
            "override_params": {
                "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
    )
    assert nim is not None
    assert nim["request_timeout"] == 18_000
    assert nim["override_params"]["chat_template_kwargs"]["enable_thinking"] is False

    google = _normalize_target(
        {
            "provider": "google",
            "api_key": "fake-google-key",
            "request_timeout": 90000,
            "override_params": {
                "model": "gemini-2.5-flash",
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            },
        }
    )
    assert google is not None
    assert google["request_timeout"] == 10_000
    assert google["override_params"]["thinking"] == {
        "type": "disabled",
        "budget_tokens": 0,
    }


def test_all_portkey_configs_have_408_failover_and_timeouts():
    for name in ("final_answer", "query_rewrite", "judge", "eval_judge_overflow"):
        raw = json.loads(
            (ROOT / "config" / "portkey" / f"{name}.json").read_text(encoding="utf-8")
        )
        codes = (raw.get("strategy") or {}).get("on_status_codes") or []
        assert 408 in codes, name
        for t in raw.get("targets") or []:
            assert "request_timeout" in t, (name, t.get("provider"))
            assert int(t["request_timeout"]) <= 18_000, (name, t)
            provider = t.get("provider")
            params = t.get("override_params") or {}
            model = str(params.get("model") or "")
            if provider == "google":
                assert int(t["request_timeout"]) <= 10_000
                thinking = params.get("thinking") or {}
                assert thinking.get("budget_tokens") == 0
                assert thinking.get("type") == "disabled"
            if "nemotron" in model.lower():
                assert params.get("chat_template_kwargs", {}).get("enable_thinking") is False


def test_loaded_final_answer_uses_flash_with_thinking_off():
    raw = json.loads(
        (ROOT / "config" / "portkey" / "final_answer.json").read_text(encoding="utf-8")
    )
    google = next(t for t in raw["targets"] if t["provider"] == "google")
    # flash-lite is 404 for new Google users; use Flash + thinking disabled.
    assert google["override_params"]["model"] == "gemini-2.5-flash"
    assert google["override_params"]["thinking"] == {
        "type": "disabled",
        "budget_tokens": 0,
    }
    # load_portkey_config may drop targets without keys — still assert normalize path
    cfg = load_portkey_config("final_answer")
    assert cfg.get("request_timeout") == 18000
