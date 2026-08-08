"""Unit tests for eval.preflight_check (no live LLM calls)."""

from __future__ import annotations

import io
import sys

from eval.preflight_check import (
    PreflightResult,
    _single_target_config,
    all_ok,
    confirm_or_abort,
    format_table,
)


def _target(provider: str = "groq", api_key: str = "abc") -> dict:
    return {
        "provider": provider,
        "api_key": api_key,
        "override_params": {"model": "llama-3.3-70b-versatile"},
    }


def test_single_target_config_has_fallback_shape_but_one_target():
    cfg = _single_target_config(_target())
    assert cfg["strategy"] == {"mode": "fallback"}
    assert len(cfg["targets"]) == 1
    assert cfg["targets"][0]["provider"] == "groq"


def test_single_target_config_does_not_mutate_input():
    target = _target()
    cfg = _single_target_config(target)
    cfg["targets"][0]["provider"] = "mutated"
    assert target["provider"] == "groq"


def test_all_ok_true_when_every_result_ok():
    results = [
        PreflightResult("groq", "llama-3.3-70b-versatile", "OK", 1.2),
        PreflightResult("nvidia_nim", "llama-3.3-nemotron-super-49b", "OK", 4.8),
    ]
    assert all_ok(results) is True


def test_all_ok_false_when_any_result_fails():
    results = [
        PreflightResult("groq", "llama-3.3-70b-versatile", "OK", 1.2),
        PreflightResult("google", "gemini-2.5-flash", "TIMEOUT", None),
    ]
    assert all_ok(results) is False


def test_format_table_includes_all_providers_and_statuses():
    results = [
        PreflightResult("groq", "llama-3.3-70b-versatile", "OK", 1.2),
        PreflightResult("google", "gemini-2.5-flash", "TIMEOUT", None),
    ]
    table = format_table(results)
    assert "groq" in table
    assert "google" in table
    assert "OK" in table
    assert "TIMEOUT" in table
    assert "1.2s" in table
    assert "-" in table  # missing latency for the timed-out provider


def test_confirm_or_abort_returns_true_immediately_when_all_ok():
    results = [PreflightResult("groq", "llama-3.3-70b-versatile", "OK", 1.2)]
    assert confirm_or_abort(results, assume_yes=False) is True


def test_confirm_or_abort_assume_yes_bypasses_failures():
    results = [PreflightResult("google", "gemini-2.5-flash", "ERROR", 1.7, "429 quota")]
    assert confirm_or_abort(results, assume_yes=True) is True


def test_confirm_or_abort_aborts_when_non_interactive_and_not_assumed(monkeypatch):
    results = [PreflightResult("openrouter", "openai/gpt-oss-20b:free", "TIMEOUT", None)]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert confirm_or_abort(results, assume_yes=False) is False


def test_confirm_or_abort_respects_interactive_yes(monkeypatch):
    results = [PreflightResult("openrouter", "openai/gpt-oss-20b:free", "TIMEOUT", None)]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert confirm_or_abort(results, assume_yes=False) is True


def test_confirm_or_abort_respects_interactive_no(monkeypatch):
    results = [PreflightResult("openrouter", "openai/gpt-oss-20b:free", "TIMEOUT", None)]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert confirm_or_abort(results, assume_yes=False) is False
