"""Pinned eval judge + eval-infra retry helpers."""

from __future__ import annotations

import time

import pytest

from eval.eval_judge_overflow import (
    DEFAULT_PINNED_JUDGE_MODEL,
    build_pinned_eval_judge_config,
    eval_judge_overflow_primary_model,
)
from eval.eval_judge_retry import call_with_eval_retry


def test_pinned_judge_is_single_target(monkeypatch):
    # Force the JSON default — ignore developer .env overrides for this assertion.
    monkeypatch.setenv("EVAL_JUDGE_MODEL", DEFAULT_PINNED_JUDGE_MODEL)
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "google")
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("SECURITY_JUDGE_MODEL", raising=False)
    cfg = build_pinned_eval_judge_config()
    assert len(cfg["targets"]) == 1
    model = cfg["targets"][0]["override_params"]["model"]
    assert model == DEFAULT_PINNED_JUDGE_MODEL
    assert eval_judge_overflow_primary_model() == DEFAULT_PINNED_JUDGE_MODEL


def test_pinned_judge_honors_env_model(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "google")
    cfg = build_pinned_eval_judge_config()
    assert len(cfg["targets"]) == 1
    assert cfg["targets"][0]["override_params"]["model"] == "gemini-2.5-flash-lite"
    assert cfg["targets"][0]["provider"] == "google"


def test_eval_retry_backoff(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("EVAL_JUDGE_MAX_RETRIES", "2")
    monkeypatch.setenv("EVAL_JUDGE_RETRY_BASE_SEC", "0.5")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert call_with_eval_retry(flaky, label="test") == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_eval_retry_exhausts():
    with pytest.raises(RuntimeError, match="always"):
        call_with_eval_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("always")),
            max_retries=1,
            base_sec=0.01,
            label="test",
        )
