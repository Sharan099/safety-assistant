"""Unit tests for the evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.eval_harness.cache import build_or_load_answers, load_cache, save_cache
from tests.eval_harness.ragas_budget import TokenTracker, _is_rate_limit_error
from tests.eval_harness.scoring import (
    answer_contains_groups,
    behavior_match,
    forbidden_in_answer,
    must_not_retrieve,
    normalize_text,
)

MAX_RETRIES = 3  # noqa: F401 — referenced in ragas_budget backoff tests


class TestAnswerContains:
    def test_groups_and_or_normalization(self):
        groups = [["42 mm", "42mm"], ["Hybrid III", "HIII"]]
        ok, failed = answer_contains_groups(
            "The ThCC limit is 42mm for Hybrid III dummy.", groups
        )
        assert ok is True
        assert failed == []

        ok2, failed2 = answer_contains_groups("42 mm only", groups)
        assert ok2 is False
        assert len(failed2) == 1

    def test_normalize_strips_commas_spaces(self):
        assert normalize_text("1,350 daN") == normalize_text("1350daN")


class TestMustNotRetrieve:
    def test_flags_evacuation_chunk_q03(self):
        docs = [
            {"id": "c1", "text": "The 50th percentile manikin can be evacuated through the door."},
            {"id": "c2", "text": "ES-2 side impact dummy criteria."},
        ]
        status, detail = must_not_retrieve(
            ["evacuation", "50th percentile manikin can be evacuated"],
            docs,
        )
        assert status == "FAIL"
        assert "evacuation" in detail.lower() or "50th" in detail.lower()


class TestBehaviorMatch:
    def test_request_blocked_fails_on_answer_item(self):
        ok, reason = behavior_match("Request blocked.", "answer")
        assert ok is False
        assert "blocked" in reason

    def test_guardrail_block_variant(self):
        ok, _ = behavior_match(
            "Your query was blocked by safety guardrails (injection).",
            "compare",
        )
        assert ok is False

    def test_abstain_passes_on_insufficient_data(self):
        ok, _ = behavior_match(
            "Insufficient data in knowledge base — no relevant passages.",
            "abstain",
        )
        assert ok is True


class TestForbiddenInAnswer:
    def test_detects_forbidden(self):
        hit, matched = forbidden_in_answer("The limit is 34 mm.", ["34 mm", "42 mm"])
        assert hit is True
        assert "34 mm" in matched


class TestRateLimitBackoff:
    def test_is_rate_limit_error(self):
        assert _is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))
        assert not _is_rate_limit_error(Exception("timeout"))

    def test_retries_then_succeeds(self, monkeypatch):
        from tests.eval_harness.ragas_budget import make_rate_limited_groq

        tracker = TokenTracker(budget=100_000)
        calls = {"n": 0}

        def fake_super_invoke(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("429 rate limit")
            return MagicMock(response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        with patch("langchain_groq.ChatGroq.invoke", fake_super_invoke):
            with patch("time.sleep"):
                llm = make_rate_limited_groq(tracker=tracker, delay_seconds=0)
                result = llm.invoke("test")
        assert calls["n"] == 3
        assert result is not None

    def test_raises_after_max_retries(self, monkeypatch):
        from tests.eval_harness.ragas_budget import make_rate_limited_groq

        tracker = TokenTracker(budget=100_000)

        def always_429(self, *args, **kwargs):
            raise Exception("429 rate limit exceeded")

        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        with patch("langchain_groq.ChatGroq.invoke", always_429):
            with patch("time.sleep"):
                llm = make_rate_limited_groq(tracker=tracker, delay_seconds=0)
                with pytest.raises(Exception) as exc:
                    llm.invoke("test")
        assert "429" in str(exc.value).lower()


class TestCacheHit:
    def test_cache_prevents_second_generation(self, tmp_path, monkeypatch):
        from tests.eval_harness import cache as cache_mod

        monkeypatch.setattr(cache_mod, "CACHE_FILE", tmp_path / "answers.json")
        calls = []

        def gen(item):
            calls.append(item["id"])
            return {"answer": "A", "contexts": ["c"], "documents": []}

        items = [{"id": "Q99", "question": "test?"}]
        build_or_load_answers(items, generate_fn=gen)
        build_or_load_answers(items, generate_fn=gen)
        assert len(calls) == 1


class TestJudgeRateLimitDegrade:
    def test_skipped_rate_limit_on_persistent_429(self):
        from tests.eval_harness.ragas_budget import judge_error_scores, TokenTracker

        tracker = TokenTracker(budget=8000)
        scores = judge_error_scores(Exception("429 Too Many Requests"), "Q01", tracker)
        assert scores["faithfulness"] == "skipped_rate_limit"
        assert scores["context_precision"] == "skipped_rate_limit"
        assert "Q01" in tracker.skipped_rate_limit
