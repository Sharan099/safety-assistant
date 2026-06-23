"""Budgeted RAGAS evaluation with Groq rate-limit safety."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from config import GROQ_MODEL, RAGAS_JUDGE_MAX_TOKENS

BACKOFF_SECONDS = (5, 15, 45)
MAX_RETRIES = 3


@dataclass
class TokenTracker:
    budget: int = 8000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    stopped_on_budget: bool = False
    skipped_rate_limit: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> bool:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        if self.total_tokens >= self.budget:
            self.stopped_on_budget = True
            return False
        return True

    def within_budget(self) -> bool:
        return not self.stopped_on_budget and self.total_tokens < self.budget


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def reference_from_item(item: dict) -> str:
    """Build ground_truth / reference string from expected_answer_contains."""
    groups = item.get("expected_answer_contains") or []
    parts = []
    for g in groups:
        if g:
            parts.append(" / ".join(g))
    return "; ".join(parts) if parts else item.get("expected_behavior", "")


def _local_answer_relevancy_scores(rows: list[dict]) -> dict[str, float | str]:
    """Embedding cosine similarity question↔answer — zero Groq tokens."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        import numpy as np

        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        scores: dict[str, float] = {}
        for row in rows:
            q_emb = emb.embed_query(row["question"])
            a_emb = emb.embed_query(row["answer"][:2000])
            sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
            scores[row["id"]] = round(max(0.0, min(1.0, sim)), 4)
        return scores
    except Exception as exc:
        return {"_error": str(exc)}


def _shim_ragas_imports() -> None:
    """Avoid optional VertexAI deps that break ragas import on minimal installs."""
    import sys
    import types

    try:
        import langchain.llms as _llms
        if not hasattr(_llms, "VertexAI"):
            class VertexAI:  # noqa: N801
                pass
            _llms.VertexAI = VertexAI
    except Exception:
        pass

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except Exception:
        vertex_mod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # noqa: N801
            pass

        vertex_mod.ChatVertexAI = ChatVertexAI
        chat_models_mod = types.ModuleType("langchain_community.chat_models")
        chat_models_mod.vertexai = vertex_mod
        lc_mod = types.ModuleType("langchain_community")
        lc_mod.chat_models = chat_models_mod
        sys.modules.setdefault("langchain_community", lc_mod)
        sys.modules.setdefault("langchain_community.chat_models", chat_models_mod)
        sys.modules["langchain_community.chat_models.vertexai"] = vertex_mod


def make_rate_limited_groq(
    tracker: TokenTracker,
    delay_seconds: float = 3.0,
    model: str | None = None,
):
    """Return a ChatGroq subclass with token tracking, delay, and 429 backoff."""
    from langchain_groq import ChatGroq

    class _RateLimitedChatGroq(ChatGroq):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._tracker = tracker
            self._delay_seconds = delay_seconds
            self._last_call = 0.0

        def _sleep_if_needed(self) -> None:
            if self._delay_seconds <= 0:
                return
            elapsed = time.time() - self._last_call
            if elapsed < self._delay_seconds:
                time.sleep(self._delay_seconds - elapsed)

        def _record_usage(self, response: Any) -> None:
            meta = getattr(response, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or meta.get("usage") or {}
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            if prompt or completion:
                self._tracker.add(prompt, completion)

        def invoke(self, *args, **kwargs):
            if not self._tracker.within_budget():
                raise RuntimeError("token_budget_exceeded")
            self._sleep_if_needed()
            last_exc: BaseException | None = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = super().invoke(*args, **kwargs)
                    self._last_call = time.time()
                    self._record_usage(result)
                    return result
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                        time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                        continue
                    raise
            raise last_exc  # type: ignore[misc]

    return _RateLimitedChatGroq(
        model=model or GROQ_MODEL,
        temperature=0,
        max_tokens=RAGAS_JUDGE_MAX_TOKENS,
    )


# Backward-compatible alias for unit tests
RateLimitedGroq = make_rate_limited_groq


def judge_error_scores(exc: BaseException, row_id: str, tracker: TokenTracker) -> dict[str, str]:
    """Map judge evaluate() failures to per-metric status strings."""
    if _is_rate_limit_error(exc):
        tracker.skipped_rate_limit.append(row_id)
        return {
            "faithfulness": "skipped_rate_limit",
            "context_precision": "skipped_rate_limit",
        }
    if "token_budget_exceeded" in str(exc):
        return {
            "faithfulness": "skipped_budget",
            "context_precision": "skipped_budget",
        }
    return {
        "faithfulness": f"error:{exc}",
        "context_precision": f"error:{exc}",
    }


def run_judge_metrics_serial(
    rows: list[dict],
    tracker: TokenTracker,
    delay_seconds: float = 3.0,
) -> dict[str, dict[str, float | str]]:
    """
    Run faithfulness + context_precision one row at a time.
    Returns per-id scores; rate-limited items get skipped_rate_limit.
    """
    if not os.getenv("GROQ_API_KEY"):
        return {r["id"]: {"faithfulness": "skipped_no_key", "context_precision": "skipped_no_key"} for r in rows}

    _shim_ragas_imports()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import context_precision, faithfulness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings

    judge_llm = LangchainLLMWrapper(
        make_rate_limited_groq(tracker=tracker, delay_seconds=delay_seconds)
    )
    evaluator_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    per_id: dict[str, dict[str, float | str]] = {}

    for row in rows:
        if not tracker.within_budget():
            per_id[row["id"]] = {
                "faithfulness": "skipped_budget",
                "context_precision": "skipped_budget",
            }
            continue

        ds = Dataset.from_list([{
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row.get("reference", ""),
        }])

        try:
            result = evaluate(
                ds,
                metrics=[faithfulness, context_precision],
                llm=judge_llm,
                embeddings=evaluator_emb,
                run_config=RunConfig(max_workers=1, timeout=120, max_retries=1),
            )
            agg = getattr(result, "_repr_dict", None) or {}
            if not agg:
                try:
                    agg = dict(result)
                except Exception:
                    scores = getattr(result, "_scores_dict", {}) or {}
                    agg = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}

            per_id[row["id"]] = {
                "faithfulness": float(agg.get("faithfulness", 0)),
                "context_precision": float(agg.get("context_precision", 0)),
            }
        except Exception as exc:
            per_id[row["id"]] = judge_error_scores(exc, row["id"], tracker)

    return per_id


def run_answer_relevancy_local(rows: list[dict]) -> dict[str, Any]:
    """Local embedding-based answer relevancy — zero Groq tokens."""
    scores = _local_answer_relevancy_scores(rows)
    if "_error" in scores:
        return {"per_id": {}, "mean": None, "error": scores["_error"], "method": "local_embeddings"}
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return {
        "per_id": scores,
        "mean": round(sum(vals) / max(len(vals), 1), 4),
        "method": "local_embeddings_cosine",
        "note": "Uses sentence-transformers/all-MiniLM-L6-v2 — no Groq tokens.",
    }


def run_ragas_budgeted(
    items: list[dict],
    cache: dict,
    *,
    subset_ids: list[str],
    abstention_ids: set[str],
    token_budget: int = 8000,
    judge_delay: float = 3.0,
    skip_judge: bool = False,
) -> dict[str, Any]:
    """Full budgeted RAGAS stage."""
    tracker = TokenTracker(budget=token_budget)

    # Non-abstention rows for answer_relevancy
    relevancy_rows = []
    for item in items:
        if item["id"] in abstention_ids:
            continue
        cached = (cache.get("items") or {}).get(item["id"])
        if not cached or not cached.get("answer"):
            continue
        relevancy_rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": cached["answer"],
            "contexts": cached.get("contexts") or [],
            "reference": reference_from_item(item),
        })

    relevancy_result = run_answer_relevancy_local(relevancy_rows)

    judge_result: dict[str, dict] = {}
    if not skip_judge:
        judge_ids = [i for i in subset_ids if i not in abstention_ids]
        judge_rows = []
        for iid in judge_ids:
            item = next((x for x in items if x["id"] == iid), None)
            cached = (cache.get("items") or {}).get(iid)
            if not item or not cached or not cached.get("answer"):
                continue
            judge_rows.append({
                "id": iid,
                "question": item["question"],
                "answer": cached["answer"],
                "contexts": cached.get("contexts") or [],
                "reference": reference_from_item(item),
            })
        judge_result = run_judge_metrics_serial(judge_rows, tracker, delay_seconds=judge_delay)

    # Aggregate judge means (ignore NaN / skipped)
    import math

    def _finite_scores(metric: str) -> list[float]:
        out: list[float] = []
        for v in judge_result.values():
            x = v.get(metric)
            if isinstance(x, (int, float)) and math.isfinite(x):
                out.append(float(x))
        return out

    faith_vals = _finite_scores("faithfulness")
    prec_vals = _finite_scores("context_precision")

    return {
        "answer_relevancy": relevancy_result,
        "judge_metrics": {
            "subset_ids": [i for i in subset_ids if i not in abstention_ids],
            "per_id": judge_result,
            "faithfulness_mean": round(sum(faith_vals) / max(len(faith_vals), 1), 4) if faith_vals else None,
            "context_precision_mean": round(sum(prec_vals) / max(len(prec_vals), 1), 4) if prec_vals else None,
        },
        "excluded_from_ragas": sorted(abstention_ids),
        "not_run_via_ragas": ["answer_correctness", "context_recall"],
        "not_run_note": "answer_correctness and context_recall are covered deterministically in Stage A.",
        "token_summary": {
            "groq_calls": tracker.calls,
            "prompt_tokens": tracker.prompt_tokens,
            "completion_tokens": tracker.completion_tokens,
            "total_tokens": tracker.total_tokens,
            "budget": token_budget,
            "stopped_on_budget": tracker.stopped_on_budget,
            "skipped_rate_limit": tracker.skipped_rate_limit,
        },
    }
