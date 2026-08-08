"""GET /metrics — per-trace and aggregate cost/latency dashboard feed."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from observability.trace import aggregate_metrics, get_trace, list_traces, save_trace

router = APIRouter()


class FaithfulnessUpdate(BaseModel):
    faithfulness_passed: bool = Field(..., description="DeepEval / gate result")


@router.get("/metrics/aggregate")
def metrics_aggregate(limit: int = 500) -> dict[str, Any]:
    return aggregate_metrics(limit=limit)


@router.get("/metrics")
def metrics_list(limit: int = 50) -> dict[str, Any]:
    rows = list_traces(limit=limit)
    return {"traces": rows, "aggregate": aggregate_metrics(limit=limit)}


def _answer_call(llm_calls: list[Any]) -> dict[str, Any]:
    """Last answer-role Portkey/LLM call (served provider / cache status live here)."""
    for call in reversed(llm_calls or []):
        if isinstance(call, dict) and call.get("role") == "answer":
            return call
    return {}


@router.get("/metrics/{trace_id}")
def metrics_get(trace_id: str) -> dict[str, Any]:
    tr = get_trace(trace_id)
    if not tr:
        raise HTTPException(404, f"trace {trace_id} not found")
    llm_calls = tr.get("llm_calls") or []
    answer = _answer_call(llm_calls)
    cache_status = (
        str(answer.get("cache_status") or "").strip()
        or str(tr.get("cache_hit_kind") or "").strip().upper()
        or ("HIT" if tr.get("served_from_cache") or tr.get("answer_cached") else "MISS")
    )
    return {
        "trace_id": tr["trace_id"],
        "question": tr.get("question"),
        "input_tokens": tr.get("input_tokens"),
        "output_tokens": tr.get("output_tokens"),
        "embedding_tokens": tr.get("embedding_tokens"),
        "rerank_calls": tr.get("rerank_calls"),
        "model": tr.get("answer_model") or tr.get("model") or answer.get("model"),
        "rewrite_model": tr.get("rewrite_model"),
        "answer_model": tr.get("answer_model") or answer.get("model"),
        "provider": tr.get("answer_provider") or answer.get("provider") or tr.get("provider"),
        "rewrite_provider": tr.get("rewrite_provider"),
        "answer_provider": tr.get("answer_provider") or answer.get("provider"),
        "target_index": answer.get("target_index"),
        "cost_usd": tr.get("cost_usd"),
        "llm_cost_usd": tr.get("llm_cost_usd"),
        "latency_ms": tr.get("latency_ms"),
        "chunk_ids": tr.get("chunk_ids") or [],
        "citations": tr.get("citations") or [],
        "not_found": tr.get("not_found"),
        "faithfulness_passed": tr.get("faithfulness_passed"),
        "answer_cached": tr.get("answer_cached"),
        "served_from_cache": tr.get("served_from_cache"),
        "cache_status": cache_status,
        "cache_hit_kind": tr.get("cache_hit_kind"),
        "prompt_cache_hit": tr.get("prompt_cache_hit"),
        "prompt_cache_tokens_saved": tr.get("prompt_cache_tokens_saved"),
        "optimizations": tr.get("optimizations") or {},
        "retrieval_queries": tr.get("retrieval_queries") or [],
        "retrieval_log": tr.get("retrieval_log") or {},
        "context_chunks_to_llm": tr.get("context_chunks_to_llm"),
        "context_tokens_est": tr.get("context_tokens_est"),
        "context_budget_mode": tr.get("context_budget_mode"),
        "n_hybrid_candidates": tr.get("n_hybrid_candidates"),
        "rerank_ran": tr.get("rerank_ran"),
        "llm_calls": llm_calls,
        "error": tr.get("error"),
    }


@router.patch("/metrics/{trace_id}/faithfulness")
def metrics_set_faithfulness(trace_id: str, body: FaithfulnessUpdate) -> dict[str, Any]:
    tr = get_trace(trace_id)
    if not tr:
        raise HTTPException(404, f"trace {trace_id} not found")
    tr["faithfulness_passed"] = body.faithfulness_passed
    save_trace(tr)
    return {"trace_id": trace_id, "faithfulness_passed": body.faithfulness_passed}
