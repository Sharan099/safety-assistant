"""Per-query trace store: tokens, cost, latency, retrieved chunks."""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from observability.prices import embedding_cost_usd, llm_cost_usd, rerank_cost_usd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "data" / "traces"

_lock = threading.Lock()
_memory: dict[str, dict[str, Any]] = {}


@dataclass
class QueryTrace:
    trace_id: str
    question: str = ""
    regulation_id: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    latency_ms: float = 0.0

    # LLM usage (sum across rewrite + answer unless broken out)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    rewrite_model: str = ""
    answer_model: str = ""
    rewrite_input_tokens: int = 0
    rewrite_output_tokens: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0

    embedding_tokens: int = 0
    embedding_model: str = ""
    rerank_calls: int = 0
    rerank_model: str = ""

    cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    rerank_cost_usd: float = 0.0

    chunk_ids: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    # Exact vector-DB queries after rewrite/expansion (determinism debugging).
    retrieval_queries: list[str] = field(default_factory=list)
    retrieval_log: dict[str, Any] = field(default_factory=dict)
    # Context actually sent to the answer LLM (after rerank + budget).
    context_chunks_to_llm: int = 0
    context_tokens_est: int = 0
    context_budget_mode: str = ""  # standard | enumerative
    n_hybrid_candidates: int = 0
    rerank_ran: bool = False

    answer_cached: bool = False
    prompt_cache_hit: bool = False
    prompt_cache_tokens_saved: int = 0
    provider: str = ""
    rewrite_provider: str = ""
    answer_provider: str = ""
    not_found: bool = False
    faithfulness_passed: bool | None = None
    error: str | None = None
    optimizations: dict[str, Any] = field(default_factory=dict)
    served_from_cache: bool = False
    cache_hit_kind: str = ""  # exact | semantic | portkey HIT | ""
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    def add_llm(
        self,
        *,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
        provider: str = "",
        cache_status: str = "",
        cost_usd: float | None = None,
        target_index: int | None = None,
        latency_ms: float = 0.0,
        retry_attempts: int = 0,
    ) -> None:
        served = (provider or "").strip()
        status = (cache_status or ("HIT" if cached else "")).strip().upper()
        is_hit = cached or status in {"HIT", "SEMANTIC HIT"}
        in_tok = 0 if is_hit else int(input_tokens)
        out_tok = 0 if is_hit else int(output_tokens)
        if cost_usd is None:
            cost_usd = (
                0.0
                if is_hit
                else llm_cost_usd(
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    provider=served or None,
                )
            )
        else:
            cost_usd = 0.0 if is_hit else float(cost_usd)

        self.input_tokens += in_tok
        self.output_tokens += out_tok
        call = {
            "role": role,
            "provider": served or "unknown",
            "model": model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(float(cost_usd), 8),
            "cache_status": status or ("HIT" if is_hit else "MISS"),
            "cached": is_hit,
            "target_index": target_index,
            "latency_ms": round(float(latency_ms), 2),
            "retry_attempts": int(retry_attempts or 0),
        }
        self.llm_calls.append(call)

        if role == "rewrite":
            self.rewrite_model = model
            self.rewrite_provider = served or self.rewrite_provider
            self.rewrite_input_tokens += in_tok
            self.rewrite_output_tokens += out_tok
        else:
            self.answer_model = model
            self.answer_provider = served or self.answer_provider
            self.provider = served or self.provider or "unknown"
            self.answer_input_tokens += in_tok
            self.answer_output_tokens += out_tok
            self.model = model
        if is_hit:
            self.optimizations["portkey_cache_hit"] = True
            if status:
                self.cache_hit_kind = status.lower().replace(" ", "_")
        if cached and not status:
            self.optimizations["llm_disk_cache"] = True

    def add_embedding(self, *, model: str, tokens: int) -> None:
        self.embedding_model = model
        self.embedding_tokens += int(tokens)

    def add_rerank(self, *, model: str, calls: int = 1) -> None:
        self.rerank_model = model
        self.rerank_calls += int(calls)

    def finalize(self) -> dict[str, Any]:
        self.ended_at = time.time()
        self.latency_ms = round((self.ended_at - self.started_at) * 1000.0, 2)

        # Cache hits: this request spent $0 / 0 tokens — do not re-bill original gen cost.
        if self.served_from_cache or self.answer_cached:
            self.input_tokens = 0
            self.output_tokens = 0
            self.rewrite_input_tokens = 0
            self.rewrite_output_tokens = 0
            self.answer_input_tokens = 0
            self.answer_output_tokens = 0
            self.embedding_tokens = 0
            self.rerank_calls = 0
            self.llm_cost_usd = 0.0
            self.embedding_cost_usd = 0.0
            self.rerank_cost_usd = 0.0
            self.cost_usd = 0.0
            self.prompt_cache_tokens_saved = 0
            self.optimizations["served_from_cache"] = True
            if self.cache_hit_kind:
                self.optimizations["cache_hit_kind"] = self.cache_hit_kind
            payload = asdict(self)
            save_trace(payload)
            _maybe_langfuse(payload)
            return payload

        self.llm_cost_usd = 0.0
        if self.llm_calls:
            # Prefer per-call costs (already provider-aware / cache-zeroed).
            self.llm_cost_usd = sum(float(c.get("cost_usd") or 0.0) for c in self.llm_calls)
        else:
            routed = bool(self.rewrite_model or self.answer_model)
            if self.rewrite_model:
                self.llm_cost_usd += llm_cost_usd(
                    model=self.rewrite_model,
                    input_tokens=self.rewrite_input_tokens,
                    output_tokens=self.rewrite_output_tokens,
                    provider=self.rewrite_provider or None,
                )
            if self.answer_model:
                self.llm_cost_usd += llm_cost_usd(
                    model=self.answer_model,
                    input_tokens=self.answer_input_tokens,
                    output_tokens=self.answer_output_tokens,
                    provider=self.answer_provider or self.provider or None,
                )
            elif not routed and self.model:
                self.llm_cost_usd = llm_cost_usd(
                    model=self.model,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    provider=self.provider or None,
                )
        if not self.model:
            self.model = self.answer_model or self.rewrite_model
        if not self.provider:
            self.provider = self.answer_provider or self.rewrite_provider
        self.embedding_cost_usd = embedding_cost_usd(
            model=self.embedding_model or "default_embedding",
            tokens=self.embedding_tokens,
        )
        self.rerank_cost_usd = rerank_cost_usd(
            model=self.rerank_model or "default_rerank",
            calls=self.rerank_calls,
        )
        # Prompt-cache savings: subtract cached input tokens at model input price
        if self.prompt_cache_tokens_saved and (self.answer_model or self.model):
            saved = llm_cost_usd(
                model=self.answer_model or self.model,
                input_tokens=self.prompt_cache_tokens_saved,
                output_tokens=0,
                provider=self.answer_provider or self.provider or None,
            )
            self.optimizations["prompt_cache_savings_usd"] = round(saved, 8)
            self.llm_cost_usd = max(0.0, self.llm_cost_usd - saved)

        self.cost_usd = round(
            self.llm_cost_usd + self.embedding_cost_usd + self.rerank_cost_usd, 8
        )
        payload = asdict(self)
        save_trace(payload)
        _maybe_langfuse(payload)
        return payload


def new_trace(question: str, *, regulation_id: str | None = None) -> QueryTrace:
    return QueryTrace(
        trace_id=str(uuid.uuid4()),
        question=question,
        regulation_id=regulation_id,
    )


def _trace_dir() -> Path:
    path = Path(os.getenv("TRACE_DIR") or DEFAULT_TRACE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_trace(payload: dict[str, Any]) -> Path:
    tid = payload["trace_id"]
    path = _trace_dir() / f"{tid}.json"
    with _lock:
        _memory[tid] = payload
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # append index line
        index = _trace_dir() / "index.jsonl"
        with index.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "trace_id": tid,
                        "question": payload.get("question"),
                        "cost_usd": payload.get("cost_usd"),
                        "latency_ms": payload.get("latency_ms"),
                        "faithfulness_passed": payload.get("faithfulness_passed"),
                        "not_found": payload.get("not_found"),
                        "ended_at": payload.get("ended_at"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def get_trace(trace_id: str) -> dict[str, Any] | None:
    with _lock:
        if trace_id in _memory:
            return _memory[trace_id]
    path = _trace_dir() / f"{trace_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    with _lock:
        _memory[trace_id] = data
    return data


def list_traces(limit: int = 100) -> list[dict[str, Any]]:
    index = _trace_dir() / "index.jsonl"
    if not index.is_file():
        # fallback: scan files
        files = sorted(_trace_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for f in files[:limit]:
            if f.name == "latest.json":
                continue
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
        return out
    rows: list[dict[str, Any]] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.reverse()
    # hydrate full traces for the dashboard when needed
    return rows[:limit]


def aggregate_metrics(limit: int = 500) -> dict[str, Any]:
    """avg cost/query, p95 latency, cost-per-passing-query (faithfulness gate)."""
    index_rows = list_traces(limit=limit)
    full: list[dict[str, Any]] = []
    for row in index_rows:
        tid = row.get("trace_id")
        if not tid:
            continue
        tr = get_trace(tid)
        if tr:
            full.append(tr)

    if not full:
        return {
            "n": 0,
            "avg_cost_usd": 0.0,
            "p95_latency_ms": 0.0,
            "cost_per_passing_query_usd": 0.0,
            "n_passing": 0,
            "avg_latency_ms": 0.0,
            "by_provider": {},
        }

    costs = [float(t.get("cost_usd") or 0.0) for t in full]
    lats = [float(t.get("latency_ms") or 0.0) for t in full]
    passing = [
        t
        for t in full
        if t.get("faithfulness_passed") is True
        or (t.get("faithfulness_passed") is None and not t.get("not_found") and not t.get("error"))
    ]
    # Strict: only traces explicitly marked faithfulness_passed=True for cost-per-passing
    strict_pass = [t for t in full if t.get("faithfulness_passed") is True]
    pass_costs = [float(t.get("cost_usd") or 0.0) for t in strict_pass] or [
        float(t.get("cost_usd") or 0.0) for t in passing if t.get("faithfulness_passed") is not False
    ]

    def p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        if len(vals) == 1:
            return vals[0]
        ordered = sorted(vals)
        idx = int(round(0.95 * (len(ordered) - 1)))
        return ordered[idx]

    return {
        "n": len(full),
        "avg_cost_usd": round(sum(costs) / len(costs), 8),
        "avg_latency_ms": round(sum(lats) / len(lats), 2),
        "p95_latency_ms": round(p95(lats), 2),
        "n_passing": len(strict_pass),
        "cost_per_passing_query_usd": round(sum(pass_costs) / len(pass_costs), 8)
        if pass_costs
        else 0.0,
        "total_cost_usd": round(sum(costs), 8),
        "by_provider": _provider_breakdown(full),
    }


def _provider_breakdown(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Count which provider actually served each LLM call (fallback frequency signal)."""
    tallies: dict[str, dict[str, Any]] = {}
    total_calls = 0

    def _p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        if len(vals) == 1:
            return vals[0]
        ordered = sorted(vals)
        idx = int(round(0.95 * (len(ordered) - 1)))
        return ordered[idx]

    for t in traces:
        calls = t.get("llm_calls") or []
        if calls:
            iterable = calls
        else:
            # Legacy traces without llm_calls — attribute answer to provider field.
            prov = (t.get("answer_provider") or t.get("provider") or "unknown").strip() or "unknown"
            iterable = [
                {
                    "provider": prov,
                    "cost_usd": t.get("llm_cost_usd") or t.get("cost_usd") or 0.0,
                    "cached": bool(t.get("served_from_cache") or t.get("answer_cached")),
                    "latency_ms": t.get("latency_ms") or 0.0,
                }
            ]
        for call in iterable:
            prov = str(call.get("provider") or "unknown").strip() or "unknown"
            bucket = tallies.setdefault(
                prov,
                {
                    "n": 0,
                    "n_cache_hit": 0,
                    "cost_usd": 0.0,
                    "_latencies": [],
                },
            )
            bucket["n"] += 1
            total_calls += 1
            if call.get("cached") or str(call.get("cache_status") or "").upper() in {
                "HIT",
                "SEMANTIC HIT",
            }:
                bucket["n_cache_hit"] += 1
            bucket["cost_usd"] = round(
                float(bucket["cost_usd"]) + float(call.get("cost_usd") or 0.0), 8
            )
            lat = float(call.get("latency_ms") or 0.0)
            if lat > 0:
                bucket["_latencies"].append(lat)
    for prov, bucket in tallies.items():
        n = int(bucket["n"])
        bucket["share"] = round(n / total_calls, 4) if total_calls else 0.0
        lats: list[float] = list(bucket.pop("_latencies", []) or [])
        bucket["avg_latency_ms"] = round(sum(lats) / len(lats), 1) if lats else 0.0
        bucket["p95_latency_ms"] = round(_p95(lats), 1) if lats else 0.0
        # Flag slow providers (NIM thinking regressions often sit well above 15s).
        bucket["slow"] = bool(bucket["p95_latency_ms"] >= 15_000)
    return dict(sorted(tallies.items(), key=lambda kv: (-kv[1]["n"], kv[0])))


def _maybe_langfuse(payload: dict[str, Any]) -> None:
    """Optional Langfuse export when keys are present."""
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    if not public or not secret:
        return
    try:
        from langfuse import Langfuse

        lf = Langfuse(public_key=public, secret_key=secret)
        lf.trace(
            id=payload["trace_id"],
            name="rag_query",
            input=payload.get("question"),
            output={"chunk_ids": payload.get("chunk_ids"), "not_found": payload.get("not_found")},
            metadata={
                "cost_usd": payload.get("cost_usd"),
                "latency_ms": payload.get("latency_ms"),
                "model": payload.get("model"),
                "input_tokens": payload.get("input_tokens"),
                "output_tokens": payload.get("output_tokens"),
                "embedding_tokens": payload.get("embedding_tokens"),
                "rerank_calls": payload.get("rerank_calls"),
            },
        )
        lf.flush()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Langfuse export skipped: %s", exc)
