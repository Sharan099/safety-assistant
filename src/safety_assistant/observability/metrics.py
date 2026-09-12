"""Prometheus metrics — the operational set from CLAUDE.md §12/§15.
Names are stable; labels are low-cardinality only (no queries, no principals)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_LAT = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

REQUESTS = Counter("sa_http_requests_total", "HTTP requests", ["route", "method", "status"])
REQUEST_LATENCY = Histogram("sa_http_request_seconds", "HTTP request latency", ["route"], buckets=_LAT)
STAGE_LATENCY = Histogram("sa_stage_seconds", "Pipeline stage latency", ["stage"], buckets=_LAT)
ANSWERS = Counter("sa_answers_total", "Answers by mode", ["mode", "abstain_reason", "route"])
CITATION_FAILURES = Counter("sa_citation_validation_failures_total", "Claims dropped by citation/numeric validation")
RETRIEVAL_NO_HIT = Counter("sa_retrieval_no_hit_total", "Retrievals that returned no evidence")
RETRIEVAL_CANDIDATES = Histogram(
    "sa_retrieval_candidates", "Fused candidates per retrieval", buckets=(0, 5, 10, 20, 40, 80)
)
LLM_CALLS = Counter("sa_llm_calls_total", "LLM calls", ["provider", "outcome"])
LLM_TOKENS = Counter("sa_llm_tokens_total", "LLM tokens", ["provider", "kind"])
RATE_LIMITED = Counter("sa_rate_limited_total", "Requests rejected by the rate limiter")
INGESTION_RUNS = Counter("sa_ingestion_runs_total", "Ingestion runs", ["status"])
INGESTION_STAGE = Histogram(
    "sa_ingestion_stage_seconds", "Ingestion stage latency", ["stage"], buckets=(0.1, 0.5, 1, 5, 15, 60, 300, 900)
)
FRESHNESS_LAG_DAYS = Gauge("sa_freshness_lag_days", "Days from publication to activation", ["regulation"])
EMBED_REUSED = Counter("sa_embeddings_reused_total", "Embeddings reused by content hash")
EMBED_COMPUTED = Counter("sa_embeddings_computed_total", "Embeddings computed")


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
