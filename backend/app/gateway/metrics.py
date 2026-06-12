"""
Gateway Prometheus metrics.

Registered on the default prometheus_client registry, so they are exposed by the
existing `GET /metrics` endpoint and scraped by the existing Prometheus job with
no scrape-config change.

Defining these as module-level singletons guarded against double-registration
keeps imports idempotent under autoreload / repeated test imports.
"""

from __future__ import annotations

from prometheus_client import REGISTRY, Counter, Histogram


def _counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    try:
        return Counter(name, doc, labels or [])
    except ValueError:
        # Already registered (e.g. reimport under --reload / tests): reuse it.
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]


def _histogram(
    name: str, doc: str, labels: list[str] | None = None, buckets=None
) -> Histogram:
    try:
        if buckets is not None:
            return Histogram(name, doc, labels or [], buckets=buckets)
        return Histogram(name, doc, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]


GATEWAY_REQUESTS_TOTAL = _counter(
    "gateway_requests_total",
    "Total gateway generate() requests",
    ["provider", "model", "tier", "outcome"],
)

GATEWAY_CACHE_HITS_TOTAL = _counter(
    "gateway_cache_hits_total",
    "Semantic cache hits",
)

GATEWAY_CACHE_MISSES_TOTAL = _counter(
    "gateway_cache_misses_total",
    "Semantic cache misses",
)

GATEWAY_FALLBACK_TOTAL = _counter(
    "gateway_fallback_total",
    "Provider failovers (primary failed, next in chain used)",
    ["from_provider", "to_provider"],
)

GATEWAY_MODEL_USAGE_TOTAL = _counter(
    "gateway_model_usage_total",
    "Successful generations served per model/tier",
    ["provider", "model", "tier"],
)

GATEWAY_COST_SAVED_USD_TOTAL = _counter(
    "gateway_cost_saved_usd_total",
    "Estimated USD saved (cache hits + routing down from baseline tier)",
)

GATEWAY_LATENCY_SECONDS = _histogram(
    "gateway_latency_seconds",
    "Gateway end-to-end latency by provider/model",
    ["provider", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# Bonus: routing score distribution per tier (useful for tuning thresholds).
GATEWAY_ROUTE_SCORE = _histogram(
    "gateway_route_score",
    "Routing complexity score (0-10) by selected tier",
    ["tier"],
    buckets=(0.0, 1.0, 2.0, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 10.0),
)
