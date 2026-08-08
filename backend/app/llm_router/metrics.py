"""Prometheus metrics for multi-LLM routing."""

from __future__ import annotations

from typing import Any

_metrics: dict[str, Any] | None = None
_available = True


def _prometheus_enabled() -> bool:
    try:
        from app.config import settings

        return bool(settings.ENABLE_PROMETHEUS)
    except Exception:
        return False


def _init() -> dict[str, Any] | None:
    global _metrics, _available
    if _metrics is not None:
        return _metrics
    if not _available or not _prometheus_enabled():
        return None
    try:
        from prometheus_client import Counter, Histogram, generate_latest
    except ImportError:
        _available = False
        return None

    _metrics = {
        "generate_latest": generate_latest,
        "requests": Counter(
            "llm_requests_total",
            "LLM router requests",
            ["provider", "model", "outcome"],
        ),
        "failovers": Counter(
            "llm_failovers_total",
            "LLM router failovers",
            ["from_provider", "to_provider", "reason"],
        ),
        "errors": Counter(
            "llm_provider_errors_total",
            "LLM provider errors",
            ["provider", "model", "error_kind"],
        ),
        "latency": Histogram(
            "llm_provider_latency_seconds",
            "LLM provider latency",
            ["provider", "model"],
            buckets=(0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
        ),
        "tokens_in": Counter(
            "llm_tokens_input_total",
            "LLM input tokens",
            ["provider", "model"],
        ),
        "tokens_out": Counter(
            "llm_tokens_output_total",
            "LLM output tokens",
            ["provider", "model"],
        ),
        "retries": Counter(
            "llm_retry_count_total",
            "LLM retry attempts",
            ["provider", "model"],
        ),
    }
    return _metrics


def record_request(
    *,
    provider: str,
    model: str,
    outcome: str,
    latency_sec: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    m = _init()
    if not m:
        return
    m["requests"].labels(provider=provider, model=model, outcome=outcome).inc()
    m["latency"].labels(provider=provider, model=model).observe(latency_sec)
    if prompt_tokens:
        m["tokens_in"].labels(provider=provider, model=model).inc(prompt_tokens)
    if completion_tokens:
        m["tokens_out"].labels(provider=provider, model=model).inc(completion_tokens)


def record_failover(*, from_provider: str, to_provider: str, reason: str) -> None:
    m = _init()
    if m:
        m["failovers"].labels(
            from_provider=from_provider, to_provider=to_provider, reason=reason
        ).inc()


def record_error(*, provider: str, model: str, error_kind: str) -> None:
    m = _init()
    if m:
        m["errors"].labels(provider=provider, model=model, error_kind=error_kind).inc()


def record_retry(*, provider: str, model: str) -> None:
    m = _init()
    if m:
        m["retries"].labels(provider=provider, model=model).inc()


def reset_for_tests() -> None:
    """No-op unless prometheus_client REGISTRY is reset in tests."""
    pass


def prometheus_payload() -> bytes | None:
    """Return router metrics exposition text when Prometheus is enabled."""
    m = _init()
    if not m:
        return None
    return m["generate_latest"]()
