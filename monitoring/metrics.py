"""Optional Prometheus metrics — disabled unless ENABLE_PROMETHEUS=true."""

from __future__ import annotations

from typing import Any

from app.config import settings

_metrics: Any = None
_available = True


def _init():
    global _metrics, _available
    if _metrics is not None or not _available:
        return _metrics
    if not settings.ENABLE_PROMETHEUS:
        return None
    try:
        from prometheus_client import Counter, Histogram, generate_latest
    except ImportError:
        _available = False
        return None

    _metrics = {
        "generate_latest": generate_latest,
        "chat_requests": Counter("rag_chat_requests_total", "Chat requests", ["route"]),
        "chat_errors": Counter("rag_chat_errors_total", "Chat errors"),
        "retrieval_latency": Histogram(
            "rag_retrieval_seconds", "Retrieval latency", buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120)
        ),
        "llm_latency": Histogram(
            "rag_llm_seconds", "LLM latency", buckets=(0.5, 1, 2, 5, 10, 30)
        ),
    }
    return _metrics


def record_chat(route: str = "regulatory") -> None:
    m = _init()
    if m:
        m["chat_requests"].labels(route=route).inc()


def record_error() -> None:
    m = _init()
    if m:
        m["chat_errors"].inc()


def record_retrieval(seconds: float) -> None:
    m = _init()
    if m:
        m["retrieval_latency"].observe(seconds)


def record_llm(seconds: float) -> None:
    m = _init()
    if m:
        m["llm_latency"].observe(seconds)


def prometheus_payload() -> bytes:
    m = _init()
    parts: list[bytes] = []
    if m:
        parts.append(m["generate_latest"]())
    try:
        from backend.app.llm_router.metrics import prometheus_payload as llm_payload

        llm = llm_payload()
        if llm:
            parts.append(llm)
    except Exception:
        pass
    if not parts:
        return b"# prometheus disabled\n"
    return b"".join(parts)
