"""OpenTelemetry setup. One provider; OTLP/HTTP exporter when
OTEL_EXPORTER_OTLP_ENDPOINT is set (otherwise spans exist in-process only).
`span()` is the single helper the pipeline uses."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False


def configure_tracing(service_name: str = "safety-assistant", app_env: str = "development") -> None:
    global _configured
    if _configured:
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "deployment.environment": app_env})
    )
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    with trace.get_tracer("safety_assistant").start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, str | bool | int | float) else str(v))
        yield s


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx and ctx.is_valid else None
