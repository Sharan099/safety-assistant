"""Request ID middleware and per-request timing context."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
request_timing_var: ContextVar[dict[str, Any]] = ContextVar("request_timing", default={})


def get_request_id() -> str:
    return request_id_var.get()


def set_timing(key: str, value: float | int | str) -> None:
    timing = dict(request_timing_var.get())
    timing[key] = value
    request_timing_var.set(timing)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:12]
        token_rid = request_id_var.set(rid)
        token_timing = request_timing_var.set({"started_at": time.perf_counter()})
        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - request_timing_var.get()["started_at"]) * 1000
            response.headers["X-Request-ID"] = rid
            response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
            return response
        finally:
            request_id_var.reset(token_rid)
            request_timing_var.reset(token_timing)
