"""Trace/request id propagation + structured access log line per request."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from safety_assistant.observability import metrics

log = logging.getLogger("safety_assistant.access")
REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request failed", extra={"request_id": rid, "path": request.url.path})
            raise
        response.headers[REQUEST_ID_HEADER] = rid
        route = getattr(request.scope.get("route"), "path", request.url.path)
        metrics.REQUESTS.labels(route=route, method=request.method, status=str(response.status_code)).inc()
        metrics.REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - t0)
        log.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        return response
