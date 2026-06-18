"""
Basic application hardening helpers for a small cloud testing deployment.

What this provides:
  - A simple in-process fixed-window rate limiter per client IP + route.
  - A security-headers middleware.

Honest limitations (do NOT treat as production-grade auth/WAF):
  - The rate limiter is in-process; with multiple workers/instances each has its
    own counters. For real cloud scale, enforce limits at the gateway/CDN or use
    a shared store (Redis). It is a basic abuse speed-bump, not DDoS protection.
  - There is no authentication yet. Add SSO/RBAC before exposing sensitive data.
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

_RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

_lock = threading.Lock()
# key -> (window_start_epoch, count)
_buckets: dict[str, tuple[int, int]] = {}


def _client_ip(request: Request) -> str:
    # Respect a single proxy hop (nginx sets X-Forwarded-For).
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, route: str, *, limit: int, window_s: int) -> None:
    """Raise HTTP 429 if the caller exceeds `limit` requests per `window_s`."""
    if not _RATE_LIMIT_ENABLED:
        return
    ip = _client_ip(request)
    key = f"{ip}:{route}"
    now = int(time.time())
    with _lock:
        window_start, count = _buckets.get(key, (now, 0))
        if now - window_start >= window_s:
            window_start, count = now, 0
        count += 1
        _buckets[key] = (window_start, count)
        # Opportunistic cleanup to bound memory.
        if len(_buckets) > 10000:
            for k, (ws, _c) in list(_buckets.items()):
                if now - ws >= window_s:
                    _buckets.pop(k, None)
    if count > limit:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please slow down and try again shortly.",
        )


def verify_dashboard_key(request: Request) -> None:
    """Protect admin dashboard routes with FEEDBACK_DASHBOARD_KEY."""
    expected = os.getenv("FEEDBACK_DASHBOARD_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Feedback dashboard is not configured (set FEEDBACK_DASHBOARD_KEY on the backend).",
        )
    provided = (request.headers.get("x-dashboard-key") or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid dashboard key")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds conservative security headers to every response."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        # HSTS only meaningful over HTTPS; harmless otherwise but opt-in.
        if os.getenv("ENABLE_HSTS", "false").lower() == "true":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response
