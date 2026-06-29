"""Classify provider failures and track process-wide disabled models."""

from __future__ import annotations

import os
import threading
import time
from enum import Enum

from loguru import logger

from backend.app.gateway.providers.base import ProviderError

CONNECT_FAIL_THRESHOLD = int(os.getenv("GATEWAY_CONNECT_FAIL_THRESHOLD", "2"))
PROVIDER_COOLDOWN_SEC = float(os.getenv("GATEWAY_PROVIDER_COOLDOWN_SEC", "300"))


class ErrorKind(str, Enum):
    TOO_LARGE = "too_large"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"  # read / overall timeout
    CONNECTION = "connection"  # TCP connect unreachable
    DECOMMISSIONED = "decommissioned"
    FATAL = "fatal"


_lock = threading.Lock()
# model_key -> (reason, expires_at or None for permanent)
_disabled: dict[str, tuple[str, float | None]] = {}
_connect_fail_streak: dict[str, int] = {}


def classify_error_text(text: str) -> ErrorKind:
    msg = (text or "").lower()
    if "model_decommissioned" in msg or "has been decommissioned" in msg:
        return ErrorKind.DECOMMISSIONED
    if "request too large" in msg or "error code: 413" in msg or " 413 " in msg:
        return ErrorKind.TOO_LARGE
    if any(
        t in msg
        for t in (
            "connecttimeout",
            "connect timeout",
            "connection attempt failed",
            "failed to respond",
            "winerror 10060",
            "network is unreachable",
            "name or service not known",
            "connection refused",
        )
    ):
        return ErrorKind.CONNECTION
    if "timeout" in msg or "timed out" in msg:
        return ErrorKind.TIMEOUT
    if "rate limit" in msg or "429" in msg or "rate_limit" in msg:
        return ErrorKind.RATE_LIMIT
    if any(t in msg for t in ("401", "403", "invalid api key", "authentication")):
        return ErrorKind.FATAL
    return ErrorKind.FATAL


def classify_error(exc: Exception) -> ErrorKind:
    if isinstance(exc, ProviderError) and exc.kind is not None:
        try:
            return ErrorKind(exc.kind)
        except ValueError:
            pass
    return classify_error_text(str(exc))


def is_disabled(key: str) -> bool:
    with _lock:
        entry = _disabled.get(key)
        if not entry:
            return False
        _, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            del _disabled[key]
            _connect_fail_streak.pop(key, None)
            return False
        return True


def disable_model(key: str, reason: str, *, cooldown_s: float | None = None) -> None:
    expires_at = time.time() + cooldown_s if cooldown_s is not None else None
    with _lock:
        if key in _disabled:
            return
        _disabled[key] = (reason, expires_at)
    ttl = f" for {cooldown_s:.0f}s" if cooldown_s is not None else ""
    logger.error(
        f"Gateway model '{key}' disabled{ttl}: {reason}. "
        "Update model_registry / .env with a supported model id."
    )


def note_provider_success(key: str) -> None:
    with _lock:
        _connect_fail_streak.pop(key, None)


def note_provider_failure(key: str, kind: ErrorKind) -> None:
    if kind == ErrorKind.RATE_LIMIT or kind == ErrorKind.DECOMMISSIONED:
        disable_model(key, f"{kind.value}", cooldown_s=None)
        return
    if kind != ErrorKind.CONNECTION:
        return
    streak = 0
    open_circuit = False
    with _lock:
        streak = _connect_fail_streak.get(key, 0) + 1
        _connect_fail_streak[key] = streak
        open_circuit = streak >= CONNECT_FAIL_THRESHOLD
    if open_circuit:
        disable_model(
            key,
            f"{streak} consecutive connection failures",
            cooldown_s=PROVIDER_COOLDOWN_SEC,
        )


def reset_disabled_for_tests() -> None:
    with _lock:
        _disabled.clear()
        _connect_fail_streak.clear()
