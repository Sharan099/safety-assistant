"""Provider/model health — temporary disable with configurable cooldown."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from loguru import logger

from backend.app.gateway.error_policy import ErrorKind

_lock = threading.Lock()
# model_key -> HealthState
_states: dict[str, "HealthState"] = {}


@dataclass
class HealthState:
    reason: str
    expires_at: float | None  # None = permanent
    failure_count: int = 0


def is_healthy(model_key: str) -> bool:
    with _lock:
        st = _states.get(model_key)
        if st is None:
            return True
        if st.expires_at is not None and time.time() >= st.expires_at:
            del _states[model_key]
            logger.info("Health: model {} cooldown expired — re-enabled", model_key)
            return True
        return False


def seconds_until_healthy(model_key: str) -> float:
    """Seconds until cooldown expires; 0 if already healthy; -1 if permanently disabled."""
    with _lock:
        st = _states.get(model_key)
        if st is None:
            return 0.0
        if st.expires_at is None:
            return -1.0
        return max(0.0, st.expires_at - time.time())


def wait_until_healthy(model_key: str, *, max_wait_sec: float = 120.0) -> bool:
    """Block until model cooldown expires (evaluation single-model mode)."""
    remaining = seconds_until_healthy(model_key)
    if remaining < 0:
        return False
    if remaining == 0:
        return True
    wait = min(remaining + 0.5, max_wait_sec)
    logger.info("Health: waiting {:.0f}s for {} cooldown (evaluation mode)", wait, model_key)
    time.sleep(wait)
    return is_healthy(model_key)


def disable(model_key: str, reason: str, *, cooldown_sec: float | None) -> None:
    expires = time.time() + cooldown_sec if cooldown_sec is not None else None
    with _lock:
        prev = _states.get(model_key)
        count = (prev.failure_count + 1) if prev else 1
        _states[model_key] = HealthState(reason=reason, expires_at=expires, failure_count=count)
    ttl = f" for {cooldown_sec:.0f}s" if cooldown_sec else " (permanent)"
    logger.warning("Health: disabled {}{}: {}", model_key, ttl, reason)


def note_success(model_key: str) -> None:
    with _lock:
        _states.pop(model_key, None)


def note_failure(model_key: str, kind: ErrorKind, *, cooldown_sec: float) -> None:
    if kind == ErrorKind.DECOMMISSIONED:
        disable(model_key, kind.value, cooldown_sec=None)
        return
    if kind in (
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
        ErrorKind.FATAL,
    ):
        # 429 / 503 / overload → cooldown; router fails over immediately.
        disable(model_key, kind.value, cooldown_sec=cooldown_sec)
        return
    if kind == ErrorKind.TOO_LARGE:
        # Prompt too large — skip model for this request only (no cooldown).
        return


def reset_for_tests() -> None:
    with _lock:
        _states.clear()
