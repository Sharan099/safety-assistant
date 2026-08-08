"""Gateway routing configuration — Groq-only."""

from __future__ import annotations

import os

DEFAULT_PRIMARY = os.getenv("GATEWAY_PRIMARY_MODEL", "groq")

ENABLE_GATEWAY = os.getenv("ENABLE_GATEWAY", "true").lower() == "true"
GATEWAY_SHADOW_MODE = os.getenv("GATEWAY_SHADOW_MODE", "false").lower() == "true"

# Retries per model for transient errors (timeout/5xx) — NOT for rate limits.
# On 429 the gateway fails over to the next chain model immediately.
GATEWAY_MODEL_RETRIES = int(os.getenv("GATEWAY_MODEL_RETRIES", "4"))
GATEWAY_RETRY_BASE_SEC = float(os.getenv("GATEWAY_RETRY_BASE_SEC", "2.0"))

# Proven separate-bucket chain (2026-07-16 header verification).
_DEFAULT_CHAIN = "groq,groq_fast,groq_instant"


def _parse_chain_keys() -> list[str]:
    raw = os.getenv("GATEWAY_FALLBACK_CHAIN", _DEFAULT_CHAIN)
    return [k.strip() for k in raw.split(",") if k.strip()]


def _build_fallback_chains() -> dict[str, list[str]]:
    keys = _parse_chain_keys()
    if not keys:
        keys = ["groq", "groq_fast", "groq_instant"]
    primaries = ("groq", "groq_fast", "groq_instant")
    return {p: keys for p in primaries}


FALLBACK_CHAINS: dict[str, list[str]] = _build_fallback_chains()
