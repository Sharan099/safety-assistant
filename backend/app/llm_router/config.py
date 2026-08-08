"""Environment-driven Groq router configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


_DEFAULT_CHAIN = "groq_70b,groq_qwen3,groq_instant"

# Legacy gateway keys → registry keys
_LEGACY_KEY_ALIASES: dict[str, str] = {
    "groq": "groq_70b",
    "groq_fast": "groq_70b",
    "groq_instant": "groq_instant",
}


@dataclass(frozen=True)
class RouterConfig:
    chain: tuple[str, ...]
    max_retries_per_model: int
    retry_base_sec: float
    cooldown_sec: float
    default_temperature: float
    default_max_tokens: int
    default_timeout_sec: float
    enable_cache: bool
    probe_groq_models: bool
    log_routing: bool

    @staticmethod
    def from_env() -> RouterConfig:
        eval_mode = _b("EVALUATION_MODE", False)
        eval_model = (os.getenv("EVALUATION_MODEL_KEY") or "groq_70b").strip()
        if eval_mode:
            raw = eval_model
        else:
            raw = os.getenv("LLM_ROUTER_CHAIN") or os.getenv(
                "GATEWAY_FALLBACK_CHAIN", _DEFAULT_CHAIN
            )
        keys: list[str] = []
        for part in raw.split(","):
            k = part.strip()
            if not k:
                continue
            keys.append(_LEGACY_KEY_ALIASES.get(k, k))
        if not keys:
            keys = [eval_model] if eval_mode else list(_DEFAULT_CHAIN.split(","))
        return RouterConfig(
            chain=tuple(keys),
            max_retries_per_model=_i("LLM_ROUTER_MAX_RETRIES", _i("GATEWAY_MODEL_RETRIES", 4)),
            retry_base_sec=_f("LLM_ROUTER_RETRY_BASE_SEC", _f("GATEWAY_RETRY_BASE_SEC", 2.0)),
            cooldown_sec=_f("LLM_ROUTER_COOLDOWN_SEC", _f("GATEWAY_RATE_LIMIT_COOLDOWN_SEC", 60.0)),
            default_temperature=_f("LLM_ROUTER_TEMPERATURE", 0.0),
            default_max_tokens=_i("LLM_ROUTER_MAX_TOKENS", _i("MAX_OUTPUT_TOKENS", 768)),
            default_timeout_sec=_f("LLM_ROUTER_TIMEOUT_SEC", _f("GATEWAY_READ_TIMEOUT", 60.0)),
            enable_cache=_b("LLM_ROUTER_CACHE", True),
            probe_groq_models=_b("GROQ_PROBE_MODELS", True),
            log_routing=_b("LLM_ROUTER_LOG", True),
        )


def api_keys() -> dict[str, str]:
    try:
        from app.config import settings

        groq = (
            settings.GROQ_API_KEY
            or os.getenv("GROQ_API_KEY")
            or os.getenv("Groq_API_KEY")
            or ""
        ).strip()
    except Exception:
        groq = (os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY") or "").strip()
    return {"groq": groq}
