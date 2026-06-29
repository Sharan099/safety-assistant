"""
Canonical model registry for the gateway failover chain.

Token limits are configurable via env vars — verify defaults against current
provider documentation before changing:
  Groq models:     https://console.groq.com/docs/models
  Anthropic models: https://docs.anthropic.com/en/docs/about-claude/models
  Groq rate limits: https://console.groq.com/docs/rate-limits
  OpenRouter models: https://openrouter.ai/docs/models
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Llama 3.1 8B decommissioned 2026-08-16 on Groq — fast tier is now GPT OSS 20B.
# NOTE: 20B fast tier narrows the fast-vs-power gap vs 70B; re-evaluate Step 5
# tiered routing against the authoritative eval — do not assume 8B-era cost split.
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")

# OpenRouter slugs — override without code changes (verify on openrouter.ai/models).
OPENROUTER_MODEL_LLAMA = os.getenv("OPENROUTER_MODEL_LLAMA", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_MODEL_GEMINI = os.getenv("OPENROUTER_MODEL_GEMINI", "google/gemini-2.5-flash")
OPENROUTER_MODEL_CLAUDE = os.getenv("OPENROUTER_MODEL_CLAUDE", "anthropic/claude-sonnet-4")


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ModelSpec:
    """One routable model with priority and token ceilings."""

    key: str
    provider: str  # groq | anthropic | openrouter
    model_id: str
    priority: int  # lower = preferred earlier in chain
    # Published context window (input + output budget at provider).
    max_context_tokens: int
    # Effective per-request input cap for routing/compression (TPM / tier limits).
    effective_request_tokens: int


# Defaults — override via GATEWAY_* env vars; re-verify when providers change tiers.
_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="groq",
        provider="groq",
        model_id=GROQ_MODEL,
        priority=1,
        max_context_tokens=_i("GATEWAY_GROQ_PRIMARY_MAX_CONTEXT", 128_000),
        # Groq on-demand free tier TPM can be ~6k–12k depending on model; 70B
        # accepts larger prompts when quota allows — verify in Groq console.
        effective_request_tokens=_i("GATEWAY_GROQ_PRIMARY_EFFECTIVE_TOKENS", 12_000),
    ),
    ModelSpec(
        key="groq_power",
        provider="groq",
        model_id=os.getenv("GROQ_TIER_MODEL_POWER", GROQ_MODEL),
        priority=1,
        max_context_tokens=_i("GATEWAY_GROQ_PRIMARY_MAX_CONTEXT", 128_000),
        effective_request_tokens=_i("GATEWAY_GROQ_PRIMARY_EFFECTIVE_TOKENS", 12_000),
    ),
    ModelSpec(
        key="groq_fast",
        provider="groq",
        model_id=GROQ_MODEL_FAST,
        priority=4,
        max_context_tokens=_i("GATEWAY_GROQ_FAST_MAX_CONTEXT", 131_072),
        # GPT OSS 20B (replaces decommissioned Llama 3.1 8B): 131k context; interim
        # judge fallback only — Claude remains authoritative judge when egress allows.
        effective_request_tokens=_i("GATEWAY_GROQ_FAST_EFFECTIVE_TOKENS", 8_000),
    ),
    ModelSpec(
        key="anthropic_sonnet",
        provider="anthropic",
        model_id=os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6"),
        priority=2,
        max_context_tokens=_i("GATEWAY_SONNET_MAX_CONTEXT", 200_000),
        effective_request_tokens=_i("GATEWAY_SONNET_EFFECTIVE_TOKENS", 16_000),
    ),
    ModelSpec(
        key="anthropic_haiku",
        provider="anthropic",
        model_id=os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5"),
        priority=3,
        max_context_tokens=_i("GATEWAY_HAIKU_MAX_CONTEXT", 200_000),
        effective_request_tokens=_i("GATEWAY_HAIKU_EFFECTIVE_TOKENS", 16_000),
    ),
    ModelSpec(
        key="openrouter_llama",
        provider="openrouter",
        model_id=OPENROUTER_MODEL_LLAMA,
        priority=5,
        # OpenRouter: meta-llama/llama-3.3-70b-instruct — 131072 context per model card.
        max_context_tokens=_i("GATEWAY_OPENROUTER_LLAMA_MAX_CONTEXT", 131_072),
        effective_request_tokens=_i("GATEWAY_OPENROUTER_LLAMA_EFFECTIVE_TOKENS", 12_000),
    ),
    ModelSpec(
        key="openrouter_gemini",
        provider="openrouter",
        model_id=OPENROUTER_MODEL_GEMINI,
        priority=6,
        # google/gemini-2.5-flash — large context; cap routing budget for cost control.
        max_context_tokens=_i("GATEWAY_OPENROUTER_GEMINI_MAX_CONTEXT", 1_048_576),
        effective_request_tokens=_i("GATEWAY_OPENROUTER_GEMINI_EFFECTIVE_TOKENS", 16_000),
    ),
    ModelSpec(
        key="openrouter_claude",
        provider="openrouter",
        model_id=OPENROUTER_MODEL_CLAUDE,
        priority=7,
        # anthropic/claude-sonnet-4 via OpenRouter — firewall workaround when direct Anthropic blocked.
        max_context_tokens=_i("GATEWAY_OPENROUTER_CLAUDE_MAX_CONTEXT", 200_000),
        effective_request_tokens=_i("GATEWAY_OPENROUTER_CLAUDE_EFFECTIVE_TOKENS", 16_000),
    ),
)

REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


def get_spec(key: str) -> ModelSpec | None:
    return REGISTRY.get(key)


def ordered_keys(primary: str) -> list[str]:
    """Failover order from config, keeping only registered models."""
    from backend.app.gateway import config as cfg

    chain = cfg.FALLBACK_CHAINS.get(primary, [primary])
    return [k for k in chain if k in REGISTRY]
