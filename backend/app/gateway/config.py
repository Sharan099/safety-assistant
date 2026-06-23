"""
Gateway configuration: model tiers, pricing, routing weights/thresholds, cache
and reliability knobs. Everything is environment-overridable so the platform
team can tune routing and pin provider model slugs without a code change.

This module is self-contained on purpose: the gateway is a separable
infrastructure component and must not depend on retrieval/graph internals for
its configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from config import GROQ_MODEL, GROQ_MODEL_FAST  # reuse configured Groq models


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


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


# ───────────────────────── master switches ─────────────────────────
# Default OFF: with the gateway disabled the workflow uses GroqLLM exactly as in
# v2.2 (full backward compatibility). Shadow mode routes + records metrics but
# always answers with Tier 1 (Groq) so production risk is zero.
ENABLE_GATEWAY = _flag("ENABLE_GATEWAY", "false")
GATEWAY_SHADOW_MODE = _flag("GATEWAY_SHADOW_MODE", "false")
GATEWAY_CANARY_PCT = _i("GATEWAY_CANARY_PCT", 100)  # % of traffic that may route

ENABLE_SEMANTIC_CACHE = _flag("ENABLE_SEMANTIC_CACHE", "true")


# ───────────────────────── providers / tiers ─────────────────────────
# Tier 1 = primary Groq (70B). Fast 8B is emergency-only (groq_fast provider).
GROQ_TIER_MODEL = os.getenv("GROQ_TIER_MODEL", GROQ_MODEL)
GROQ_TIER_MODEL_POWER = os.getenv("GROQ_TIER_MODEL_POWER", GROQ_MODEL)
GROQ_TIER_MODEL_FAST = os.getenv("GROQ_TIER_MODEL_FAST", GROQ_MODEL_FAST)
# Exact Anthropic slugs are env-pinned (production teams set the current ids).
CLAUDE_HAIKU_MODEL = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5")
CLAUDE_SONNET_MODEL = os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-5")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


@dataclass(frozen=True)
class TierSpec:
    tier: int
    provider: str          # "groq" | "anthropic"
    model: str
    purpose: str
    # USD per 1M tokens (rough public list rates; override via env in config maps)
    price_in_per_m: float
    price_out_per_m: float


# Tier definitions. Pricing is used for cost estimation, cost-saved accounting
# and as a tie-breaker near tier boundaries. Rates are overridable.
TIERS: dict[int, TierSpec] = {
    1: TierSpec(
        tier=1,
        provider="groq",
        model=GROQ_TIER_MODEL,
        purpose="simple single-value lookups (definitions, limits, citation)",
        price_in_per_m=_f("PRICE_GROQ_IN", 0.59),
        price_out_per_m=_f("PRICE_GROQ_OUT", 0.79),
    ),
    2: TierSpec(
        tier=2,
        provider="groq",
        model=GROQ_TIER_MODEL_POWER,
        purpose="comparisons, dummy/criteria mapping, multi-regulation reasoning",
        price_in_per_m=_f("PRICE_GROQ_POWER_IN", 0.59),
        price_out_per_m=_f("PRICE_GROQ_POWER_OUT", 0.79),
    ),
    3: TierSpec(
        tier=3,
        provider="anthropic",
        model=CLAUDE_SONNET_MODEL,
        purpose="advanced reasoning (debugging, analysis, planning, code)",
        price_in_per_m=_f("PRICE_SONNET_IN", 3.00),
        price_out_per_m=_f("PRICE_SONNET_OUT", 15.00),
    ),
}

# Tier 2 (legacy) used Anthropic Haiku — kept as optional escalation via failover.

# Baseline tier used to compute "cost saved" when we route DOWN from the most
# capable model. cost_saved = baseline_cost - actual_cost (never negative).
COST_BASELINE_TIER = _i("GATEWAY_COST_BASELINE_TIER", 3)


# ───────────────────────── routing weights & thresholds ─────────────────────────
@dataclass(frozen=True)
class RoutingWeights:
    prompt_length: float = _f("W_PROMPT_LENGTH", 0.10)
    query_complexity: float = _f("W_QUERY_COMPLEXITY", 0.18)
    grounding_confidence: float = _f("W_GROUNDING", 0.15)
    conversation_depth: float = _f("W_CONV_DEPTH", 0.08)
    presence_of_code: float = _f("W_CODE", 0.13)
    reasoning_tasks: float = _f("W_REASONING", 0.16)
    retrieval_confidence: float = _f("W_RETRIEVAL", 0.10)
    feedback_history: float = _f("W_FEEDBACK", 0.05)
    model_performance: float = _f("W_MODEL_PERF", 0.03)
    estimated_cost: float = _f("W_COST", 0.02)


WEIGHTS = RoutingWeights()

# score < TIER1_MAX -> Tier 1; < TIER2_MAX -> Tier 2; else Tier 3.
TIER1_MAX_SCORE = _f("TIER1_MAX_SCORE", 3.5)
TIER2_MAX_SCORE = _f("TIER2_MAX_SCORE", 6.5)

# Normalisation anchors.
PROMPT_TOKENS_SATURATION = _i("PROMPT_TOKENS_SATURATION", 2000)  # tokens -> 1.0
CONV_DEPTH_SATURATION = _i("CONV_DEPTH_SATURATION", 8)           # turns -> 1.0

# Keyword cues (lowercased, substring match). Override via comma-separated env.
REASONING_KEYWORDS: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv(
        "REASONING_KEYWORDS",
        "debug,derive,prove,why,explain why,root cause,trade-off,tradeoff,"
        "step by step,plan,design,analyse,analyze,compare,evaluate,optimise,"
        "optimize,reason,implications,architecture",
    ).split(",")
    if s.strip()
)

CODE_KEYWORDS: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv(
        "CODE_KEYWORDS",
        "def ,class ,import ,function,traceback,stacktrace,compile,"
        "python,javascript,typescript,sql,regex,api call,snippet",
    ).split(",")
    if s.strip()
)


# ───────────────────────── reliability ─────────────────────────
PROVIDER_TIMEOUT_S = _f("GATEWAY_PROVIDER_TIMEOUT_S", 20.0)
MAX_RETRIES = _i("GATEWAY_MAX_RETRIES", 1)          # retries per provider
BACKOFF_BASE_S = _f("GATEWAY_BACKOFF_BASE_S", 0.5)  # exponential base
BACKOFF_MAX_S = _f("GATEWAY_BACKOFF_MAX_S", 8.0)

# Fast-tier (8B) generation caps when failover lands on Groq instant.
FAST_FALLBACK_MAX_TOKENS_DEFAULT = _i("FAST_FALLBACK_MAX_TOKENS_DEFAULT", 900)
FAST_FALLBACK_MAX_TOKENS_LIST = _i("FAST_FALLBACK_MAX_TOKENS_LIST", 550)
FAST_FALLBACK_FREQUENCY_PENALTY = _f("FAST_FALLBACK_FREQUENCY_PENALTY", 0.35)

# Fallback chains keyed by the originally-selected provider.
# Fallback chains — never downgrade from 70B to 8B; fast tier is last resort only.
FALLBACK_CHAINS: dict[str, list[str]] = {
    "groq": ["groq", "anthropic_haiku", "anthropic_sonnet", "groq_fast"],
    "groq_power": ["groq_power", "groq", "anthropic_haiku", "anthropic_sonnet", "groq_fast"],
    "anthropic_haiku": ["anthropic_haiku", "anthropic_sonnet", "groq", "groq_fast"],
    "anthropic_sonnet": ["anthropic_sonnet", "anthropic_haiku", "groq", "groq_fast"],
    "groq_fast": ["groq_fast"],
}


# ───────────────────────── cache ─────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_INDEX_NAME = os.getenv("CACHE_INDEX_NAME", "psa_gw_cache")
CACHE_KEY_PREFIX = os.getenv("CACHE_KEY_PREFIX", "psa:gw:cache:")
CACHE_SIM_THRESHOLD = _f("CACHE_SIM_THRESHOLD", 0.95)  # cosine >= -> hit
CACHE_TTL_S = _i("CACHE_TTL_S", 86_400)                # 24h
CACHE_EMBED_DIM = _i("CACHE_EMBED_DIM", 768)           # BAAI/bge-base-en-v1.5
CACHE_FALLBACK_SCAN_LIMIT = _i("CACHE_FALLBACK_SCAN_LIMIT", 500)


@dataclass
class GatewaySettings:
    """Snapshot of the active configuration (handy for /health and tests)."""

    enabled: bool = ENABLE_GATEWAY
    shadow: bool = GATEWAY_SHADOW_MODE
    canary_pct: int = GATEWAY_CANARY_PCT
    cache_enabled: bool = ENABLE_SEMANTIC_CACHE
    tiers: dict[int, TierSpec] = field(default_factory=lambda: TIERS)


def provider_key_for_tier(tier: int) -> str:
    """Map a tier to the provider key used by the fallback chains."""
    if tier == 1:
        return "groq"
    if tier == 2:
        return "groq_power"
    return "anthropic_sonnet"


def tier_for_provider_key(provider_key: str) -> int:
    return {
        "groq": 1,
        "groq_power": 2,
        "groq_fast": 1,
        "anthropic_haiku": 2,
        "anthropic_sonnet": 3,
    }.get(provider_key, 1)
