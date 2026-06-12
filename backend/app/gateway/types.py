"""
Gateway data contracts.

`GatewayResult` is intentionally a SUPERSET of the dict returned by
`GroqLLM.generate()` (keys: answer, latency_ms, prompt_tokens,
completion_tokens, model). `as_legacy_dict()` produces that exact superset so
the gateway is a drop-in replacement behind `RAGWorkflow._get_llm()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingContext:
    """All signals the classifier may use. Everything except `prompt` is
    optional so the gateway also works with a bare `.generate(prompt)` call."""

    prompt: str
    query: str = ""
    # Reused verbatim from RAGState["grounding"] (assess_grounding output).
    grounding: dict[str, Any] = field(default_factory=dict)
    # Retrieval signals (doc_count, best_semantic, reranker_used, ...).
    retrieval: dict[str, Any] = field(default_factory=dict)
    conversation_depth: int = 0
    has_code: bool | None = None  # None => classifier detects from text
    user_id: str | None = None
    session_id: str | None = None
    # Aggregate feedback/perf snapshots (injected by the workflow/store).
    feedback_downvote_rate: float = 0.0
    model_performance: dict[str, float] = field(default_factory=dict)
    # Detected regulation scope (for cache namespacing); e.g. ["UN_R14"].
    scope: list[str] = field(default_factory=list)

    @property
    def grounding_confidence(self) -> float:
        val = self.grounding.get("confidence")
        return float(val) if isinstance(val, (int, float)) else 0.0

    @property
    def retrieval_confidence(self) -> float:
        # Prefer rerank probability, else best semantic cosine, else grounding.
        for key in ("best_rerank_prob", "best_semantic"):
            v = self.grounding.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        v = self.retrieval.get("best_semantic")
        if isinstance(v, (int, float)):
            return float(v)
        return self.grounding_confidence


@dataclass
class RouteDecision:
    score: float                       # 0..10
    reasons: list[str]
    tier: int                          # 1 | 2 | 3
    provider: str                      # provider key (groq|anthropic_haiku|...)
    model: str
    signals: dict[str, float] = field(default_factory=dict)  # raw 0..1 signals

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


@dataclass
class ProviderResponse:
    """Normalised provider output (OpenAI-compatible content)."""

    answer: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class GatewayResult:
    answer: str
    model: str
    provider: str
    tier: int
    latency_ms: float
    cache_hit: bool = False
    fallback_used: bool = False
    route_score: float | None = None
    route_reasons: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    # Carried through for cache hits so the workflow can reuse them.
    citations: list[dict[str, Any]] | None = None
    grounding: dict[str, Any] | None = None
    error: str | None = None

    def as_legacy_dict(self) -> dict[str, Any]:
        """Superset of GroqLLM.generate() — safe drop-in for the workflow."""
        return {
            # ---- original GroqLLM keys (unchanged contract) ----
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            # ---- additive gateway fields ----
            "provider": self.provider,
            "tier": self.tier,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "route_score": self.route_score,
            "route_reasons": self.route_reasons,
            "cost_usd": self.cost_usd,
            "cost_saved_usd": self.cost_saved_usd,
            "error": self.error,
        }
