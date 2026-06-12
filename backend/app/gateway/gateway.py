"""
LLMGateway — the orchestrator and drop-in replacement for GroqLLM.

Pipeline per request:
    1. semantic cache lookup (reuse BGE embeddings)        -> hit short-circuits
    2. classify complexity -> RouteDecision {score, reasons, tier}
    3. shadow / canary gating
    4. route to provider with timeout + retry + failover
    5. store result in cache; emit metrics; account cost saved

The public surface mirrors GroqLLM:
    gateway.generate(prompt) -> dict (superset of GroqLLM.generate())

A richer overload accepts a RoutingContext for full routing fidelity:
    gateway.generate(prompt, routing_context=ctx)
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable

from loguru import logger

from config import LLM_MAX_TOKENS, LLM_TEMPERATURE, SYSTEM_PROMPT

from backend.app.gateway import classifier
from backend.app.gateway import config as cfg
from backend.app.gateway import metrics as gm
from backend.app.gateway.cache import SemanticCache
from backend.app.gateway.providers import build_default_registry
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.router import Router
from backend.app.gateway.types import GatewayResult, RouteDecision, RoutingContext


class LLMGateway:
    def __init__(
        self,
        *,
        embed_fn: Callable[[str], Any] | None = None,
        providers: dict[str, Provider] | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        self._providers = providers or build_default_registry()
        self._router = Router(self._providers)
        if cache is not None:
            self._cache = cache
        elif embed_fn is not None:
            self._cache = SemanticCache(embed_fn)
        else:
            self._cache = None  # cache disabled (no embedder available)

    # ───────────────────────── public API ─────────────────────────
    def generate(
        self, prompt: str, routing_context: RoutingContext | None = None
    ) -> dict[str, Any]:
        """Drop-in for GroqLLM.generate(). Returns a superset dict."""
        result = self.complete(prompt, routing_context)
        return result.as_legacy_dict()

    def complete(
        self, prompt: str, routing_context: RoutingContext | None = None
    ) -> GatewayResult:
        t0 = time.perf_counter()
        ctx = routing_context or RoutingContext(prompt=prompt, query=prompt)
        ctx.prompt = ctx.prompt or prompt

        # 1. Semantic cache --------------------------------------------------
        decision = classifier.classify(ctx)  # needed for cost-saved accounting
        if self._cache is not None:
            hit = self._cache.lookup(prompt, ctx.scope)
            if hit:
                gm.GATEWAY_CACHE_HITS_TOTAL.inc()
                saved = self._estimate_cost(decision.tier, prompt, hit["answer"])
                gm.GATEWAY_COST_SAVED_USD_TOTAL.inc(saved)
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                gm.GATEWAY_REQUESTS_TOTAL.labels(
                    provider="cache",
                    model=hit.get("model", "cache"),
                    tier=str(decision.tier),
                    outcome="cache_hit",
                ).inc()
                logger.info(
                    f"Gateway cache HIT (sim={hit.get('similarity')}) "
                    f"in {latency_ms}ms"
                )
                return GatewayResult(
                    answer=hit["answer"],
                    model=hit.get("model", "cache"),
                    provider="cache",
                    tier=decision.tier,
                    latency_ms=latency_ms,
                    cache_hit=True,
                    route_score=decision.score,
                    route_reasons=decision.reasons,
                    cost_saved_usd=saved,
                    citations=hit.get("citations"),
                    grounding=hit.get("grounding"),
                )
            gm.GATEWAY_CACHE_MISSES_TOTAL.inc()

        # 2/3. Shadow & canary gating ---------------------------------------
        effective = self._apply_gating(decision)

        gm.GATEWAY_ROUTE_SCORE.labels(tier=str(decision.tier)).observe(decision.score)

        # 4. Route with failover --------------------------------------------
        messages = self._build_messages(prompt, ctx)
        try:
            outcome = self._router.route(
                effective.provider,
                messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        except ProviderError as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            gm.GATEWAY_REQUESTS_TOTAL.labels(
                provider=effective.provider,
                model=effective.model,
                tier=str(effective.tier),
                outcome="error",
            ).inc()
            logger.error(f"Gateway routing failed: {exc}")
            return GatewayResult(
                answer=f"LLM error: {exc}",
                model=effective.model,
                provider=effective.provider,
                tier=effective.tier,
                latency_ms=latency_ms,
                route_score=decision.score,
                route_reasons=decision.reasons,
                error=str(exc),
            )

        resp = outcome.response
        served_tier = cfg.tier_for_provider_key(outcome.provider_key)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 5. Cache store + accounting + metrics -----------------------------
        if self._cache is not None and not effective_is_shadow(decision, effective):
            self._cache.store(
                prompt=prompt,
                answer=resp.answer,
                model=resp.model,
                scope=ctx.scope,
                citations=ctx.retrieval.get("citations"),
                grounding=ctx.grounding,
            )

        cost = self._token_cost(served_tier, resp.prompt_tokens, resp.completion_tokens,
                                prompt, resp.answer)
        saved = self._downgrade_savings(served_tier, resp.prompt_tokens,
                                        resp.completion_tokens, prompt, resp.answer)
        if saved > 0:
            gm.GATEWAY_COST_SAVED_USD_TOTAL.inc(saved)

        gm.GATEWAY_MODEL_USAGE_TOTAL.labels(
            provider=resp.provider, model=resp.model, tier=str(served_tier)
        ).inc()
        gm.GATEWAY_REQUESTS_TOTAL.labels(
            provider=resp.provider,
            model=resp.model,
            tier=str(served_tier),
            outcome="success",
        ).inc()
        gm.GATEWAY_LATENCY_SECONDS.labels(
            provider=resp.provider, model=resp.model
        ).observe(latency_ms / 1000.0)

        logger.info(
            f"Gateway served tier={served_tier} model={resp.model} "
            f"score={decision.score:.2f} fallback={outcome.fallback_used} "
            f"in {latency_ms}ms"
        )
        return GatewayResult(
            answer=resp.answer,
            model=resp.model,
            provider=resp.provider,
            tier=served_tier,
            latency_ms=latency_ms,
            cache_hit=False,
            fallback_used=outcome.fallback_used,
            route_score=decision.score,
            route_reasons=decision.reasons,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=cost,
            cost_saved_usd=saved,
        )

    # ───────────────────────── helpers ─────────────────────────
    def route_preview(self, ctx: RoutingContext) -> RouteDecision:
        """Expose the routing decision without calling a provider (shadow/debug)."""
        return classifier.classify(ctx)

    def _apply_gating(self, decision: RouteDecision) -> RouteDecision:
        """Shadow mode and canary keep traffic on Tier 1 (Groq) for safety."""
        if cfg.GATEWAY_SHADOW_MODE:
            return _force_tier1(decision)
        if cfg.GATEWAY_CANARY_PCT < 100:
            if random.randint(1, 100) > cfg.GATEWAY_CANARY_PCT:
                return _force_tier1(decision)
        return decision

    def _build_messages(
        self, prompt: str, ctx: RoutingContext
    ) -> list[dict[str, str]]:
        # Preserve the existing system prompt so answers stay consistent.
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _estimate_cost(
        self, tier: int, prompt: str, answer: str
    ) -> float:
        spec = cfg.TIERS.get(tier, cfg.TIERS[1])
        p_tok = _approx_tokens(prompt)
        c_tok = _approx_tokens(answer)
        return (p_tok * spec.price_in_per_m + c_tok * spec.price_out_per_m) / 1_000_000

    def _token_cost(
        self, tier: int, p_tok: int | None, c_tok: int | None,
        prompt: str, answer: str
    ) -> float:
        spec = cfg.TIERS.get(tier, cfg.TIERS[1])
        pt = p_tok if p_tok is not None else _approx_tokens(prompt)
        ct = c_tok if c_tok is not None else _approx_tokens(answer)
        return (pt * spec.price_in_per_m + ct * spec.price_out_per_m) / 1_000_000

    def _downgrade_savings(
        self, served_tier: int, p_tok: int | None, c_tok: int | None,
        prompt: str, answer: str
    ) -> float:
        """Saving vs always using the configured baseline (most-capable) tier."""
        baseline = cfg.COST_BASELINE_TIER
        if served_tier >= baseline:
            return 0.0
        base_cost = self._token_cost(baseline, p_tok, c_tok, prompt, answer)
        actual = self._token_cost(served_tier, p_tok, c_tok, prompt, answer)
        return max(0.0, base_cost - actual)


def _force_tier1(decision: RouteDecision) -> RouteDecision:
    spec = cfg.TIERS[1]
    return RouteDecision(
        score=decision.score,
        reasons=decision.reasons + ["shadow/canary: forced Tier 1"],
        tier=1,
        provider="groq",
        model=spec.model,
        signals=decision.signals,
    )


def effective_is_shadow(original: RouteDecision, effective: RouteDecision) -> bool:
    return cfg.GATEWAY_SHADOW_MODE and effective.tier == 1 and original.tier != 1


def _approx_tokens(text: str) -> int:
    return max(1, int(len((text or "").split()) * 1.33))
