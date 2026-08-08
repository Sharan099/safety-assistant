"""Groq LLM router with model failover — single entry for all generation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from backend.app.gateway import cache
from backend.app.gateway.error_policy import ErrorKind, classify_error
from backend.app.gateway.prompt_budget import compress_messages, fit_messages_for_model
from backend.app.gateway.providers.base import ProviderError
from backend.app.llm_router import health, metrics, tokens
from backend.app.llm_router.config import RouterConfig
from backend.app.llm_router.provider_registry import ProviderRegistry
from backend.app.llm_router.registry import (
    ModelSpec,
    filter_groq_available,
    ordered_chain,
)


@dataclass
class RoutingStep:
    model_key: str
    model_id: str
    provider: str
    outcome: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    detail: str = ""
    retry_count: int = 0
    fallback_reason: str = ""


@dataclass
class RouterResult:
    text: str
    model_key: str
    model_id: str
    provider: str
    evidence_only: bool = False
    cache_hit: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    retry_count: int = 0
    steps: list[RoutingStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "provider": self.provider,
            "evidence_only": self.evidence_only,
            "cache_hit": self.cache_hit,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 2),
            "retry_count": self.retry_count,
            "steps": [
                {
                    "model_key": s.model_key,
                    "model_id": s.model_id,
                    "provider": s.provider,
                    "outcome": s.outcome,
                    "latency_ms": round(s.latency_ms, 2),
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "cache_creation_input_tokens": s.cache_creation_input_tokens,
                    "cache_read_input_tokens": s.cache_read_input_tokens,
                    "detail": s.detail,
                    "retry_count": s.retry_count,
                    "fallback_reason": s.fallback_reason,
                }
                for s in self.steps
            ],
        }


class LLMRouter:
    """Groq router with automatic failover across Groq models."""

    _RETRYABLE = frozenset({ErrorKind.TIMEOUT})

    def __init__(
        self,
        *,
        config: RouterConfig | None = None,
        providers: ProviderRegistry | None = None,
        use_cache: bool | None = None,
    ):
        self.config = config or RouterConfig.from_env()
        self.providers = providers or ProviderRegistry()
        self.use_cache = self.config.enable_cache if use_cache is None else use_cache
        self._groq_available = filter_groq_available(self.config.probe_groq_models)

    def _is_evaluation_mode(self) -> bool:
        return len(self.config.chain) == 1 and os.getenv(
            "EVALUATION_MODE", ""
        ).lower() in ("1", "true", "yes", "on")

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        context_chunks: list[dict[str, Any]] | None = None,
    ) -> RouterResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete(
            messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            context_chunks=context_chunks,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        context_chunks: list[dict[str, Any]] | None = None,
    ) -> RouterResult:
        start = time.time()
        max_out = max_output_tokens or self.config.default_max_tokens
        temp = self.config.default_temperature if temperature is None else temperature
        chain = self._eligible_chain()
        steps: list[RoutingStep] = []
        total_retries = 0

        if self.use_cache and chain:
            cached = cache.get(messages, chain[0].key)
            if cached:
                return RouterResult(
                    text=cached.text,
                    model_key=cached.model_key,
                    model_id=cached.model_id,
                    provider=cached.provider,
                    cache_hit=True,
                    prompt_tokens=cached.prompt_tokens,
                    completion_tokens=cached.completion_tokens,
                    latency_ms=(time.time() - start) * 1000,
                    steps=steps,
                )

        prev_provider = ""
        for spec in chain:
            if not health.is_healthy(spec.key):
                # Evaluation single-model mode: wait for cooldown instead of evidence-only.
                if self._is_evaluation_mode():
                    if health.wait_until_healthy(spec.key, max_wait_sec=self.config.cooldown_sec + 5):
                        pass  # retry this model
                    else:
                        steps.append(
                            RoutingStep(
                                model_key=spec.key,
                                model_id=spec.model_id,
                                provider=spec.provider,
                                outcome="skipped",
                                latency_ms=0.0,
                                detail="cooldown",
                                fallback_reason="health_disabled",
                            )
                        )
                        continue
                else:
                    steps.append(
                        RoutingStep(
                            model_key=spec.key,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            outcome="skipped",
                            latency_ms=0.0,
                            detail="cooldown",
                            fallback_reason="health_disabled",
                        )
                    )
                    continue

            fb_reason = f"failover_from_{prev_provider}" if prev_provider else ""
            result, retries = self._invoke(spec, messages, max_out, temp, steps, fb_reason)
            total_retries += retries
            if result is not None:
                result.latency_ms = (time.time() - start) * 1000
                result.retry_count = total_retries
                result.steps = steps
                if self.config.log_routing:
                    logger.info(
                        "LLM router success provider={} model={} key={} latency_ms={:.0f} retries={}",
                        result.provider,
                        result.model_id,
                        result.model_key,
                        result.latency_ms,
                        total_retries,
                    )
                return result
            prev_provider = spec.provider
            if steps:
                metrics.record_failover(
                    from_provider=steps[-1].provider,
                    to_provider="next",
                    reason=steps[-1].outcome,
                )
                # Evaluation mode: rate-limit → wait and retry same model (up to 5).
                if self._is_evaluation_mode() and steps[-1].outcome == "rate_limit":
                    recovered = None
                    for attempt in range(5):
                        if not health.wait_until_healthy(
                            spec.key, max_wait_sec=self.config.cooldown_sec + 5
                        ):
                            break
                        result2, retries2 = self._invoke(
                            spec,
                            messages,
                            max_out,
                            temp,
                            steps,
                            f"eval_mode_rate_limit_wait_{attempt + 1}",
                        )
                        total_retries += retries2
                        if result2 is not None:
                            recovered = result2
                            break
                        if not steps or steps[-1].outcome != "rate_limit":
                            break
                    if recovered is not None:
                        recovered.latency_ms = (time.time() - start) * 1000
                        recovered.retry_count = total_retries
                        recovered.steps = steps
                        return recovered

        text = self._evidence_only(context_chunks or [], messages)
        steps.append(
            RoutingStep(
                model_key="evidence_only",
                model_id="none",
                provider="registry",
                outcome="success",
                latency_ms=0.0,
                detail="all providers failed or unavailable",
            )
        )
        return RouterResult(
            text=text,
            model_key="evidence_only",
            model_id="none",
            provider="registry",
            evidence_only=True,
            latency_ms=(time.time() - start) * 1000,
            retry_count=total_retries,
            steps=steps,
        )

    def _eligible_chain(self) -> list[ModelSpec]:
        chain = ordered_chain(self.config)
        if not self._groq_available:
            return chain
        return [s for s in chain if s.model_id in self._groq_available]

    def _invoke(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        max_output_tokens: int,
        temperature: float,
        steps: list[RoutingStep],
        fallback_reason: str,
    ) -> tuple[RouterResult | None, int]:
        provider = self.providers.get(spec.provider)
        if not provider.is_configured():
            steps.append(
                RoutingStep(
                    model_key=spec.key,
                    model_id=spec.model_id,
                    provider=spec.provider,
                    outcome="skipped",
                    latency_ms=0.0,
                    detail="not_configured",
                    fallback_reason=fallback_reason,
                )
            )
            return None, 0

        fitted = fit_messages_for_model(messages, spec, max_output_tokens=max_output_tokens)
        compress_level = 0
        retries = 0
        while compress_level <= 3:
            attempt_messages = (
                fitted
                if compress_level == 0
                else compress_messages(
                    fitted, spec, level=compress_level, max_output_tokens=max_output_tokens
                )
            )
            too_large_retry = False
            for attempt in range(self.config.max_retries_per_model):
                step_start = time.time()
                try:
                    raw = provider.complete(
                        spec.model_id,
                        attempt_messages,
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                        timeout=self.config.default_timeout_sec,
                    )
                    latency = (time.time() - step_start) * 1000
                    health.note_success(spec.key)
                    metrics.record_request(
                        provider=spec.provider,
                        model=spec.model_id,
                        outcome="success",
                        latency_sec=latency / 1000,
                        prompt_tokens=raw.prompt_tokens,
                        completion_tokens=raw.completion_tokens,
                    )
                    tokens.record(
                        spec.provider,
                        input_tokens=raw.prompt_tokens,
                        output_tokens=raw.completion_tokens,
                    )
                    steps.append(
                        RoutingStep(
                            model_key=spec.key,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            outcome="success",
                            latency_ms=latency,
                            prompt_tokens=raw.prompt_tokens,
                            completion_tokens=raw.completion_tokens,
                            detail=str(raw.get("finish_reason") or ""),
                            retry_count=attempt,
                            fallback_reason=fallback_reason,
                        )
                    )
                    if self.use_cache:
                        cache.put(
                            messages,
                            spec.key,
                            text=raw.text,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            prompt_tokens=raw.prompt_tokens,
                            completion_tokens=raw.completion_tokens,
                        )
                    return (
                        RouterResult(
                            text=raw.text,
                            model_key=spec.key,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            prompt_tokens=raw.prompt_tokens,
                            completion_tokens=raw.completion_tokens,
                            finish_reason=str(raw.get("finish_reason") or ""),
                        ),
                        retries + attempt,
                    )
                except ProviderError as exc:
                    kind = classify_error(exc)
                    detail = str(exc)[:200]
                    if exc.status_code:
                        detail = f"HTTP {exc.status_code}: {detail}"
                    latency = (time.time() - step_start) * 1000
                    metrics.record_error(
                        provider=spec.provider, model=spec.model_id, error_kind=kind.value
                    )
                    metrics.record_request(
                        provider=spec.provider,
                        model=spec.model_id,
                        outcome=kind.value,
                        latency_sec=latency / 1000,
                    )
                    steps.append(
                        RoutingStep(
                            model_key=spec.key,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            outcome=kind.value,
                            latency_ms=latency,
                            detail=detail,
                            retry_count=attempt,
                            fallback_reason=fallback_reason,
                        )
                    )
                    if kind == ErrorKind.TOO_LARGE and compress_level < 3:
                        compress_level += 1
                        too_large_retry = True
                        break
                    if kind in (ErrorKind.RATE_LIMIT, ErrorKind.CONNECTION, ErrorKind.TIMEOUT):
                        health.note_failure(spec.key, kind, cooldown_sec=self.config.cooldown_sec)
                    elif kind == ErrorKind.FATAL:
                        health.note_failure(spec.key, kind, cooldown_sec=self.config.cooldown_sec)
                    if kind in self._RETRYABLE and attempt < self.config.max_retries_per_model - 1:
                        retries += 1
                        metrics.record_retry(provider=spec.provider, model=spec.model_id)
                        delay = self.config.retry_base_sec * (2**attempt)
                        if self.config.log_routing:
                            logger.warning(
                                "LLM router {} on {} — retry {}/{} in {:.1f}s",
                                kind.value,
                                spec.key,
                                attempt + 1,
                                self.config.max_retries_per_model,
                                delay,
                            )
                        time.sleep(delay)
                        continue
                    return None, retries + attempt
            if too_large_retry:
                continue
            return None, retries
        return None, retries

    @staticmethod
    def _evidence_only(chunks: list[dict[str, Any]], messages: list[dict[str, str]]) -> str:
        if not chunks:
            user = next((m["content"] for m in messages if m.get("role") == "user"), "")
            return (
                "[evidence-only] No LLM summary available. "
                "I could not find grounded passages to answer your question.\n"
                f"Question: {user[:300]}"
            )
        lines = ["[evidence-only] LLM unavailable — retrieved passages only:\n"]
        for idx, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.get("chunk_text", "")
            if chunk.get("confidential_tier") is True or chunk.get("chunk_type") == "test_record":
                chunk_text = (
                    "[REDACTED: Confidential data cannot be displayed via "
                    "unpermitted evidence-only fallback]"
                )
            lines.append(
                f"[{idx}] {chunk.get('regulation_code', '?')} | "
                f"{chunk.get('document_name', '?')} p.{chunk.get('page_number', '?')}\n"
                f"{chunk_text[:600]}"
            )
        return "\n\n".join(lines)


_router_singleton: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = LLMRouter()
    return _router_singleton


def reset_router_for_tests() -> None:
    global _router_singleton
    _router_singleton = None
    health.reset_for_tests()
    tokens.reset_for_tests()
    cache.clear_for_tests()
