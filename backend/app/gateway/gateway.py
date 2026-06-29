"""LLM gateway: failover, compression, cache, evidence-only fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from backend.app.gateway import cache, config
from backend.app.gateway.error_policy import (
    ErrorKind,
    classify_error,
    is_disabled,
    note_provider_failure,
    note_provider_success,
    reset_disabled_for_tests,
)
from backend.app.gateway.model_registry import ModelSpec, get_spec, ordered_keys
from backend.app.gateway.prompt_budget import compress_messages, count_message_tokens, estimate_tokens, fit_messages_for_model
from backend.app.gateway.providers.anthropic_provider import AnthropicProvider
from backend.app.gateway.providers.base import ProviderError
from backend.app.gateway.providers.groq_provider import GroqProvider
from backend.app.gateway.providers.openrouter_provider import OpenRouterProvider


@dataclass
class RoutingStep:
    model_key: str
    model_id: str
    provider: str
    outcome: str  # success | rate_limit | too_large | error | skipped
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    detail: str = ""


@dataclass
class GatewayResult:
    text: str
    model_key: str
    model_id: str
    provider: str
    evidence_only: bool = False
    cache_hit: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
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
            "latency_ms": round(self.latency_ms, 2),
            "steps": [
                {
                    "model_key": s.model_key,
                    "model_id": s.model_id,
                    "provider": s.provider,
                    "outcome": s.outcome,
                    "latency_ms": round(s.latency_ms, 2),
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "detail": s.detail,
                }
                for s in self.steps
            ],
        }


class LLMGateway:
    def __init__(
        self,
        *,
        groq: GroqProvider | None = None,
        anthropic: AnthropicProvider | None = None,
        openrouter: OpenRouterProvider | None = None,
        primary: str | None = None,
        use_cache: bool = True,
    ):
        self.groq = groq or GroqProvider()
        self.anthropic = anthropic or AnthropicProvider()
        self.openrouter = openrouter or OpenRouterProvider()
        self.primary = primary or config.DEFAULT_PRIMARY
        self.use_cache = use_cache

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int = 1024,
        context_chunks: list[dict[str, Any]] | None = None,
    ) -> GatewayResult:
        start = time.time()
        chain = ordered_keys(self.primary)
        steps: list[RoutingStep] = []

        # Check if confidential context is present
        has_confidential = False
        if context_chunks:
            for chunk in context_chunks:
                if chunk.get("confidential_tier") is True or chunk.get("chunk_type") == "test_record":
                    has_confidential = True
                    break

        if self.use_cache:
            cached = cache.get(messages, chain[0] if chain else self.primary)
            if cached:
                return GatewayResult(
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

        for model_key in chain:
            if is_disabled(model_key):
                steps.append(
                    RoutingStep(
                        model_key=model_key,
                        model_id="",
                        provider="",
                        outcome="skipped",
                        latency_ms=0.0,
                        detail="disabled",
                    )
                )
                continue
            spec = get_spec(model_key)
            if not spec:
                continue

            # Confidentiality Gate check at transmission time for each model key
            if has_confidential:
                from registry.harness_security import is_model_authorized
                if not is_model_authorized(model_key, spec.model_id):
                    steps.append(
                        RoutingStep(
                            model_key=model_key,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            outcome="skipped",
                            latency_ms=0.0,
                            detail="unauthorized_confidential",
                        )
                    )
                    continue

            fitted = fit_messages_for_model(messages, spec, max_output_tokens=max_output_tokens)
            compress_level = 0
            while compress_level <= 3:
                attempt_messages = (
                    fitted if compress_level == 0 else compress_messages(fitted, spec, level=compress_level, max_output_tokens=max_output_tokens)
                )
                step_start = time.time()
                try:
                    result = self._call_provider(spec, attempt_messages, max_output_tokens)
                    step = RoutingStep(
                        model_key=model_key,
                        model_id=spec.model_id,
                        provider=spec.provider,
                        outcome="success",
                        latency_ms=(time.time() - step_start) * 1000,
                        prompt_tokens=result["prompt_tokens"],
                        completion_tokens=result["completion_tokens"],
                    )
                    steps.append(step)
                    note_provider_success(model_key)
                    if spec.provider == "openrouter":
                        from backend.app.gateway.openrouter_spend import note_openrouter_request

                        spend = note_openrouter_request(
                            prompt_tokens=result["prompt_tokens"],
                            completion_tokens=result["completion_tokens"],
                        )
                        logger.warning(
                            "OpenRouter failover served (paid tier +5.5% fee): "
                            f"model_key={model_key} model_id={spec.model_id} "
                            f"prompt_tokens={result['prompt_tokens']} "
                            f"completion_tokens={result['completion_tokens']} "
                            f"daily_requests={spend['daily_requests']} "
                            f"daily_prompt_tokens={spend['daily_prompt_tokens']} "
                            f"daily_completion_tokens={spend['daily_completion_tokens']}"
                        )
                    if self.use_cache:
                        cache.put(
                            messages,
                            model_key,
                            text=result["text"],
                            model_id=spec.model_id,
                            provider=spec.provider,
                            prompt_tokens=result["prompt_tokens"],
                            completion_tokens=result["completion_tokens"],
                        )
                    return GatewayResult(
                        text=result["text"],
                        model_key=model_key,
                        model_id=spec.model_id,
                        provider=spec.provider,
                        prompt_tokens=result["prompt_tokens"],
                        completion_tokens=result["completion_tokens"],
                        latency_ms=(time.time() - start) * 1000,
                        steps=steps,
                    )
                except ProviderError as exc:
                    kind = classify_error(exc)
                    steps.append(
                        RoutingStep(
                            model_key=model_key,
                            model_id=spec.model_id,
                            provider=spec.provider,
                            outcome=kind.value,
                            latency_ms=(time.time() - step_start) * 1000,
                            detail=str(exc)[:200],
                        )
                    )
                    if kind == ErrorKind.TOO_LARGE and compress_level < 3:
                        compress_level += 1
                        continue
                    note_provider_failure(model_key, kind)
                    break

        # Evidence-only fallback
        text = self._evidence_only(context_chunks or [], messages)
        est_prompt = count_message_tokens(messages)
        est_completion = estimate_tokens(text)
        steps.append(
            RoutingStep(
                model_key="evidence_only",
                model_id="none",
                provider="registry",
                outcome="success",
                latency_ms=0.0,
                prompt_tokens=est_prompt,
                completion_tokens=est_completion,
                detail="all providers failed or unavailable",
            )
        )
        return GatewayResult(
            text=text,
            model_key="evidence_only",
            model_id="none",
            provider="registry",
            evidence_only=True,
            prompt_tokens=est_prompt,
            completion_tokens=est_completion,
            latency_ms=(time.time() - start) * 1000,
            steps=steps,
        )

    def _call_provider(self, spec: ModelSpec, messages: list[dict[str, str]], max_output_tokens: int) -> dict[str, Any]:
        if spec.provider == "groq":
            return self.groq.complete(spec.model_id, messages, max_tokens=max_output_tokens)
        if spec.provider == "anthropic":
            return self.anthropic.complete(spec.model_id, messages, max_tokens=max_output_tokens)
        if spec.provider == "openrouter":
            return self.openrouter.complete(spec.model_id, messages, max_tokens=max_output_tokens)
        raise ProviderError(f"Unknown provider {spec.provider}", kind=ErrorKind.FATAL)

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
            chunk_text = chunk.get('chunk_text', '')
            if chunk.get("confidential_tier") is True or chunk.get("chunk_type") == "test_record":
                chunk_text = "[REDACTED: Confidential data cannot be displayed via unpermitted evidence-only fallback]"
            lines.append(
                f"[{idx}] {chunk.get('regulation_code', '?')} | "
                f"{chunk.get('document_name', '?')} p.{chunk.get('page_number', '?')}\n"
                f"{chunk_text[:600]}"
            )
        return "\n\n".join(lines)


def reset_gateway_state_for_tests() -> None:
    reset_disabled_for_tests()
    cache.clear_for_tests()
    from backend.app.gateway.openrouter_spend import reset_for_tests

    reset_for_tests()
