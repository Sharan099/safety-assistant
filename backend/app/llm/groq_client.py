"""Groq LLM client — llama-3.3-70b-versatile (unchanged from original)."""

import os
import time

from loguru import logger

from config import GROQ_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, SYSTEM_PROMPT

from backend.app.gateway.errors import GENERATION_UNAVAILABLE_MESSAGE, is_raw_provider_error
from backend.app.gateway.fallback_safeguards import (
    is_fast_groq_model,
    sanitize_fast_model_output,
)


class GroqLLM:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")

        try:
            from groq import Groq
        except ImportError as exc:
            raise EnvironmentError(
                "Groq SDK is not installed. Run: pip install -r backend/requirements.txt"
            ) from exc

        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL

    def generate(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        try:
            kwargs: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            }
            if is_fast_groq_model(self.model):
                from backend.app.gateway import config as gw_cfg
                kwargs["frequency_penalty"] = gw_cfg.FAST_FALLBACK_FREQUENCY_PENALTY
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.error(f"LLM failed: {exc}")
            return {
                "answer": GENERATION_UNAVAILABLE_MESSAGE,
                "latency_ms": latency_ms,
                "model": self.model,
                "error": "provider_failed",
                "generation_failed": True,
            }
        answer = (response.choices[0].message.content or "").strip()
        if is_fast_groq_model(self.model):
            answer, _ = sanitize_fast_model_output(
                answer, provider_key="groq", model=self.model
            )
        if is_raw_provider_error(answer):
            answer = GENERATION_UNAVAILABLE_MESSAGE
        usage = getattr(response, "usage", None)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"LLM completed in {latency_ms}ms")
        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "model": self.model,
        }
