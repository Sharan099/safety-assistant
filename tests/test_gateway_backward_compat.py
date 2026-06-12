"""Backward-compatibility & contract tests.

Guarantees:
  - GatewayResult.as_legacy_dict() is a SUPERSET of GroqLLM.generate() keys, so
    the gateway is a safe drop-in behind RAGWorkflow._get_llm().
  - With ENABLE_GATEWAY disabled, services.get_gateway() returns None and the
    workflow keeps using GroqLLM.
  - gateway.generate(prompt) returns the legacy dict shape end-to-end.
"""

import pytest

from backend.app.gateway.gateway import LLMGateway
from backend.app.gateway.providers.base import Provider
from backend.app.gateway.types import GatewayResult, ProviderResponse

# Keys the original GroqLLM.generate() contract guarantees.
LEGACY_KEYS = {"answer", "latency_ms", "prompt_tokens", "completion_tokens", "model"}


class OkProvider(Provider):
    def __init__(self, key="groq"):
        self.key = key

    @property
    def provider_name(self):
        return "groq"

    @property
    def model(self):
        return "groq-model"

    def available(self):
        return True

    def chat(self, messages, *, temperature, max_tokens, timeout_s):
        return ProviderResponse(
            answer="ok", model=self.model, provider="groq",
            latency_ms=1.0, prompt_tokens=3, completion_tokens=2,
        )


def test_gateway_result_is_legacy_superset():
    res = GatewayResult(
        answer="x", model="m", provider="groq", tier=1, latency_ms=5.0,
        prompt_tokens=1, completion_tokens=1,
    )
    d = res.as_legacy_dict()
    assert LEGACY_KEYS.issubset(d.keys())
    # additive fields present but optional for legacy consumers
    assert "provider" in d and "tier" in d and "cache_hit" in d


def test_generate_returns_legacy_dict():
    gw = LLMGateway(providers={"groq": OkProvider()}, cache=None)
    out = gw.generate("What is UN R14?")
    assert LEGACY_KEYS.issubset(out.keys())
    assert out["answer"] == "ok"
    assert out["model"] == "groq-model"


def test_get_gateway_none_when_disabled(monkeypatch):
    import backend.app.core.settings as settings
    import backend.app.core.services as services

    monkeypatch.setattr(settings, "ENABLE_GATEWAY", False)
    services._gateway = None  # reset singleton
    assert services.get_gateway() is None
