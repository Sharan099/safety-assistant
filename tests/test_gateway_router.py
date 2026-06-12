"""Router failover / retry tests with mock providers (no network, no real sleep)."""

import pytest

from backend.app.gateway import config as cfg
from backend.app.gateway.providers.base import Provider, ProviderError
from backend.app.gateway.router import Router
from backend.app.gateway.types import ProviderResponse


class FakeProvider(Provider):
    def __init__(self, key, *, behaviour):
        self.key = key
        self._behaviour = behaviour  # "ok" | "retryable" | "fatal"
        self.calls = 0

    @property
    def provider_name(self):
        return self.key

    @property
    def model(self):
        return f"{self.key}-model"

    def available(self):
        return True

    def chat(self, messages, *, temperature, max_tokens, timeout_s):
        self.calls += 1
        if self._behaviour == "ok":
            return ProviderResponse(
                answer=f"answer from {self.key}",
                model=self.model,
                provider=self.key,
                latency_ms=1.0,
                prompt_tokens=10,
                completion_tokens=5,
            )
        if self._behaviour == "fatal":
            raise ProviderError("bad request", retryable=False)
        raise ProviderError("timeout", retryable=True)


@pytest.fixture(autouse=True)
def _fast_router(monkeypatch):
    monkeypatch.setattr(cfg, "MAX_RETRIES", 1)
    monkeypatch.setattr(cfg, "BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(cfg, "BACKOFF_MAX_S", 0.0)


def _msgs():
    return [{"role": "user", "content": "hi"}]


def test_primary_success_no_fallback():
    providers = {"groq": FakeProvider("groq", behaviour="ok")}
    outcome = Router(providers).route("groq", _msgs(), temperature=0, max_tokens=10)
    assert outcome.provider_key == "groq"
    assert outcome.fallback_used is False
    assert outcome.response.answer == "answer from groq"


def test_failover_to_next_provider():
    providers = {
        "groq": FakeProvider("groq", behaviour="retryable"),
        "anthropic_haiku": FakeProvider("anthropic_haiku", behaviour="ok"),
        "anthropic_sonnet": FakeProvider("anthropic_sonnet", behaviour="ok"),
    }
    outcome = Router(providers).route("groq", _msgs(), temperature=0, max_tokens=10)
    assert outcome.provider_key == "anthropic_haiku"
    assert outcome.fallback_used is True
    # primary retried (MAX_RETRIES+1) times before failing over
    assert providers["groq"].calls == cfg.MAX_RETRIES + 1


def test_fatal_error_does_not_retry_but_fails_over():
    providers = {
        "groq": FakeProvider("groq", behaviour="fatal"),
        "anthropic_haiku": FakeProvider("anthropic_haiku", behaviour="ok"),
        "anthropic_sonnet": FakeProvider("anthropic_sonnet", behaviour="ok"),
    }
    outcome = Router(providers).route("groq", _msgs(), temperature=0, max_tokens=10)
    assert providers["groq"].calls == 1  # no retry on fatal
    assert outcome.provider_key == "anthropic_haiku"


def test_all_fail_raises():
    providers = {
        "anthropic_sonnet": FakeProvider("anthropic_sonnet", behaviour="retryable"),
        "anthropic_haiku": FakeProvider("anthropic_haiku", behaviour="retryable"),
    }
    with pytest.raises(ProviderError):
        Router(providers).route(
            "anthropic_sonnet", _msgs(), temperature=0, max_tokens=10
        )
