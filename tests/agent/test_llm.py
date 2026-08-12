"""FreeLLMAPIProvider is tested against a mocked HTTP transport (httpx.MockTransport),
not the live proxy: the proxy running on this dev machine (docs/ADR/0004)
requires a credential we don't have (confirmed: it returns "Invalid API key"
for unauthenticated /v1/models). That 401 is itself evidence the
TRD.md Section 30 failure path matters — see test_unavailable_on_http_error.
"""

import httpx
import pytest

from packages.agent.llm import (
    FreeLLMAPIProvider,
    LLMMessage,
    LLMUnavailableError,
    MockProvider,
    get_provider,
)
from packages.domain.db import Settings


def test_mock_provider_returns_canned_response() -> None:
    provider = MockProvider("hello")
    result = provider.complete([LLMMessage(role="user", content="hi")])
    assert result.content == "hello"
    assert result.provider == "mock"


def test_freellmapi_requires_model() -> None:
    with pytest.raises(ValueError, match="LLM_MODEL"):
        FreeLLMAPIProvider(base_url="http://localhost:3001/v1", api_key="k", model="")


def test_freellmapi_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "model": "some-free-model",
                "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
        )

    provider = FreeLLMAPIProvider(
        base_url="http://localhost:3001/v1",
        api_key="test-key",
        model="some-free-model",
        transport=httpx.MockTransport(handler),
    )
    result = provider.complete([LLMMessage(role="user", content="what is the answer?")])
    assert result.content == "42"
    assert result.provider == "freellmapi"
    assert result.finish_reason == "stop"


def test_freellmapi_unavailable_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    provider = FreeLLMAPIProvider(
        base_url="http://localhost:3001/v1",
        api_key="wrong-key",
        model="some-free-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMUnavailableError):
        provider.complete([LLMMessage(role="user", content="hi")])


def test_freellmapi_unavailable_on_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = FreeLLMAPIProvider(
        base_url="http://localhost:3001/v1", api_key="k", model="m", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMUnavailableError):
        provider.complete([LLMMessage(role="user", content="hi")])


def test_get_provider_mock() -> None:
    settings = Settings(llm_provider="mock")
    provider = get_provider(settings)
    assert isinstance(provider, MockProvider)


def test_get_provider_unknown_raises() -> None:
    settings = Settings(llm_provider="not-a-real-provider")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_provider(settings)
