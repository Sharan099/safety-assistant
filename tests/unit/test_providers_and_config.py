import json

import httpx
import pytest
from pydantic import BaseModel

from safety_assistant.config.settings import Settings
from safety_assistant.providers.embeddings import ProviderConfigurationError, build_embedding_provider
from safety_assistant.providers.llm import (
    LLMBadRequest,
    LLMMessage,
    LLMRateLimited,
    LLMResponse,
    LLMSchemaError,
    LLMUnavailable,
)
from safety_assistant.providers.llm.openai_compatible import OpenAICompatibleProvider

PROD: dict[str, str] = {
    "app_env": "production",
    "auth_mode": "api_key",
    "database_url": "postgresql+psycopg://u:strong@db/x",
    # explicit real providers: the test conftest exports the fake ones via env
    "embedding_provider": "fastembed",
    "llm_provider": "none",
    "session_secret": "production-session-secret-of-adequate-length",
    "dev_login_enabled": "false",
}


def test_production_accepts_a_real_configuration() -> None:
    Settings(**PROD)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"embedding_provider": "hashing"},
        {"llm_provider": "mock"},
        {"auth_mode": "none"},
        {"database_url": "postgresql+psycopg://u:change_me@db/x"},
        {"dev_login_enabled": "true"},
        {"session_secret": "short"},
    ],
)
def test_production_refuses_fakes_and_dev_defaults(bad: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="refusing to start in production"):
        Settings(**{**PROD, **bad})  # type: ignore[arg-type]


def test_hashing_provider_only_in_test_profile() -> None:
    with pytest.raises(ProviderConfigurationError):
        build_embedding_provider(Settings(app_env="development", embedding_provider="hashing"))
    p = build_embedding_provider(Settings(app_env="test", embedding_provider="hashing"))
    assert p.dimensions == 384 and p.embed_query("tibia index") == p.embed_query("tibia index")


class Out(BaseModel):
    answer: str


def _provider(handler):  # type: ignore[no-untyped-def]
    return OpenAICompatibleProvider(
        base_url="http://llm.test/v1", api_key="k", model="m", transport=httpx.MockTransport(handler)
    )


@pytest.mark.parametrize("status,exc", [(429, LLMRateLimited), (503, LLMUnavailable), (400, LLMBadRequest)])
def test_http_errors_are_classified(status: int, exc: type[Exception], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("safety_assistant.providers.llm.openai_compatible.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": "x"})

    with pytest.raises(exc):
        _provider(handler).generate([LLMMessage(role="user", content="hi")])
    assert calls["n"] == (3 if status != 400 else 1)  # retryable errors are retried, 4xx is not


def test_schema_output_is_validated() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        content = '```json\n{"answer": "42"}\n```'
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "model": "m"}
        )

    resp = _provider(ok).generate([LLMMessage(role="user", content="q")], schema=Out)
    assert isinstance(resp.parsed, Out) and resp.parsed.answer == "42"

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    with pytest.raises(LLMSchemaError):
        _provider(bad).generate([LLMMessage(role="user", content="q")], schema=Out)


def test_gateway_usage_with_nested_details_is_flattened_to_int_counters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
            "cost_details": {"upstream_inference_cost": 0},
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": usage})

    resp = _provider(handler).generate([LLMMessage(role="user", content="q")])
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_model_is_mandatory() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(base_url="http://x", api_key="", model="")


def test_call_budget_bounds_retries_to_one_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow gateway costs at most `timeout` seconds in total, not attempts × timeout."""
    clock = {"t": 0.0}
    monkeypatch.setattr("safety_assistant.providers.llm.openai_compatible.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("safety_assistant.providers.llm.openai_compatible.time.sleep", lambda s: None)
    calls = {"n": 0}

    def slow(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        clock["t"] += 20.0  # each attempt burns 20 s of a 30 s budget
        raise httpx.ReadTimeout("slow", request=request)

    p = OpenAICompatibleProvider(
        base_url="http://llm.test/v1", api_key="k", model="m", timeout=30.0, transport=httpx.MockTransport(slow)
    )
    with pytest.raises(LLMUnavailable):
        p.generate([LLMMessage(role="user", content="hi")])
    assert calls["n"] == 2  # the second attempt runs with the 10 s left; a third would exceed the budget


def test_extract_json_tolerates_prose_fences_and_trailing_commas() -> None:
    from safety_assistant.providers.llm.openai_compatible import extract_json

    assert extract_json('```json\n{"answer": "42"}\n```') == {"answer": "42"}
    assert extract_json('Here is the result:\n{"answer": "42", "claims": [],}\nThanks') == {
        "answer": "42",
        "claims": [],
    }
    with pytest.raises(json.JSONDecodeError):
        extract_json("no object here")


def test_fallback_model_answers_when_primary_fails() -> None:
    from safety_assistant.providers.llm.factory import FallbackLLM

    class Down:
        name, model = "mock", "small"

        def generate(self, messages, *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
            raise LLMUnavailable("timeout")

    class Up:
        name, model = "mock", "big"

        def generate(self, messages, *, schema=None, temperature=0.0, max_tokens=1024):  # type: ignore[no-untyped-def]
            return LLMResponse(content="ok", model=self.model, provider=self.name)

    r = FallbackLLM(Down(), Up()).generate([LLMMessage(role="user", content="hi")])
    assert r.model == "big"  # the answering model is recorded, not the configured primary
