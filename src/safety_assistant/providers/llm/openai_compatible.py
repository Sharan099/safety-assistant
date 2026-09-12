"""OpenAI-compatible chat-completions client (works for OpenAI, Azure-style
proxies, vLLM, Ollama's /v1, FreeLLMAPI). Sync httpx; CPU cost is nil, the
API layer offloads to a thread like every provider call."""

from __future__ import annotations

import json
import random
import time

import httpx
from pydantic import BaseModel, ValidationError

from safety_assistant.providers.llm.base import (
    LLMBadRequest,
    LLMError,
    LLMMessage,
    LLMRateLimited,
    LLMResponse,
    LLMSchemaError,
    LLMUnavailable,
)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 0.5


def _classify(exc: httpx.HTTPStatusError) -> LLMError:
    status = exc.response.status_code
    if status == 429:
        return LLMRateLimited(f"rate limited ({status})")
    if status >= 500:
        return LLMUnavailable(f"provider error {status}")
    return LLMBadRequest(f"provider rejected request ({status}): {exc.response.text[:200]}")


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM_MODEL must be set — no safe default model exists")
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        last: LLMError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.post(f"{self._base_url}/chat/completions", json=payload, headers=self._headers)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
            except httpx.HTTPStatusError as exc:
                last = _classify(exc)
            except httpx.HTTPError as exc:
                last = LLMUnavailable(f"request failed: {exc}")
            if not last.retryable or attempt == _MAX_ATTEMPTS - 1:
                raise last
            time.sleep(_BACKOFF_BASE_S * (2**attempt) + random.uniform(0, 0.2))  # noqa: S311 — jitter, not crypto
        raise LLMUnavailable("exhausted retries")  # pragma: no cover — loop always returns or raises

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload)
        try:
            choice = data["choices"][0]  # type: ignore[index]
            content: str = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"unexpected response shape: {str(data)[:200]}") from exc

        parsed = None
        if schema is not None:
            try:
                parsed = schema.model_validate(json.loads(_strip_fences(content)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise LLMSchemaError(f"output does not satisfy {schema.__name__}: {exc}") from exc
        return LLMResponse(
            content=content,
            model=str(data.get("model", self.model)),
            provider=self.name,
            finish_reason=choice.get("finish_reason"),
            usage=_int_usage(data.get("usage")),
            parsed=parsed,
        )


def _int_usage(raw: object) -> dict[str, int] | None:
    """Keep only integer counters; gateways nest per-provider detail dicts under usage."""
    if not isinstance(raw, dict):
        return None
    return {k: v for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)} or None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    return t.strip()
