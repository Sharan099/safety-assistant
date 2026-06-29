"""Groq chat-completions provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from backend.app.gateway.error_policy import ErrorKind, classify_error_text
from backend.app.gateway.http_timeout import provider_timeout
from backend.app.gateway.providers.base import ProviderError


class GroqProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError("GROQ_API_KEY not set", kind=ErrorKind.FATAL.value)
        http_timeout = provider_timeout(read=timeout or 60.0)
        try:
            with httpx.Client(timeout=http_timeout, trust_env=True) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
            if resp.status_code >= 400:
                kind = classify_error_text(resp.text)
                raise ProviderError(resp.text, kind=kind.value, status_code=resp.status_code)
            data = resp.json()
            usage = data.get("usage") or {}
            return {
                "text": data["choices"][0]["message"]["content"],
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except ProviderError:
            raise
        except httpx.ConnectTimeout as exc:
            raise ProviderError(str(exc), kind=ErrorKind.CONNECTION.value) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(str(exc), kind=ErrorKind.CONNECTION.value) from exc
        except Exception as exc:
            raise ProviderError(str(exc), kind=classify_error_text(str(exc)).value) from exc
