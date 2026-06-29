"""Anthropic messages API provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from backend.app.gateway.error_policy import ErrorKind, classify_error_text
from backend.app.gateway.http_timeout import provider_timeout
from backend.app.gateway.providers.base import ProviderError


class AnthropicProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

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
            raise ProviderError("ANTHROPIC_API_KEY not set", kind=ErrorKind.FATAL.value)
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_parts = [m["content"] for m in messages if m.get("role") == "user"]
        system = "\n\n".join(system_parts) if system_parts else ""
        user = "\n\n".join(user_parts)
        http_timeout = provider_timeout(read=timeout or 60.0)
        try:
            with httpx.Client(timeout=http_timeout, trust_env=True) as client:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                    },
                )
            if resp.status_code >= 400:
                kind = classify_error_text(resp.text)
                raise ProviderError(resp.text, kind=kind.value, status_code=resp.status_code)
            data = resp.json()
            usage = data.get("usage") or {}
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            return {
                "text": text,
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
            }
        except ProviderError:
            raise
        except httpx.ConnectTimeout as exc:
            raise ProviderError(str(exc), kind=ErrorKind.CONNECTION.value) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(str(exc), kind=ErrorKind.CONNECTION.value) from exc
        except Exception as exc:
            raise ProviderError(str(exc), kind=classify_error_text(str(exc)).value) from exc
