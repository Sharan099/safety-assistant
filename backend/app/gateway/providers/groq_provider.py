"""Groq OpenAI-compatible chat completions provider."""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from backend.app.gateway.error_policy import ErrorKind, classify_error_text
from backend.app.gateway.http_timeout import provider_timeout
from backend.app.gateway.providers.base import ProviderError

_GROQ_DIRECT = "https://api.groq.com/openai/v1"


def _resolve_groq_api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    return (os.getenv('GROQ_API_KEY') or os.getenv('Groq_API_KEY') or '').strip()


def _openai_base_url() -> str:
    gateway = (os.getenv("PORTKEY_GATEWAY_URL") or "").strip().rstrip("/")
    return gateway if gateway else _GROQ_DIRECT


def _request_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "AutoSafety-RAG-Gateway/1.0",
    }
    if (os.getenv("PORTKEY_GATEWAY_URL") or "").strip():
        headers["x-portkey-provider"] = (
            os.getenv("PORTKEY_PROVIDER") or "groq"
        ).strip() or "groq"
    return headers


class GroqProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = _resolve_groq_api_key(api_key)

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
            raise ProviderError('GROQ_API_KEY not set', kind=ErrorKind.FATAL.value)
        http_timeout = provider_timeout(read=timeout or 60.0)
        try:
            with httpx.Client(timeout=http_timeout, trust_env=True) as client:
                resp = client.post(
                    f"{_openai_base_url()}/chat/completions",
                    headers=_request_headers(self.api_key),
                    json={
                        'model': model_id,
                        'messages': messages,
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                    },
                )
            if resp.status_code >= 400:
                kind = classify_error_text(resp.text)
                raise ProviderError(resp.text, kind=kind.value, status_code=resp.status_code)
            data = resp.json()
            usage = data.get('usage') or {}
            choice = (data.get('choices') or [{}])[0]
            message = choice.get('message') or {}
            text = message.get('content') or ''
            if not text and isinstance(message.get('reasoning'), str):
                text = message['reasoning']
            logger.info(
                'groq usage model={} input={} output={} finish={}',
                model_id,
                int(usage.get('prompt_tokens', 0)),
                int(usage.get('completion_tokens', 0)),
                choice.get('finish_reason'),
            )
            return {
                'text': text,
                'prompt_tokens': int(usage.get('prompt_tokens', 0)),
                'completion_tokens': int(usage.get('completion_tokens', 0)),
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'finish_reason': str(choice.get('finish_reason') or ''),
            }
        except ProviderError:
            raise
        except httpx.ConnectTimeout as exc:
            raise ProviderError(str(exc), kind=ErrorKind.CONNECTION.value) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(str(exc), kind=ErrorKind.CONNECTION.value) from exc
        except Exception as exc:
            raise ProviderError(str(exc), kind=classify_error_text(str(exc)).value) from exc

    def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            with httpx.Client(timeout=15.0, trust_env=True) as client:
                resp = client.get(
                    f"{_openai_base_url()}/models",
                    headers=_request_headers(self.api_key),
                )
            if resp.status_code >= 400:
                return []
            data = resp.json()
            return [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        except Exception:
            return []
