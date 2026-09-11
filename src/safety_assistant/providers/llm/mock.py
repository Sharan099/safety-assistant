"""Deterministic TEST-ONLY provider. Returns canned text, or when a schema is
requested, validates a canned JSON payload against it."""

from __future__ import annotations

import json

from pydantic import BaseModel

from safety_assistant.providers.llm.base import LLMMessage, LLMResponse, LLMSchemaError


class MockLLMProvider:
    name = "mock"
    model = "mock"

    def __init__(self, response: str = "This is a mock response.", json_payload: dict[str, object] | None = None):
        self._response = response
        self._json = json_payload

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        parsed = None
        content = self._response
        if schema is not None:
            if self._json is None:
                raise LLMSchemaError("mock has no JSON payload configured for schema output")
            parsed = schema.model_validate(self._json)
            content = json.dumps(self._json)
        return LLMResponse(content=content, model=self.model, provider=self.name, finish_reason="stop", parsed=parsed)
