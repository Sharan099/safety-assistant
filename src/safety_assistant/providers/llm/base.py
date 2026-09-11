"""LLM provider contract with classified errors.

Rules (CLAUDE.md §2.3, §22): never silently substitute a different model or
fake output; classify failures so callers can retry a 429 but not a 400;
structured output is validated against the requested schema in code.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    parsed: Any | None = None  # populated when a schema was requested


class LLMError(Exception):
    """Base for every provider failure. `retryable` drives the bounded retry policy."""

    retryable = False


class LLMRateLimited(LLMError):
    retryable = True


class LLMUnavailable(LLMError):
    """Connection failure, timeout, or 5xx."""

    retryable = True


class LLMBadRequest(LLMError):
    """4xx other than 429 — retrying identical input cannot help."""


class LLMSchemaError(LLMError):
    """The model returned text that does not satisfy the requested schema."""


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
