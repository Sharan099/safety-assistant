"""Chat / query input validation."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field, field_validator

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_Q = 4000
_MIN_Q = 3


class ValidationError(ValueError):
    """User-facing validation failure."""


def sanitize_question(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw or "")
    text = _CTRL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_question(raw: str) -> str:
    q = sanitize_question(raw)
    if len(q) < _MIN_Q:
        raise ValidationError("Question is too short — ask a concrete regulation question.")
    if len(q) > _MAX_Q:
        raise ValidationError(f"Question exceeds {_MAX_Q} characters.")
    # Block obvious prompt-injection dumps without rejecting normal engineering questions.
    if q.count("```") >= 4 or len(re.findall(r"(?i)ignore (all|previous) instructions", q)) > 0:
        raise ValidationError("Question looks like a prompt-injection attempt.")
    return q


class ChatInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_Q)
    regulation_id: str | None = Field(default=None, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
    conversation_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _q(cls, v: str) -> str:
        return validate_question(v)

    @field_validator("regulation_id")
    @classmethod
    def _reg(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        cleaned = re.sub(r"[^A-Za-z0-9._\-]", "", str(v).strip())
        return cleaned or None

    @field_validator("conversation_id")
    @classmethod
    def _cid(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        cleaned = re.sub(r"[^A-Za-z0-9\-]", "", str(v).strip())
        return cleaned or None
