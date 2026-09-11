"""Grounded generation contract. `AnswerService` lives in `generation.service`
and is imported lazily to avoid the generation ↔ agents import cycle."""

from __future__ import annotations

from typing import Any

from safety_assistant.generation.citations import citation_views, validate_draft
from safety_assistant.generation.grounding import GateDecision, evaluate_gate, rewrite_query
from safety_assistant.generation.schemas import AnswerResponse, Claim, GroundedDraft, ValidationReport

__all__ = [
    "AnswerResponse",
    "AnswerService",
    "Claim",
    "GateDecision",
    "GroundedDraft",
    "ValidationReport",
    "citation_views",
    "evaluate_gate",
    "rewrite_query",
    "validate_draft",
]


def __getattr__(name: str) -> Any:
    if name == "AnswerService":
        from safety_assistant.generation.service import AnswerService

        return AnswerService
    raise AttributeError(name)
