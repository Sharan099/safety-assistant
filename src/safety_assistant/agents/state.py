"""Typed agent state and hard budgets (ENGINEERING.md §9)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypedDict

from safety_assistant.generation.grounding import GateDecision
from safety_assistant.generation.schemas import GroundedDraft, ValidationReport
from safety_assistant.retrieval.context import Evidence
from safety_assistant.retrieval.filters import ScopeFilter
from safety_assistant.retrieval.service import RetrievalResult


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_retrieval_attempts: int = 3  # standard + one rewrite, or one per compared regulation (≤2) + rewrite
    max_llm_calls: int = 1
    max_tool_calls: int = 8
    timeout_seconds: float = 45.0
    context_tokens: int = 6000

    def check_deadline(self, started: float) -> None:
        if time.perf_counter() - started > self.timeout_seconds:
            raise BudgetExceeded(f"timeout budget {self.timeout_seconds}s exceeded")


class AgentState(TypedDict, total=False):
    # input
    query: str
    base_scope: ScopeFilter
    k: int | None
    today: Any  # datetime.date | None
    trace_id: str
    started: float
    budget: Budget
    # plan
    intent: str
    scope: ScopeFilter
    regulation_keys: list[str]
    rewrites: list[str]
    route: str
    # work
    retrieval: RetrievalResult | None
    sub_results: list[RetrievalResult]
    evidence: list[Evidence]
    extra_context: str | None  # e.g. change-analysis diff, rendered as data for the model
    conversation_context: str | None  # earlier turns, wording context only — never evidence
    # the engineer's stated project (vehicle category, mass, market): data for the model, never evidence
    project_context: str | None
    decision: GateDecision | None
    draft: GroundedDraft | None
    validation: ValidationReport | None
    # accounting
    retrieval_attempts: int
    llm_calls: int
    tool_calls: int
    timings: dict[str, float]
    warnings: list[str]
    versions: dict[str, Any]
    tokens: dict[str, int] | None
    # output
    mode: str
    abstain_reason: str | None
    message: str | None
