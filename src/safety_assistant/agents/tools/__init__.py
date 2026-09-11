"""Narrow, typed tools the agent may call. No shell, no SQL strings, no network.
Every call is counted against the budget; every call is session-bound and
scope-filtered (authorization travels inside `ScopeFilter`)."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.agents.state import AgentState, BudgetExceeded
from safety_assistant.ingestion.diff import VersionDiff, diff_versions, find_version, versions_of
from safety_assistant.persistence.models import Regulation, RegulationVersion, Section
from safety_assistant.retrieval import RetrievalService, ScopeFilter
from safety_assistant.retrieval.service import RetrievalResult


def _spend(state: AgentState, kind: str) -> None:
    budget = state["budget"]
    budget.check_deadline(state["started"])
    state["tool_calls"] = state.get("tool_calls", 0) + 1
    if state["tool_calls"] > budget.max_tool_calls:
        raise BudgetExceeded(f"tool-call budget {budget.max_tool_calls} exceeded at {kind}")
    if kind == "retrieve":
        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
        if state["retrieval_attempts"] > budget.max_retrieval_attempts:
            raise BudgetExceeded(f"retrieval budget {budget.max_retrieval_attempts} exceeded")


@dataclass
class Tools:
    session: Session
    retrieval: RetrievalService

    def search_regulations(
        self, state: AgentState, query: str, scope: ScopeFilter, *, k: int | None = None
    ) -> RetrievalResult:
        _spend(state, "retrieve")
        return self.retrieval.search(self.session, query, scope=scope, k=k or state.get("k"), today=state.get("today"))

    def get_regulation_versions(self, state: AgentState, regulation_key: str) -> list[RegulationVersion]:
        _spend(state, "versions")
        return versions_of(self.session, regulation_key)

    def compare_versions(
        self, state: AgentState, regulation_key: str, from_label: str | None = None, to_label: str | None = None
    ) -> VersionDiff | None:
        """Diff two versions; defaults to the two most recent verified texts."""
        _spend(state, "diff")
        versions = versions_of(self.session, regulation_key)
        if len(versions) < 2 and not (from_label and to_label):
            return None
        a = find_version(self.session, regulation_key, from_label) if from_label else versions[-2]
        b = find_version(self.session, regulation_key, to_label) if to_label else versions[-1]
        if a is None or b is None or a.id == b.id:
            return None
        return diff_versions(self.session, a, b)

    def get_section(self, state: AgentState, version_id: uuid.UUID, path: str) -> Section | None:
        _spend(state, "section")
        return self.session.scalar(select(Section).where(Section.version_id == version_id, Section.path == path))

    def regulation_exists(self, regulation_key: str) -> bool:
        return self.session.scalar(select(Regulation.id).where(Regulation.regulation_key == regulation_key)) is not None

    def effective_version_on(self, regulation_key: str, day: datetime.date) -> RegulationVersion | None:
        for v in reversed(versions_of(self.session, regulation_key)):
            if (v.valid_from is None or v.valid_from <= day) and (v.valid_to is None or v.valid_to > day):
                return v
        return None
