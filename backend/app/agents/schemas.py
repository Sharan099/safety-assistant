"""Pydantic schemas for crew agent outputs and shared state."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class MetricRow(BaseModel):
    name: str
    target: str
    actual: str
    unit: str = ""
    status: Literal["pass", "fail", "marginal", "unknown"] = "unknown"


class SimulationOutput(BaseModel):
    status: Literal["ok", "insufficient_data"] = "ok"
    metrics: list[MetricRow] = Field(default_factory=list)


class RegulationCheck(BaseModel):
    metric: str
    body: str
    regulation_id: str
    limit: str
    result: Literal["pass", "fail", "not_found", "unknown"]
    source: str = ""


class RegulationOutput(BaseModel):
    status: Literal["ok", "insufficient_data"] = "ok"
    checks: list[RegulationCheck] = Field(default_factory=list)


class RootCauseItem(BaseModel):
    metric: str
    hypothesis: str
    evidence: str
    confidence: Literal["high", "medium", "low"] = "medium"


class RootCauseOutput(BaseModel):
    status: Literal["ok", "insufficient_data"] = "ok"
    root_causes: list[RootCauseItem] = Field(default_factory=list)


class SimilarCase(BaseModel):
    program: str
    failure_mode: str
    fix_applied: str
    outcome: str


class KnowledgeOutput(BaseModel):
    status: Literal["ok", "insufficient_data"] = "ok"
    similar_cases: list[SimilarCase] = Field(default_factory=list)


class CountermeasureItem(BaseModel):
    action: str
    targets_metric: str
    expected_effect: str
    effort: Literal["low", "medium", "high"] = "medium"
    rank: int = 1


class CountermeasureOutput(BaseModel):
    status: Literal["ok", "insufficient_data"] = "ok"
    countermeasures: list[CountermeasureItem] = Field(default_factory=list)


class JiraTicket(BaseModel):
    title: str
    description: str
    component: str
    priority: Literal["P1", "P2", "P3", "P4"] = "P2"


class ProgramManagerOutput(BaseModel):
    status: Literal["ok", "insufficient_data"] = "ok"
    report_markdown: str = ""
    jira_tickets: list[JiraTicket] = Field(default_factory=list)


class CrewReport(BaseModel):
    summary: str = ""
    failing_metrics: list[str] = Field(default_factory=list)
    root_cause: list[str] = Field(default_factory=list)
    similar_cases: list[str] = Field(default_factory=list)
    countermeasures: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class CrewState(TypedDict, total=False):
    crash_input: str
    crash_summary: str
    vehicle: str
    user_id: str | None
    session_id: str | None
    agent_queries: dict[str, str]
    agent_outputs: dict[str, Any]
    citations: list[dict[str, Any]]
    timing: dict[str, Any]
    report: dict[str, Any]


AGENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "simulation": SimulationOutput,
    "regulation": RegulationOutput,
    "root_cause": RootCauseOutput,
    "knowledge": KnowledgeOutput,
    "countermeasure": CountermeasureOutput,
    "program_manager": ProgramManagerOutput,
}
