"""API request/response models.

Distinct from `packages.domain` (ORM) and reuses `packages.analysis.models`
(already Pydantic) directly for analysis results — no need to redeclare
those. See UI_UX_DESIGN_BRIEF.md for what the eventual UI reads from these.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pydantic import BaseModel


class RunSummary(BaseModel):
    id: uuid.UUID
    run_id: str
    vehicle_name: str
    model_version: str
    solver: str | None
    solver_version: str | None
    impact_type: str | None
    impact_speed: float | None
    quality_status: str
    created_at: datetime.datetime


class RunDetail(RunSummary):
    dummy_version: str | None
    seat_configuration: dict[str, Any] | None
    restraint_configuration: dict[str, Any] | None
    result_processing_version: str | None
    metadata: dict[str, Any] | None


class CreateInvestigationRequest(BaseModel):
    run_a_id: str
    run_b_id: str
    question: str
    primary_metric: str | None = None
    title: str | None = None


class EngineerReviewRequest(BaseModel):
    # ACCEPT | REJECT | MODIFY | REQUEST_SIGNAL | REQUEST_SOURCE |
    # REQUEST_CONTROLLED_COMPARISON | MARK_INCONCLUSIVE — APP_FLOW.md Section 16
    decision: str
    comment: str | None = None


class EvidenceSummary(BaseModel):
    id: uuid.UUID
    evidence_type: str
    source_type: str
    content: str | None
    created_at: datetime.datetime


class HypothesisSummary(BaseModel):
    id: uuid.UUID
    title: str
    description: str | None
    status: str
    confidence_basis: str | None
    created_at: datetime.datetime


class AgentRunResult(BaseModel):
    investigation_id: uuid.UUID
    state: str
    blocked_reason: str | None
    hypotheses: list[HypothesisSummary]
    evidence_count: int


class InvestigationSummary(BaseModel):
    id: uuid.UUID
    title: str
    question: str
    primary_metric: str | None = None
    state: str
    decision: str | None
    run_a_id: str
    run_b_id: str
    created_at: datetime.datetime


class CopilotMessageRequest(BaseModel):
    message: str


class CopilotToolActivity(BaseModel):
    tool_name: str
    status: str
    result_summary: str


class CopilotMessageSummary(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    evidence_refs: list[str]
    suggested_actions: list[str]
    unknowns: list[str]
    llm_degraded: bool
    tool_activity: list[CopilotToolActivity] = []
    created_at: datetime.datetime
