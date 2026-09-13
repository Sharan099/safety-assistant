"""Structured generation contract (CLAUDE.md §10). The model fills `GroundedDraft`;
the service wraps it into `AnswerResponse` after programmatic validation."""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, Field

AbstainReason = Literal[
    "no_evidence",
    "requested_regulation_not_in_evidence",
    "no_version_valid_on_date",
    "weak_evidence",
    "ambiguous_query",
    "generation_unavailable",
    "validation_failed",
]


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)
    # REQUIREMENT = quoted/paraphrased regulation text; INTERPRETATION = the model's reading of it.
    kind: Literal["REQUIREMENT", "INTERPRETATION"] = "REQUIREMENT"


class GroundedDraft(BaseModel):
    """What the LLM must return."""

    answer: str
    claims: list[Claim]
    warnings: list[str] = []
    insufficient_evidence: bool = False


class CitationView(BaseModel):
    evidence_id: str
    label: str
    regulation_key: str
    version_label: str
    section_path: str
    page_start: int | None
    page_end: int | None
    source_sha256: str
    source_uri: str | None
    valid_from: datetime.date | None
    valid_to: datetime.date | None
    version_status: str


class ClaimValidation(BaseModel):
    claim_index: int
    status: Literal["SUPPORTED", "UNSUPPORTED_EVIDENCE_ID", "NUMERIC_MISMATCH", "NO_EVIDENCE"]
    detail: str | None = None


class ValidationReport(BaseModel):
    ok: bool
    claims: list[ClaimValidation]
    unknown_evidence_ids: list[str]
    dropped_claims: int


class AnswerScope(BaseModel):
    regulation_keys: list[str]
    as_of: datetime.date | None
    intent: str
    historical: bool


class AnswerResponse(BaseModel):
    trace_id: str
    query: str
    mode: Literal["GENERATED", "EVIDENCE_ONLY", "ABSTAINED"]
    answer: str | None
    claims: list[Claim]
    citations: list[CitationView]
    warnings: list[str]
    abstain_reason: AbstainReason | None = None
    scope: AnswerScope
    evidence: list[dict[str, object]]  # full Evidence records for the UI
    validation: ValidationReport | None
    versions: dict[str, object]
    latency_ms: dict[str, float]
    tokens: dict[str, int] | None = None  # provider usage counters when the LLM reports them
