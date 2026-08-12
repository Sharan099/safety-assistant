"""Investigation entities — BACKEND_SCHEMA.md §4, §29-45.

Investigation, InvestigationRun, InvestigationMetric, QualityGateResult,
ComparabilityAssessment, ConfigurationDiff, SignalAnalysis, AnalysisEvent,
Hypothesis, Evidence, Claim, HypothesisEvidenceLink, EvidenceContradiction,
Finding, RecommendedAction, ControlledComparisonRequest, EngineerReview.

Core chain (BACKEND_SCHEMA.md §1): Run -> Analysis -> Evidence -> Claim ->
Hypothesis -> Finding -> Decision. Never collapse these into one AI response.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import ARRAY, Double, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.domain.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class Investigation(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "investigations"

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    title: Mapped[str] = mapped_column(Text)
    question: Mapped[str] = mapped_column(Text)
    primary_metric_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # DRAFT | RUNS_SELECTED | IDENTITY_CHECK | QUALITY_CHECK |
    # COMPARABILITY_CHECK | GLOBAL_RESPONSE | CONFIGURATION_ANALYSIS |
    # SIGNAL_ANALYSIS | MECHANISM_REVIEW | HYPOTHESIS_ANALYSIS |
    # EVIDENCE_REVIEW | ENGINEER_REVIEW | DECISION | FOLLOW_UP | BLOCKED | CLOSED
    state: Mapped[str] = mapped_column(Text, default="DRAFT", index=True)
    # ACCEPTED_EXPLANATION | PARTIALLY_SUPPORTED | INCONCLUSIVE |
    # INVALID_COMPARISON | REQUIRES_CONTROLLED_RERUN
    decision: Mapped[str | None] = mapped_column(Text)

    updated_at: Mapped[datetime.datetime | None] = mapped_column()
    closed_at: Mapped[datetime.datetime | None] = mapped_column()


class InvestigationRun(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "investigation_runs"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    simulation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.id"), index=True
    )
    # BASELINE | COMPARISON | PHYSICAL_TEST_REFERENCE
    role: Mapped[str] = mapped_column(Text)


class InvestigationMetric(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "investigation_metrics"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    signal_definition_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signal_definitions.id"))

    priority: Mapped[int] = mapped_column(Integer, default=0)
    is_primary: Mapped[bool] = mapped_column(default=False)
    reason: Mapped[str | None] = mapped_column(Text)


class QualityGateResult(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "quality_gate_results"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    simulation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.id"), index=True
    )

    check_type: Mapped[str] = mapped_column(Text)
    # PASS | WARNING | FAIL | UNKNOWN
    status: Mapped[str] = mapped_column(Text)
    value: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    threshold: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    explanation: Mapped[str | None] = mapped_column(Text)
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"))


class ComparabilityAssessment(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "comparability_assessments"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    dimension: Mapped[str] = mapped_column(Text)
    # COMPARABLE | CONDITIONAL | NOT_COMPARABLE | NOT_ESTABLISHED | UNKNOWN
    status: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text)
    evidence_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))


class ConfigurationDiff(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "configuration_diffs"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    component_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("components.id"))

    path: Mapped[str] = mapped_column(Text)

    run_a_value: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    run_b_value: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # SAME | CHANGED | UNKNOWN
    change_status: Mapped[str] = mapped_column(Text)
    # INTENTIONAL | DEPENDENCY | UNINTENTIONAL | UNKNOWN
    change_classification: Mapped[str] = mapped_column(Text, default="UNKNOWN")


class SignalAnalysis(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "signal_analyses"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    signal_definition_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signal_definitions.id"))

    run_a_signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"))
    run_b_signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"))

    alignment_method: Mapped[str | None] = mapped_column(Text)
    filtering_method: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    algorithm_version: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class AnalysisEvent(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "analysis_events"

    signal_analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_analyses.id"), index=True
    )

    event_type: Mapped[str] = mapped_column(Text)
    time_ms: Mapped[float] = mapped_column(Double)

    algorithm_version: Mapped[str] = mapped_column(Text)
    threshold: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    window: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    alignment_method: Mapped[str | None] = mapped_column(Text)

    source_signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"))

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)


class Hypothesis(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "hypotheses"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    # PROPOSED | SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | REJECTED | INCONCLUSIVE
    status: Mapped[str] = mapped_column(Text, default="PROPOSED")
    confidence_basis: Mapped[str | None] = mapped_column(Text)

    affected_components: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    missing_evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    recommended_isolation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    created_by: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime.datetime | None] = mapped_column()


class Evidence(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "evidence"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    # OBSERVED | CALCULATED | DOCUMENTARY | HISTORICAL | INFERRED
    evidence_type: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(Text)
    source_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    source_locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    content: Mapped[str | None] = mapped_column(Text)
    value: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    calculation_version: Mapped[str | None] = mapped_column(Text)
    source_hash: Mapped[str | None] = mapped_column(Text)


class Claim(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "claims"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    claim: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="PROPOSED")
    created_by: Mapped[str] = mapped_column(Text)


class HypothesisEvidenceLink(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "hypothesis_evidence_links"

    hypothesis_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hypotheses.id"), index=True)
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"), index=True)

    # SUPPORTS | CONTRADICTS | CONTEXT
    relationship: Mapped[str] = mapped_column(Text)
    weight: Mapped[float | None] = mapped_column(Double)
    notes: Mapped[str | None] = mapped_column(Text)


class EvidenceContradiction(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "evidence_contradictions"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    evidence_a_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"))
    evidence_b_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"))

    description: Mapped[str | None] = mapped_column(Text)


class Finding(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "findings"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    finding: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="DRAFT")

    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    reviewed_at: Mapped[datetime.datetime | None] = mapped_column()


class RecommendedAction(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "recommended_actions"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)

    action_type: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    priority: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="PROPOSED")


class ControlledComparisonRequest(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "controlled_comparison_requests"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    hypothesis_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hypotheses.id"))

    requested_change: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    controlled_variables: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    reason: Mapped[str | None] = mapped_column(Text)
    # RECORDED — V1 records requests only; it never autonomously launches a
    # commercial solver job (BACKEND_SCHEMA.md §44).
    status: Mapped[str] = mapped_column(Text, default="RECORDED")


class EngineerReview(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "engineer_reviews"

    investigation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("investigations.id"), index=True)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    # ACCEPT | REJECT | MODIFY | REQUEST_SIGNAL | REQUEST_SOURCE |
    # REQUEST_CONTROLLED_COMPARISON | MARK_INCONCLUSIVE
    decision: Mapped[str] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
