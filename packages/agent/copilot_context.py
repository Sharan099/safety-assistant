"""build_investigation_context() — PRD_COPILOT_UPDATE.md Section 4,
CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 2.

Every query here is filtered strictly by `investigation_id` (or by a
simulation_run/signal_analysis that itself belongs to that investigation) —
no cross-investigation leakage, and nothing sent to the LLM beyond this
structured summary (PRD_COPILOT_UPDATE.md Section 4: "Do not send the
entire database to the LLM").
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from packages.agent.tools import get_run_manifest
from packages.domain.core import SignalDefinition, SimulationRun
from packages.domain.investigation import (
    AnalysisEvent,
    ComparabilityAssessment,
    ConfigurationDiff,
    EngineerReview,
    Evidence,
    Hypothesis,
    HypothesisEvidenceLink,
    Investigation,
    InvestigationRun,
    QualityGateResult,
    SignalAnalysis,
)

# Evidence/hypothesis content is capped, not the whole row text, to keep the
# context small and bounded regardless of how much an investigation
# accumulates — PRD_COPILOT_UPDATE.md Section 4.
MAX_CONTENT_CHARS = 400
MAX_EVIDENCE_ITEMS = 40
MAX_HYPOTHESES = 10


class InvestigationContext(BaseModel):
    investigation_id: uuid.UUID
    question: str
    run_a: dict[str, Any]
    run_b: dict[str, Any]
    primary_metric: str | None
    investigation_state: str

    quality_results: dict[str, list[dict[str, Any]]]  # run_id -> checks
    comparability_results: list[dict[str, Any]]
    configuration_diffs: list[dict[str, Any]]

    selected_signals: list[str]
    signal_analysis_results: list[dict[str, Any]]
    divergence_events: list[dict[str, Any]]

    current_evidence: list[dict[str, Any]]
    current_hypotheses: list[dict[str, Any]]

    engineer_review_state: dict[str, Any] | None


def _get_investigation_or_raise(session: Session, investigation_id: uuid.UUID) -> Investigation:
    investigation = session.get(Investigation, investigation_id)
    if investigation is None:
        raise ValueError(f"investigation not found: {investigation_id}")
    return investigation


def build_investigation_context(session: Session, investigation_id: uuid.UUID) -> InvestigationContext:
    investigation = _get_investigation_or_raise(session, investigation_id)

    run_rows = session.query(InvestigationRun).filter_by(investigation_id=investigation.id).all()
    by_role = {r.role: r.simulation_run_id for r in run_rows}
    if "BASELINE" not in by_role or "COMPARISON" not in by_role:
        raise ValueError(f"investigation {investigation_id} is missing BASELINE/COMPARISON runs")
    run_a = session.get(SimulationRun, by_role["BASELINE"])
    run_b = session.get(SimulationRun, by_role["COMPARISON"])
    assert run_a is not None and run_b is not None

    primary_metric = None
    if investigation.primary_metric_id is not None:
        signal_def = session.get(SignalDefinition, investigation.primary_metric_id)
        primary_metric = signal_def.canonical_name if signal_def is not None else None

    quality_rows = session.query(QualityGateResult).filter_by(investigation_id=investigation.id).all()
    quality_results: dict[str, list[dict[str, Any]]] = {run_a.run_id: [], run_b.run_id: []}
    run_id_by_pk = {run_a.id: run_a.run_id, run_b.id: run_b.run_id}
    for q in quality_rows:
        label = run_id_by_pk.get(q.simulation_run_id)
        if label is not None:
            quality_results[label].append(
                {"check_type": q.check_type, "status": q.status, "explanation": q.explanation}
            )

    comparability_rows = session.query(ComparabilityAssessment).filter_by(investigation_id=investigation.id).all()
    comparability_results = [
        {"dimension": c.dimension, "status": c.status, "explanation": c.explanation} for c in comparability_rows
    ]

    config_rows = session.query(ConfigurationDiff).filter_by(investigation_id=investigation.id).all()
    configuration_diffs = [
        {
            "path": c.path,
            "run_a_value": c.run_a_value,
            "run_b_value": c.run_b_value,
            "change_status": c.change_status,
            "change_classification": c.change_classification,
        }
        for c in config_rows
        if c.change_status != "SAME"
    ]

    signal_analysis_rows = session.query(SignalAnalysis).filter_by(investigation_id=investigation.id).all()
    signal_def_ids = {s.signal_definition_id for s in signal_analysis_rows}
    signal_def_names = {
        d.id: d.canonical_name
        for d in session.query(SignalDefinition).filter(SignalDefinition.id.in_(signal_def_ids)).all()
    }
    signal_analysis_results = [
        {
            "signal": signal_def_names.get(s.signal_definition_id, "unknown"),
            "algorithm_version": s.algorithm_version,
            "metrics": s.metrics,
        }
        for s in signal_analysis_rows
    ]
    selected_signals = sorted({str(r["signal"]) for r in signal_analysis_results})

    analysis_ids = [s.id for s in signal_analysis_rows]
    event_rows = (
        session.query(AnalysisEvent).filter(AnalysisEvent.signal_analysis_id.in_(analysis_ids)).all()
        if analysis_ids
        else []
    )
    signal_analysis_by_id = {s.id: s for s in signal_analysis_rows}
    divergence_events = [
        {
            "signal": signal_def_names.get(signal_analysis_by_id[e.signal_analysis_id].signal_definition_id, "unknown")
            if e.signal_analysis_id in signal_analysis_by_id
            else "unknown",
            "event_type": e.event_type,
            "time_ms": e.time_ms,
        }
        for e in event_rows
    ]

    evidence_rows = (
        session.query(Evidence)
        .filter_by(investigation_id=investigation.id)
        .order_by(Evidence.created_at)
        .limit(MAX_EVIDENCE_ITEMS)
        .all()
    )
    current_evidence = [
        {
            "id": str(e.id),
            "evidence_type": e.evidence_type,
            "source_type": e.source_type,
            "content": (e.content or "")[:MAX_CONTENT_CHARS],
        }
        for e in evidence_rows
    ]

    hypothesis_rows = (
        session.query(Hypothesis)
        .filter_by(investigation_id=investigation.id)
        .order_by(Hypothesis.created_at.desc())
        .limit(MAX_HYPOTHESES)
        .all()
    )
    hypothesis_ids = [h.id for h in hypothesis_rows]
    link_rows = (
        session.query(HypothesisEvidenceLink).filter(HypothesisEvidenceLink.hypothesis_id.in_(hypothesis_ids)).all()
        if hypothesis_ids
        else []
    )
    links_by_hypothesis: dict[uuid.UUID, list[HypothesisEvidenceLink]] = {}
    for link in link_rows:
        links_by_hypothesis.setdefault(link.hypothesis_id, []).append(link)

    current_hypotheses = [
        {
            "id": str(h.id),
            "title": h.title,
            "description": (h.description or "")[:MAX_CONTENT_CHARS],
            "status": h.status,
            "confidence_basis": h.confidence_basis,
            "supporting_evidence_ids": [
                str(link.evidence_id) for link in links_by_hypothesis.get(h.id, []) if link.relationship == "SUPPORTS"
            ],
            "contradicting_evidence_ids": [
                str(link.evidence_id)
                for link in links_by_hypothesis.get(h.id, [])
                if link.relationship == "CONTRADICTS"
            ],
        }
        for h in hypothesis_rows
    ]

    latest_review = (
        session.query(EngineerReview)
        .filter_by(investigation_id=investigation.id)
        .order_by(EngineerReview.created_at.desc())
        .first()
    )
    engineer_review_state = (
        {"decision": latest_review.decision, "comment": latest_review.comment} if latest_review is not None else None
    )

    return InvestigationContext(
        investigation_id=investigation.id,
        question=investigation.question,
        run_a=get_run_manifest(run_a),
        run_b=get_run_manifest(run_b),
        primary_metric=primary_metric,
        investigation_state=investigation.state,
        quality_results=quality_results,
        comparability_results=comparability_results,
        configuration_diffs=configuration_diffs,
        selected_signals=selected_signals,
        signal_analysis_results=signal_analysis_results,
        divergence_events=divergence_events,
        current_evidence=current_evidence,
        current_hypotheses=current_hypotheses,
        engineer_review_state=engineer_review_state,
    )
