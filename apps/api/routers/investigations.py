"""Investigation lifecycle — APP_FLOW.md §4-9, PRD.md PR-001..PR-006.

Each analysis endpoint (quality / global-response / configuration-diff /
comparability / signal analysis) is self-sufficient: it loads what it needs
from the two runs and recomputes, rather than requiring the others to have
been called first over HTTP. `assess_comparability` internally reuses the
same `packages.analysis` functions the dedicated endpoints call — one
implementation, several entry points.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db, get_or_create_default_project, get_or_create_default_user
from apps.api.parquet_io import read_signal
from apps.api.schemas import (
    AgentRunResult,
    CreateInvestigationRequest,
    EngineerReviewRequest,
    EvidenceSummary,
    HypothesisSummary,
    InvestigationSummary,
)
from packages.agent.graph import run_investigation
from packages.agent.llm import get_provider
from packages.analysis.comparability import assess_comparability
from packages.analysis.configuration import compare_configuration
from packages.analysis.global_response import compare_global_response
from packages.analysis.models import (
    ComparabilitySummary,
    ConfigDiffEntry,
    GlobalResponseComparison,
    QualityGateSummary,
    SignalAnalysisResult,
)
from packages.analysis.quality import run_quality_gate
from packages.analysis.signals import analyze_signal_pair
from packages.domain.core import Signal, SignalDefinition, SimulationRun
from packages.domain.db import get_settings
from packages.domain.investigation import (
    AnalysisEvent,
    ComparabilityAssessment,
    ConfigurationDiff,
    EngineerReview,
    Evidence,
    Hypothesis,
    Investigation,
    InvestigationRun,
    QualityGateResult,
    SignalAnalysis,
)

router = APIRouter(tags=["investigations"])


def _get_run_or_404(session: Session, run_id: str) -> SimulationRun:
    run = session.query(SimulationRun).filter_by(run_id=run_id).one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return run


def _get_investigation_or_404(session: Session, investigation_id: uuid.UUID) -> Investigation:
    inv = session.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail=f"investigation not found: {investigation_id}")
    return inv


def _investigation_run_ids(session: Session, investigation: Investigation) -> tuple[SimulationRun, SimulationRun]:
    rows = session.query(InvestigationRun).filter_by(investigation_id=investigation.id).all()
    by_role = {r.role: r.simulation_run_id for r in rows}
    if "BASELINE" not in by_role or "COMPARISON" not in by_role:
        raise HTTPException(status_code=500, detail="investigation is missing BASELINE/COMPARISON runs")
    run_a = session.get(SimulationRun, by_role["BASELINE"])
    run_b = session.get(SimulationRun, by_role["COMPARISON"])
    assert run_a is not None and run_b is not None
    return run_a, run_b


def _config(run: SimulationRun) -> dict[str, Any]:
    config = (run.metadata_ or {}).get("config")
    if config is None:
        raise HTTPException(status_code=422, detail=f"run {run.run_id} has no recorded configuration to diff")
    return dict(config)


def _summary(
    session: Session, investigation: Investigation, run_a: SimulationRun, run_b: SimulationRun
) -> InvestigationSummary:
    primary_metric = None
    if investigation.primary_metric_id is not None:
        signal_def = session.get(SignalDefinition, investigation.primary_metric_id)
        primary_metric = signal_def.canonical_name if signal_def is not None else None
    return InvestigationSummary(
        id=investigation.id,
        title=investigation.title,
        question=investigation.question,
        primary_metric=primary_metric,
        state=investigation.state,
        decision=investigation.decision,
        run_a_id=run_a.run_id,
        run_b_id=run_b.run_id,
        created_at=investigation.created_at,
    )


@router.post("/investigations", response_model=InvestigationSummary, status_code=201)
def create_investigation(body: CreateInvestigationRequest, session: Session = Depends(get_db)) -> InvestigationSummary:
    run_a = _get_run_or_404(session, body.run_a_id)
    run_b = _get_run_or_404(session, body.run_b_id)
    project = get_or_create_default_project(session)
    user = get_or_create_default_user(session)

    # primary_metric is a canonical signal name (e.g. "chest_deflection");
    # Investigation.primary_metric_id is the SignalDefinition it resolves to.
    # An unrecognized name is left NULL rather than guessed — PR-002:
    # "Unknown values must remain explicitly unknown."
    primary_metric_id = None
    if body.primary_metric:
        signal_def = session.query(SignalDefinition).filter_by(canonical_name=body.primary_metric).one_or_none()
        if signal_def is None:
            raise HTTPException(status_code=422, detail=f"unknown primary_metric: {body.primary_metric!r}")
        primary_metric_id = signal_def.id

    investigation = Investigation(
        project_id=project.id,
        created_by=user.id,
        title=body.title or f"{run_a.run_id} vs {run_b.run_id}",
        question=body.question,
        primary_metric_id=primary_metric_id,
        state="RUNS_SELECTED",
    )
    session.add(investigation)
    session.flush()

    session.add_all(
        [
            InvestigationRun(investigation_id=investigation.id, simulation_run_id=run_a.id, role="BASELINE"),
            InvestigationRun(investigation_id=investigation.id, simulation_run_id=run_b.id, role="COMPARISON"),
        ]
    )
    session.commit()

    return _summary(session, investigation, run_a, run_b)


@router.get("/investigations/{investigation_id}", response_model=InvestigationSummary)
def get_investigation(investigation_id: uuid.UUID, session: Session = Depends(get_db)) -> InvestigationSummary:
    investigation = _get_investigation_or_404(session, investigation_id)
    run_a, run_b = _investigation_run_ids(session, investigation)
    return _summary(session, investigation, run_a, run_b)


@router.post("/investigations/{investigation_id}/quality", response_model=dict[str, QualityGateSummary])
def compute_quality(investigation_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, QualityGateSummary]:
    investigation = _get_investigation_or_404(session, investigation_id)
    run_a, run_b = _investigation_run_ids(session, investigation)

    summary_a = run_quality_gate(run_a.run_id, (run_a.metadata_ or {}).get("quality_raw"))
    summary_b = run_quality_gate(run_b.run_id, (run_b.metadata_ or {}).get("quality_raw"))

    session.query(QualityGateResult).filter_by(investigation_id=investigation.id).delete()
    for run, summary in ((run_a, summary_a), (run_b, summary_b)):
        for check in summary.checks:
            session.add(
                QualityGateResult(
                    investigation_id=investigation.id,
                    simulation_run_id=run.id,
                    check_type=check.check_type,
                    status=check.status,
                    value=check.value,
                    threshold=check.threshold,
                    explanation=check.explanation,
                )
            )
    investigation.state = "QUALITY_CHECK"
    session.commit()
    return {"run_a": summary_a, "run_b": summary_b}


@router.post("/investigations/{investigation_id}/global-response", response_model=GlobalResponseComparison)
def compute_global_response(
    investigation_id: uuid.UUID, session: Session = Depends(get_db)
) -> GlobalResponseComparison:
    investigation = _get_investigation_or_404(session, investigation_id)
    run_a, run_b = _investigation_run_ids(session, investigation)

    signal_a = (
        session.query(Signal)
        .join(SignalDefinition)
        .filter(Signal.simulation_run_id == run_a.id, SignalDefinition.canonical_name == "vehicle_pulse")
        .one_or_none()
    )
    signal_b = (
        session.query(Signal)
        .join(SignalDefinition)
        .filter(Signal.simulation_run_id == run_b.id, SignalDefinition.canonical_name == "vehicle_pulse")
        .one_or_none()
    )
    if signal_a is None or signal_b is None:
        raise HTTPException(status_code=422, detail="vehicle_pulse signal not available for one or both runs")

    time_a, values_a = read_signal(signal_a.storage_uri, "vehicle_pulse")
    _time_b, values_b = read_signal(signal_b.storage_uri, "vehicle_pulse")

    comparison = compare_global_response(run_a.run_id, run_b.run_id, time_a, values_a, values_b)
    investigation.state = "GLOBAL_RESPONSE"
    session.commit()
    return comparison


@router.post("/investigations/{investigation_id}/configuration-diff", response_model=list[ConfigDiffEntry])
def compute_configuration_diff(
    investigation_id: uuid.UUID, session: Session = Depends(get_db)
) -> list[ConfigDiffEntry]:
    investigation = _get_investigation_or_404(session, investigation_id)
    run_a, run_b = _investigation_run_ids(session, investigation)

    diffs = compare_configuration(_config(run_a), _config(run_b))

    session.query(ConfigurationDiff).filter_by(investigation_id=investigation.id).delete()
    for diff in diffs:
        session.add(
            ConfigurationDiff(
                investigation_id=investigation.id,
                path=diff.path,
                run_a_value={"value": diff.run_a_value},
                run_b_value={"value": diff.run_b_value},
                change_status=diff.change_status,
                change_classification=diff.change_classification,
            )
        )
    investigation.state = "CONFIGURATION_ANALYSIS"
    session.commit()
    return diffs


@router.post("/investigations/{investigation_id}/comparability", response_model=ComparabilitySummary)
def compute_comparability(investigation_id: uuid.UUID, session: Session = Depends(get_db)) -> ComparabilitySummary:
    investigation = _get_investigation_or_404(session, investigation_id)
    run_a, run_b = _investigation_run_ids(session, investigation)

    quality_a = run_quality_gate(run_a.run_id, (run_a.metadata_ or {}).get("quality_raw"))
    quality_b = run_quality_gate(run_b.run_id, (run_b.metadata_ or {}).get("quality_raw"))

    global_response: GlobalResponseComparison | None = None
    config_diffs: list[ConfigDiffEntry] | None = None
    try:
        signal_a = (
            session.query(Signal)
            .join(SignalDefinition)
            .filter(Signal.simulation_run_id == run_a.id, SignalDefinition.canonical_name == "vehicle_pulse")
            .one_or_none()
        )
        signal_b = (
            session.query(Signal)
            .join(SignalDefinition)
            .filter(Signal.simulation_run_id == run_b.id, SignalDefinition.canonical_name == "vehicle_pulse")
            .one_or_none()
        )
        if signal_a is not None and signal_b is not None:
            time_a, values_a = read_signal(signal_a.storage_uri, "vehicle_pulse")
            _time_b, values_b = read_signal(signal_b.storage_uri, "vehicle_pulse")
            global_response = compare_global_response(run_a.run_id, run_b.run_id, time_a, values_a, values_b)
    except (OSError, ValueError):
        global_response = None

    if (run_a.metadata_ or {}).get("config") is not None and (run_b.metadata_ or {}).get("config") is not None:
        config_diffs = compare_configuration(_config(run_a), _config(run_b))

    summary = assess_comparability(
        run_a.run_id,
        run_b.run_id,
        quality_a,
        quality_b,
        global_response=global_response,
        configuration_diffs=config_diffs,
        result_processing_version_a=run_a.result_processing_version,
        result_processing_version_b=run_b.result_processing_version,
    )

    session.query(ComparabilityAssessment).filter_by(investigation_id=investigation.id).delete()
    for dim in summary.dimensions:
        session.add(
            ComparabilityAssessment(
                investigation_id=investigation.id,
                dimension=dim.dimension,
                status=dim.status,
                explanation=dim.explanation,
            )
        )
    investigation.state = "COMPARABILITY_CHECK"
    session.commit()
    return summary


@router.post("/investigations/{investigation_id}/signals/{signal_name}/analyze", response_model=SignalAnalysisResult)
def analyze_signal(
    investigation_id: uuid.UUID, signal_name: str, session: Session = Depends(get_db)
) -> SignalAnalysisResult:
    investigation = _get_investigation_or_404(session, investigation_id)
    run_a, run_b = _investigation_run_ids(session, investigation)

    signal_def = session.query(SignalDefinition).filter_by(canonical_name=signal_name).one_or_none()
    if signal_def is None:
        raise HTTPException(status_code=404, detail=f"unknown signal: {signal_name}")

    signal_a = (
        session.query(Signal).filter_by(simulation_run_id=run_a.id, signal_definition_id=signal_def.id).one_or_none()
    )
    signal_b = (
        session.query(Signal).filter_by(simulation_run_id=run_b.id, signal_definition_id=signal_def.id).one_or_none()
    )
    if signal_a is None or signal_b is None:
        raise HTTPException(status_code=422, detail=f"signal {signal_name} not available for one or both runs")

    time_s, values_a = read_signal(signal_a.storage_uri, signal_name)
    _t, values_b = read_signal(signal_b.storage_uri, signal_name)

    result = analyze_signal_pair(signal_name, run_a.run_id, run_b.run_id, time_s, values_a, values_b)

    analysis_row = SignalAnalysis(
        investigation_id=investigation.id,
        signal_definition_id=signal_def.id,
        run_a_signal_id=signal_a.id,
        run_b_signal_id=signal_b.id,
        alignment_method="index-aligned",
        filtering_method="none",
        metrics={
            "correlation": result.correlation,
            "run_a_peak": result.run_a_features.peak,
            "run_b_peak": result.run_b_features.peak,
        },
        algorithm_version=result.provenance.algorithm_version,
    )
    session.add(analysis_row)
    session.flush()

    if result.divergence is not None:
        session.add(
            AnalysisEvent(
                signal_analysis_id=analysis_row.id,
                event_type="first_divergence",
                time_ms=result.divergence.time_ms,
                algorithm_version=result.divergence.provenance.algorithm_version,
                threshold=result.divergence.threshold,
                window=result.divergence.window,
                alignment_method=result.divergence.alignment_method,
                source_signal_id=signal_a.id,
            )
        )

    investigation.state = "SIGNAL_ANALYSIS"
    session.commit()
    return result


@router.post("/investigations/{investigation_id}/run-agent", response_model=AgentRunResult)
def run_agent(investigation_id: uuid.UUID, session: Session = Depends(get_db)) -> AgentRunResult:
    """The full LangGraph investigation graph — TRD.md §17. Runs the whole
    deterministic pipeline (quality -> global response -> configuration diff
    -> comparability -> signal analysis -> knowledge/historical retrieval)
    and drafts one hypothesis, then stops at ENGINEER_REVIEW (or BLOCKED on
    quality failure) for a human decision — the agent never finalizes one
    itself (PRD.md §19 Human-in-the-Loop)."""
    investigation = _get_investigation_or_404(session, investigation_id)

    llm = None
    settings = get_settings()
    if settings.llm_provider and settings.llm_provider != "none":
        try:
            llm = get_provider(settings)
        except ValueError:
            llm = None  # unconfigured provider — proceed deterministic-only, per TRD.md §30

    final_state = run_investigation(session, investigation.id, llm=llm)
    session.refresh(investigation)

    hypotheses = (
        session.query(Hypothesis)
        .filter_by(investigation_id=investigation.id)
        .order_by(Hypothesis.created_at.desc())
        .limit(len(final_state.get("hypotheses", [])) or 1)
        .all()
    )
    evidence_count = session.query(Evidence).filter_by(investigation_id=investigation.id).count()

    return AgentRunResult(
        investigation_id=investigation.id,
        state=investigation.state,
        blocked_reason=final_state.get("blocked_reason"),
        hypotheses=[
            HypothesisSummary(
                id=h.id,
                title=h.title,
                description=h.description,
                status=h.status,
                confidence_basis=h.confidence_basis,
                created_at=h.created_at,
            )
            for h in hypotheses
        ],
        evidence_count=evidence_count,
    )


@router.get("/investigations/{investigation_id}/evidence", response_model=list[EvidenceSummary])
def list_evidence(investigation_id: uuid.UUID, session: Session = Depends(get_db)) -> list[EvidenceSummary]:
    _get_investigation_or_404(session, investigation_id)
    rows = session.query(Evidence).filter_by(investigation_id=investigation_id).order_by(Evidence.created_at).all()
    return [
        EvidenceSummary(
            id=e.id,
            evidence_type=e.evidence_type,
            source_type=e.source_type,
            content=e.content,
            created_at=e.created_at,
        )
        for e in rows
    ]


@router.get("/investigations/{investigation_id}/hypotheses", response_model=list[HypothesisSummary])
def list_hypotheses(investigation_id: uuid.UUID, session: Session = Depends(get_db)) -> list[HypothesisSummary]:
    _get_investigation_or_404(session, investigation_id)
    rows = session.query(Hypothesis).filter_by(investigation_id=investigation_id).order_by(Hypothesis.created_at).all()
    return [
        HypothesisSummary(
            id=h.id,
            title=h.title,
            description=h.description,
            status=h.status,
            confidence_basis=h.confidence_basis,
            created_at=h.created_at,
        )
        for h in rows
    ]


@router.post("/investigations/{investigation_id}/review", status_code=201)
def submit_engineer_review(
    investigation_id: uuid.UUID, body: EngineerReviewRequest, session: Session = Depends(get_db)
) -> dict[str, Any]:
    """APP_FLOW.md §16: the human decision boundary. The agent stops at
    ENGINEER_REVIEW/BLOCKED; only an explicit review moves the investigation
    to DECISION — never automatic."""
    investigation = _get_investigation_or_404(session, investigation_id)
    user = get_or_create_default_user(session)

    valid_decisions = {
        "ACCEPT",
        "REJECT",
        "MODIFY",
        "REQUEST_SIGNAL",
        "REQUEST_SOURCE",
        "REQUEST_CONTROLLED_COMPARISON",
        "MARK_INCONCLUSIVE",
    }
    if body.decision not in valid_decisions:
        raise HTTPException(status_code=422, detail=f"decision must be one of {sorted(valid_decisions)}")

    review = EngineerReview(
        investigation_id=investigation.id, reviewer_id=user.id, decision=body.decision, comment=body.comment
    )
    session.add(review)

    investigation.state = "DECISION" if body.decision in ("ACCEPT", "REJECT") else "FOLLOW_UP"
    session.commit()
    return {"review_id": str(review.id), "investigation_state": investigation.state}
