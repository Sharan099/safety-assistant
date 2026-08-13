"""Shared DB-persistence helpers for the deterministic analysis steps —
used by both `apps/api/routers/investigations.py`'s individual step
endpoints and `packages/agent/graph.py`'s LangGraph nodes, so the two paths
to running an investigation leave identical database state.

Found via a Copilot-context test (tests/agent/test_copilot_context.py):
`run_investigation()` computed quality/comparability/configuration-diff/
signal-analysis results into the LangGraph state dict but never wrote
QualityGateResult/ComparabilityAssessment/ConfigurationDiff/SignalAnalysis
rows — only the API's step-by-step endpoints did. An investigation run
purely through "Run full agent" therefore looked, to anything reading the
database (the Copilot context builder, a report, another engineer), as if
none of that analysis had happened at all. Fixed by extracting this module
and having both call sites use it — one implementation, not two that drift.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from apps.api.parquet_io import read_signal
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
from packages.domain.investigation import (
    AnalysisEvent,
    ComparabilityAssessment,
    ConfigurationDiff,
    Investigation,
    QualityGateResult,
    SignalAnalysis,
)


def _find_signal(session: Session, run: SimulationRun, canonical_name: str) -> Signal | None:
    return (
        session.query(Signal)
        .join(SignalDefinition)
        .filter(Signal.simulation_run_id == run.id, SignalDefinition.canonical_name == canonical_name)
        .one_or_none()
    )


def persist_quality_results(
    session: Session, investigation: Investigation, run_a: SimulationRun, run_b: SimulationRun
) -> tuple[QualityGateSummary, QualityGateSummary]:
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
    session.flush()
    return summary_a, summary_b


def compute_global_response(
    session: Session, run_a: SimulationRun, run_b: SimulationRun
) -> GlobalResponseComparison | None:
    """Not persisted to its own table — BACKEND_SCHEMA.md has none for it —
    but exposed here so both call sites share one implementation."""
    signal_a = _find_signal(session, run_a, "vehicle_pulse")
    signal_b = _find_signal(session, run_b, "vehicle_pulse")
    if signal_a is None or signal_b is None:
        return None
    try:
        time_a, values_a = read_signal(signal_a.storage_uri, "vehicle_pulse")
        _t, values_b = read_signal(signal_b.storage_uri, "vehicle_pulse")
    except OSError:
        return None
    return compare_global_response(run_a.run_id, run_b.run_id, time_a, values_a, values_b)


def persist_configuration_diff(
    session: Session, investigation: Investigation, run_a: SimulationRun, run_b: SimulationRun
) -> list[ConfigDiffEntry] | None:
    config_a = (run_a.metadata_ or {}).get("config")
    config_b = (run_b.metadata_ or {}).get("config")
    if config_a is None or config_b is None:
        return None

    diffs = compare_configuration(config_a, config_b)

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
    session.flush()
    return diffs


def persist_comparability(
    session: Session,
    investigation: Investigation,
    run_a: SimulationRun,
    run_b: SimulationRun,
    quality_a: QualityGateSummary,
    quality_b: QualityGateSummary,
    *,
    global_response: GlobalResponseComparison | None,
    configuration_diffs: list[ConfigDiffEntry] | None,
) -> ComparabilitySummary:
    summary = assess_comparability(
        run_a.run_id,
        run_b.run_id,
        quality_a,
        quality_b,
        global_response=global_response,
        configuration_diffs=configuration_diffs,
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
    session.flush()
    return summary


def persist_signal_analysis(
    session: Session, investigation: Investigation, run_a: SimulationRun, run_b: SimulationRun, signal_name: str
) -> SignalAnalysisResult | None:
    signal_def = session.query(SignalDefinition).filter_by(canonical_name=signal_name).one_or_none()
    if signal_def is None:
        return None

    signal_a = (
        session.query(Signal).filter_by(simulation_run_id=run_a.id, signal_definition_id=signal_def.id).one_or_none()
    )
    signal_b = (
        session.query(Signal).filter_by(simulation_run_id=run_b.id, signal_definition_id=signal_def.id).one_or_none()
    )
    if signal_a is None or signal_b is None:
        return None

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
    session.flush()
    return result
