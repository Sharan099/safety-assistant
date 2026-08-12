"""Deterministic tool layer — TRD.md Section 23.

Thin wrappers around packages.analysis / packages.retrieval / packages.domain
so the graph (packages/agent/graph.py) never performs an engineering
calculation itself — it only calls these. Each tool returns a plain dict
(JSON-serializable, matches TRD.md Section 23: "Tools return structured
JSON/Pydantic objects") so it can sit directly in `InvestigationState`.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from apps.api.parquet_io import read_signal
from packages.analysis.comparability import assess_comparability
from packages.analysis.configuration import compare_configuration
from packages.analysis.global_response import compare_global_response
from packages.analysis.quality import run_quality_gate
from packages.analysis.signals import analyze_signal_pair
from packages.domain.core import Signal, SignalDefinition, SimulationRun
from packages.retrieval.search import SourceFilter, retrieve

# PR-007 example: a metric-specific signal plan. Only signals the synthetic
# benchmark actually models (packages/analysis/synthetic.py DEFAULT_PARAMS)
# are listed — a real deployment would source this from SignalDefinition
# relationships instead of a hardcoded table.
SIGNAL_PLANS: dict[str, list[str]] = {
    "chest_deflection": [
        "chest_acceleration",
        "chest_velocity",
        "belt_force",
        "pelvis_acceleration",
        "torso_rotation",
        "airbag_pressure",
        "vehicle_pulse",
    ],
    "chest_acceleration": ["chest_deflection", "belt_force", "vehicle_pulse"],
    "belt_force": ["chest_deflection", "chest_acceleration", "pelvis_acceleration"],
    "pelvis_acceleration": ["belt_force", "torso_rotation", "vehicle_pulse"],
}


def load_run(session: Session, run_id: str) -> SimulationRun:
    run = session.query(SimulationRun).filter_by(run_id=run_id).one_or_none()
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    return run


def get_run_manifest(run: SimulationRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "solver": run.solver,
        "solver_version": run.solver_version,
        "dummy_version": run.dummy_version,
        "impact_type": run.impact_type,
        "impact_speed": run.impact_speed,
        "quality_status": run.quality_status,
        "result_processing_version": run.result_processing_version,
    }


def tool_run_quality_gate(run: SimulationRun) -> dict[str, Any]:
    summary = run_quality_gate(run.run_id, (run.metadata_ or {}).get("quality_raw"))
    return summary.model_dump()


def tool_compare_global_response(session: Session, run_a: SimulationRun, run_b: SimulationRun) -> dict[str, Any] | None:
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
        return None
    time_a, values_a = read_signal(signal_a.storage_uri, "vehicle_pulse")
    _t, values_b = read_signal(signal_b.storage_uri, "vehicle_pulse")
    return compare_global_response(run_a.run_id, run_b.run_id, time_a, values_a, values_b).model_dump()


def tool_compare_configuration(run_a: SimulationRun, run_b: SimulationRun) -> list[dict[str, Any]] | None:
    config_a = (run_a.metadata_ or {}).get("config")
    config_b = (run_b.metadata_ or {}).get("config")
    if config_a is None or config_b is None:
        return None
    return [d.model_dump() for d in compare_configuration(config_a, config_b)]


def tool_assess_comparability(
    run_a: SimulationRun,
    run_b: SimulationRun,
    quality_a: dict[str, Any],
    quality_b: dict[str, Any],
    global_response: dict[str, Any] | None,
    configuration_diff: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    from packages.analysis.models import (
        ConfigDiffEntry,
        GlobalResponseComparison,
        QualityGateSummary,
    )

    summary = assess_comparability(
        run_a.run_id,
        run_b.run_id,
        QualityGateSummary.model_validate(quality_a),
        QualityGateSummary.model_validate(quality_b),
        global_response=GlobalResponseComparison.model_validate(global_response) if global_response else None,
        configuration_diffs=[ConfigDiffEntry.model_validate(d) for d in configuration_diff]
        if configuration_diff
        else None,
        result_processing_version_a=run_a.result_processing_version,
        result_processing_version_b=run_b.result_processing_version,
    )
    return summary.model_dump()


def select_signal_plan(primary_metric: str) -> list[str]:
    return [primary_metric, *SIGNAL_PLANS.get(primary_metric, [])]


def tool_analyze_signal(
    session: Session, run_a: SimulationRun, run_b: SimulationRun, signal_name: str
) -> dict[str, Any] | None:
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
    return result.model_dump()


def tool_retrieve_knowledge(session: Session, query_text: str, *, limit: int = 5) -> list[dict[str, Any]]:
    results = retrieve(session, query_text, filters=SourceFilter(), limit=limit)
    return [r.model_dump(mode="json") for r in results]


def tool_retrieve_historical_cases(
    session: Session, primary_metric: str, exclude_investigation_id: object
) -> list[dict[str, Any]]:
    """PRD.md PR-010: similar historical cases. V1 queries past *closed*
    investigations sharing the same primary-metric canonical signal — no
    dedicated "historical case" table exists yet (BACKEND_SCHEMA.md has no
    such entity); this is legitimately empty until investigations have been
    completed and closed. An empty list is NOT_AVAILABLE, never fabricated.
    """
    from packages.domain.investigation import Finding, Investigation

    rows = (
        session.query(Investigation, Finding)
        .join(Finding, Finding.investigation_id == Investigation.id)
        .filter(Investigation.state == "CLOSED", Investigation.id != exclude_investigation_id)
        .filter(Investigation.question.ilike(f"%{primary_metric}%"))
        .limit(5)
        .all()
    )
    return [
        {"investigation_id": str(inv.id), "title": inv.title, "question": inv.question, "finding": finding.finding}
        for inv, finding in rows
    ]
