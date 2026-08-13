"""Generate the synthetic CAE benchmark and persist it.

Writes:
  - data/parquet/<RUN_ID>/{metadata,occupant,restraint,vehicle}.parquet
    (TRD.md §8 layout — raw signal samples never go in Postgres)
  - data/synthetic/scenarios.yaml (ground truth: changed factor, expected
    signal changes, allowed/disallowed conclusions — PRD.md §13)
  - Postgres rows (Organization/Project/Vehicle/ModelVersion/SimulationRun/
    SignalDefinition/Signal) so the runs are queryable (IMPLEMENTATION_PLAN.md
    Phase Gate 1: "Two runs can be registered and queried")

Idempotent: re-running clears and rewrites both the Parquet tree and the
previously-loaded synthetic rows (identified by `metadata.synthetic: true`)
rather than accumulating duplicates.

Usage:
    uv run python scripts/generate_synthetic_dataset.py
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

import pandas as pd
import yaml
from sqlalchemy import delete
from sqlalchemy.orm import Session

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.analysis.synthetic import RunFixture, ScenarioResult, generate_all  # noqa: E402
from packages.domain.core import (  # noqa: E402
    ModelVersion,
    Organization,
    Project,
    Signal,
    SignalDefinition,
    SimulationRun,
    Vehicle,
)
from packages.domain.db import get_engine  # noqa: E402
from packages.domain.investigation import (  # noqa: E402
    AnalysisEvent,
    ComparabilityAssessment,
    ConfigurationDiff,
    ControlledComparisonRequest,
    EngineerReview,
    Evidence,
    EvidenceContradiction,
    Finding,
    Hypothesis,
    HypothesisEvidenceLink,
    Investigation,
    InvestigationMetric,
    InvestigationRun,
    QualityGateResult,
    RecommendedAction,
    SignalAnalysis,
)

SIGNAL_GROUPS = {
    "occupant": ["chest_acceleration", "chest_deflection", "chest_velocity", "pelvis_acceleration", "torso_rotation"],
    "restraint": ["belt_force", "airbag_pressure"],
    "vehicle": ["vehicle_pulse"],
}

SYNTHETIC_ORG_KEY = "SYNTHETIC"
SYNTHETIC_PROJECT_NAME = "Synthetic Benchmark"
SYNTHETIC_VEHICLE_NAME = "Synthetic Sedan A"


def _write_parquet(run: RunFixture, out_dir: pathlib.Path) -> dict[str, str]:
    run_dir = out_dir / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    uris: dict[str, str] = {}
    for group, names in SIGNAL_GROUPS.items():
        df = pd.DataFrame({"time_s": run.time_s, **{n: run.signals[n] for n in names}})
        path = run_dir / f"{group}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        uris[group] = path.as_posix()

    meta_path = run_dir / "metadata.parquet"
    pd.DataFrame([{"run_id": run.run_id, "label": run.label, "model_version": run.model_version}]).to_parquet(
        meta_path, engine="pyarrow", index=False
    )
    uris["metadata"] = meta_path.as_posix()
    return uris


def _write_manifest(results: list[ScenarioResult], out_path: pathlib.Path) -> None:
    manifest = {
        "schema_version": 1,
        "scenarios": [
            {
                "scenario_id": r.spec.scenario_id,
                "title": r.spec.title,
                "changed_factor": r.spec.changed_factor,
                "description": r.spec.description,
                "run_a_id": r.run_a.run_id,
                "run_b_id": r.run_b.run_id,
                "expected_signal_changes": r.spec.expected_signal_changes,
                "allowed_conclusions": r.spec.allowed_conclusions,
                "disallowed_conclusions": r.spec.disallowed_conclusions,
            }
            for r in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)


def _get_or_create_org(session: Session) -> Organization:
    org = session.query(Organization).filter_by(name=SYNTHETIC_ORG_KEY).one_or_none()
    if org is None:
        org = Organization(name=SYNTHETIC_ORG_KEY)
        session.add(org)
        session.flush()
    return org


def _clear_investigations_referencing(session: Session, run_ids: Sequence[object]) -> None:
    """Any real investigation an engineer created against these runs (e.g. via
    the API while exploring) must be torn down before the runs themselves can
    be deleted — Artifact.simulation_run_id-style FK chains, but through
    InvestigationRun. Regenerating the benchmark is a full reset, not a
    partial one; see docs/ADR/0008."""
    investigation_ids = [
        row[0]
        for row in session.query(InvestigationRun.investigation_id)
        .filter(InvestigationRun.simulation_run_id.in_(run_ids))
        .all()
    ]
    if not investigation_ids:
        return

    hypothesis_ids = [
        row[0] for row in session.query(Hypothesis.id).filter(Hypothesis.investigation_id.in_(investigation_ids)).all()
    ]
    signal_analysis_ids = [
        row[0]
        for row in session.query(SignalAnalysis.id).filter(SignalAnalysis.investigation_id.in_(investigation_ids)).all()
    ]

    session.execute(delete(HypothesisEvidenceLink).where(HypothesisEvidenceLink.hypothesis_id.in_(hypothesis_ids)))
    session.execute(delete(AnalysisEvent).where(AnalysisEvent.signal_analysis_id.in_(signal_analysis_ids)))
    session.execute(delete(SignalAnalysis).where(SignalAnalysis.investigation_id.in_(investigation_ids)))
    session.execute(delete(EvidenceContradiction).where(EvidenceContradiction.investigation_id.in_(investigation_ids)))
    session.execute(delete(QualityGateResult).where(QualityGateResult.investigation_id.in_(investigation_ids)))
    session.execute(
        delete(ComparabilityAssessment).where(ComparabilityAssessment.investigation_id.in_(investigation_ids))
    )
    session.execute(delete(ConfigurationDiff).where(ConfigurationDiff.investigation_id.in_(investigation_ids)))
    session.execute(
        delete(ControlledComparisonRequest).where(ControlledComparisonRequest.investigation_id.in_(investigation_ids))
    )
    session.execute(delete(Hypothesis).where(Hypothesis.investigation_id.in_(investigation_ids)))
    session.execute(delete(Evidence).where(Evidence.investigation_id.in_(investigation_ids)))
    session.execute(delete(Finding).where(Finding.investigation_id.in_(investigation_ids)))
    session.execute(delete(RecommendedAction).where(RecommendedAction.investigation_id.in_(investigation_ids)))
    session.execute(delete(EngineerReview).where(EngineerReview.investigation_id.in_(investigation_ids)))
    session.execute(delete(InvestigationMetric).where(InvestigationMetric.investigation_id.in_(investigation_ids)))
    session.execute(delete(InvestigationRun).where(InvestigationRun.investigation_id.in_(investigation_ids)))
    session.execute(delete(Investigation).where(Investigation.id.in_(investigation_ids)))
    session.flush()


def _clear_previous_load(session: Session, project: Project | None) -> None:
    if project is None:
        return
    run_ids = [r.id for r in session.query(SimulationRun).filter_by(project_id=project.id).all()]
    if run_ids:
        _clear_investigations_referencing(session, run_ids)
        session.execute(delete(Signal).where(Signal.simulation_run_id.in_(run_ids)))
        session.execute(delete(SimulationRun).where(SimulationRun.id.in_(run_ids)))
    session.query(ModelVersion).filter(
        ModelVersion.vehicle_id.in_(session.query(Vehicle.id).filter_by(project_id=project.id))
    ).delete(synchronize_session=False)
    session.query(Vehicle).filter_by(project_id=project.id).delete(synchronize_session=False)
    session.flush()


def _load_run_into_db(
    session: Session,
    run: RunFixture,
    *,
    project: Project,
    vehicle: Vehicle,
    signal_defs: dict[str, SignalDefinition],
    parquet_uris: dict[str, str],
    scenario_id: str,
) -> None:
    model_version = ModelVersion(vehicle_id=vehicle.id, version=run.model_version, metadata_={"synthetic": True})
    session.add(model_version)
    session.flush()

    quality_status = "PASS" if run.quality_raw["termination"] == "NORMAL" else "FAIL"
    sim_run = SimulationRun(
        project_id=project.id,
        vehicle_id=vehicle.id,
        model_version_id=model_version.id,
        run_id=run.run_id,
        solver="SYNTHETIC",
        solver_version="v1",
        dummy_version=run.config["dummy"]["type"],
        impact_type="FRONTAL",
        impact_speed=50.0,
        seat_configuration=run.config["seat"],
        restraint_configuration=run.config["restraint"],
        result_processing_version=run.config.get("result_processing_version", "CFC180"),
        status="COMPLETE",
        quality_status=quality_status,
        metadata_={
            "synthetic": True,
            "scenario_id": scenario_id,
            "label": run.label,
            "quality_raw": run.quality_raw,
            "config": run.config,
        },
    )
    session.add(sim_run)
    session.flush()

    group_by_signal = {name: group for group, names in SIGNAL_GROUPS.items() for name in names}
    for name in run.signals:
        session.add(
            Signal(
                simulation_run_id=sim_run.id,
                signal_definition_id=signal_defs[name].id,
                name=name,
                storage_uri=parquet_uris[group_by_signal[name]],
                sampling_rate=5000.0,
                start_time=float(run.time_s[0]),
                end_time=float(run.time_s[-1]),
                unit=signal_defs[name].unit,
            )
        )
    session.flush()


def _get_or_create_signal_definitions(session: Session) -> dict[str, SignalDefinition]:
    from packages.analysis.synthetic import DEFAULT_PARAMS

    defs: dict[str, SignalDefinition] = {}
    for name, params in DEFAULT_PARAMS.items():
        existing = session.query(SignalDefinition).filter_by(canonical_name=name).one_or_none()
        if existing is None:
            existing = SignalDefinition(name=name, canonical_name=name, domain="occupant", unit=params["unit"])
            session.add(existing)
            session.flush()
        defs[name] = existing
    return defs


def main() -> None:
    results = generate_all()

    parquet_root = ROOT / "data" / "parquet"
    _write_manifest(results, ROOT / "data" / "synthetic" / "scenarios.yaml")

    engine = get_engine()
    with Session(engine) as session:
        org = _get_or_create_org(session)
        project = session.query(Project).filter_by(organization_id=org.id, name=SYNTHETIC_PROJECT_NAME).one_or_none()
        _clear_previous_load(session, project)
        if project is None:
            project = Project(organization_id=org.id, name=SYNTHETIC_PROJECT_NAME, status="ACTIVE")
            session.add(project)
            session.flush()

        vehicle = Vehicle(project_id=project.id, name=SYNTHETIC_VEHICLE_NAME, vehicle_type="SEDAN")
        session.add(vehicle)
        session.flush()

        signal_defs = _get_or_create_signal_definitions(session)

        for result in results:
            uris_a = _write_parquet(result.run_a, parquet_root)
            uris_b = _write_parquet(result.run_b, parquet_root)
            _load_run_into_db(
                session,
                result.run_a,
                project=project,
                vehicle=vehicle,
                signal_defs=signal_defs,
                parquet_uris=uris_a,
                scenario_id=result.spec.scenario_id,
            )
            _load_run_into_db(
                session,
                result.run_b,
                project=project,
                vehicle=vehicle,
                signal_defs=signal_defs,
                parquet_uris=uris_b,
                scenario_id=result.spec.scenario_id,
            )

        session.commit()

    print(f"Wrote {len(results)} scenarios ({2 * len(results)} runs) to {parquet_root} and Postgres.")


if __name__ == "__main__":
    main()
