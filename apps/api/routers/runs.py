"""Run Browser / Run Identity — APP_FLOW.md §5, UI_UX_DESIGN_BRIEF.md §8-9."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import RunDetail, RunSummary
from packages.domain.core import ModelVersion, SimulationRun, Vehicle

router = APIRouter(tags=["runs"])


def _to_summary(run: SimulationRun, vehicle_name: str, model_version: str) -> RunSummary:
    return RunSummary(
        id=run.id,
        run_id=run.run_id,
        vehicle_name=vehicle_name,
        model_version=model_version,
        solver=run.solver,
        solver_version=run.solver_version,
        impact_type=run.impact_type,
        impact_speed=run.impact_speed,
        quality_status=run.quality_status,
        created_at=run.created_at,
    )


@router.get("/runs", response_model=list[RunSummary])
def list_runs(session: Session = Depends(get_db)) -> list[RunSummary]:
    rows = (
        session.query(SimulationRun, Vehicle.name, ModelVersion.version)
        .join(Vehicle, SimulationRun.vehicle_id == Vehicle.id)
        .join(ModelVersion, SimulationRun.model_version_id == ModelVersion.id)
        .order_by(SimulationRun.run_id)
    )
    return [_to_summary(run, vehicle_name, model_version) for run, vehicle_name, model_version in rows.all()]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, session: Session = Depends(get_db)) -> RunDetail:
    row = (
        session.query(SimulationRun, Vehicle.name, ModelVersion.version)
        .join(Vehicle, SimulationRun.vehicle_id == Vehicle.id)
        .join(ModelVersion, SimulationRun.model_version_id == ModelVersion.id)
        .filter(SimulationRun.run_id == run_id)
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    run, vehicle_name, model_version = row
    return RunDetail(
        **_to_summary(run, vehicle_name, model_version).model_dump(),
        dummy_version=run.dummy_version,
        seat_configuration=run.seat_configuration,
        restraint_configuration=run.restraint_configuration,
        result_processing_version=run.result_processing_version,
        metadata=run.metadata_,
    )
