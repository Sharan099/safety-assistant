"""Round-trip the core chain: Org -> User -> Project -> Vehicle ->
ModelVersion -> SimulationRun -> Artifact -> SignalDefinition -> Signal.

Also exercises the artifacts<->simulation_runs<->model_versions FK cycle
resolved in docs/ADR/0005 — this is the actual regression test for that fix.
"""

import uuid

import pytest
import sqlalchemy.exc
from sqlalchemy.orm import Session
from tests.legacy.domain.conftest import requires_db

from packages.domain.core import (
    Artifact,
    ModelVersion,
    Organization,
    Project,
    Signal,
    SignalDefinition,
    SimulationRun,
    User,
    Vehicle,
)


@requires_db
def test_artifact_simulation_run_fk_is_actually_enforced(session: Session) -> None:
    """docs/ADR/0005's correction: a valid insert succeeding proves nothing
    about whether a FK constraint exists — only a *rejected* invalid insert
    does. This is what test_core_chain_round_trip below was missing."""
    artifact = Artifact(
        simulation_run_id=uuid.uuid4(),  # does not exist
        artifact_type="RESULT_DATABASE",
        filename="d3plot",
        storage_uri="file:///data/artifacts/does-not-exist/d3plot",
        sha256="b" * 64,
    )
    session.add(artifact)
    with pytest.raises(sqlalchemy.exc.IntegrityError, match="fk_artifacts_simulation_run_id|foreign key"):
        session.flush()
    # Leave cleanup to the `session` fixture's teardown (transaction.rollback())
    # — rolling back here too risks a double-rollback on the same underlying
    # connection-level transaction the fixture owns.


@requires_db
def test_core_chain_round_trip(session: Session) -> None:
    org = Organization(name="ACME Automotive")
    session.add(org)
    session.flush()

    user = User(organization_id=org.id, email="engineer@example.com", display_name="E. Engineer", role="ENGINEER")
    project = Project(organization_id=org.id, name="Project X")
    session.add_all([user, project])
    session.flush()

    vehicle = Vehicle(project_id=project.id, name="Sedan A", vehicle_type="SEDAN")
    session.add(vehicle)
    session.flush()

    model_version = ModelVersion(vehicle_id=vehicle.id, version="v12.3")
    session.add(model_version)
    session.flush()

    run = SimulationRun(
        project_id=project.id,
        vehicle_id=vehicle.id,
        model_version_id=model_version.id,
        run_id="RUN-0041",
        solver="LS-DYNA",
        solver_version="R17",
        impact_type="FRONTAL",
        impact_speed=50.0,
        quality_status="PASS",
    )
    session.add(run)
    session.flush()

    # Regression: Artifact -> SimulationRun uses use_alter (docs/ADR/0005);
    # confirm the FK is actually enforced end-to-end, not just deferred away.
    artifact = Artifact(
        simulation_run_id=run.id,
        artifact_type="RESULT_DATABASE",
        filename="d3plot",
        storage_uri="file:///data/artifacts/RUN-0041/d3plot",
        sha256="a" * 64,
    )
    session.add(artifact)
    session.flush()

    model_version.source_artifact_id = artifact.id  # the other side of the cycle
    session.flush()

    # A distinct canonical_name from the real synthetic-benchmark signal
    # definitions (scripts/generate_synthetic_dataset.py) — those rows are
    # committed outside this test's transaction and would collide otherwise.
    signal_def = SignalDefinition(name="Chest Deflection (test)", canonical_name="chest_deflection__test", unit="mm")
    session.add(signal_def)
    session.flush()

    signal = Signal(
        simulation_run_id=run.id,
        signal_definition_id=signal_def.id,
        name="chest_deflection",
        storage_uri="file:///data/parquet/RUN-0041/occupant.parquet",
        unit="mm",
    )
    session.add(signal)
    session.flush()

    session.expire_all()
    reloaded = session.get(SimulationRun, run.id)
    assert reloaded is not None
    assert reloaded.run_id == "RUN-0041"
    assert reloaded.quality_status == "PASS"

    reloaded_artifact = session.get(Artifact, artifact.id)
    assert reloaded_artifact is not None
    assert reloaded_artifact.simulation_run_id == run.id

    reloaded_model_version = session.get(ModelVersion, model_version.id)
    assert reloaded_model_version is not None
    assert reloaded_model_version.source_artifact_id == artifact.id
