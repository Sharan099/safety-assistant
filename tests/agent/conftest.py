"""Shared test helper for tests/agent — creates an Investigation row against
real seeded synthetic runs, without going through the API layer.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from packages.domain.core import Organization, Project, SignalDefinition, SimulationRun, User
from packages.domain.investigation import Investigation, InvestigationRun


def make_investigation(
    session: Session, run_a_id: str, run_b_id: str, question: str, *, primary_metric: str = "chest_deflection"
) -> Investigation:
    project = session.query(Project).filter_by(name="__test_project").one_or_none()
    if project is None:
        org = Organization(name="__test_org")
        session.add(org)
        session.flush()
        project = Project(organization_id=org.id, name="__test_project")
        session.add(project)
        session.flush()

    user = session.query(User).filter_by(email="__test@local").one_or_none()
    if user is None:
        user = User(organization_id=project.organization_id, email="__test@local", display_name="Test", role="ENGINEER")
        session.add(user)
        session.flush()

    signal_def = session.query(SignalDefinition).filter_by(canonical_name=primary_metric).one()
    run_a = session.query(SimulationRun).filter_by(run_id=run_a_id).one()
    run_b = session.query(SimulationRun).filter_by(run_id=run_b_id).one()

    investigation = Investigation(
        project_id=project.id,
        created_by=user.id,
        title=f"{run_a_id} vs {run_b_id}",
        question=question,
        primary_metric_id=signal_def.id,
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
    session.flush()
    return investigation
