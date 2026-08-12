"""End-to-end LangGraph investigation runs against real SCN-00x synthetic
runs (ingested by scripts/generate_synthetic_dataset.py).
"""

import uuid

from sqlalchemy.orm import Session

from packages.agent.graph import run_investigation
from packages.domain.core import Project, SignalDefinition, SimulationRun, User
from packages.domain.investigation import Hypothesis, Investigation, InvestigationRun
from tests.conftest import requires_db


def _make_investigation(session: Session, run_a_id: str, run_b_id: str, question: str) -> Investigation:
    project = session.query(Project).filter_by(name="Default Project").one_or_none()
    if project is None:
        from packages.domain.core import Organization

        org = Organization(name="__test_org")
        session.add(org)
        session.flush()
        project = Project(organization_id=org.id, name="__test_project")
        session.add(project)
        session.flush()

    user = session.query(User).filter_by(email="engineer@local").one_or_none()
    if user is None:
        user = User(organization_id=project.organization_id, email="__test@local", display_name="Test", role="ENGINEER")
        session.add(user)
        session.flush()

    signal_def = session.query(SignalDefinition).filter_by(canonical_name="chest_deflection").one()
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


@requires_db
def test_scn001_produces_a_hypothesis_and_reaches_engineer_review(session: Session) -> None:
    investigation = _make_investigation(session, "SCN-001-RUN-A", "SCN-001-RUN-B", "Why did chest deflection increase?")
    final_state = run_investigation(session, investigation.id)

    session.refresh(investigation)
    assert investigation.state == "ENGINEER_REVIEW"
    assert final_state.get("blocked_reason") is None
    assert len(final_state["hypotheses"]) == 1

    hypothesis = session.get(Hypothesis, uuid.UUID(final_state["hypotheses"][0]["id"]))
    assert hypothesis is not None
    assert "force_limiter" in str(hypothesis.affected_components) or "restraint" in hypothesis.title.lower()


@requires_db
def test_scn010_quality_failure_blocks_the_agent(session: Session) -> None:
    investigation = _make_investigation(session, "SCN-010-RUN-A", "SCN-010-RUN-B", "Did Run B complete normally?")
    final_state = run_investigation(session, investigation.id)

    session.refresh(investigation)
    assert investigation.state == "BLOCKED"
    assert final_state["blocked_reason"] == "quality_gate_failed"
    # Blocked runs never reach hypothesis generation.
    assert "hypotheses" not in final_state or not final_state["hypotheses"]
