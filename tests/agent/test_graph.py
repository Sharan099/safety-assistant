"""End-to-end LangGraph investigation runs against real SCN-00x synthetic
runs (ingested by scripts/generate_synthetic_dataset.py).
"""

import uuid

from sqlalchemy.orm import Session

from packages.agent.graph import run_investigation
from packages.domain.investigation import Hypothesis
from tests.agent.conftest import make_investigation
from tests.conftest import requires_db


@requires_db
def test_scn001_produces_a_hypothesis_and_reaches_engineer_review(session: Session) -> None:
    investigation = make_investigation(session, "SCN-001-RUN-A", "SCN-001-RUN-B", "Why did chest deflection increase?")
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
    investigation = make_investigation(session, "SCN-010-RUN-A", "SCN-010-RUN-B", "Did Run B complete normally?")
    final_state = run_investigation(session, investigation.id)

    session.refresh(investigation)
    assert investigation.state == "BLOCKED"
    assert final_state["blocked_reason"] == "quality_gate_failed"
    # Blocked runs never reach hypothesis generation.
    assert "hypotheses" not in final_state or not final_state["hypotheses"]
