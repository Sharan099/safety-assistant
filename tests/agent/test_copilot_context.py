"""build_investigation_context() — correctness and cross-investigation
isolation (CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 2/12).
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from packages.agent.copilot_context import build_investigation_context
from packages.agent.graph import run_investigation
from tests.agent.conftest import make_investigation
from tests.conftest import requires_db


@requires_db
def test_context_reflects_investigation_state_after_agent_run(session: Session) -> None:
    investigation = make_investigation(session, "SCN-001-RUN-A", "SCN-001-RUN-B", "Why did chest deflection increase?")
    run_investigation(session, investigation.id)

    context = build_investigation_context(session, investigation.id)

    assert context.investigation_id == investigation.id
    assert context.question == "Why did chest deflection increase?"
    assert context.investigation_state == "ENGINEER_REVIEW"
    assert context.run_a["run_id"] == "SCN-001-RUN-A"
    assert context.run_b["run_id"] == "SCN-001-RUN-B"
    assert context.primary_metric == "chest_deflection"

    assert context.quality_results["SCN-001-RUN-A"]
    assert all(c["status"] == "PASS" for c in context.quality_results["SCN-001-RUN-A"])

    assert any(c["dimension"] == "causal_isolation" for c in context.comparability_results)
    assert any(d["path"].startswith("restraint") for d in context.configuration_diffs)
    assert "chest_deflection" in context.selected_signals
    assert any(e["signal"] == "chest_deflection" for e in context.divergence_events)

    assert context.current_evidence
    assert context.current_hypotheses
    assert context.current_hypotheses[0]["title"]

    assert context.engineer_review_state is None  # no review submitted yet in this test


@requires_db
def test_context_has_no_cross_investigation_leakage(session: Session) -> None:
    inv_1 = make_investigation(session, "SCN-001-RUN-A", "SCN-001-RUN-B", "Investigation one")
    inv_2 = make_investigation(session, "SCN-002-RUN-A", "SCN-002-RUN-B", "Investigation two")

    run_investigation(session, inv_1.id)
    run_investigation(session, inv_2.id)

    context_1 = build_investigation_context(session, inv_1.id)
    context_2 = build_investigation_context(session, inv_2.id)

    ids_1 = {e["id"] for e in context_1.current_evidence}
    ids_2 = {e["id"] for e in context_2.current_evidence}
    assert ids_1.isdisjoint(ids_2)

    assert context_1.run_a["run_id"] == "SCN-001-RUN-A"
    assert context_2.run_a["run_id"] == "SCN-002-RUN-A"
    assert context_1.question == "Investigation one"
    assert context_2.question == "Investigation two"


@requires_db
def test_context_raises_for_unknown_investigation(session: Session) -> None:
    with pytest.raises(ValueError, match="not found"):
        build_investigation_context(session, uuid.uuid4())
