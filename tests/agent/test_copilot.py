"""The Copilot LangGraph — CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 12
(agent: tool selection, evidence grounding, hypothesis challenge, unknown
handling) and Phase 13 (the full acceptance-test conversation).
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from packages.agent.copilot import run_copilot_turn, stream_copilot_turn
from packages.agent.graph import run_investigation
from packages.agent.llm import LLMMessage, LLMResponse, LLMUnavailableError
from packages.domain.copilot import CopilotConversation, CopilotMessage, CopilotToolCall
from packages.domain.investigation import Hypothesis
from tests.agent.conftest import make_investigation
from tests.conftest import requires_db


class _AlwaysFailsLLM:
    """Simulates an LLM provider that's configured but unreachable —
    TRD.md Section 30 / PRD_COPILOT_UPDATE.md Section 11."""

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.2, max_tokens: int = 1024) -> LLMResponse:
        raise LLMUnavailableError("simulated provider outage")


def _scn001_investigation(session: Session) -> uuid.UUID:
    investigation = make_investigation(session, "SCN-001-RUN-A", "SCN-001-RUN-B", "Why did chest deflection increase?")
    run_investigation(session, investigation.id)
    return investigation.id


@requires_db
def test_acceptance_conversation_scn001(session: Session) -> None:
    """CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 13, verbatim."""
    investigation_id = _scn001_investigation(session)

    r1 = run_copilot_turn(
        session, investigation_id, "Why is the restraint configuration currently a leading contributor?"
    )
    assert r1["intent"] == "EXPLAIN_EVIDENCE"
    assert r1["response"]
    assert (
        "restraint" in r1["response"].lower()
        or "force_limiter" in r1["response"].lower()
        or "force-limiter" in r1["response"].lower()
    )

    r2 = run_copilot_turn(session, investigation_id, "What evidence contradicts this hypothesis?")
    assert r2["intent"] == "EXPLAIN_EVIDENCE"
    # SCN-001's hypothesis has no CONTRADICTS-linked evidence — must say so,
    # and explicitly distinguish it from "proven" (PRD_COPILOT_UPDATE.md
    # Section 6), not just stay silent on the distinction.
    assert "no direct contradiction" in r2["response"].lower()
    assert "does not mean" in r2["response"].lower()

    r3 = run_copilot_turn(
        session, investigation_id, "I disagree. Investigate torso rotation as an alternative explanation."
    )
    assert r3["intent"] == "CHALLENGE_HYPOTHESIS"
    assert r3.get("new_hypothesis_id")
    new_hypothesis = session.get(Hypothesis, uuid.UUID(r3["new_hypothesis_id"]))
    assert new_hypothesis is not None
    assert new_hypothesis.status == "PROPOSED"  # never auto-finalized
    assert "torso" in new_hypothesis.title.lower()

    r4 = run_copilot_turn(session, investigation_id, "What should I analyze next?")
    assert r4["intent"] == "REQUEST_NEXT_ANALYSIS"
    assert r4["response"]

    r5 = run_copilot_turn(session, investigation_id, "Show me the source for your LS-DYNA-related claim.")
    assert r5["intent"] == "SHOW_SOURCE"
    assert r5["response"]

    # Every turn persisted.
    conversation = session.query(CopilotConversation).filter_by(investigation_id=investigation_id).one()
    messages = (
        session.query(CopilotMessage)
        .filter_by(conversation_id=conversation.id)
        .order_by(CopilotMessage.created_at)
        .all()
    )
    assert len(messages) == 10  # 5 user + 5 assistant
    assert [m.role for m in messages] == ["user", "assistant"] * 5


@requires_db
def test_tool_selection_compare_runs(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    result = run_copilot_turn(session, investigation_id, "Compare the crash pulse")
    assert result["intent"] == "COMPARE_RUNS"
    tool_names = {t["tool_name"] for t in result["tool_calls"]}
    assert "compare_global_response" in tool_names


@requires_db
def test_tool_selection_analyze_divergence(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    result = run_copilot_turn(session, investigation_id, "When did belt force first diverge?")
    assert result["intent"] == "ANALYZE_DIVERGENCE"
    tool_names = {t["tool_name"] for t in result["tool_calls"]}
    assert "analyze_signal" in tool_names


@requires_db
def test_evidence_grounding_response_cites_real_evidence_ids(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    result = run_copilot_turn(session, investigation_id, "Why is this hypothesis leading?")
    assert result["evidence_refs"]
    # Every cited ID must be a real, parseable UUID — not invented text.
    for ref in result["evidence_refs"]:
        uuid.UUID(ref)


@requires_db
def test_unknown_handling_no_historical_cases(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    result = run_copilot_turn(session, investigation_id, "Find similar historical cases")
    assert result["intent"] == "RETRIEVE_HISTORY"
    assert result["unknowns"]
    assert "no historical cases" in result["response"].lower()


@requires_db
def test_llm_failure_does_not_block_deterministic_response(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    result = run_copilot_turn(session, investigation_id, "Why is this hypothesis leading?", llm=_AlwaysFailsLLM())
    assert result["llm_degraded"] is True
    assert result["response"]  # deterministic text still returned


@requires_db
def test_invalid_investigation_raises(session: Session) -> None:
    with pytest.raises(ValueError, match="not found"):
        run_copilot_turn(session, uuid.uuid4(), "hello")


@requires_db
def test_empty_message_rejected(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    with pytest.raises(ValueError, match="empty"):
        run_copilot_turn(session, investigation_id, "   ")


@requires_db
def test_stream_yields_one_step_per_node_then_final(session: Session) -> None:
    """This is what gives the UI real-time workflow visibility — verify the
    generator actually yields incrementally (one event per graph node), not
    just a single event at the end."""
    investigation_id = _scn001_investigation(session)

    events = list(stream_copilot_turn(session, investigation_id, "Compare the crash pulse"))

    step_events = [e for e in events if e.type == "step"]
    final_events = [e for e in events if e.type == "final"]

    assert [e.node for e in step_events] == ["load_context", "classify_intent", "dispatch_tools", "ground_response"]
    assert all(e.detail for e in step_events)
    assert len(final_events) == 1
    assert final_events[0].state is not None
    assert final_events[0].state["intent"] == "COMPARE_RUNS"
    # The final event comes after every step event, not interleaved.
    assert events[-1].type == "final"


@requires_db
def test_tool_calls_persisted(session: Session) -> None:
    investigation_id = _scn001_investigation(session)
    run_copilot_turn(session, investigation_id, "Compare the crash pulse")

    conversation = session.query(CopilotConversation).filter_by(investigation_id=investigation_id).one()
    assistant_message = (
        session.query(CopilotMessage)
        .filter_by(conversation_id=conversation.id, role="assistant")
        .order_by(CopilotMessage.created_at.desc())
        .first()
    )
    assert assistant_message is not None
    tool_calls = session.query(CopilotToolCall).filter_by(message_id=assistant_message.id).all()
    assert tool_calls
    assert tool_calls[0].tool_name == "compare_global_response"
    assert tool_calls[0].status == "SUCCESS"
