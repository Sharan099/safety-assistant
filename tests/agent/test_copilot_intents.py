"""Locks in CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 6's example mappings
and the Phase 13 acceptance-test conversation's five questions.
"""

from packages.agent.copilot_intents import classify_intent, extract_signal


def test_phase6_examples() -> None:
    assert classify_intent("Compare the crash pulse") == "COMPARE_RUNS"
    assert classify_intent("When did belt force first diverge?") == "ANALYZE_DIVERGENCE"
    assert classify_intent("What changed?") == "GENERAL_INVESTIGATION_QUESTION"  # too generic to force a tool
    assert classify_intent("Analyze chest acceleration") == "ANALYZE_SIGNAL"
    assert classify_intent("What does LS-DYNA documentation say?") == "RETRIEVE_KNOWLEDGE"
    assert classify_intent("Find similar cases") == "RETRIEVE_HISTORY"


def test_acceptance_conversation_questions() -> None:
    assert classify_intent("Why is the restraint configuration currently a leading contributor?") == "EXPLAIN_EVIDENCE"
    assert classify_intent("What evidence contradicts this hypothesis?") == "EXPLAIN_EVIDENCE"
    assert (
        classify_intent("I disagree. Investigate torso rotation as an alternative explanation.")
        == "CHALLENGE_HYPOTHESIS"
    )
    assert classify_intent("What should I analyze next?") == "REQUEST_NEXT_ANALYSIS"
    assert classify_intent("Show me the source for your LS-DYNA-related claim.") == "SHOW_SOURCE"


def test_extract_signal_from_challenge_message() -> None:
    assert extract_signal("I disagree. Investigate torso rotation as an alternative explanation.") == "torso_rotation"
    assert extract_signal("Analyze chest acceleration") == "chest_acceleration"
    assert extract_signal("What should I analyze next?") is None


def test_controlled_comparison_intent() -> None:
    assert classify_intent("What controlled simulation should I run?") == "CONTROLLED_COMPARISON"
