"""Unit tests for definition-seeking vs compliance prefer-limit gating."""

from retrieval.value_limit import (
    is_compliance_prefer_limit_query,
    is_definition_seeking_query,
)


def test_definition_seeking_reess_and_isolation():
    assert is_definition_seeking_query("What does REESS mean?")
    assert is_definition_seeking_query("Define isolation resistance in UN R94.")
    assert is_definition_seeking_query(
        "What does Soft Tissue Criterion (VC) mean in UN R95?"
    )
    assert not is_definition_seeking_query(
        "Side impact Soft Tissue Criterion VC was 0.8 m/s. Does this satisfy UN R95?"
    )


def test_prefer_limit_skips_definitions_keeps_compliance():
    assert not is_compliance_prefer_limit_query("What does REESS mean?")
    assert not is_compliance_prefer_limit_query("Define isolation resistance.")
    assert not is_compliance_prefer_limit_query(
        "What does Soft Tissue Criterion (VC) mean in UN R95?"
    )
    assert is_compliance_prefer_limit_query(
        "REESS remained mounted, but electrolyte entered the passenger "
        "compartment. Does the vehicle comply?"
    )
    assert is_compliance_prefer_limit_query(
        "Side impact Soft Tissue Criterion VC was 0.8 m/s. Does this satisfy UN R95?"
    )
    # Named criterion + limit language still prefers limits.
    assert is_compliance_prefer_limit_query("What is the HPC limit in UN R95?")
    # Bare named criterion without verdict/limit cues must not demote definitions.
    assert not is_compliance_prefer_limit_query("Tell me about Soft Tissue Criterion VC")
