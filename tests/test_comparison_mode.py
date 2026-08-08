"""Comparison-mode classifier and topic query helpers."""

from __future__ import annotations

from retrieval.comparison import (
    comparison_side_query,
    detect_named_regulations,
    has_comparative_cue,
    is_comparison_mode_query,
)
from retrieval.enumerative import is_comparative_query


def test_comparison_mode_detects_differ_from_and_compare():
    assert is_comparison_mode_query(
        "How does UN R95 differ from UN R94 in impact direction?"
    )
    assert is_comparison_mode_query("Compare UN R94 and UN R95 in one sentence.")
    assert is_comparison_mode_query(
        "Compare UN Regulation No. 94 and UN Regulation No. 95 in terms of "
        "collision direction addressed, test objective, and primary occupant "
        "injury assessment focus."
    )
    assert is_comparison_mode_query(
        "Why are UN Regulation No. 16 safety-belt and retractor requirements "
        "relevant when assessing frontal occupant protection tested under UN "
        "Regulation No. 94?"
    )
    assert is_comparison_mode_query("Compare UN R94 and UN R95 injury criteria.")
    assert is_comparison_mode_query("Compare UN R94 and UN R95 impact direction.")


def test_comparison_mode_rejects_single_reg_lookup():
    assert not is_comparison_mode_query("What is the HPC limit in UN R94?")
    assert not is_comparison_mode_query("What vehicles are covered under UN R95?")
    # Comparative cue alone without two named regs is not pairwise comparison mode.
    assert not is_comparison_mode_query("Compare injury criteria in one sentence.")


def test_detect_named_regulations_order():
    regs = detect_named_regulations(
        "How does UN R95 differ from UN R94 in impact direction?"
    )
    assert regs == ["UN-ECE-R95", "UN-ECE-R94"]
    assert has_comparative_cue("differ from")
    assert is_comparative_query(
        "How does UN R95 differ from UN R94 in impact direction?"
    )


def test_comparison_side_query_keeps_topic():
    topic = comparison_side_query(
        "How does UN R95 differ from UN R94 in impact direction?"
    )
    low = topic.lower()
    assert "impact" in low or "direction" in low
    assert "frontal" in low and "lateral" in low
    assert "R94" not in topic and "R95" not in topic


def test_comparison_side_query_one_sentence_defaults_to_crash_contrast():
    topic = comparison_side_query("Compare UN R94 and UN R95 in one sentence.")
    low = topic.lower()
    assert "frontal" in low
    assert "lateral" in low or "side" in low


def test_comparison_side_query_for_regulation_biases_crash_type():
    from retrieval.comparison import comparison_side_query_for_regulation

    q = "How does UN R95 differ from UN R94 in impact direction?"
    r94 = comparison_side_query_for_regulation(q, "UN-ECE-R94").lower()
    r95 = comparison_side_query_for_regulation(q, "UN-ECE-R95").lower()
    assert "frontal" in r94
    assert "lateral" in r95 or "side" in r95


def test_balance_comparison_chunks_round_robin():
    from types import SimpleNamespace

    from retrieval.comparison import balance_comparison_chunks

    per = {
        "UN-ECE-R94": [
            SimpleNamespace(chunk_id="a", regulation_id="UN-ECE-R94"),
            SimpleNamespace(chunk_id="b", regulation_id="UN-ECE-R94"),
        ],
        "UN-ECE-R95": [
            SimpleNamespace(chunk_id="c", regulation_id="UN-ECE-R95"),
            SimpleNamespace(chunk_id="d", regulation_id="UN-ECE-R95"),
        ],
    }
    merged = balance_comparison_chunks(per, max_total=3)
    regs = [c.regulation_id for c in merged]
    assert regs.count("UN-ECE-R94") >= 1
    assert regs.count("UN-ECE-R95") >= 1
