"""Tests for sub-clause merging in clause_splitter."""

from ingestion.clause_splitter import (
    merge_subclause_blocks,
    merge_trailing_subclauses,
    split_body_by_clauses,
)


def test_merge_trailing_subclauses_attaches_pdf_style_subclause():
    blocks = [
        ("2.12", "§2.12", '"Emergency locking retractor (type 4)" actuated in an emergency by:', None),
        ("2.12.4.1", "§2.12.4.1", "Deceleration of the vehicle (single sensitivity).", None),
        ("2.12.4.2", "§2.12.4.2", "A combination of deceleration of the vehicle.", None),
        ("2.12.5", "§2.12.5", "Next clause", None),
    ]
    merged = merge_trailing_subclauses(blocks)
    assert len(merged) == 2
    assert "Deceleration of the vehicle (single sensitivity)" in merged[0][2]
    assert merged[1][0] == "2.12.5"


def test_merge_subclause_blocks_keeps_elr_with_test_conditions():
    body = (
        "2.12.4.\n"
        '"Emergency locking retractor (type 4)" means a retractor actuated in an emergency by:\n'
        "2.12.4.1.\n"
        "Deceleration of the vehicle (single sensitivity).\n"
        "2.12.4.2.\n"
        "A combination of deceleration of the vehicle.\n"
        "2.12.5.\n"
        '"Emergency locking retractor with higher response threshold"\n'
    )
    blocks = split_body_by_clauses(body)
    merged = merge_subclause_blocks(blocks)
    elr = next(b for b in merged if b[0] == "2.12.4")
    text = elr[2]
    assert "Emergency locking retractor (type 4)" in text
    assert "2.12.4.1" in text
    assert "Deceleration of the vehicle (single sensitivity)" in text
    assert "2.12.5" not in text
