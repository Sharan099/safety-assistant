"""Part C: applicability-grouped prompt context."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.retrieval.applicability_grouping import (
    format_grouped_context,
    should_use_grouped_context,
)


def _doc(cat: str, test: str, text: str) -> dict:
    return {
        "text": text,
        "applies_to_category": [cat],
        "anchorage_test_type": test,
        "regulation": "UN_R14",
    }


def _cite(i: int) -> dict:
    return {
        "marker": f"S{i}",
        "label": f"UN R14 §6.4.{i}",
        "doc_type_label": "Legal regulation",
        "doc_type": "legal_regulation",
    }


def test_grouped_context_splits_distinct_applicability():
    docs = [
        _doc("M1_N1", "three_point_retractor", "Load 1350 daN M1"),
        _doc("M3_N3", "lap_belt", "Load 675 daN M3"),
        _doc("M1_N1", "three_point_no_retractor", "Load 900 daN rear"),
    ]
    cites = [_cite(i) for i in range(1, 4)]
    ctx = format_grouped_context(docs, cites, char_cap=2000)
    assert "APPLICABILITY GROUP G1" in ctx
    assert "APPLICABILITY GROUP G2" in ctx
    assert "[S1]" in ctx and "[S2]" in ctx


def test_should_use_grouped_for_broad_k():
    docs = [_doc("M1_N1", "t", "x") for _ in range(10)]
    assert should_use_grouped_context(docs, breadth_label="broad")
