"""Part A extraction fidelity — quick structural gate (no full Docling OCR in CI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_extraction_fidelity import (  # noqa: E402
    CROSS_REFS,
    REPORT_PATH,
    REQUIRED_TABLES,
    check_cross_references,
    check_table_structure,
    run_audit,
)


@pytest.fixture(scope="module")
def r14_md() -> str:
    path = ROOT / "output" / "markdown" / "UN_R14.md"
    if not path.is_file():
        pytest.skip("UN_R14 markdown not available")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def r16_md() -> str:
    path = ROOT / "output" / "markdown" / "UN_R16.md"
    if not path.is_file():
        pytest.skip("UN_R16 markdown not available")
    return path.read_text(encoding="utf-8")


def test_r14_r16_cross_references_present(r14_md, r16_md):
    for spec in CROSS_REFS:
        md = r14_md if spec.regulation == "UN_R14" else r16_md
        issues = check_cross_references(md, spec)
        blocking = [i for i in issues if i.get("severity") == "blocking"]
        assert not blocking, blocking


def test_critical_numeric_anchors_in_markdown(r14_md, r16_md):
    for token in ("1,350", "0.2 second", "450"):
        assert token in r14_md
    for token in ("2.5", "5,000", "Hz"):
        assert token in r16_md


def test_annex6_content_present_even_if_not_pipe_table(r14_md):
    """Annex 6 body must exist; structured pipe table is a separate blocking check."""
    assert "Annex 6" in r14_md
    assert "M1" in r14_md
    assert "Ø" in r14_md or "O:" in r14_md  # symbol or legend


def test_quick_audit_writes_report():
    report = run_audit(mode="quick")
    assert REPORT_PATH.is_file()
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert data["mode"] == "quick"
    assert "discrepancies" in data


def test_table_structure_issues_detected_for_linearized_annex6(r14_md):
    """Documents known Part A finding: Annex 6 is linearized, not pipe-table."""
    spec = next(s for s in REQUIRED_TABLES if s.name.startswith("Annex 6 anchorage"))
    issues = check_table_structure(r14_md, spec)
    assert any(i["type"] == "table_not_structured" for i in issues)
