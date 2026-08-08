"""Deterministic compliance path: parse → lookup → Python compare."""

from __future__ import annotations

from pathlib import Path

from generation.compliance import (
    compare_values,
    evaluate_compliance,
    extract_measured_criteria,
    is_compliance_check_query,
    render_compliance_answer,
)
from ingestion.extract_limits import seed_known_limits, spot_check


COMPOSITE_Q = (
    "The test recorded an HPC of 920, chest compression of 32 mm, "
    "and fuel leakage of 40 g/min. Does the vehicle pass?"
)


def test_seed_and_spot_check(tmp_path: Path):
    seed_known_limits(directory=tmp_path)
    results = spot_check(directory=tmp_path)
    assert all(r["ok"] for r in results), results


def test_classifier_and_extract_composite():
    assert is_compliance_check_query(COMPOSITE_Q)
    measured = extract_measured_criteria(COMPOSITE_Q)
    names = " ".join(m.criterion_query.lower() for m in measured)
    assert len(measured) == 3
    assert "hpc" in names
    assert "chest" in names or "compression" in names
    assert "fuel" in names or "leakage" in names
    values = {round(m.measured_value, 5) for m in measured}
    assert 920 in values
    assert 32 in values
    assert 40 in values


def test_compare_operators():
    assert compare_values(35, "<=", 30) == "FAIL"
    assert compare_values(30, "<=", 30) == "PASS"
    assert compare_values(920, "<=", 1000) == "PASS"
    assert compare_values(40, "<=", 30) == "FAIL"


def test_composite_evaluates_all_three_including_fuel(tmp_path: Path, monkeypatch):
    seed_known_limits(directory=tmp_path)
    monkeypatch.setenv("LIMITS_DIR", str(tmp_path))
    # Clear any cached path assumptions by re-seeding via env.
    from ingestion import extract_limits as el

    monkeypatch.setattr(el, "DEFAULT_LIMITS_DIR", tmp_path)

    result = evaluate_compliance(COMPOSITE_Q)
    assert result is not None
    assert len(result.criteria) == 3
    by_name = {c.criterion.lower(): c for c in result.criteria}
    # Fuel must not be silently dropped.
    fuel = next(c for c in result.criteria if "fuel" in c.criterion.lower() or "leak" in c.criterion.lower())
    assert fuel.verdict == "FAIL"
    assert fuel.measured == 40
    assert fuel.limit == 30
    assert fuel.verdict != "LIMIT_NOT_FOUND"

    hpc = next(c for c in result.criteria if "head" in c.criterion.lower() or c.measured == 920)
    assert hpc.verdict == "PASS"
    assert hpc.measured == 920

    chest = next(c for c in result.criteria if c.measured == 32)
    assert chest.verdict == "PASS"
    assert chest.limit == 42

    # Overall FAIL because fuel failed — not an implied pass from HPC+ThCC alone.
    assert result.overall_verdict == "FAIL"
    text = render_compliance_answer(result)
    assert "40" in text
    assert "30" in text
    assert "FAIL" in text
    assert "limit not found" not in text.lower()


def test_missing_limit_flagged_explicitly(tmp_path: Path, monkeypatch):
    seed_known_limits(directory=tmp_path)
    monkeypatch.setattr("ingestion.extract_limits.DEFAULT_LIMITS_DIR", tmp_path)
    monkeypatch.setenv("LIMITS_DIR", str(tmp_path))
    q = "Our custom FooCriterion was 12 mm. Does the vehicle pass UN R95?"
    # May not classify if FooCriterion isn't a known cue — force via fuel-style
    # unknown alias by using a known unit pattern with nonsense name won't work.
    # Instead: delete fuel from table and ask fuel question.
    from ingestion.extract_limits import LimitsTable, load_limits_table, save_limits_table

    table = load_limits_table("UN-ECE-R95", directory=tmp_path)
    assert table is not None
    table.limits = [r for r in table.limits if "fuel" not in r.criterion_name.lower()]
    save_limits_table(table, directory=tmp_path)

    result = evaluate_compliance(
        "If the fuel leakage rate is 40 g/min, does the vehicle pass?",
        regulation_id="UN-ECE-R95",
    )
    assert result is not None
    assert any(c.verdict == "LIMIT_NOT_FOUND" for c in result.criteria)
    assert "limit not found" in result.answer_text.lower()
