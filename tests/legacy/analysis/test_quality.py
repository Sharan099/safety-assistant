from packages.analysis.quality import run_quality_gate
from packages.analysis.synthetic import generate_scenario


def test_nominal_run_passes() -> None:
    result = generate_scenario("SCN-001")
    summary = run_quality_gate(result.run_a.run_id, result.run_a.quality_raw)
    assert summary.overall_status == "PASS"
    assert all(c.status == "PASS" for c in summary.checks)


def test_scn007_contact_energy_warning() -> None:
    result = generate_scenario("SCN-007")
    summary = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)
    assert summary.overall_status == "WARNING"
    contact = next(c for c in summary.checks if c.check_type == "contact_energy")
    assert contact.status == "WARNING"


def test_scn010_run_b_fails() -> None:
    result = generate_scenario("SCN-010")
    summary = run_quality_gate(result.run_b.run_id, result.run_b.quality_raw)
    assert summary.overall_status == "FAIL"
    statuses = {c.check_type: c.status for c in summary.checks}
    assert statuses["solver_termination"] == "FAIL"
    assert statuses["final_time"] == "FAIL"
    assert statuses["result_database_completeness"] == "FAIL"
    assert statuses["errors_warnings"] == "FAIL"


def test_missing_diagnostics_are_unknown_not_pass() -> None:
    summary = run_quality_gate("RUN-X", None)
    assert summary.overall_status == "UNKNOWN"
    assert all(c.status == "UNKNOWN" for c in summary.checks)
