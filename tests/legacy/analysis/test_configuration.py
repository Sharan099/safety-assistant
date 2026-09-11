from packages.analysis.configuration import changed_paths, compare_configuration
from packages.analysis.synthetic import generate_scenario


def test_scn001_belt_paths_changed_others_same() -> None:
    result = generate_scenario("SCN-001")
    diffs = compare_configuration(result.run_a.config, result.run_b.config)
    changed = set(changed_paths(diffs))
    assert "restraint.force_limiter.level_n" in changed
    assert "restraint.belt.webbing_revision" in changed
    assert "seat.fore_aft_position_mm" not in changed
    assert "dummy.posture" not in changed

    limiter = next(d for d in diffs if d.path == "restraint.force_limiter.level_n")
    assert limiter.run_a_value == 4000
    assert limiter.run_b_value == 3500
    assert limiter.change_status == "CHANGED"
    # A pure diff cannot know intent — PRD.md PR-006 classifications require
    # a changelog/reviewer this function does not have.
    assert limiter.change_classification == "UNKNOWN"


def test_scn009_no_configuration_change_recorded() -> None:
    result = generate_scenario("SCN-009")
    diffs = compare_configuration(result.run_a.config, result.run_b.config)
    assert changed_paths(diffs) == []
    assert all(d.change_status == "SAME" for d in diffs)
