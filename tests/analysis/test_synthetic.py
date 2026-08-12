"""Phase-5 gate: the synthetic benchmark itself is deterministic and complete.

PRD.md §13 requires each of the 10 scenarios to carry Run A, Run B, a known
changed factor, expected signal changes, and allowed/disallowed conclusions.
"""

import numpy as np

from packages.analysis.synthetic import DEFAULT_PARAMS, generate_all, generate_scenario

EXPECTED_SCENARIO_IDS = [f"SCN-{i:03d}" for i in range(1, 11)]


def test_ten_scenarios_generated() -> None:
    results = generate_all()
    assert [r.spec.scenario_id for r in results] == EXPECTED_SCENARIO_IDS


def test_every_scenario_has_ground_truth() -> None:
    for result in generate_all():
        spec = result.spec
        assert spec.changed_factor
        assert spec.description
        assert spec.allowed_conclusions, spec.scenario_id
        assert spec.disallowed_conclusions, spec.scenario_id
        # SCN-010 (quality failure) legitimately expects zero signal changes —
        # the point of that scenario is that no signal claim should be made.
        if spec.scenario_id != "SCN-010":
            assert spec.expected_signal_changes, spec.scenario_id


def test_runs_carry_all_baseline_signals() -> None:
    for result in generate_all():
        for run in (result.run_a, result.run_b):
            assert set(run.signals) == set(DEFAULT_PARAMS)
            for array in run.signals.values():
                assert array.shape == run.time_s.shape
                assert np.isfinite(array).all()


def test_generation_is_deterministic() -> None:
    first = generate_scenario("SCN-001")
    second = generate_scenario("SCN-001")
    for name in first.run_a.signals:
        np.testing.assert_array_equal(first.run_a.signals[name], second.run_a.signals[name])
        np.testing.assert_array_equal(first.run_b.signals[name], second.run_b.signals[name])


def test_scn001_belt_force_and_deflection_differ_others_do_not() -> None:
    result = generate_scenario("SCN-001")
    a, b = result.run_a.signals, result.run_b.signals
    for name in set(DEFAULT_PARAMS) - {"belt_force", "chest_deflection", "chest_acceleration"}:
        # unaffected signals: same generating parameters, only RNG noise differs
        assert abs(float(a[name].max()) - float(b[name].max())) < 3.0, name
    assert float(b["belt_force"].max()) < float(a["belt_force"].max())
    assert float(b["chest_deflection"].max()) > float(a["chest_deflection"].max())


def test_scn010_run_b_fails_quality_raw() -> None:
    result = generate_scenario("SCN-010")
    assert result.run_b.quality_raw["termination"] == "ERROR"
    assert result.run_b.quality_raw["final_time_s"] < result.run_b.quality_raw["target_time_s"]
    assert result.run_a.quality_raw["termination"] == "NORMAL"
