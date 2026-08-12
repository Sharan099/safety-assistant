from packages.analysis.signals import analyze_signal_pair, calculate_signal_features, detect_first_divergence
from packages.analysis.synthetic import generate_scenario


def test_calculate_signal_features_matches_known_pulse() -> None:
    result = generate_scenario("SCN-001")
    features = calculate_signal_features(
        result.run_a.signals["chest_acceleration"],
        result.run_a.time_s,
        run_id=result.run_a.run_id,
        signal_name="chest_acceleration",
    )
    assert 40.0 < features.peak < 50.0  # baseline peak is 45.0 g, +/- noise
    assert 40.0 < features.time_to_peak_ms < 50.0  # baseline t_peak is 0.045s
    assert features.provenance.algorithm == "calculate_signal_features"


def test_scn001_belt_force_and_deflection_diverge() -> None:
    result = generate_scenario("SCN-001")
    for name in ("belt_force", "chest_deflection"):
        event = detect_first_divergence(
            result.run_a.signals[name],
            result.run_b.signals[name],
            result.run_a.time_s,
            signal_name=name,
            run_a_id=result.run_a.run_id,
            run_b_id=result.run_b.run_id,
        )
        assert event is not None, name
        assert 0.0 < event.time_ms < 150.0


def test_scn001_unaffected_signal_does_not_diverge() -> None:
    result = generate_scenario("SCN-001")
    event = detect_first_divergence(
        result.run_a.signals["torso_rotation"],
        result.run_b.signals["torso_rotation"],
        result.run_a.time_s,
        signal_name="torso_rotation",
        run_a_id=result.run_a.run_id,
        run_b_id=result.run_b.run_id,
    )
    assert event is None


def test_analyze_signal_pair_bundles_features_and_divergence() -> None:
    result = generate_scenario("SCN-001")
    analysis = analyze_signal_pair(
        "belt_force",
        result.run_a.run_id,
        result.run_b.run_id,
        result.run_a.time_s,
        result.run_a.signals["belt_force"],
        result.run_b.signals["belt_force"],
    )
    assert analysis.divergence is not None
    assert -1.0 <= analysis.correlation <= 1.0
