from packages.analysis.global_response import compare_global_response
from packages.analysis.synthetic import generate_scenario


def test_scn004_global_pulse_differs_materially() -> None:
    result = generate_scenario("SCN-004")
    comparison = compare_global_response(
        result.run_a.run_id,
        result.run_b.run_id,
        result.run_a.time_s,
        result.run_a.signals["vehicle_pulse"],
        result.run_b.signals["vehicle_pulse"],
    )
    assert comparison.material_difference_detected is True


def test_scn001_global_pulse_unaffected() -> None:
    result = generate_scenario("SCN-001")
    comparison = compare_global_response(
        result.run_a.run_id,
        result.run_b.run_id,
        result.run_a.time_s,
        result.run_a.signals["vehicle_pulse"],
        result.run_b.signals["vehicle_pulse"],
    )
    assert comparison.material_difference_detected is False
