"""compare_global_response() — PRD.md PR-005.

Compares global crash response (currently: vehicle crash pulse — the only
global-scale signal the synthetic benchmark models; a real solver adapter
would add barrier/load response, energy, intrusion, structural deformation)
before any occupant-response hypothesis is allowed to attribute a difference
to a local component. PR-005: "Before attributing an occupant response to a
local component, compare relevant global behavior."
"""

from __future__ import annotations

import numpy as np

from packages.analysis.models import GlobalResponseComparison, GlobalResponseMetric, Provenance
from packages.analysis.signals import calculate_signal_features

ALGORITHM_VERSION = "compare_global_response v0.1.0"

# A global metric differing by more than this fraction of Run A's value is
# "materially different" — below it, treat as noise-level variation.
MATERIAL_DELTA_FRACTION = 0.10


def compare_global_response(
    run_a_id: str,
    run_b_id: str,
    time_s: np.ndarray,
    global_signal_a: np.ndarray,
    global_signal_b: np.ndarray,
    *,
    signal_name: str = "vehicle_pulse",
) -> GlobalResponseComparison:
    features_a = calculate_signal_features(global_signal_a, time_s, run_id=run_a_id, signal_name=signal_name)
    features_b = calculate_signal_features(global_signal_b, time_s, run_id=run_b_id, signal_name=signal_name)

    metrics: list[GlobalResponseMetric] = []
    for metric_name, value_a, value_b in [
        (f"{signal_name}_peak", features_a.peak, features_b.peak),
        (f"{signal_name}_time_to_peak_ms", features_a.time_to_peak_ms, features_b.time_to_peak_ms),
    ]:
        delta = value_b - value_a
        delta_pct = (delta / abs(value_a)) * 100.0 if value_a else None
        material = abs(delta) > MATERIAL_DELTA_FRACTION * abs(value_a) if value_a else value_b != 0
        metrics.append(
            GlobalResponseMetric(
                name=metric_name,
                run_a_value=value_a,
                run_b_value=value_b,
                delta=delta,
                delta_pct=delta_pct,
                materially_different=material,
                provenance=Provenance(
                    algorithm="compare_global_response",
                    algorithm_version=ALGORITHM_VERSION,
                    parameters={"material_delta_fraction": MATERIAL_DELTA_FRACTION},
                ),
            )
        )

    return GlobalResponseComparison(
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        metrics=metrics,
        material_difference_detected=any(m.materially_different for m in metrics),
    )
