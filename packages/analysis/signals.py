"""calculate_signal_features() / detect_first_divergence() — PRD.md PR-007/PR-008.

PR-008 is explicit: "The system must never infer the target metric's
divergence from another signal's divergence without evidence." Every call
here is per-signal and independent; nothing in this module aggregates or
propagates a divergence result across signals — that inference, if ever
made, belongs to the agent/hypothesis layer, and only as a labelled
hypothesis, never a fact.
"""

from __future__ import annotations

import numpy as np

from packages.analysis.models import DivergenceEvent, Provenance, SignalAnalysisResult, SignalFeatureSet

FEATURES_ALGORITHM_VERSION = "calculate_signal_features v0.1.0"
DIVERGENCE_ALGORITHM_VERSION = "detect_first_divergence v0.1.0"

DEFAULT_DIVERGENCE_THRESHOLD_FRACTION = 0.10
DEFAULT_MIN_SUSTAIN_SAMPLES = 5


def calculate_signal_features(
    signal: np.ndarray, time_s: np.ndarray, *, run_id: str, signal_name: str
) -> SignalFeatureSet:
    peak_idx = int(np.argmax(np.abs(signal)))
    peak = float(signal[peak_idx])
    time_to_peak_ms = float(time_s[peak_idx] * 1000.0)

    half_peak = abs(peak) / 2.0
    above_half = np.abs(signal) >= half_peak
    duration_ms: float | None = None
    if above_half.any():
        idx = np.flatnonzero(above_half)
        duration_ms = float((time_s[idx[-1]] - time_s[idx[0]]) * 1000.0)

    rise_time_ms: float | None = None
    ten_pct, ninety_pct = abs(peak) * 0.10, abs(peak) * 0.90
    pre_peak = np.abs(signal[: peak_idx + 1])
    above_ten = np.flatnonzero(pre_peak >= ten_pct)
    above_ninety = np.flatnonzero(pre_peak >= ninety_pct)
    if above_ten.size and above_ninety.size:
        rise_time_ms = float((time_s[above_ninety[0]] - time_s[above_ten[0]]) * 1000.0)

    integral = float(np.trapezoid(signal, time_s))

    return SignalFeatureSet(
        signal=signal_name,
        run_id=run_id,
        peak=peak,
        time_to_peak_ms=time_to_peak_ms,
        rise_time_ms=rise_time_ms,
        duration_ms=duration_ms,
        integral=integral,
        provenance=Provenance(algorithm="calculate_signal_features", algorithm_version=FEATURES_ALGORITHM_VERSION),
    )


def detect_first_divergence(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    time_s: np.ndarray,
    *,
    signal_name: str,
    run_a_id: str,
    run_b_id: str,
    threshold_fraction: float = DEFAULT_DIVERGENCE_THRESHOLD_FRACTION,
    min_sustain_samples: int = DEFAULT_MIN_SUSTAIN_SAMPLES,
) -> DivergenceEvent | None:
    """First index where |a-b| exceeds `threshold_fraction` of the larger
    peak magnitude and stays exceeded for `min_sustain_samples` samples
    (guards against a single noisy sample producing a false event)."""
    scale = max(float(np.max(np.abs(signal_a))), float(np.max(np.abs(signal_b))), 1e-9)
    threshold_abs = threshold_fraction * scale
    diff = np.abs(signal_a - signal_b)
    exceeds = diff > threshold_abs

    n = len(exceeds)
    for i in range(n - min_sustain_samples + 1):
        if exceeds[i : i + min_sustain_samples].all():
            return DivergenceEvent(
                signal=signal_name,
                time_ms=float(time_s[i] * 1000.0),
                threshold={"fraction": threshold_fraction, "absolute": threshold_abs},
                window={"min_sustain_samples": min_sustain_samples},
                alignment_method="index-aligned (identical sampling grid)",
                source_run_a=run_a_id,
                source_run_b=run_b_id,
                provenance=Provenance(
                    algorithm="detect_first_divergence",
                    algorithm_version=DIVERGENCE_ALGORITHM_VERSION,
                    parameters={
                        "threshold_fraction": threshold_fraction,
                        "min_sustain_samples": min_sustain_samples,
                    },
                ),
            )
    return None


def analyze_signal_pair(
    signal_name: str,
    run_a_id: str,
    run_b_id: str,
    time_s: np.ndarray,
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> SignalAnalysisResult:
    features_a = calculate_signal_features(signal_a, time_s, run_id=run_a_id, signal_name=signal_name)
    features_b = calculate_signal_features(signal_b, time_s, run_id=run_b_id, signal_name=signal_name)
    correlation = float(np.corrcoef(signal_a, signal_b)[0, 1])
    divergence = detect_first_divergence(
        signal_a, signal_b, time_s, signal_name=signal_name, run_a_id=run_a_id, run_b_id=run_b_id
    )
    return SignalAnalysisResult(
        signal=signal_name,
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        run_a_features=features_a,
        run_b_features=features_b,
        correlation=correlation,
        divergence=divergence,
        provenance=Provenance(algorithm="analyze_signal_pair", algorithm_version="analyze_signal_pair v0.1.0"),
    )
