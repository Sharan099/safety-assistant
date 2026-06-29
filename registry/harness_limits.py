"""Harness limit parsing and deterministic pass/fail / margin math (reuse at ingest + query time)."""

from __future__ import annotations

import re


def verify_criterion_matches_clause(criterion: str, chunk_text: str) -> bool:
    text_lower = chunk_text.lower()
    crit_lower = criterion.lower()
    if crit_lower == "thcc":
        return any(w in text_lower for w in ["thcc", "thorax", "chest", "deflection", "compression"])
    if crit_lower in ("hic", "hic36"):
        return any(w in text_lower for w in ["hic", "head", "injury"])
    return crit_lower in text_lower


def extract_limit_details(criterion: str, chunk_text: str) -> tuple[float, str, str] | None:
    """
    Extract limit from clause text.
    Returns (limit_value, direction, unit) where direction is 'ceiling' or 'floor'.
    """
    text_lower = chunk_text.lower()
    crit_lower = criterion.lower()

    if not verify_criterion_matches_clause(criterion, chunk_text):
        return None

    pattern = r"(\d+(?:\.\d+)?)\s*(mm|g|kn|ms)\b"
    matches = list(re.finditer(pattern, text_lower))
    if not matches:
        if crit_lower == "thcc" and ("94" in text_lower or "r94" in text_lower):
            return 42.0, "ceiling", "mm"
        if crit_lower in ("hic", "hic36"):
            return 1000.0, "ceiling", "none"
        return None

    match = matches[0]
    try:
        val = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2)

    start_idx = max(0, match.start() - 60)
    end_idx = min(len(text_lower), match.end() + 60)
    prefix = text_lower[start_idx : match.start()]
    suffix = text_lower[match.end() : end_idx]
    context = prefix + " " + suffix

    floor_words = ["above", "greater than", "greater", "minimum", "min", "at least", "stay above", "more than", "exceeds"]
    ceil_words = ["not exceed", "not to exceed", "less than", "less", "maximum", "max", "below", "limit"]

    is_floor = any(w in context for w in floor_words)
    is_ceil = any(w in context for w in ceil_words)

    if is_floor and not is_ceil:
        direction = "floor"
    elif is_ceil and not is_floor:
        direction = "ceiling"
    else:
        if crit_lower in ("hic", "hic36", "thcc"):
            direction = "ceiling"
        else:
            return None

    return val, direction, unit


def derive_pass_fail(value: float, limit: float, direction: str) -> str:
    if direction == "ceiling":
        return "PASS" if value <= limit else "FAIL"
    return "PASS" if value >= limit else "FAIL"


def compute_headroom_mm(measured: float, limit: float, direction: str) -> float:
    """Signed headroom in mm (positive = margin to limit for ceiling criteria)."""
    if direction == "ceiling":
        return round(limit - measured, 4)
    return round(measured - limit, 4)


def headroom_pct_of_limit(headroom: float, limit: float) -> float:
    if limit == 0:
        return 0.0
    return round(abs(headroom) / limit * 100.0, 2)
