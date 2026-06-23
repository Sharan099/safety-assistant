"""R14 clause dependency map: §6.3 general conditions → §6.4 load clauses."""

from __future__ import annotations

import re
from typing import Any

# Denormalized general test requirements (§6.3.2 angle, §6.3.3 duration).
R14_ANGLE_SNIPPET = (
    "General traction angle (§6.3.2): tractive force at 10 degrees ±5° above the "
    "horizontal, in a plane parallel to the median longitudinal plane of the vehicle."
)
R14_DURATION_INLINE = (
    "General hold time (§6.3.3): belt anchorages must withstand the specified load "
    "for not less than 0.2 second."
)

# §6.3.5 seat configuration → primary §6.4 test clause.
TEST_CONFIGURATION_MAP: dict[str, dict[str, Any]] = {
    "6.4.1": {
        "applicable_test_configuration": "front_outboard_retractor_pulley",
        "seat_position": "front_outboard",
        "retractor_required": "required",
        "primary_clause": "6.4.1",
        "label": "Front outboard — three-point belt with retractor/pulley (§6.3.5.1 → §6.4.1)",
    },
    "6.4.2": {
        "applicable_test_configuration": "rear_centre_no_retractor",
        "seat_position": "rear_centre",
        "retractor_required": "not_required",
        "primary_clause": "6.4.2",
        "label": "Rear outboard / centre — three-point without retractor (§6.3.5.2 → §6.4.2)",
    },
    "6.4.3": {
        "applicable_test_configuration": "lap_belt_only",
        "seat_position": "any",
        "retractor_required": "not_required",
        "primary_clause": "6.4.3",
        "label": "Lap belt lower anchorage (§6.3.5.2 / §6.3.6 → §6.4.3)",
    },
    "6.4.5": {
        "applicable_test_configuration": "special_type_belt",
        "seat_position": "any",
        "retractor_required": "optional",
        "primary_clause": "6.4.5",
        "label": "Special-type belt geometry (§6.3.5.3 → §6.4.5)",
    },
}

# §5.4.2.x location/angle rules by seat category (summary for metadata).
LOCATION_ANGLE_CLAUSES: dict[str, str] = {
    "5.4.2.1": "M1 front outboard — angle constant 60° ±10° (adjustable seats)",
    "5.4.2.2": "M1 front centre — angle rules per seat adjustability",
    "5.4.2.3": "M1 rear — angle constant 60° ±10°",
    "5.4.2.4": "M2/M3/N2/N3 — angle constant 60° ±10°",
    "5.4.2.5": "Effective anchorage / special cases",
}


def _norm_clause(clause: str | None) -> str:
    if not clause:
        return ""
    return clause.strip().rstrip(".")


def resolve_test_configuration(clause_number: str | None) -> dict[str, Any]:
    """Map §6.4.x clause to seat/retractor test configuration metadata."""
    c = _norm_clause(clause_number)
    for key, meta in TEST_CONFIGURATION_MAP.items():
        if c == key or c.startswith(key + "."):
            return dict(meta)
    if c.startswith("6.4.1"):
        return dict(TEST_CONFIGURATION_MAP["6.4.1"])
    if c.startswith("6.4.2"):
        return dict(TEST_CONFIGURATION_MAP["6.4.2"])
    if c.startswith("6.4.3"):
        return dict(TEST_CONFIGURATION_MAP["6.4.3"])
    if c.startswith("6.4.5"):
        return dict(TEST_CONFIGURATION_MAP["6.4.5"])
    return {}


def applicable_general_conditions(clause_number: str | None, regulation: str) -> list[str]:
    c = _norm_clause(clause_number)
    if regulation != "UN_R14":
        return []
    if c.startswith("6.4") or c.startswith("6.3.5"):
        return ["6.3.2", "6.3.3"]
    if c.startswith("5.4.2"):
        return [c] if c in LOCATION_ANGLE_CLAUSES else ["5.4.2"]
    return []


def build_denormalized_block(clause_number: str | None, regulation: str) -> str:
    """Inline angle + duration for §6.4 load clauses (self-sufficient retrieval hit)."""
    c = _norm_clause(clause_number)
    if regulation != "UN_R14" or not c.startswith("6.4"):
        return ""
    parts = [R14_ANGLE_SNIPPET, R14_DURATION_INLINE]
    loc = None
    for loc_clause, desc in LOCATION_ANGLE_CLAUSES.items():
        if c.startswith("6.4.1") and loc_clause == "5.4.2.1":
            loc = f"Location/angle note ({loc_clause}): {desc}"
            break
        if c.startswith("6.4.2") and loc_clause in ("5.4.2.2", "5.4.2.3"):
            loc = f"Location/angle note ({loc_clause}): {desc}"
            break
    if loc:
        parts.append(loc)
    cfg = resolve_test_configuration(c)
    if cfg.get("label"):
        parts.append(f"Test configuration (§6.3.5): {cfg['label']}")
    return "\n".join(parts)


def infer_test_configuration_from_query(query: str) -> str | None:
    """Best-effort §6.4 clause family from seat + retractor language in query."""
    q = query.lower()
    rear_centre = any(
        x in q for x in ("rear centre", "rear center", "centre seat", "center seat")
    )
    no_retractor = any(
        x in q for x in ("no retractor", "without retractor", "without a retractor")
    )
    lap_only = "lap belt" in q or "lap-belt" in q
    front_outboard = "front outboard" in q or ("front" in q and "outboard" in q)
    if rear_centre and no_retractor:
        return "6.4.2"
    if lap_only and no_retractor:
        return "6.4.3"
    if front_outboard and "retractor" in q:
        return "6.4.1"
    return None


def enrich_clause_dependency_meta(
    *,
    regulation: str,
    clause_number: str | None,
) -> dict[str, Any]:
    c = _norm_clause(clause_number)
    meta: dict[str, Any] = {}
    gen = applicable_general_conditions(c, regulation)
    if gen:
        meta["applicable_general_conditions"] = gen
    cfg = resolve_test_configuration(c)
    if cfg:
        meta.update({
            k: cfg[k]
            for k in (
                "applicable_test_configuration",
                "seat_position",
                "retractor_required",
            )
            if k in cfg
        })
    if c.startswith("5.4.2") and c in LOCATION_ANGLE_CLAUSES:
        meta["location_angle_rule"] = LOCATION_ANGLE_CLAUSES[c]
    return meta
