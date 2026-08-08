"""Normalize regulation aliases used by the agent tools."""

from __future__ import annotations

import re

# Canonical ids match ingestion / catalog keys.
_ALIASES: dict[str, str] = {
    "r94": "UN-ECE-R94",
    "un-ece-r94": "UN-ECE-R94",
    "un r94": "UN-ECE-R94",
    "unece r94": "UN-ECE-R94",
    "regulation 94": "UN-ECE-R94",
    "r95": "UN-ECE-R95",
    "un-ece-r95": "UN-ECE-R95",
    "un r95": "UN-ECE-R95",
    "r16": "UN-ECE-R16",
    "un-ece-r16": "UN-ECE-R16",
    "r129": "UN-ECE-R129",
    "un-ece-r129": "UN-ECE-R129",
    "fmvss 208": "FMVSS-208",
    "fmvss208": "FMVSS-208",
    "fmvss-208": "FMVSS-208",
    "49 cfr 571.208": "FMVSS-208",
}


def normalize_regulation(raw: str | None) -> str | None:
    if not raw:
        return None
    text = re.sub(r"\s+", " ", str(raw).strip())
    key = text.lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    m = re.search(r"(?:un[-\s]?ece[-\s]?)?r\s*(\d+)", key, re.I)
    if m:
        return _ALIASES.get(f"r{m.group(1)}") or f"UN-ECE-R{m.group(1)}"
    if key.startswith("fmvss"):
        return "FMVSS-208" if "208" in key else text.upper()
    return text


def is_indexed(regulation_id: str | None) -> bool:
    """True when ``regulation_id`` is present in the live Qdrant catalog."""
    rid = normalize_regulation(regulation_id)
    if not rid:
        return False
    try:
        from retrieval.retrieve import indexed_regulation_ids

        return rid in indexed_regulation_ids()
    except Exception:  # noqa: BLE001
        return False
