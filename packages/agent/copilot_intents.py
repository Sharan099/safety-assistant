"""Deterministic intent classification for the Copilot —
CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 5/6.

Rule-based, not LLM-based: "tool selection" needs to be a testable,
predictable mapping (Phase 12: "Add: ### Agent: tool selection..."), and
CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 6 gives literal examples
("Compare the crash pulse" -> compare_global_response) that read as a
pattern table, not a prompt for an LLM to improvise over.
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal[
    "EXPLAIN_EVIDENCE",
    "COMPARE_RUNS",
    "ANALYZE_SIGNAL",
    "ANALYZE_DIVERGENCE",
    "RETRIEVE_KNOWLEDGE",
    "RETRIEVE_HISTORY",
    "CHALLENGE_HYPOTHESIS",
    "REQUEST_NEXT_ANALYSIS",
    "CONTROLLED_COMPARISON",
    "SHOW_SOURCE",
    "GENERAL_INVESTIGATION_QUESTION",
]

# Every signal packages/analysis/synthetic.py models — the only ones any
# tool can actually act on. Longest-first so "chest_deflection" doesn't
# accidentally get shadowed by a shorter alias.
KNOWN_SIGNALS = [
    "chest_deflection",
    "chest_acceleration",
    "chest_velocity",
    "belt_force",
    "pelvis_acceleration",
    "torso_rotation",
    "airbag_pressure",
    "vehicle_pulse",
]


def extract_signal(message: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")
    for signal in KNOWN_SIGNALS:
        if signal in normalized:
            return signal
    return None


def classify_intent(message: str) -> Intent:
    text = message.lower()

    if any(p in text for p in ("i disagree", "disagree", "alternative explanation", "what about", "instead")):
        return "CHALLENGE_HYPOTHESIS"

    if "source" in text and any(p in text for p in ("show", "cite", "where", "which document")):
        return "SHOW_SOURCE"

    if any(
        p in text for p in ("similar case", "historical case", "seen this before", "precedent", "prior investigation")
    ):
        return "RETRIEVE_HISTORY"

    if any(p in text for p in ("documentation", "regulation", "ls-dyna", "lsdyna", "un r", "says about", "what does")):
        return "RETRIEVE_KNOWLEDGE"

    if "compare" in text and any(p in text for p in ("crash pulse", "global", "pulse", "runs")):
        return "COMPARE_RUNS"

    if any(p in text for p in ("diverge", "divergence", "when did")):
        return "ANALYZE_DIVERGENCE"

    if any(
        p in text
        for p in (
            "what should i",
            "what next",
            "next analysis",
            "what's missing",
            "what is missing",
            "missing evidence",
        )
    ):
        return "REQUEST_NEXT_ANALYSIS"

    # Gated on an actual signal name being present — "analyze" alone (e.g.
    # "what should I analyze next?", already routed above) isn't enough.
    if ("analyze" in text or "analyse" in text) and extract_signal(message) is not None:
        return "ANALYZE_SIGNAL"

    if any(p in text for p in ("controlled comparison", "controlled simulation", "isolate", "controlled rerun")):
        return "CONTROLLED_COMPARISON"

    if any(p in text for p in ("why is", "why did", "contradict", "supports", "leading", "hypothesis")):
        return "EXPLAIN_EVIDENCE"

    return "GENERAL_INVESTIGATION_QUESTION"
