"""Clause-topic detection and query → allowed-topic mapping (Phase 2)."""

from __future__ import annotations

import os
import re

CLAUSE_TOPICS = (
    "dummy_spec",
    "injury_criteria",
    "test_setup",
    "barrier",
    "scope",
    "evacuation",
    "approval_admin",
    "instrumentation",
    "general",
)

ENABLE_CLAUSE_TOPIC_FILTER = (
    os.getenv("ENABLE_CLAUSE_TOPIC_FILTER", "true").lower() == "true"
)

# Evacuation before dummy — evacuation text mentions "manikin".
_EVACUATION_KW = (
    "evacuat", "egress", "exit the vehicle", "door opening", "can be removed",
    "removed from the vehicle", "exit the car", "manikin can be evacuated",
    "50th percentile manikin can be evacuated",
)
_DUMMY_KW = (
    "dummy", "manikin", "mannequin", "anthropomorphic", "hybrid iii", "hybrid-iii",
    "es-2", "es2", "eurosid", "worldsid", "h350", "h3 50", "50th percentile male",
    "50th percentile dummy", "test device",
)
_INJURY_KW = (
    "injury criterion", "injury criteria", "hic", "chest deflection", "thorax",
    "viscous", "rib deflection", "abdomen", "pelvis", "femur", "nic", "aic",
    "performance criterion", "tolerance limit",
)
_TEST_SETUP_KW = (
    "test speed", "impact speed", "test procedure", "test configuration",
    "dynamic test", "sled test", "vehicle preparation", "test conditions",
)
_BARRIER_KW = (
    "barrier", "deformable", "odb", "mdb", "rigid barrier", "offset",
    "full-width", "full width", "moving deformable",
    "rigid pole", "pole barrier",
)
_SCOPE_KW = (
    "scope", "field of application", "this regulation applies", "definitions",
    "applies to", "purpose of this",
)
_APPROVAL_KW = (
    "approval", "homologation", "type approval", "application for approval",
    "contracting parties", "extension of approval", "conformity of production",
)
_INSTRUMENTATION_KW = (
    "instrumentation", "accelerometer", "transducer", "data acquisition",
    "filtering", "cfc", "sampling rate",
)

_HEADING_INJURY_RE = re.compile(
    r"injury\s+criteri|head\s+injury|thorax|pelvis|abdomen|hic\b|nic\b|aic\b",
    re.I,
)
_HEADING_DUMMY_RE = re.compile(
    r"\bdummy\b|manikin|mannequin|worldsid|eurosid|hybrid\s*iii|test\s+device",
    re.I,
)
_HEADING_TEST_SETUP_RE = re.compile(
    r"test\s+(speed|procedure|configuration|conditions)|impact\s+speed",
    re.I,
)


def _strip_context_block(text: str) -> str:
    """Remove ingestion CONTEXT preamble so pole/regulation keywords do not skew topics."""
    if text.startswith("CONTEXT:"):
        parts = text.split("---\n", 1)
        if len(parts) == 2:
            return parts[1]
    return text


def _heading_priority_topic(heading_path: str, section_title: str) -> str | None:
    blob = f"{heading_path} {section_title}"
    if _HEADING_INJURY_RE.search(blob):
        return "injury_criteria"
    if _HEADING_DUMMY_RE.search(blob):
        return "dummy_spec"
    if _HEADING_TEST_SETUP_RE.search(blob):
        return "test_setup"
    return None


def detect_clause_topic(
    text: str,
    heading_path: str = "",
    section_title: str = "",
) -> str:
    """Rules-based clause_topic from headings + keywords."""
    heading_topic = _heading_priority_topic(heading_path, section_title)
    if heading_topic:
        return heading_topic

    body = _strip_context_block(text)
    blob = f"{heading_path} {section_title} {body}".lower()

    if any(k in blob for k in _EVACUATION_KW):
        return "evacuation"
    if any(k in blob for k in _APPROVAL_KW):
        return "approval_admin"
    if any(k in blob for k in _INSTRUMENTATION_KW):
        return "instrumentation"
    if any(k in blob for k in _INJURY_KW):
        return "injury_criteria"
    if any(k in blob for k in _DUMMY_KW):
        return "dummy_spec"
    if any(k in blob for k in _TEST_SETUP_KW):
        return "test_setup"
    if any(k in blob for k in _BARRIER_KW):
        return "barrier"
    if any(k in blob for k in _SCOPE_KW):
        return "scope"
    return "general"


def detect_allowed_clause_topics(query: str) -> frozenset[str] | None:
    """
    Map query intent → allowed clause_topic values (hard filter when set).
    Returns None when no topic filter should apply.
    """
    if not ENABLE_CLAUSE_TOPIC_FILTER:
        return None

    q = query.lower()

    if re.search(
        r"\b(which|what)\s+(dummy|manikin|mannequin|test device)\b", q
    ) or "dummy and injury" in q or "dummy is used" in q:
        return frozenset({"dummy_spec", "injury_criteria"})

    if any(
        k in q
        for k in (
            "injury criteria", "injury criterion", "injury limits",
            "performance criteria", "performance criterion", "criteria are used", "criteria used",
        )
    ):
        return frozenset({
            "injury_criteria", "dummy_spec", "test_setup", "barrier", "general",
        })

    if any(
        k in q
        for k in (
            "test setup", "barrier", "impact speed", "test configuration",
            "type of frontal", "type of impact", "test type", "deformable",
        )
    ):
        return frozenset({"test_setup", "barrier"})

    if any(k in q for k in ("scope", "govern", "what does", "field of application")):
        return frozenset({"scope", "approval_admin", "general"})

    if any(k in q for k in ("anchorage load", "test load", "daN", "strength")):
        return frozenset({"test_setup", "injury_criteria", "general"})

    return None


def chunk_passes_topic_filter(
    chunk: dict,
    allowed: frozenset[str] | None,
) -> bool:
    """Hard exclude chunks whose clause_topic is outside the allowed set."""
    if not allowed:
        return True
    topic = chunk.get("clause_topic") or "general"
    if topic in allowed:
        return True
    # Mis-tagged chunks: honour structural feature flags when topics were inferred wrong.
    if "injury_criteria" in allowed and chunk.get("has_injury_criteria"):
        return True
    if "dummy_spec" in allowed and (chunk.get("dummy") or chunk.get("dummy_type")):
        return True
    if "test_setup" in allowed and chunk.get("has_test_procedure"):
        return True
    heading = f"{chunk.get('heading_path', '')} {chunk.get('section_title', '')}"
    if _HEADING_INJURY_RE.search(heading) and "injury_criteria" in allowed:
        return True
    if _HEADING_DUMMY_RE.search(heading) and "dummy_spec" in allowed:
        return True
    return False
