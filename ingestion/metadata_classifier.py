"""
Chunk-level metadata classifier — rules + keywords (no LLM).

Assigns test_type and value_type per chunk during ingestion; document-level
defaults come from document_registry.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.core.document_registry import (
    ENGINEERING_REFERENCE,
    INTERNAL_DOCUMENT,
    LEGAL_REGULATION,
    RATING_PROTOCOL,
    get_document_meta,
)

# Canonical doc_type values for chunks (maps registry types)
DOC_TYPE_LEGAL = "legal"
DOC_TYPE_RATING = "rating"
DOC_TYPE_REFERENCE = "reference"
DOC_TYPE_INTERNAL = "internal"
DOC_TYPE_SIM = "sim_report"
DOC_TYPE_TEST = "test_report"

CHUNK_TYPES = ("prose", "table", "figure", "procedure_step", "paragraph", "section", "event")
IMPACT_MODES = ("frontal", "side", "pole_side", "rear", "pedestrian", "belt", "general")
REGIONS = ("EU", "US", "global")
AUTHORITIES = ("UN-ECE", "UNECE", "NHTSA/FMVSS", "EuroNCAP", "internal")
DUMMY_TYPES_VOCAB = ("Hybrid III", "H3-50M", "ES-2", "WorldSID", "BioRID", "Q-series")
VEHICLE_PROGRAMS = ("PROG_X",)

TEST_TYPES = (
    "frontal", "side", "pole_side", "rear", "pedestrian",
    "belt", "seat", "general",
)

VALUE_TYPES = (
    "legal_limit", "rating_threshold", "target", "measured", None,
)

DUMMY_TYPES = (
    "H3-50M", "ES-2", "WorldSID", "Q-series", "BioRID", None,
)

_REGION_MAP = {
    "UN_R14": "EU", "UN_R16": "EU", "UN_R17": "EU", "UN_R94": "EU",
    "UN_R95": "EU", "UN_R135": "EU", "UN_R137": "EU", "UN_R127": "EU",
    "FMVSS": "US",
    "EURO_NCAP": "EU",
    "CAE_REFERENCE": "global", "SAFETY_REFERENCE": "global",
}

_AUTHORITY_MAP = {
    "UN_R14": "UN-ECE", "UN_R16": "UN-ECE", "UN_R17": "UNECE",
    "UN_R94": "UNECE", "UN_R95": "UNECE", "UN_R135": "UNECE", "UN_R137": "UNECE", "UN_R127": "UNECE",
    "FMVSS": "NHTSA/FMVSS",
    "EURO_NCAP": "EuroNCAP",
    "CAE_REFERENCE": "internal", "SAFETY_REFERENCE": "internal",
}
_PROG_X_PREFIX = "PROG_X"

_DOC_TYPE_MAP = {
    LEGAL_REGULATION: DOC_TYPE_LEGAL,
    RATING_PROTOCOL: DOC_TYPE_RATING,
    ENGINEERING_REFERENCE: DOC_TYPE_REFERENCE,
    INTERNAL_DOCUMENT: DOC_TYPE_INTERNAL,
}

_FRONTAL_KW = (
    "frontal", "odb", "full-width frontal", "full width frontal", "r94", "r137",
    "fmvss 208", "fmvss208", "head-on", "offset deformable",
)
_SIDE_KW = ("side impact", "lateral", "worldsid", "r95", "es-2", "es2re")
_POLE_KW = ("pole", "r135", "pole side")
_REAR_KW = ("rear impact", "rear-end", "whiplash", "rear collision")
_PED_KW = ("pedestrian", "vru", "leg impact", "head impact", "biorid", "q3", "q6")
_BELT_KW = ("safety belt", "seat belt", "restraint", "anchorage", "r14", "r16", "isofix")
_SEAT_KW = ("seat strength", "head restraint", "r17", "seat back", "headrest")

_LEGAL_KW = (
    "shall", "regulation", "approval", "conformity", "paragraph", "annex",
    "legal limit", "maximum", "minimum", "not exceed", "requirement",
)
_RATING_KW = (
    "euro ncap", "score", "points", "star", "rating", "assessment protocol",
    "adult occupant", "child occupant", "scoring", "threshold for",
)
_TARGET_KW = (
    "target", "design objective", "internal limit", "program target",
    "objective value", "design criterion",
)
_MEASURED_KW = ("measured", "test result", "actual value", "recorded", "achieved")

_NUMERIC_LIMIT_RE = re.compile(
    r"(chest\s+deflection|head\s+injury|hic|aic|nic|pelvis|femur|"
    r"deflection|acceleration|force|load)\s*[:\s]*\d",
    re.I,
)
_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|g|kN|daN|m/s|ms|kPa|N)", re.I)
_CLAUSE_RE = re.compile(r"(?:§|section|paragraph|annex)\s*[\d.]+", re.I)


def _parse_frontmatter_fields(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _normalize_dummy(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    if "hybrid iii" in low or "h3" in low:
        return "Hybrid III"
    if "worldsid" in low:
        return "WorldSID"
    if "es-2" in low or "es2" in low:
        return "ES-2"
    if "biorid" in low:
        return "BioRID"
    if "q-series" in low or re.search(r"\bq[36]\b", low):
        return "Q-series"
    return None


def _extract_test_id(text: str) -> str | None:
    m = re.search(r"\b(FT-PROG-X-\d{3}|CAE-PROG-X-\d{3}|RCA-PROG-X-\d{3})\b", text)
    return m.group(1) if m else None


def _extract_cae_version(text: str) -> str | None:
    m = re.search(r"PROG_X_v[\d.]+(?:_\d{8})?", text)
    return m.group(0) if m else None


def _extract_impact_mode(text: str, test_type: str) -> str:
    low = text.lower()
    if "impact_mode" in low:
        m = re.search(r"impact_mode\s*\|\s*(\w+)", low)
        if m:
            mode = m.group(1)
            if mode in IMPACT_MODES:
                return mode
    return test_type if test_type in IMPACT_MODES else "general"


def _registry_doc_type(regulation: str) -> str:
    meta = get_document_meta(regulation)
    return _DOC_TYPE_MAP.get(meta.doc_type, DOC_TYPE_REFERENCE)


def _detect_test_type(text: str, pdf_name: str, regulation: str) -> str:
    blob = f"{pdf_name} {text}".lower()
    if any(k in blob for k in _POLE_KW):
        return "pole_side"
    if any(k in blob for k in _SIDE_KW):
        return "side"
    if any(k in blob for k in _REAR_KW):
        return "rear"
    if any(k in blob for k in _PED_KW):
        return "pedestrian"
    if any(k in blob for k in _BELT_KW):
        return "belt"
    if any(k in blob for k in _SEAT_KW):
        return "seat"
    if any(k in blob for k in _FRONTAL_KW):
        return "frontal"
    if "chest deflection" in blob and "side" not in blob and "lateral" not in blob:
        return "frontal"
    if regulation in ("UN_R94", "UN_R137", "FMVSS"):
        return "frontal"
    if regulation == "UN_R95":
        return "side"
    if regulation == "UN_R135":
        return "pole_side"
    if regulation in ("UN_R14", "UN_R16"):
        return "belt"
    if regulation == "UN_R17":
        return "seat"
    if regulation == "EURO_NCAP":
        if "side" in blob:
            return "side"
        if "rear" in blob:
            return "rear"
        if "vru" in blob or "pedestrian" in blob:
            return "pedestrian"
        if "frontal" in blob:
            return "frontal"
    return "general"


def _detect_value_type(text: str, doc_type: str, regulation: str) -> str | None:
    low = text.lower()
    if doc_type == DOC_TYPE_REFERENCE:
        if any(k in low for k in _TARGET_KW):
            return "target"
        return None
    if doc_type == DOC_TYPE_RATING:
        if any(k in low for k in ("score", "points", "star rating", "0 points", "scoring")):
            return "rating_threshold"
        if _NUMERIC_LIMIT_RE.search(text) or _UNIT_RE.search(text):
            return "rating_threshold"
        return "rating_threshold" if "ncap" in low else None
    if doc_type == DOC_TYPE_LEGAL:
        if _NUMERIC_LIMIT_RE.search(text) or (
            _UNIT_RE.search(text) and any(k in low for k in _LEGAL_KW)
        ):
            return "legal_limit"
        if any(k in low for k in _LEGAL_KW) and _UNIT_RE.search(text):
            return "legal_limit"
    if any(k in low for k in _MEASURED_KW):
        return "measured"
    if any(k in low for k in _TARGET_KW):
        return "target"
    return None


def _detect_dummy(text: str) -> str | None:
    low = text.lower()
    if "h3-50m" in low or "h350m" in low:
        return "H3-50M"
    if "worldsid" in low:
        return "WorldSID"
    if "es-2" in low or "es2re" in low:
        return "ES-2"
    if "biorid" in low:
        return "BioRID"
    if re.search(r"\bq[36]\b", low) or "q-series" in low or "q series" in low:
        return "Q-series"
    return None


from backend.app.retrieval.clause_topic import detect_clause_topic


def classify_chunk(
    *,
    regulation: str,
    pdf_name: str,
    text: str,
    clause_number: str | None = None,
    heading_path: str = "",
    section_title: str = "",
) -> dict[str, Any]:
    """Return full metadata dict for a chunk."""
    fm = _parse_frontmatter_fields(text)
    is_synthetic = fm.get("is_synthetic", "").lower() == "true" or regulation.startswith(_PROG_X_PREFIX)
    meta = get_document_meta(regulation)
    doc_type = fm.get("doc_type") or _registry_doc_type(regulation)
    if doc_type not in (DOC_TYPE_LEGAL, DOC_TYPE_RATING, DOC_TYPE_REFERENCE, DOC_TYPE_INTERNAL):
        doc_type = _registry_doc_type(regulation)
    doc_id = regulation
    test_type = _detect_test_type(text, pdf_name, regulation)
    value_type = _detect_value_type(text, doc_type, regulation)
    dummy = _normalize_dummy(fm.get("dummy_type")) or _detect_dummy(text)
    clause_topic = detect_clause_topic(
        text, heading_path=heading_path, section_title=section_title
    )
    vehicle_program = fm.get("vehicle_program")
    if vehicle_program and vehicle_program not in VEHICLE_PROGRAMS:
        vehicle_program = None
    test_id = _extract_test_id(text)
    cae_model_version = _extract_cae_version(text)
    impact_mode = _extract_impact_mode(text, test_type)
    region = fm.get("region") or _REGION_MAP.get(regulation, "global")
    if region not in REGIONS:
        region = "global"
    authority = fm.get("authority") or _AUTHORITY_MAP.get(regulation, meta.authority)
    if authority not in AUTHORITIES and authority != meta.authority:
        authority = meta.authority if meta.authority in AUTHORITIES else "internal"

    clause = clause_number
    if not clause and _CLAUSE_RE.search(text):
        m = _CLAUSE_RE.search(text)
        clause = m.group(0) if m else None
    if not clause and heading_path:
        m = re.search(r"§[\d.]+", heading_path)
        if m:
            clause = m.group(0)

    revision = meta.indexed_revision
    authority_tier = fm.get("authority_tier") or meta.authority_tier
    license_status = fm.get("license_status") or meta.license_status
    if license_status not in ("public_domain", "licensed", "proprietary_internal", "review_needed"):
        license_status = meta.license_status

    return {
        "doc_id": doc_id,
        "regulation_id": doc_id,
        "doc_type": doc_type,
        "authority_tier": authority_tier,
        "authority": authority,
        "region": region,
        "jurisdiction": test_type,
        "test_type": test_type,
        "impact_mode": impact_mode,
        "dummy": dummy,
        "dummy_type": dummy,
        "value_type": value_type,
        "clause": clause,
        "revision": revision,
        "clause_topic": clause_topic,
        "is_synthetic": is_synthetic,
        "vehicle_program": vehicle_program,
        "test_id": test_id,
        "CAE_model_version": cae_model_version,
        "license_status": license_status,
    }


def classify_document_defaults(regulation: str, pdf_name: str) -> dict[str, Any]:
    """Document-level defaults when chunk text is empty."""
    return classify_chunk(
        regulation=regulation,
        pdf_name=pdf_name,
        text=pdf_name,
        clause_number=None,
    )
