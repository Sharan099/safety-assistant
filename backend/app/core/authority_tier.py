"""Closed authority-tier vocabulary — ordered by binding force (highest first)."""

from __future__ import annotations

from typing import Any

# Tier constants (closed vocabulary — do not add values without schema migration).
LEGAL_BINDING = "legal_binding"
RATING_PROTOCOL = "rating_protocol"
ENGINEERING_REF = "engineering_ref"
OEM_INTERNAL = "oem_internal"
HISTORICAL_DATA = "historical_data"
SYNTHETIC = "synthetic"

AUTHORITY_TIERS: tuple[str, ...] = (
    LEGAL_BINDING,
    RATING_PROTOCOL,
    ENGINEERING_REF,
    OEM_INTERNAL,
    HISTORICAL_DATA,
    SYNTHETIC,
)

TIER_ORDER: dict[str, int] = {t: i for i, t in enumerate(AUTHORITY_TIERS)}

TIER_BADGE: dict[str, str] = {
    LEGAL_BINDING: "LEGAL",
    RATING_PROTOCOL: "RATING",
    ENGINEERING_REF: "ENG-REF",
    OEM_INTERNAL: "OEM",
    HISTORICAL_DATA: "HISTORICAL",
    SYNTHETIC: "SYNTHETIC",
}

TIER_LABEL: dict[str, str] = {
    LEGAL_BINDING: "Legal regulation (binding — must comply)",
    RATING_PROTOCOL: "Rating protocol (not legally binding)",
    ENGINEERING_REF: "Engineering reference (advisory)",
    OEM_INTERNAL: "OEM internal standard (proprietary advisory)",
    HISTORICAL_DATA: "Historical test / investigation evidence",
    SYNTHETIC: "Synthetic placeholder data",
}

# Map legacy doc_type → default authority tier.
_DOC_TYPE_DEFAULT_TIER: dict[str, str] = {
    "legal": LEGAL_BINDING,
    "rating": RATING_PROTOCOL,
    "reference": ENGINEERING_REF,
    "internal": HISTORICAL_DATA,
}

_LICENSE_STATUSES: tuple[str, ...] = (
    "public_domain",
    "licensed",
    "proprietary_internal",
    "review_needed",
)


def is_valid_tier(tier: str | None) -> bool:
    return tier in TIER_ORDER


def tier_rank(tier: str | None) -> int:
    return TIER_ORDER.get(tier or "", 99)


def is_binding_tier(tier: str | None) -> bool:
    return tier == LEGAL_BINDING


def is_advisory_tier(tier: str | None) -> bool:
    return tier in (ENGINEERING_REF, OEM_INTERNAL, HISTORICAL_DATA, RATING_PROTOCOL)


def resolve_authority_tier(
    *,
    regulation: str | None = None,
    doc_type: str | None = None,
    is_synthetic: bool = False,
    explicit_tier: str | None = None,
    document_kind: str | None = None,
) -> str:
    """Resolve chunk/document authority tier from registry + chunk signals."""
    if explicit_tier and is_valid_tier(explicit_tier):
        return explicit_tier
    if is_synthetic or (regulation or "").startswith("PROG_X"):
        return SYNTHETIC
    if document_kind == "oem_standard":
        return OEM_INTERNAL
    if document_kind in ("test_report", "rca", "crash_investigation"):
        return HISTORICAL_DATA
    if doc_type:
        return _DOC_TYPE_DEFAULT_TIER.get(doc_type, ENGINEERING_REF)
    from backend.app.core.document_registry import get_document_meta

    meta = get_document_meta(regulation)
    return getattr(meta, "authority_tier", None) or _DOC_TYPE_DEFAULT_TIER.get(
        _registry_doc_type_to_chunk(meta.doc_type), ENGINEERING_REF
    )


def _registry_doc_type_to_chunk(registry_doc_type: str) -> str:
    from backend.app.core.document_registry import (
        ENGINEERING_REFERENCE,
        INTERNAL_DOCUMENT,
        LEGAL_REGULATION,
        RATING_PROTOCOL,
    )

    return {
        LEGAL_REGULATION: "legal",
        RATING_PROTOCOL: "rating",
        ENGINEERING_REFERENCE: "reference",
        INTERNAL_DOCUMENT: "internal",
    }.get(registry_doc_type, "reference")


def badge_for_tier(tier: str | None) -> str:
    return TIER_BADGE.get(tier or "", "REF")


def label_for_tier(tier: str | None) -> str:
    return TIER_LABEL.get(tier or "", "Unknown authority tier")


def chunk_authority_tier(chunk: dict[str, Any]) -> str:
    return resolve_authority_tier(
        regulation=chunk.get("regulation") or chunk.get("doc_id"),
        doc_type=chunk.get("doc_type"),
        is_synthetic=bool(chunk.get("is_synthetic")),
        explicit_tier=chunk.get("authority_tier"),
        document_kind=chunk.get("document_kind"),
    )
