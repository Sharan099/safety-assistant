"""
Document registry — single source of truth for document provenance.

Item 1 (grounding & anti-hallucination) needs three facts about every source
that the chunk metadata does NOT currently carry in a structured form:

  1. The *document type* — a legal regulation (binding, e.g. UN/ECE, FMVSS) vs a
     rating protocol (consumer assessment, NOT legally binding, e.g. Euro NCAP).
     These must never be blurred.
  2. The *revision / amendment* the indexed copy corresponds to.
  3. Whether other revisions exist, so the answer can prompt the user to confirm
     which version applies.

The revision strings below were extracted from the actual indexed corpus text
(not invented):
  - UN R14: Regulation No. 14, Revision 7, 09 series of amendments,
            entry into force 29 December 2018 (ECE/TRANS/WP.29/2018/44).
  - UN R16: Regulation No. 16, Revision 10, 07 series of amendments,
            entry into force 10 February 2018.

Entries for documents not yet indexed (FMVSS, Euro NCAP, …) are seeded so the
legal-vs-rating distinction is enforced the moment those documents are added.
Fields marked `verified=False` must be confirmed before they are trusted.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.core.authority_tier import (
    ENGINEERING_REF,
    HISTORICAL_DATA,
    LEGAL_BINDING,
    OEM_INTERNAL,
    RATING_PROTOCOL,
    SYNTHETIC,
)

# Document type taxonomy. Keep legal vs rating strictly separated.
LEGAL_REGULATION = "legal_regulation"   # binding (UN/ECE, FMVSS, EU directives)
RATING_PROTOCOL = "rating_protocol"     # consumer assessment (Euro NCAP, IIHS)
ENGINEERING_REFERENCE = "reference"     # ISO / internal CAE references (non-binding)
INTERNAL_DOCUMENT = "internal"          # synthetic / program internal docs (pilot)

DOC_TYPE_LABEL = {
    LEGAL_REGULATION: "Legal regulation (binding)",
    RATING_PROTOCOL: "Rating protocol (consumer assessment — not legally binding)",
    ENGINEERING_REFERENCE: "Engineering reference (non-binding)",
    INTERNAL_DOCUMENT: "Internal document (synthetic / program data)",
}


@dataclass(frozen=True)
class DocumentMeta:
    code: str                       # internal regulation code, e.g. "UN_R14"
    display_name: str               # human label, e.g. "UN R14"
    full_title: str
    doc_type: str                   # one of the constants above
    authority_tier: str             # closed vocabulary — binding force (Part A)
    authority: str                  # issuing body
    region: str = "global"          # EU / US / global
    impact_mode: str = "general"    # frontal / side / belt / general — hard filter dimension
    license_status: str = "public_domain"
    indexed_revision: str | None = None
    known_revisions: tuple[str, ...] = ()
    in_force: str | None = None
    legal_reference: str | None = None
    verified: bool = False

    @property
    def is_legal(self) -> bool:
        return self.authority_tier == LEGAL_BINDING

    @property
    def has_multiple_revisions(self) -> bool:
        # A legal regulation that has been revised many times: confirming the
        # applicable revision matters for compliance.
        return len(self.known_revisions) > 1


_REGISTRY: dict[str, DocumentMeta] = {
    "UN_R14": DocumentMeta(
        code="UN_R14",
        display_name="UN R14",
        full_title="Uniform provisions concerning the approval of vehicles with "
        "regard to safety-belt anchorages, ISOFIX anchorage systems, ISOFIX top "
        "tether anchorages and i-Size seating positions",
        doc_type=LEGAL_REGULATION,
        authority_tier=LEGAL_BINDING,
        authority="UN-ECE",
        region="EU",
        impact_mode="belt",
        license_status="public_domain",
        indexed_revision="Revision 7 (09 series of amendments)",
        known_revisions=(
            "Revision 6 (08 series)",
            "Revision 7 (09 series)",
        ),
        in_force="29 December 2018",
        legal_reference="ECE/TRANS/WP.29/2018/44 — Addendum 13",
        verified=True,
    ),
    "UN_R16": DocumentMeta(
        code="UN_R16",
        display_name="UN R16",
        full_title="Uniform provisions concerning the approval of safety-belts, "
        "restraint systems, child restraint systems and ISOFIX child restraint "
        "systems for occupants of power-driven vehicles",
        doc_type=LEGAL_REGULATION,
        authority_tier=LEGAL_BINDING,
        authority="UN-ECE",
        region="EU",
        impact_mode="belt",
        license_status="public_domain",
        indexed_revision="Revision 10 (07 series of amendments)",
        known_revisions=(
            "Revision 9 (07 series)",
            "Revision 10 (07 series)",
        ),
        in_force="10 February 2018",
        legal_reference="E/ECE/324/Rev.1/Add.15/Rev.10",
        verified=True,
    ),
    # ---- Seeded (NOT yet indexed); revisions unverified, do not trust blindly ----
    "UN_R17": DocumentMeta(
        code="UN_R17", display_name="UN R17",
        full_title="Seats, their anchorages and head restraints",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="UNECE (UN Regulation)", region="EU", impact_mode="seat",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "UN_R94": DocumentMeta(
        code="UN_R94", display_name="UN R94",
        full_title="Protection of occupants in the event of a frontal collision",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="UNECE (UN Regulation)", region="EU", impact_mode="frontal",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "UN_R95": DocumentMeta(
        code="UN_R95", display_name="UN R95",
        full_title="Protection of occupants in the event of a lateral collision",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="UNECE (UN Regulation)", region="EU", impact_mode="side",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "UN_R135": DocumentMeta(
        code="UN_R135", display_name="UN R135",
        full_title="Protection of occupants in the event of a pole side impact",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="UNECE (UN Regulation)", region="EU", impact_mode="pole_side",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "UN_R137": DocumentMeta(
        code="UN_R137", display_name="UN R137",
        full_title="Frontal impact with focus on restraint systems",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="UNECE (UN Regulation)", region="EU", impact_mode="frontal",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "UN_R127": DocumentMeta(
        code="UN_R127", display_name="UN R127",
        full_title="Steering equipment",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="UNECE (UN Regulation)", region="EU", impact_mode="general",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "FMVSS": DocumentMeta(
        code="FMVSS", display_name="FMVSS",
        full_title="Federal Motor Vehicle Safety Standards (49 CFR Part 571)",
        doc_type=LEGAL_REGULATION, authority_tier=LEGAL_BINDING,
        authority="NHTSA (United States)", region="US", impact_mode="general",
        license_status="public_domain", indexed_revision=None, verified=False,
    ),
    "EURO_NCAP": DocumentMeta(
        code="EURO_NCAP", display_name="Euro NCAP",
        full_title="Euro NCAP assessment protocols",
        doc_type=RATING_PROTOCOL, authority_tier=RATING_PROTOCOL,
        authority="Euro NCAP (consumer programme)", region="EU", impact_mode="general",
        license_status="licensed", indexed_revision=None, verified=False,
    ),
    "CAE_REFERENCE": DocumentMeta(
        code="CAE_REFERENCE", display_name="CAE Companion",
        full_title="CAE Companion handbook (engineering reference — not a legal regulation)",
        doc_type=ENGINEERING_REFERENCE, authority_tier=ENGINEERING_REF,
        authority="Internal reference", region="global", impact_mode="general",
        license_status="licensed", indexed_revision=None, verified=False,
    ),
    "SAFETY_REFERENCE": DocumentMeta(
        code="SAFETY_REFERENCE", display_name="Safety Companion",
        full_title="Safety Companion handbook (engineering reference — not a legal regulation)",
        doc_type=ENGINEERING_REFERENCE, authority_tier=ENGINEERING_REF,
        authority="Internal reference", region="global", impact_mode="general",
        license_status="licensed", indexed_revision=None, verified=False,
    ),
    # ---- Synthetic PROG_X internal docs (pilot) ----
    "PROG_X_FT_001": DocumentMeta(
        code="PROG_X_FT_001", display_name="PROG_X FT-001",
        full_title="PROG_X frontal crash test report FT-PROG-X-001 (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="frontal",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
    "PROG_X_FT_002": DocumentMeta(
        code="PROG_X_FT_002", display_name="PROG_X FT-002",
        full_title="PROG_X frontal crash test report FT-PROG-X-002 (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="frontal",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
    "PROG_X_CAE_001": DocumentMeta(
        code="PROG_X_CAE_001", display_name="PROG_X CAE-001",
        full_title="PROG_X CAE correlation vs FT-001 (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="general",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
    "PROG_X_CAE_002": DocumentMeta(
        code="PROG_X_CAE_002", display_name="PROG_X CAE-002",
        full_title="PROG_X CAE correlation vs FT-002 (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="general",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
    "PROG_X_RCA_001": DocumentMeta(
        code="PROG_X_RCA_001", display_name="PROG_X RCA-001",
        full_title="PROG_X root cause analysis anchorage margin (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="belt",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
    "PROG_X_DR": DocumentMeta(
        code="PROG_X_DR", display_name="PROG_X DR minutes",
        full_title="PROG_X design review minutes Feb 2026 (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="general",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
    "PROG_X_STATUS": DocumentMeta(
        code="PROG_X_STATUS", display_name="PROG_X status",
        full_title="PROG_X project status summary (SYNTHETIC)",
        doc_type=INTERNAL_DOCUMENT, authority_tier=SYNTHETIC,
        authority="internal", region="global", impact_mode="general",
        license_status="proprietary_internal", indexed_revision="synthetic", verified=False,
    ),
}


# ── Indexed corpus lock (Phase 0 pilot) ─────────────────────────────────────
# Pilot scope: UN R14 + UN R16 only. Expand when scaling to full corpus.
INDEXED_LEGAL_CORPUS: frozenset[str] = frozenset({
    "UN_R14", "UN_R16", "UN_R17", "UN_R94", "UN_R95", "UN_R127", "UN_R135", "UN_R137", "FMVSS",
})

# FMVSS_210 is NOT indexed — answers must not cite it.
GHOST_REGULATIONS: frozenset[str] = frozenset({"FMVSS_210", "UN_R210"})


@dataclass(frozen=True)
class RequirementCluster:
    name: str
    members: tuple[str, ...]
    validated_by: tuple[str, ...] = ()
    query_triggers: tuple[str, ...] = ()


REQUIREMENT_CLUSTERS: dict[str, RequirementCluster] = {
    "belt_restraint_design": RequirementCluster(
        name="belt_restraint_design",
        members=("UN_R14", "UN_R16", "UN_R17"),
        validated_by=("UN_R94", "UN_R137", "UN_R95"),
        query_triggers=(
            "belt", "restraint", "seat belt", "seatbelt", "anchorage",
            "govern", "which regulations", "which regs",
        ),
    ),
    "frontal": RequirementCluster(
        name="frontal",
        members=("UN_R94", "UN_R137", "FMVSS"),
        validated_by=(),
        query_triggers=("frontal", "head-on", "odb", "full-width frontal"),
    ),
    "side": RequirementCluster(
        name="side",
        members=("UN_R95",),
        validated_by=(),
        query_triggers=("side impact", "lateral collision", "lateral impact"),
    ),
    "pole_side": RequirementCluster(
        name="pole_side",
        members=("UN_R135",),
        validated_by=(),
        query_triggers=("pole side", "pole impact", "psi", "worldsid pole"),
    ),
    "frontal_and_side": RequirementCluster(
        name="frontal_and_side",
        members=("UN_R94", "UN_R95"),
        validated_by=("UN_R137", "FMVSS"),
        query_triggers=(),
    ),
}


def is_indexed_regulation(code: str | None) -> bool:
    if not code:
        return False
    norm = code.upper().replace(" ", "_")
    if norm in GHOST_REGULATIONS:
        return False
    if norm in INDEXED_LEGAL_CORPUS:
        return True
    return norm in _REGISTRY


def match_requirement_cluster(query: str) -> str | None:
    q = query.lower()
    best: str | None = None
    best_hits = 0
    for name, cluster in REQUIREMENT_CLUSTERS.items():
        hits = sum(1 for t in cluster.query_triggers if t in q)
        if hits > best_hits:
            best_hits = hits
            best = name
    return best if best_hits > 0 else None


def cluster_member_codes(cluster_name: str | None) -> tuple[str, ...]:
    if not cluster_name:
        return ()
    c = REQUIREMENT_CLUSTERS.get(cluster_name)
    return c.members if c else ()


def cluster_boost_for_regulation(reg_code: str, cluster_name: str | None) -> float:
    """Return metadata multiplier for cluster-aware ranking."""
    if not cluster_name:
        return 1.0
    cluster = REQUIREMENT_CLUSTERS.get(cluster_name)
    if not cluster:
        return 1.0
    norm = (reg_code or "").upper().replace(" ", "_")
    if norm in cluster.members:
        return float(os.getenv("CLUSTER_MEMBER_BOOST", "2.0"))
    if norm in cluster.validated_by:
        return float(os.getenv("CLUSTER_VALIDATED_BOOST", "1.3"))
    if norm in GHOST_REGULATIONS:
        return 0.0
    return 1.0


_UNKNOWN = DocumentMeta(
    code="UNKNOWN",
    display_name="Unknown document",
    full_title="Unregistered source",
    doc_type=ENGINEERING_REFERENCE,
    authority_tier=ENGINEERING_REF,
    authority="Unknown",
    region="global",
    impact_mode="general",
    license_status="review_needed",
    indexed_revision=None,
    verified=False,
)


_NCAP_REGISTRY_FILE = (
    Path(__file__).resolve().parents[3] / "data" / "manifest" / "ncap_registry.json"
)


def _load_ncap_registry() -> None:
    """Merge persisted NCAP document entries (written by fetch_ncap_data.py)."""
    if not _NCAP_REGISTRY_FILE.exists():
        return
    try:
        entries = json.loads(_NCAP_REGISTRY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    for code, raw in entries.items():
        _REGISTRY[code] = DocumentMeta(**raw)


def register_ncap_document(
    code: str,
    *,
    display_name: str,
    year: int,
    make: str,
    model: str,
    vehicle_id: int,
    impact_mode: str = "general",
) -> DocumentMeta:
    """Register one NHTSA NCAP historical document and persist to manifest."""
    norm = code.strip().upper()
    meta = DocumentMeta(
        code=norm,
        display_name=display_name,
        full_title=(
            f"NHTSA NCAP safety ratings — {year} {make} {model} "
            f"(VehicleId {vehicle_id}, star ratings + media)"
        ),
        doc_type=INTERNAL_DOCUMENT,
        authority_tier=HISTORICAL_DATA,
        authority="NHTSA",
        region="US",
        impact_mode=impact_mode,
        license_status="public",
        indexed_revision=str(year),
        verified=True,
    )
    _REGISTRY[norm] = meta
    _persist_ncap_registry()
    return meta


def _persist_ncap_registry() -> None:
    ncap = {
        k: {
            "code": m.code,
            "display_name": m.display_name,
            "full_title": m.full_title,
            "doc_type": m.doc_type,
            "authority_tier": m.authority_tier,
            "authority": m.authority,
            "region": m.region,
            "impact_mode": m.impact_mode,
            "license_status": m.license_status,
            "indexed_revision": m.indexed_revision,
            "known_revisions": list(m.known_revisions),
            "in_force": m.in_force,
            "legal_reference": m.legal_reference,
            "verified": m.verified,
        }
        for k, m in _REGISTRY.items()
        if k.startswith("NCAP_")
    }
    _NCAP_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _NCAP_REGISTRY_FILE.write_text(json.dumps(ncap, indent=2), encoding="utf-8")


def get_document_meta(regulation_code: str | None) -> DocumentMeta:
    """Look up provenance for a regulation code; never raises."""
    if not regulation_code:
        return _UNKNOWN
    norm = regulation_code.strip().upper()
    if norm in _REGISTRY:
        return _REGISTRY[norm]
    # Fallback for NCAP codes not yet persisted (e.g. fresh fetch before reload).
    m = re.match(r"^NCAP_(\d{4})_([A-Z0-9_]+)_(\d+)$", norm)
    if m:
        year, slug, vid = m.group(1), m.group(2), m.group(3)
        return DocumentMeta(
            code=norm,
            display_name=f"NCAP {slug.replace('_', ' ').title()}",
            full_title=f"NHTSA NCAP safety ratings ({year}, VehicleId {vid})",
            doc_type=INTERNAL_DOCUMENT,
            authority_tier=HISTORICAL_DATA,
            authority="NHTSA",
            region="US",
            impact_mode="general",
            license_status="public",
            indexed_revision=year,
            verified=False,
        )
    return _UNKNOWN


_load_ncap_registry()


def doc_type_label(doc_type: str) -> str:
    return DOC_TYPE_LABEL.get(doc_type, doc_type)


# Aliases for regulation detection in user queries (longest match first at runtime).
REG_QUERY_ALIASES: dict[str, str] = {
    "fmvss 208": "FMVSS",
    "fmvss208": "FMVSS",
    "fmvss 210": "FMVSS",
    "fmvss": "FMVSS",
    "un regulation no. 14": "UN_R14",
    "regulation no. 14": "UN_R14",
    "un r14": "UN_R14",
    "r14": "UN_R14",
    "un regulation no. 16": "UN_R16",
    "un r16": "UN_R16",
    "r16": "UN_R16",
    "un regulation no. 17": "UN_R17",
    "un r17": "UN_R17",
    "r17": "UN_R17",
    "un regulation no. 94": "UN_R94",
    "un r94": "UN_R94",
    "r94": "UN_R94",
    "un regulation no. 95": "UN_R95",
    "un r95": "UN_R95",
    "r95": "UN_R95",
    "un regulation no. 135": "UN_R135",
    "un r135": "UN_R135",
    "r135": "UN_R135",
    "un regulation no. 137": "UN_R137",
    "un r127": "UN_R127", "r127": "UN_R127",
    "un r137": "UN_R137",
    "r137": "UN_R137",
    "euro ncap": "EURO_NCAP",
    "euroncap": "EURO_NCAP",
    "nhtsa ncap": "NCAP_NHTSA",
    "nhtsa": "NCAP_NHTSA",
}

# Golden-set / external ids that map to indexed corpus regulation codes.
REG_CORPUS_ALIASES: dict[str, tuple[str, ...]] = {
    "FMVSS_208": ("FMVSS",),
    "FMVSS_210": ("FMVSS",),
}


def detect_regulations_in_query(query: str) -> list[str]:
    """Return unique indexed regulation codes named in the query (registry order)."""
    q = query.lower()
    found: list[str] = []
    for alias, code in sorted(REG_QUERY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q and code not in found:
            found.append(code)
    return found


def regulation_matches_corpus(expected: str, chunk_reg: str) -> bool:
    """True if chunk regulation satisfies an expected doc id (e.g. FMVSS_208 → FMVSS)."""
    exp = expected.upper().replace(" ", "_")
    reg = (chunk_reg or "").upper().replace(" ", "_")
    if not reg:
        return False
    if exp in ("NCAP_NHTSA", "NCAP") and reg.startswith("NCAP_"):
        return True
    if exp == reg or exp in reg or reg in exp:
        return True
    for alias, codes in REG_CORPUS_ALIASES.items():
        if exp == alias.upper() and reg in [c.upper() for c in codes]:
            return True
    return False
