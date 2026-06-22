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

from dataclasses import dataclass, field

# Document type taxonomy. Keep legal vs rating strictly separated.
LEGAL_REGULATION = "legal_regulation"   # binding (UN/ECE, FMVSS, EU directives)
RATING_PROTOCOL = "rating_protocol"     # consumer assessment (Euro NCAP, IIHS)
ENGINEERING_REFERENCE = "reference"     # ISO / internal CAE references (non-binding)

DOC_TYPE_LABEL = {
    LEGAL_REGULATION: "Legal regulation (binding)",
    RATING_PROTOCOL: "Rating protocol (consumer assessment — not legally binding)",
    ENGINEERING_REFERENCE: "Engineering reference (non-binding)",
}


@dataclass(frozen=True)
class DocumentMeta:
    code: str                       # internal regulation code, e.g. "UN_R14"
    display_name: str               # human label, e.g. "UN R14"
    full_title: str
    doc_type: str                   # one of the constants above
    authority: str                  # issuing body
    indexed_revision: str | None    # the revision the corpus copy maps to
    known_revisions: tuple[str, ...] = ()  # other revisions known to exist
    in_force: str | None = None
    legal_reference: str | None = None
    verified: bool = False          # True only when revision was confirmed

    @property
    def is_legal(self) -> bool:
        return self.doc_type == LEGAL_REGULATION

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
        authority="UNECE (UN Regulation)",
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
        authority="UNECE (UN Regulation)",
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
        doc_type=LEGAL_REGULATION, authority="UNECE (UN Regulation)",
        indexed_revision=None, verified=False,
    ),
    "UN_R94": DocumentMeta(
        code="UN_R94", display_name="UN R94",
        full_title="Protection of occupants in the event of a frontal collision",
        doc_type=LEGAL_REGULATION, authority="UNECE (UN Regulation)",
        indexed_revision=None, verified=False,
    ),
    "UN_R95": DocumentMeta(
        code="UN_R95", display_name="UN R95",
        full_title="Protection of occupants in the event of a lateral collision",
        doc_type=LEGAL_REGULATION, authority="UNECE (UN Regulation)",
        indexed_revision=None, verified=False,
    ),
    "UN_R135": DocumentMeta(
        code="UN_R135", display_name="UN R135",
        full_title="Protection of occupants in the event of a pole side impact",
        doc_type=LEGAL_REGULATION, authority="UNECE (UN Regulation)",
        indexed_revision=None, verified=False,
    ),
    "UN_R137": DocumentMeta(
        code="UN_R137", display_name="UN R137",
        full_title="Frontal impact with focus on restraint systems",
        doc_type=LEGAL_REGULATION, authority="UNECE (UN Regulation)",
        indexed_revision=None, verified=False,
    ),
    "FMVSS": DocumentMeta(
        code="FMVSS", display_name="FMVSS",
        full_title="Federal Motor Vehicle Safety Standards (49 CFR Part 571)",
        doc_type=LEGAL_REGULATION, authority="NHTSA (United States)",
        indexed_revision=None, verified=False,
    ),
    "EURO_NCAP": DocumentMeta(
        code="EURO_NCAP", display_name="Euro NCAP",
        full_title="Euro NCAP assessment protocols",
        doc_type=RATING_PROTOCOL, authority="Euro NCAP (consumer programme)",
        indexed_revision=None, verified=False,
    ),
    "CAE_REFERENCE": DocumentMeta(
        code="CAE_REFERENCE", display_name="CAE Companion",
        full_title="CAE Companion handbook (engineering reference — not a legal regulation)",
        doc_type=ENGINEERING_REFERENCE, authority="Internal reference",
        indexed_revision=None, verified=False,
    ),
    "SAFETY_REFERENCE": DocumentMeta(
        code="SAFETY_REFERENCE", display_name="Safety Companion",
        full_title="Safety Companion handbook (engineering reference — not a legal regulation)",
        doc_type=ENGINEERING_REFERENCE, authority="Internal reference",
        indexed_revision=None, verified=False,
    ),
}

_UNKNOWN = DocumentMeta(
    code="UNKNOWN",
    display_name="Unknown document",
    full_title="Unregistered source",
    doc_type=ENGINEERING_REFERENCE,
    authority="Unknown",
    indexed_revision=None,
    verified=False,
)


def get_document_meta(regulation_code: str | None) -> DocumentMeta:
    """Look up provenance for a regulation code; never raises."""
    if not regulation_code:
        return _UNKNOWN
    return _REGISTRY.get(regulation_code.strip().upper(), _UNKNOWN)


def doc_type_label(doc_type: str) -> str:
    return DOC_TYPE_LABEL.get(doc_type, doc_type)
