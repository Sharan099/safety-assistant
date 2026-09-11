"""Programmatic citation validation (CLAUDE.md §10).

- every cited evidence id must exist in the bundle;
- every claim needs at least one evidence id;
- numbers (with decimal comma or point) that appear in a REQUIREMENT claim must
  appear in at least one cited evidence text (or its parent/related context) —
  the numeric-fidelity check that catches silent rounding or invention;
- citation views are built from the evidence records, never from model text.
"""

from __future__ import annotations

import re

from safety_assistant.generation.schemas import (
    CitationView,
    Claim,
    ClaimValidation,
    GroundedDraft,
    ValidationReport,
)
from safety_assistant.retrieval.context import Evidence

_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\w])")
_EVIDENCE_ID_RE = re.compile(r"^E\d+$")


def canonical_number(raw: str) -> str:
    """'1,3' → '1.3' (decimal comma), '1,000' → '1000' (thousands), '3,500 kg' → '3500',
    '1.25' → '1.25'. A comma followed by exactly three digits with no other separator
    is a thousands separator; any other comma is a decimal mark."""
    t = raw.replace(" ", "")
    if "," in t and "." not in t:
        head, _, tail = t.rpartition(",")
        t = (
            f"{head}{tail}"
            if len(tail) == 3 and t.count(",") >= 1 and head.replace(",", "").isdigit()
            else f"{head}.{tail}"
        )
        t = t.replace(",", "")
    elif "," in t:
        t = t.replace(",", "")
    try:
        return str(float(t)).rstrip("0").rstrip(".") if "." in t else str(int(t))
    except ValueError:
        return t


def claimed_numbers(text: str) -> set[str]:
    """Canonical numbers in a claim, ignoring one-digit list markers."""
    return {canonical_number(m.group(1)) for m in _NUMBER_RE.finditer(text) if not _is_trivial(m.group(1))}


def _evidence_text(e: Evidence) -> str:
    parts = [e.content, e.parent_context or ""] + [r.excerpt for r in e.related]
    return "\n".join(parts)


def validate_draft(draft: GroundedDraft, evidence: list[Evidence]) -> tuple[list[Claim], ValidationReport]:
    by_id = {e.evidence_id: e for e in evidence}
    results: list[ClaimValidation] = []
    unknown: list[str] = []
    kept: list[Claim] = []
    for i, claim in enumerate(draft.claims):
        if not claim.evidence_ids:
            results.append(ClaimValidation(claim_index=i, status="NO_EVIDENCE"))
            continue
        bad = [x for x in claim.evidence_ids if not _EVIDENCE_ID_RE.match(x) or x not in by_id]
        if bad:
            unknown.extend(bad)
            results.append(ClaimValidation(claim_index=i, status="UNSUPPORTED_EVIDENCE_ID", detail=", ".join(bad)))
            continue
        if claim.kind == "REQUIREMENT":
            cited = "\n".join(_evidence_text(by_id[x]) for x in claim.evidence_ids)
            have = {canonical_number(m.group(1)) for m in _NUMBER_RE.finditer(cited)}
            unmatched = sorted(n for n in claimed_numbers(claim.text) if n not in have)
            if unmatched:
                results.append(
                    ClaimValidation(
                        claim_index=i,
                        status="NUMERIC_MISMATCH",
                        detail=f"not in cited evidence: {', '.join(unmatched)}",
                    )
                )
                continue
        results.append(ClaimValidation(claim_index=i, status="SUPPORTED"))
        kept.append(claim)
    report = ValidationReport(
        ok=all(r.status == "SUPPORTED" for r in results) and not unknown,
        claims=results,
        unknown_evidence_ids=sorted(set(unknown)),
        dropped_claims=len(draft.claims) - len(kept),
    )
    return kept, report


def _is_trivial(n: str) -> bool:
    """Single digits like list markers ('1', '2') are not treated as regulatory values."""
    return n.isdigit() and len(n) == 1


def citation_views(evidence: list[Evidence], used_ids: set[str] | None = None) -> list[CitationView]:
    out = []
    for e in evidence:
        if used_ids is not None and e.evidence_id not in used_ids:
            continue
        out.append(
            CitationView(
                evidence_id=e.evidence_id,
                label=e.citation_label,
                regulation_key=e.regulation_key,
                version_label=e.version_label,
                section_path=e.section_path,
                page_start=e.page_start,
                page_end=e.page_end,
                source_sha256=e.source_sha256,
                source_uri=e.source_uri,
                valid_from=e.valid_from,
                valid_to=e.valid_to,
                version_status=e.version_status,
            )
        )
    return out
