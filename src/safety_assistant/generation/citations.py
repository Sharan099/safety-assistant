"""Programmatic citation validation (ENGINEERING.md §10).

- every cited evidence id must exist in the bundle;
- every claim needs at least one evidence id;
- numbers (with decimal comma or point) that appear in a REQUIREMENT claim must
  appear in at least one cited evidence text (or its parent/related context) —
  the numeric-fidelity check that catches silent rounding or invention;
- a CALCULATION claim may state derived numbers (a unit conversion, a margin against a
  limit) but at least one of its numbers must come from the cited evidence; the answer
  carries a warning that derived values must be checked;
- a claim's citations are pruned to the ids that support it (numbers present, or the best
  content-word overlap), so a sibling regulation's twin clause is not cited for a claim it
  does not state;
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
from safety_assistant.retrieval.filters import shared_term_count, significant_tokens

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
    """Text a claim's numbers may come from: the chunk, its parent/related excerpts, and the evidence
    attributes the prompt exposes (clause path, citation label, version label, dates, pages) —
    models legitimately repeat "§7.4.1.4.2", "Rev.7" or "2021" when attributing a requirement."""
    parts = [e.content, e.parent_context or ""] + [r.excerpt for r in e.related]
    parts += [e.citation_label, e.section_path, e.version_label, e.regulation_key]
    parts += [str(d) for d in (e.valid_from, e.valid_to, e.published_at) if d]
    parts += [str(pg) for pg in (e.page_start, e.page_end) if pg]
    return "\n".join(parts)


def validate_draft(
    draft: GroundedDraft, evidence: list[Evidence], *, question: str = ""
) -> tuple[list[Claim], ValidationReport]:
    """Numbers the engineer stated in the question ("our result is 44 mm") are known inputs, not
    fabrications: a claim may repeat them next to the limit it cites. Every other number must be in
    the cited evidence."""
    by_id = {e.evidence_id: e for e in evidence}
    stated = claimed_numbers(question)
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
        if claim.kind in ("REQUIREMENT", "CALCULATION"):
            cited = "\n".join(_evidence_text(by_id[x]) for x in claim.evidence_ids)
            have = {canonical_number(m.group(1)) for m in _NUMBER_RE.finditer(cited)}
            claimed = claimed_numbers(claim.text)
            unmatched = sorted(n for n in claimed if n not in have and n not in stated)
            # A calculation may introduce derived numbers, but must start from numbers the evidence states.
            invalid = unmatched if claim.kind == "REQUIREMENT" else ([] if claimed & have else sorted(claimed))
            if invalid:
                results.append(
                    ClaimValidation(
                        claim_index=i,
                        status="NUMERIC_MISMATCH",
                        detail=f"not in cited evidence: {', '.join(invalid)}",
                    )
                )
                continue
        results.append(ClaimValidation(claim_index=i, status="SUPPORTED"))
        kept.append(_prune_citations(claim, by_id))
    report = ValidationReport(
        ok=all(r.status == "SUPPORTED" for r in results) and not unknown,
        claims=results,
        unknown_evidence_ids=sorted(set(unknown)),
        dropped_claims=len(draft.claims) - len(kept),
    )
    return kept, report


def _prune_citations(claim: Claim, by_id: dict[str, Evidence]) -> Claim:
    """Keep only the cited ids that actually support the claim: for a numeric claim the ones whose text
    contains its numbers, otherwise the ones sharing most of its content words. Models pad citations
    with every evidence block that mentions the topic (a sibling regulation's twin clause); that padding
    is what makes a citation wrong without making the claim wrong."""
    if len(claim.evidence_ids) <= 1:
        return claim
    numbers = claimed_numbers(claim.text)
    scored: list[tuple[float, str]] = []
    for eid in claim.evidence_ids:
        text = _evidence_text(by_id[eid])
        if numbers:
            have = {canonical_number(m.group(1)) for m in _NUMBER_RE.finditer(text)}
            scored.append((len(numbers & have) / len(numbers), eid))
        else:
            terms = significant_tokens(claim.text)
            scored.append((shared_term_count(claim.text, text) / len(terms) if terms else 1.0, eid))
    best = max(sc for sc, _ in scored)
    keep = [eid for sc, eid in scored if sc >= best and sc > 0] or list(claim.evidence_ids)
    return claim.model_copy(update={"evidence_ids": keep})


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
