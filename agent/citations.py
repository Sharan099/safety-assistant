"""Citation extraction and faithfulness gates for agent outputs."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

CITATION_RE = re.compile(
    r"\[(?P<reg>[^\]]+?)\s*§(?P<section>[^,\]\s]+)\s*,\s*p\.(?P<page>\d+|\?)\]"
)

# Split on sentence-ish boundaries while keeping trailing punctuation with the clause.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


class ClaimCheck(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    grounded: bool = False
    reason: str = ""


def extract_citations(text: str) -> list[str]:
    return [m.group(0) for m in CITATION_RE.finditer(text or "")]


def allowed_citation_set(citations: Iterable[str]) -> set[str]:
    return {c.strip() for c in citations if c and str(c).strip()}


def split_claims(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Tables / multi-line memos: gate per line so citations stay attached to their cell.
    if "\n" in text and ("|" in text or text.lstrip().startswith("#")):
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    parts = _SENT_SPLIT.split(text)
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Attach trailing "… 1.2 [cite]" fragments to the previous sentence.
        bare = CITATION_RE.sub("", part).strip(" .;:")
        if merged and extract_citations(part) and len(bare.split()) <= 3:
            merged[-1] = f"{merged[-1]} {part}"
            continue
        merged.append(part)
    return merged


def _looks_factual(sentence: str) -> bool:
    s = sentence.strip()
    if not s or s.startswith("#") or s.startswith("---"):
        return False
    # Markdown table rows: only gate cells that assert facts without a citation / Note.
    if s.startswith("|"):
        if re.match(r"^\|[\s\-:|]+\|$", s):
            return False
        if "regulation" in s.lower() and "cited" in s.lower():
            return False
        if "note:" in s.lower() or CITATION_RE.search(s):
            return False
        # Bare table cell with numbers/limits and no cite → flag
        return bool(re.search(r"\d", s) or re.search(r"\b(shall|must|limit)\b", s, re.I))
    if re.match(r"^[-*|]+$", s):
        return False
    lower = s.lower()
    if lower.startswith(("note:", "disclaimer:", "mock provider", "(mock")):
        return False
    if "not in the local index" in lower or "insufficient evidence" in lower:
        return False
    if lower.startswith("| regulation |") or lower.startswith("|---|"):
        return False
    # Pure instructions / soft language without numbers or limits often still need cites
    # if they assert requirements — treat as factual when they contain digits or "shall"/"must"/limit words.
    if re.search(r"\d", s) or re.search(
        r"\b(shall|must|limit|maximum|minimum|requirement|criterion|deflection|hic|tti|vc)\b",
        s,
        re.I,
    ):
        return True
    # Short connective lines without substance
    if len(s.split()) < 6:
        return False
    return True


def check_claims(
    text: str,
    *,
    allowed: Sequence[str] | set[str],
) -> list[ClaimCheck]:
    allowed_set = allowed_citation_set(allowed)
    out: list[ClaimCheck] = []
    for sent in split_claims(text):
        if not _looks_factual(sent):
            continue
        cites = extract_citations(sent)
        if not cites:
            out.append(
                ClaimCheck(
                    text=sent,
                    citations=[],
                    grounded=False,
                    reason="missing_citation",
                )
            )
            continue
        bad = [c for c in cites if c not in allowed_set]
        if bad and allowed_set:
            out.append(
                ClaimCheck(
                    text=sent,
                    citations=cites,
                    grounded=False,
                    reason=f"unknown_citation:{bad[0]}",
                )
            )
        else:
            out.append(
                ClaimCheck(
                    text=sent,
                    citations=cites,
                    grounded=True,
                    reason="ok",
                )
            )
    return out


def citation_coverage(checks: Sequence[ClaimCheck]) -> float:
    if not checks:
        return 1.0
    return sum(1 for c in checks if c.grounded) / len(checks)


def strip_ungrounded_sentences(text: str, *, allowed: Sequence[str] | set[str]) -> str:
    """Drop factual sentences that fail the citation gate; keep structure/headers."""
    allowed_set = allowed_citation_set(allowed)
    lines = (text or "").splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith(("#", "|", "-", "*")) and not _looks_factual(stripped):
            kept.append(line)
            continue
        # Multi-sentence lines: filter claim-by-claim
        claims = split_claims(stripped) or [stripped]
        ok_parts: list[str] = []
        for claim in claims:
            if not _looks_factual(claim):
                ok_parts.append(claim)
                continue
            cites = extract_citations(claim)
            if cites and (not allowed_set or all(c in allowed_set for c in cites)):
                ok_parts.append(claim)
            # else drop
        if ok_parts:
            kept.append(" ".join(ok_parts))
        elif stripped.startswith(("#", "|")):
            kept.append(line)
    body = "\n".join(kept).strip()
    if not body:
        return (
            "Insufficient grounded evidence in retrieved passages to support a cited answer. "
            "No uncited claims were retained."
        )
    return body


def enforce_grounded_text(text: str, *, allowed: Sequence[str] | set[str]) -> tuple[str, list[ClaimCheck]]:
    checks = check_claims(text, allowed=allowed)
    if all(c.grounded for c in checks):
        return text, checks
    cleaned = strip_ungrounded_sentences(text, allowed=allowed)
    return cleaned, check_claims(cleaned, allowed=allowed)
