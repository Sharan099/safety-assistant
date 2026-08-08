"""Post-extraction guard: numbered UNECE clauses must not be caption-typed.

Docling's layout model occasionally absorbs the clause *after* a figure into the
``caption`` role (Stage 1: R94 Figure 3 → ``5.2.1.7`` TCFC / 8 kN). Chunking
skips ``CAPTION`` items, so those clauses silently vanish from the index.

This module scans every extracted caption-typed element for clause-number
patterns and flags hits so the failure is caught at ingestion automatically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from docling_core.types.doc import DocItemLabel, DoclingDocument

logger = logging.getLogger(__name__)

# UNECE-style clause numbers inside caption text, e.g. "5.2.1.7." or "6.3.5.1".
# Require ≥ two dots (three numeric segments) so "Figure 3" / "Table 1" never match.
CLAUSE_IN_CAPTION_RE = re.compile(r"(?m)(?<!\d)(\d+(?:\.\d+){2,})\.?(?=\s|[A-Za-z]|$)")


@dataclass(frozen=True)
class CaptionClauseViolation:
    """One caption-typed element whose text looks like a numbered clause."""

    page_number: int | None
    clause_numbers: tuple[str, ...]
    text_preview: str
    label: str = "caption"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label_name(item: Any) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "value", None) or str(label or type(item).__name__)


def _page_of(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page = getattr(prov[0], "page_no", None)
    return int(page) if page is not None else None


def _item_text(item: Any) -> str:
    return (getattr(item, "text", None) or "").strip()


def _is_caption_item(item: Any) -> bool:
    label = getattr(item, "label", None)
    if label == DocItemLabel.CAPTION:
        return True
    return _label_name(item).lower() == "caption"


def find_clause_numbers_in_text(text: str) -> list[str]:
    """Return unique clause-number hits (``\\d+.\\d+.\\d+``…) in *text*."""
    seen: list[str] = []
    for m in CLAUSE_IN_CAPTION_RE.finditer(text or ""):
        num = m.group(1)
        if num not in seen:
            seen.append(num)
    return seen


def find_caption_clause_violations(
    doc: DoclingDocument,
    *,
    pages: Iterable[int] | None = None,
) -> list[CaptionClauseViolation]:
    """Scan caption-typed elements; return those containing clause-number patterns."""
    page_filter = {int(p) for p in pages} if pages is not None else None
    violations: list[CaptionClauseViolation] = []
    for item, _lvl in doc.iterate_items():
        if not _is_caption_item(item):
            continue
        page = _page_of(item)
        if page_filter is not None and (page is None or page not in page_filter):
            continue
        text = _item_text(item)
        if not text:
            continue
        clauses = find_clause_numbers_in_text(text)
        if not clauses:
            continue
        violations.append(
            CaptionClauseViolation(
                page_number=page,
                clause_numbers=tuple(clauses),
                text_preview=text.splitlines()[0][:160],
            )
        )
    return violations


def validate_no_clause_as_caption(
    doc: DoclingDocument,
    *,
    pages: Iterable[int] | None = None,
    raise_on_violation: bool = False,
) -> list[CaptionClauseViolation]:
    """Flag caption-typed elements that contain numbered clauses.

    Always logs a warning when violations are found. When
    ``raise_on_violation`` is True, raises ``CaptionClauseGuardError`` so CI /
    strict ingest can fail closed.
    """
    violations = find_caption_clause_violations(doc, pages=pages)
    if not violations:
        logger.info("Caption clause guard: OK (no clause-as-caption hits)")
        return []

    pages_hit = sorted({v.page_number for v in violations if v.page_number is not None})
    logger.warning(
        "Caption clause guard: %d caption element(s) contain clause-number "
        "pattern(s) on page(s) %s — Docling mislabeled body clauses as caption",
        len(violations),
        pages_hit or "?",
    )
    for v in violations:
        logger.warning(
            "  page=%s clauses=%s preview=%r",
            v.page_number,
            list(v.clause_numbers),
            v.text_preview,
        )
    if raise_on_violation:
        raise CaptionClauseGuardError(violations)
    return violations


def flagged_pages(violations: Iterable[CaptionClauseViolation]) -> list[int]:
    """Unique 1-indexed pages that need LightOnOCR remediation."""
    return sorted(
        {int(v.page_number) for v in violations if v.page_number is not None}
    )


class CaptionClauseGuardError(RuntimeError):
    """Raised when caption-typed elements contain numbered clause text."""

    def __init__(self, violations: list[CaptionClauseViolation]) -> None:
        self.violations = violations
        pages = flagged_pages(violations)
        super().__init__(
            f"{len(violations)} caption element(s) contain clause numbers "
            f"on page(s) {pages}: "
            + "; ".join(v.text_preview for v in violations[:3])
        )
