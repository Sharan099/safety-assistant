"""Post-Docling page validator that flags LightOnOCR fallback candidates.

Runs immediately after Docling extraction (before chunking). Flags are
reason-coded so review / reconciliation know *why* a page was selected.

Triggers
--------
1. ``clause_in_caption`` — caption-typed element contains a UNECE clause number
   (the tibia-force / Figure 3 failure class).
2. ``text_density_anomaly`` — page text length is significantly below the
   rolling baseline for the same document (dropped footnotes / formulas).
3. ``figure_or_table`` — page contains a figure or table (highest-risk zones;
   flagged preemptively for verification).
4. ``section_discontinuity`` — last clause on page N and first on page N+1 are
   not plausible neighbors in the document numbering scheme.

Output is persisted next to the Docling export as
``{stem}.extraction_flags.json``.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from docling_core.types.doc import DocItemLabel, DoclingDocument

from ingestion.caption_guard import (
    find_caption_clause_violations,
    find_clause_numbers_in_text,
)

logger = logging.getLogger(__name__)

_CLAUSE_LINE_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\.\s+\S")
_CLAUSE_ANY_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+){1,})\.?(?=\s|[A-Za-z]|$)")

# Density: flag when page chars < baseline * ratio (after min baseline floor).
_DENSITY_RATIO = 0.40
_DENSITY_MIN_BASELINE = 400  # ignore thin pages in baseline (covers / blanks)
_DENSITY_ROLLING_WINDOW = 11  # odd window centered on the page


@dataclass(frozen=True)
class PageFlag:
    """One trigger firing on one page."""

    page_number: int
    trigger: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionValidationResult:
    """Per-document validation outcome."""

    flags: list[PageFlag] = field(default_factory=list)
    page_text_lengths: dict[int, int] = field(default_factory=dict)
    page_clause_sequences: dict[int, list[str]] = field(default_factory=dict)
    density_baseline: dict[int, float] = field(default_factory=dict)

    @property
    def flagged_pages(self) -> list[int]:
        return sorted({f.page_number for f in self.flags})

    def flags_for_page(self, page: int) -> list[PageFlag]:
        return [f for f in self.flags if f.page_number == page]

    def triggers_for_page(self, page: int) -> list[str]:
        return [f.trigger for f in self.flags_for_page(page)]

    def to_dict(self) -> dict[str, Any]:
        by_page: dict[str, list[dict[str, Any]]] = {}
        for f in self.flags:
            by_page.setdefault(str(f.page_number), []).append(f.to_dict())
        return {
            "flagged_pages": self.flagged_pages,
            "flag_count": len(self.flags),
            "flags_by_page": by_page,
            "flags": [f.to_dict() for f in self.flags],
            "page_text_lengths": {str(k): v for k, v in sorted(self.page_text_lengths.items())},
            "density_baseline": {
                str(k): round(v, 1) for k, v in sorted(self.density_baseline.items())
            },
            "page_clause_sequences": {
                str(k): v for k, v in sorted(self.page_clause_sequences.items())
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


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


def _is_figure_or_table(item: Any) -> bool:
    from docling_core.types.doc import TableItem

    if isinstance(item, TableItem):
        return True
    label = getattr(item, "label", None)
    if label in (
        DocItemLabel.TABLE,
        DocItemLabel.PICTURE,
        getattr(DocItemLabel, "CHART", None),
    ):
        return True
    name = _label_name(item).lower()
    return name in {"table", "picture", "figure", "chart"}


def _collect_page_texts(doc: DoclingDocument) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}
    for item, _lvl in doc.iterate_items():
        page = _page_of(item)
        if page is None:
            continue
        text = _item_text(item)
        if not text:
            label = _label_name(item).lower()
            if label in {"picture", "figure", "table"}:
                text = f"[{label}]"
            else:
                continue
        pages.setdefault(int(page), []).append(text)
    return pages


def page_clause_sequence(texts: Sequence[str]) -> list[str]:
    """Ordered unique-ish clause ids appearing at line starts in page texts."""
    seq: list[str] = []
    for text in texts:
        for m in _CLAUSE_LINE_RE.finditer(text or ""):
            num = m.group(1)
            if not seq or seq[-1] != num:
                seq.append(num)
    return seq


def _parse_clause_tuple(num: str) -> tuple[int, ...] | None:
    parts = (num or "").strip().rstrip(".").split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def clauses_are_plausible_neighbors(prev: str, nxt: str) -> bool:
    """True when *nxt* could reasonably follow *prev* across a page break.

    Accepts: exact next sibling, child of prev, jump to a new top-level / annex
    style id, or same-prefix continuation within a small sibling gap (≤ 2).
    Rejects: clear regressions (nxt < prev under same parent) or large sibling
    gaps that suggest a dropped mid-page clause.
    """
    a = _parse_clause_tuple(prev)
    b = _parse_clause_tuple(nxt)
    if a is None or b is None:
        # Annex / Appendix / non-numeric — treat as structural boundary, OK.
        return True
    if a == b:
        return True
    # Child continuation: 5.2 → 5.2.1
    if len(b) > len(a) and b[: len(a)] == a:
        return True
    # Parent rewind then sibling: 5.2.1.7 → 5.2.2
    if len(b) < len(a) and a[: len(b)][:-1] == b[:-1]:
        return abs(b[-1] - a[len(b) - 1]) <= 2 if len(a) >= len(b) else True
    # Same depth siblings
    if len(a) == len(b) and a[:-1] == b[:-1]:
        gap = b[-1] - a[-1]
        return 0 < gap <= 2
    # Prefix share with small drift
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    if shared == 0:
        # New major section (e.g. 5.x → 6.x) or annex jump
        return abs(a[0] - b[0]) <= 1 or b[0] > a[0]
    if shared == len(b):
        return True
    # Divergence at shared+1
    if shared < min(len(a), len(b)):
        gap = b[shared] - a[shared]
        return gap >= 0 and gap <= 2
    return True


def _rolling_baseline(lengths: dict[int, int], page: int) -> float:
    pages = sorted(lengths)
    if not pages:
        return 0.0
    # Use nearby "normal" pages (≥ min baseline) as reference.
    half = _DENSITY_ROLLING_WINDOW // 2
    try:
        idx = pages.index(page)
    except ValueError:
        return float(statistics.median(lengths.values()) if lengths else 0.0)
    window_pages = pages[max(0, idx - half) : idx + half + 1]
    vals = [
        lengths[p]
        for p in window_pages
        if p != page and lengths[p] >= _DENSITY_MIN_BASELINE
    ]
    if len(vals) < 3:
        vals = [v for v in lengths.values() if v >= _DENSITY_MIN_BASELINE]
    if not vals:
        vals = list(lengths.values())
    return float(statistics.median(vals)) if vals else 0.0


def validate_extraction(
    doc: DoclingDocument,
    *,
    density_ratio: float = _DENSITY_RATIO,
    flag_figures_tables: bool = True,
    flag_density: bool = True,
    flag_discontinuity: bool = True,
    flag_clause_in_caption: bool = True,
) -> ExtractionValidationResult:
    """Scan a DoclingDocument and return reason-coded page flags."""
    result = ExtractionValidationResult()
    page_texts = _collect_page_texts(doc)

    for page, texts in page_texts.items():
        joined = "\n".join(texts)
        result.page_text_lengths[page] = len(joined)
        result.page_clause_sequences[page] = page_clause_sequence(texts)

    # 1) Clause-in-caption
    if flag_clause_in_caption:
        for v in find_caption_clause_violations(doc):
            if v.page_number is None:
                continue
            result.flags.append(
                PageFlag(
                    page_number=int(v.page_number),
                    trigger="clause_in_caption",
                    detail=(
                        f"caption contains clause(s) {list(v.clause_numbers)}: "
                        f"{v.text_preview!r}"
                    ),
                    evidence={
                        "clause_numbers": list(v.clause_numbers),
                        "text_preview": v.text_preview,
                    },
                )
            )

    # 2) Text-density anomaly
    if flag_density and result.page_text_lengths:
        for page, length in result.page_text_lengths.items():
            baseline = _rolling_baseline(result.page_text_lengths, page)
            result.density_baseline[page] = baseline
            if baseline < _DENSITY_MIN_BASELINE:
                continue
            if length < baseline * density_ratio:
                result.flags.append(
                    PageFlag(
                        page_number=page,
                        trigger="text_density_anomaly",
                        detail=(
                            f"page text length {length} < "
                            f"{density_ratio:.0%} of baseline {baseline:.0f}"
                        ),
                        evidence={
                            "length": length,
                            "baseline": round(baseline, 1),
                            "ratio": round(length / baseline, 3) if baseline else 0.0,
                        },
                    )
                )

    # 3) Figure / table pages
    if flag_figures_tables:
        fig_pages: set[int] = set()
        for item, _lvl in doc.iterate_items():
            if not _is_figure_or_table(item):
                continue
            page = _page_of(item)
            if page is None:
                continue
            fig_pages.add(int(page))
        for page in sorted(fig_pages):
            result.flags.append(
                PageFlag(
                    page_number=page,
                    trigger="figure_or_table",
                    detail="page contains figure/table element (preemptive)",
                    evidence={"has_figure_or_table": True},
                )
            )

    # 4) Section-number discontinuity across consecutive pages
    if flag_discontinuity:
        pages = sorted(result.page_clause_sequences)
        for i in range(len(pages) - 1):
            p0, p1 = pages[i], pages[i + 1]
            if p1 != p0 + 1:
                continue  # only adjacent page numbers
            seq0 = result.page_clause_sequences.get(p0) or []
            seq1 = result.page_clause_sequences.get(p1) or []
            if not seq0 or not seq1:
                continue
            last, first = seq0[-1], seq1[0]
            if clauses_are_plausible_neighbors(last, first):
                continue
            detail = (
                f"clause discontinuity page {p0}→{p1}: "
                f"last={last!r} first={first!r}"
            )
            evidence = {"last_clause": last, "first_clause": first, "pair": [p0, p1]}
            for page in (p0, p1):
                result.flags.append(
                    PageFlag(
                        page_number=page,
                        trigger="section_discontinuity",
                        detail=detail,
                        evidence=evidence,
                    )
                )

    logger.info(
        "Extraction validator: %d flag(s) on %d page(s)",
        len(result.flags),
        len(result.flagged_pages),
    )
    return result


def trigger_resolved_after_vlm(
    trigger: str,
    *,
    doc: DoclingDocument,
    page_number: int,
    vlm_markdown: str,
    original_length: int | None = None,
    baseline: float | None = None,
    density_ratio: float = _DENSITY_RATIO,
) -> tuple[bool, str]:
    """Return (resolved, note) for one trigger after LightOnOCR + Docling mutate."""
    if trigger == "clause_in_caption":
        left = find_caption_clause_violations(doc, pages=[page_number])
        if left:
            return False, f"clause_in_caption still present ({len(left)})"
        # Also require the clause text to appear in non-caption form or VLM md.
        return True, "clause_in_caption cleared"

    if trigger == "text_density_anomaly":
        vlm_len = len(vlm_markdown or "")
        if baseline and vlm_len >= baseline * density_ratio:
            return True, f"vlm density recovered ({vlm_len} vs baseline {baseline:.0f})"
        # Post-mutation Docling page length
        page_texts = _collect_page_texts(doc).get(page_number) or []
        new_len = len("\n".join(page_texts))
        if baseline and new_len >= baseline * density_ratio:
            return True, f"docling density recovered ({new_len})"
        if original_length and vlm_len > original_length * 1.25:
            return True, f"vlm longer than docling ({vlm_len}>{original_length})"
        return False, f"density still low (vlm={vlm_len}, docling={new_len})"

    if trigger == "figure_or_table":
        # Resolved when VLM recovers at least one figure/table cue or clauses.
        has_fig = bool(
            re.search(r"(?i)\b(figure|table)\s+\d+", vlm_markdown or "")
        ) or ("![image]" in (vlm_markdown or ""))
        has_clause = bool(_CLAUSE_LINE_RE.search(vlm_markdown or ""))
        if has_fig or has_clause:
            return True, "figure/table page verified via VLM output"
        return False, "vlm produced no figure/table/clause cues"

    if trigger == "section_discontinuity":
        # Soft: resolved if VLM has a plausible clause sequence on this page.
        vlm_clauses = _CLAUSE_LINE_RE.findall(vlm_markdown or "")
        if vlm_clauses:
            return True, f"vlm clauses present: {vlm_clauses[:6]}"
        return False, "vlm missing clause cues for discontinuity check"

    return False, f"unknown trigger {trigger!r}"


def write_review_queue_entry(
    *,
    review_dir: str | Path,
    regulation_id: str,
    pdf_stem: str,
    page_number: int,
    triggers: Sequence[str],
    unresolved: Sequence[dict[str, Any]],
    docling_markdown: str,
    vlm_markdown: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a side-by-side disagreement for human review."""
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    out = review_dir / f"{pdf_stem}_p{page_number:04d}.json"
    payload = {
        "regulation_id": regulation_id,
        "pdf_stem": pdf_stem,
        "page_number": page_number,
        "triggers": list(triggers),
        "unresolved": list(unresolved),
        "docling_markdown": docling_markdown,
        "vlm_markdown": vlm_markdown,
        **(extra or {}),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Also write a readable side-by-side markdown.
    md = review_dir / f"{pdf_stem}_p{page_number:04d}.md"
    md.write_text(
        f"# OCR review queue — {pdf_stem} page {page_number}\n\n"
        f"**Triggers:** {', '.join(triggers)}\n\n"
        f"## Unresolved\n```json\n{json.dumps(list(unresolved), indent=2)}\n```\n\n"
        f"## Docling\n\n{docling_markdown}\n\n"
        f"## LightOnOCR\n\n{vlm_markdown}\n",
        encoding="utf-8",
    )
    return out


def pages_needing_lighton(
    result: ExtractionValidationResult,
    *,
    include_triggers: Iterable[str] | None = None,
) -> list[int]:
    """Pages that should be re-processed with LightOnOCR."""
    allowed = set(include_triggers) if include_triggers is not None else None
    pages: set[int] = set()
    for f in result.flags:
        if allowed is not None and f.trigger not in allowed:
            continue
        pages.add(f.page_number)
    return sorted(pages)
