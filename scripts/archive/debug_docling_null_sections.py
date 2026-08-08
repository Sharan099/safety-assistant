"""Debug: show how Docling represents annex/table headings for audit null-section samples.

Read-only inspection â€” does not modify the index.

Usage::

    uv run python scripts/archive/debug_docling_null_sections.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from ingestion.parse import parse_pdf

ANNEX_HINTS = (
    "annex",
    "test procedures for the vehicles",
    "honeycomb",
    "certification procedure",
    "deformable barrier",
)
TABLE_HINTS = (
    "femur force",
    "frequency response",
    "honeycomb axes",
    "head performance criterion",
)


def _pick_samples(samples: list[dict], hints: tuple[str, ...], n: int = 5) -> list[dict]:
    picked: list[dict] = []
    for s in samples:
        blob = f"{s.get('section_title') or ''} {s.get('text_preview') or ''}".lower()
        if any(h in blob for h in hints):
            picked.append(s)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for s in samples:
            if s not in picked:
                picked.append(s)
            if len(picked) >= n:
                break
    return picked[:n]


def _dump_context(doc, *, needle: str, page_hint: int | None, window: int = 8) -> None:
    needle_l = needle.lower().strip()
    items = list(doc.iterate_items())
    hit_idxs: list[int] = []
    for i, (item, level) in enumerate(items):
        text = (getattr(item, "text", None) or "").strip()
        if needle_l and needle_l[:40] in text.lower():
            hit_idxs.append(i)
        label = str(getattr(item, "label", "") or "")
        if "table" in label.lower() and needle_l[:20] in text.lower():
            hit_idxs.append(i)

    if not hit_idxs and page_hint is not None:
        # Fall back: print all headings near that page
        print(f"  (no exact text hit for {needle!r}; dumping headings on page ~{page_hint})")
        for item, level in items:
            prov = getattr(item, "prov", None) or []
            page = getattr(prov[0], "page_no", None) if prov else None
            if page is None or abs(int(page) - int(page_hint)) > 1:
                continue
            from ingestion.chunk import _is_heading, _heading_level, _item_text

            if not _is_heading(item) and str(getattr(item, "label", "")) != "table":
                continue
            kind = type(item).__name__
            label = getattr(item, "label", None)
            hlev = _heading_level(item, level) if _is_heading(item) else level
            print(
                f"  page={page} walk_level={level} heading_level={hlev} "
                f"type={kind} label={label} text={_item_text(item)[:120]!r}"
            )
        return

    from ingestion.chunk import _heading_level, _is_heading, _item_text, _page_number

    for hi in hit_idxs[:2]:
        start = max(0, hi - window)
        end = min(len(items), hi + window + 1)
        print(f"  --- context around item[{hi}] ---")
        for j in range(start, end):
            item, level = items[j]
            mark = ">>>" if j == hi else "   "
            kind = type(item).__name__
            label = getattr(item, "label", None)
            page = _page_number(item)
            text = _item_text(item).replace("\n", " ")[:140]
            extra = ""
            if _is_heading(item):
                extra = f" heading_level={_heading_level(item, level)}"
            print(
                f"  {mark} [{j}] page={page} walk_level={level}{extra} "
                f"type={kind} label={label} text={text!r}"
            )


def main() -> int:
    load_dotenv(ROOT / ".env")
    samples_path = ROOT / "audit_samples.json"
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = data.get("samples") or []

    annex_samples = _pick_samples(samples, ANNEX_HINTS, 5)
    table_samples = _pick_samples(
        [s for s in samples if s.get("content_type") == "table" or "criterion" in (s.get("section_title") or "").lower() or "frequency" in (s.get("section_title") or "").lower()],
        TABLE_HINTS,
        5,
    )
    # Ensure femur + frequency are included even if classified as clause
    for want in ("Femur force criterion", "Frequency response curve"):
        for s in samples:
            if (s.get("section_title") or "") == want and s not in table_samples:
                table_samples.insert(0, s)
        table_samples = table_samples[:5]

    pages = sorted(
        {
            int(s["page_number"])
            for s in annex_samples + table_samples
            if s.get("page_number") is not None
        }
    )
    if not pages:
        print("No page numbers in samples")
        return 1

    pdf = ROOT / "data" / "pdfs" / "UN_R94.pdf"
    # Parse a window covering sample pages (Docling page_range is 1-indexed inclusive)
    lo, hi = max(1, min(pages) - 1), max(pages) + 1
    print(f"Parsing {pdf.name} pages {lo}-{hi} for Docling structure dumpâ€¦")
    doc = parse_pdf(pdf, page_range=(lo, hi), export_dir=ROOT / "data" / "docling")

    print("\n========== ANNEX-RELATED NULL-SECTION SAMPLES (Docling context) ==========")
    for i, s in enumerate(annex_samples, 1):
        print(
            f"\n[{i}] audit title={s.get('section_title')!r} "
            f"page={s.get('page_number')} type={s.get('content_type')}"
        )
        _dump_context(
            doc,
            needle=str(s.get("section_title") or ""),
            page_hint=s.get("page_number"),
        )

    print("\n========== TABLE / CRITERION NULL-SECTION SAMPLES ==========")
    for i, s in enumerate(table_samples, 1):
        print(
            f"\n[{i}] audit title={s.get('section_title')!r} "
            f"page={s.get('page_number')} type={s.get('content_type')}"
        )
        print(f"    preview: {(s.get('text_preview') or '')[:220]}")
        _dump_context(
            doc,
            needle=str(s.get("section_title") or ""),
            page_hint=s.get("page_number"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
