"""Diff Docling vs LightOnOCR per-page markdown for the figure-boundary corpus.

Pairs ``docling_pages/page_NNN.md`` with ``lightonocr_pages/page_NNN.md``,
skips pages LightOn has not produced yet, compares section-number sets and
figure/caption content, and writes a ranked markdown report.

The known reference case (R94 Figure 3 / clause 5.2.1.7 misattribution) is
**test page 4** (source R94 p.12). It is always listed first in the report.

Usage::

    python scripts/diff_ocr_outputs.py
    python scripts/diff_ocr_outputs.py --out scripts/ocr_diff_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "ocr_compare" / "figure_boundary_v1"
DOCLING_DIR = OUT_DIR / "docling_pages"
LIGHTON_DIR = OUT_DIR / "lightonocr_pages"
MANIFEST = OUT_DIR / "manifest.json"
DEFAULT_REPORT = ROOT / "scripts" / "ocr_diff_report.md"

# UNECE-style clause numbers: 5.2.1.6, 6.3.5.1, Annex-ish 1.5., etc.
_SECTION_RE = re.compile(
    r"(?<![\w./])("
    r"\d+(?:\.\d+){1,5}"  # 5.2 / 5.2.1.6 / 6.3.5.1.2
    r")(?!\.\d)"  # don't leave a trailing unfinished segment
    r"(?=\s|[.:,;)\]]|$)"
)

_FIG_MENTION_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bfigure\s*\d+\b"
    r"|\bfig\.?\s*\d+\b"
    r"|<!--\s*picture\s*-->"
    r"|<!--\s*caption\s*-->"
    r"|!\[image\]\([^)]+\)"
    r"|\[\s*picture\s*\]"
    r")"
)

_CAPTION_LINE_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"<!--\s*caption\s*-->\s*$|"
    r"figure\s+\d+\b.*|"
    r"\*\*[^*]*criterion[^*]*\*\*"
    r")"
)

# Docling role pollution: numbered clause absorbed into caption.
_CLAUSE_AS_CAPTION_RE = re.compile(
    r"(?is)<!--\s*caption\s*-->\s*\n\s*(\d+(?:\.\d+){1,5}\.?[^\n]*)"
)

# R94 Figure 3 reference: test PDF page 4 == source R94 page 12.
REFERENCE_TEST_PAGE = 4
REFERENCE_LABEL = (
    "R94 Figure 3 / 5.2.1.7 misattribution (test p.4 = source R94 p.12)"
)


@dataclass
class PageDiff:
    page: int
    not_yet_processed: bool = False
    docling_chars: int = 0
    lighton_chars: int = 0
    docling_sections: set[str] = field(default_factory=set)
    lighton_sections: set[str] = field(default_factory=set)
    section_only_docling: set[str] = field(default_factory=set)
    section_only_lighton: set[str] = field(default_factory=set)
    section_disagreement: bool = False
    lighton_figure_extra: list[str] = field(default_factory=list)
    docling_clause_as_caption: list[str] = field(default_factory=list)
    figure_content_lighton_only: bool = False
    disagreement_score: float = 0.0
    diff_preview: list[str] = field(default_factory=list)
    source_hint: str = ""


def _page_num(path: Path) -> int | None:
    m = re.search(r"page_(\d+)\.md$", path.name, re.I)
    return int(m.group(1)) if m else None


def _normalize_text(text: str) -> str:
    # Strip Docling role markers for fairer textual compare.
    text = re.sub(r"<!--\s*\w+\s*-->", "\n", text)
    text = re.sub(r"!\[image\]\([^)]+\)", "[IMAGE]", text)
    text = re.sub(r"\[\s*picture\s*\]", "[IMAGE]", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_sections(text: str) -> set[str]:
    found: set[str] = set()
    for m in _SECTION_RE.finditer(text or ""):
        tok = m.group(1).rstrip(".")
        # Drop trivial decimals that are clearly measurements when alone?
        # Keep all multi-dot and 2-segment forms; skip lone "1.0" style if only one dot
        # and looks like a measurement (ends after unit context) — keep simple for now.
        parts = tok.split(".")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            # Keep 5.2-style; drop 1.0 / 0.81 style (leading zero or value-like).
            if parts[0] == "0" or (len(parts[1]) > 1 and parts[1].startswith("0")):
                continue
        found.add(tok)
    return found


def _figure_snippets(text: str, *, limit: int = 12) -> list[str]:
    lines = (text or "").splitlines()
    out: list[str] = []
    for i, ln in enumerate(lines):
        if _FIG_MENTION_RE.search(ln) or _CAPTION_LINE_RE.match(ln.strip()):
            snippet = ln.strip()
            if not snippet and i + 1 < len(lines):
                snippet = f"{ln.strip()} | {lines[i + 1].strip()}"
            if snippet and snippet not in out:
                out.append(snippet[:160])
            if len(out) >= limit:
                break
    return out


def _lighton_figure_missing_from_docling(d_text: str, l_text: str) -> list[str]:
    """Figure/caption lines in LightOn that have no counterpart cue in Docling."""
    d_norm = _normalize_text(d_text).lower()
    extras: list[str] = []
    for snip in _figure_snippets(l_text):
        key = re.sub(r"\s+", " ", snip).strip().lower()
        key = re.sub(r"[*_`#]+", "", key)
        # Require a distinctive fragment (≥ a figure label or short caption).
        probe = key[:80]
        if not probe:
            continue
        # Docling often has [picture] without the caption words LightOn recovered.
        fig_m = re.search(r"figure\s*(\d+)", probe, re.I)
        if fig_m:
            label = f"figure {fig_m.group(1)}"
            if label not in d_norm and f"fig. {fig_m.group(1)}" not in d_norm:
                extras.append(snip)
                continue
            # Label present — still flag if LightOn has caption prose Docling lacks.
            # e.g. "Femur force criterion"
            prose = re.sub(r"figure\s*\d+", "", probe, flags=re.I).strip(" :-*")
            if len(prose) >= 8 and prose not in d_norm:
                extras.append(snip)
            continue
        if "image" in probe or "picture" in probe:
            if "[image]" not in d_norm and "picture" not in d_norm:
                extras.append(snip)
            continue
        if probe not in d_norm:
            extras.append(snip)
    # Deduplicate preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for e in extras:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def _load_manifest_hints() -> dict[int, str]:
    if not MANIFEST.is_file():
        return {}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    hints: dict[int, str] = {}
    for row in data.get("pages") or []:
        tp = int(row.get("test_page") or 0)
        if not tp:
            continue
        reg = row.get("regulation_id") or row.get("regulation") or "?"
        sp = row.get("source_page")
        roles = ",".join(row.get("roles") or [])
        hints[tp] = f"{reg} source p.{sp}" + (f" [{roles}]" if roles else "")
    return hints


def diff_page(
    page: int,
    *,
    docling_path: Path | None,
    lighton_path: Path | None,
    source_hint: str = "",
) -> PageDiff:
    result = PageDiff(page=page, source_hint=source_hint)
    if lighton_path is None or not lighton_path.is_file():
        result.not_yet_processed = True
        return result
    if docling_path is None or not docling_path.is_file():
        result.not_yet_processed = True
        result.diff_preview = ["(docling page missing)"]
        return result

    d_raw = docling_path.read_text(encoding="utf-8", errors="replace")
    l_raw = lighton_path.read_text(encoding="utf-8", errors="replace")
    result.docling_chars = len(d_raw)
    result.lighton_chars = len(l_raw)
    result.docling_sections = _extract_sections(d_raw)
    result.lighton_sections = _extract_sections(l_raw)
    result.section_only_docling = result.docling_sections - result.lighton_sections
    result.section_only_lighton = result.lighton_sections - result.docling_sections
    result.section_disagreement = bool(
        result.section_only_docling or result.section_only_lighton
    )
    result.docling_clause_as_caption = [
        m.group(1).strip()[:120] for m in _CLAUSE_AS_CAPTION_RE.finditer(d_raw)
    ]
    result.lighton_figure_extra = _lighton_figure_missing_from_docling(d_raw, l_raw)
    result.figure_content_lighton_only = bool(result.lighton_figure_extra)

    d_lines = _normalize_text(d_raw).splitlines()
    l_lines = _normalize_text(l_raw).splitlines()
    udiff = list(
        unified_diff(
            d_lines,
            l_lines,
            fromfile=f"docling/page_{page:03d}.md",
            tofile=f"lighton/page_{page:03d}.md",
            lineterm="",
            n=1,
        )
    )
    # Keep a short preview (skip huge dumps).
    result.diff_preview = udiff[:80]

    # Disagreement score: section symmetric diff + clause-as-caption + figure extras.
    result.disagreement_score = (
        3.0 * len(result.section_only_docling | result.section_only_lighton)
        + 5.0 * len(result.docling_clause_as_caption)
        + 2.0 * len(result.lighton_figure_extra)
        + min(2.0, abs(result.docling_chars - result.lighton_chars) / 2000.0)
    )
    return result


def rank_pages(pages: list[PageDiff]) -> list[PageDiff]:
    comparable = [p for p in pages if not p.not_yet_processed]
    pending = [p for p in pages if p.not_yet_processed]
    comparable.sort(key=lambda p: (-p.disagreement_score, p.page))
    # Always pin reference page first when present.
    ref = next((p for p in comparable if p.page == REFERENCE_TEST_PAGE), None)
    if ref is not None:
        comparable = [ref] + [p for p in comparable if p.page != REFERENCE_TEST_PAGE]
    return comparable + pending


def write_report(pages: list[PageDiff], path: Path) -> dict[str, Any]:
    ranked = rank_pages(pages)
    processed = [p for p in pages if not p.not_yet_processed]
    pending = [p for p in pages if p.not_yet_processed]
    section_disagreements = [p for p in processed if p.section_disagreement]
    figure_extras = [p for p in processed if p.figure_content_lighton_only]
    clause_captions = [p for p in processed if p.docling_clause_as_caption]

    lines: list[str] = []
    lines.append("# Docling vs LightOnOCR — per-page OCR diff")
    lines.append("")
    lines.append(f"Corpus: `{OUT_DIR.as_posix()}`")
    lines.append(
        f"Paired pages: **{len(processed)}** processed, "
        f"**{len(pending)}** not yet processed (LightOn missing)."
    )
    lines.append(
        f"Section-number disagreements: **{len(section_disagreements)}** / {len(processed)}"
    )
    lines.append(
        f"Pages where LightOn has figure/caption text Docling lacks: "
        f"**{len(figure_extras)}** / {len(processed)}"
    )
    lines.append(
        f"Docling clause-as-caption pollution pages: **{len(clause_captions)}**"
    )
    lines.append("")
    lines.append("## Reference case (always first)")
    lines.append("")
    lines.append(f"**{REFERENCE_LABEL}**")
    lines.append("")

    ref = next((p for p in pages if p.page == REFERENCE_TEST_PAGE), None)
    if ref is None or ref.not_yet_processed:
        lines.append("_Reference page not yet processed by LightOn._")
    else:
        lines.append(f"- Source: `{ref.source_hint or 'n/a'}`")
        lines.append(
            f"- Section disagreement: **{'YES' if ref.section_disagreement else 'no'}**"
        )
        lines.append(
            f"- Docling sections: `{sorted(ref.docling_sections, key=_section_sort)}`"
        )
        lines.append(
            f"- LightOn sections: `{sorted(ref.lighton_sections, key=_section_sort)}`"
        )
        if ref.section_only_docling:
            lines.append(
                f"- Only in Docling: `{sorted(ref.section_only_docling, key=_section_sort)}`"
            )
        if ref.section_only_lighton:
            lines.append(
                f"- Only in LightOn: `{sorted(ref.section_only_lighton, key=_section_sort)}`"
            )
        if ref.docling_clause_as_caption:
            lines.append(
                "- Docling tagged numbered clause(s) as `caption`: "
                + "; ".join(f"`{c}`" for c in ref.docling_clause_as_caption)
            )
        else:
            lines.append("- Docling clause-as-caption: none detected on this page")
        if ref.lighton_figure_extra:
            lines.append(
                "- LightOn figure/caption extras vs Docling: "
                + "; ".join(f"`{e}`" for e in ref.lighton_figure_extra[:6])
            )
        # Explicit 5.2.1.7 attribution check
        d_path = DOCLING_DIR / f"page_{REFERENCE_TEST_PAGE:03d}.md"
        l_path = LIGHTON_DIR / f"page_{REFERENCE_TEST_PAGE:03d}.md"
        d_raw = d_path.read_text(encoding="utf-8", errors="replace")
        l_raw = l_path.read_text(encoding="utf-8", errors="replace")
        lines.append("")
        lines.append("### Attribution of `5.2.1.7`")
        lines.append("")
        if _CLAUSE_AS_CAPTION_RE.search(d_raw):
            lines.append(
                "- **Docling:** `5.2.1.7` (or sibling) appears under `<!-- caption -->` "
                "after Figure 3 — **misattribution confirmed**."
            )
        else:
            lines.append("- **Docling:** no clause-as-caption pattern matched.")
        if re.search(r"5\.2\.1\.7", l_raw) and not re.search(
            r"<!--\s*caption\s*-->\s*\n\s*5\.2\.1\.7", l_raw
        ):
            lines.append(
                "- **LightOn:** `5.2.1.7` present as body text after Figure 3 / caption "
                "— **boundary held**."
            )
        elif re.search(r"5\.2\.1\.7", l_raw):
            lines.append("- **LightOn:** `5.2.1.7` present (check role manually).")
        else:
            lines.append("- **LightOn:** `5.2.1.7` not found on page.")
        lines.append("")
        lines.append("<details><summary>Unified diff preview (normalized)</summary>")
        lines.append("")
        lines.append("```diff")
        lines.extend(ref.diff_preview[:60] or ["(no textual diff)"])
        lines.append("```")
        lines.append("")
        lines.append("</details>")

    lines.append("")
    lines.append("## Ranked pages (by disagreement score)")
    lines.append("")

    for p in ranked:
        if p.not_yet_processed:
            lines.append(f"### Page {p.page:03d} — not yet processed")
            lines.append("")
            continue
        if p.page == REFERENCE_TEST_PAGE:
            heading_note = " ★ REFERENCE"
        else:
            heading_note = ""
        lines.append(
            f"### Page {p.page:03d}{heading_note} — score={p.disagreement_score:.1f}"
        )
        if p.source_hint:
            lines.append(f"- Source: `{p.source_hint}`")
        lines.append(
            f"- Chars: Docling={p.docling_chars}, LightOn={p.lighton_chars}"
        )
        lines.append(
            f"- Section disagreement: **{'YES' if p.section_disagreement else 'no'}** "
            f"(docling={len(p.docling_sections)}, lighton={len(p.lighton_sections)})"
        )
        if p.section_only_docling:
            lines.append(
                f"  - only Docling: `{sorted(p.section_only_docling, key=_section_sort)[:20]}`"
            )
        if p.section_only_lighton:
            lines.append(
                f"  - only LightOn: `{sorted(p.section_only_lighton, key=_section_sort)[:20]}`"
            )
        if p.docling_clause_as_caption:
            lines.append(
                "- Docling clause-as-caption: "
                + "; ".join(f"`{c}`" for c in p.docling_clause_as_caption)
            )
        if p.figure_content_lighton_only:
            lines.append(
                "- LightOn figure/caption not in Docling: "
                + "; ".join(f"`{e}`" for e in p.lighton_figure_extra[:8])
            )
        if p.disagreement_score > 0 and p.diff_preview:
            lines.append("")
            lines.append("<details><summary>Diff preview</summary>")
            lines.append("")
            lines.append("```diff")
            lines.extend(p.diff_preview[:40])
            lines.append("```")
            lines.append("")
            lines.append("</details>")
        lines.append("")

    if pending:
        lines.append("## Not yet processed")
        lines.append("")
        lines.append(
            ", ".join(f"{p.page:03d}" for p in sorted(pending, key=lambda x: x.page))
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "processed": len(processed),
        "pending": len(pending),
        "section_disagreements": len(section_disagreements),
        "figure_extras": len(figure_extras),
        "clause_as_caption": len(clause_captions),
        "reference_page": REFERENCE_TEST_PAGE,
        "reference_section_disagreement": bool(
            ref and not ref.not_yet_processed and ref.section_disagreement
        ),
        "reference_clause_as_caption": bool(
            ref and ref.docling_clause_as_caption
        ),
        "report": str(path),
    }


def _section_sort(s: str) -> tuple:
    try:
        return tuple(int(p) for p in s.split("."))
    except ValueError:
        return (999, s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown report path (default: scripts/ocr_diff_report.md)",
    )
    ap.add_argument(
        "--docling-dir",
        type=Path,
        default=DOCLING_DIR,
    )
    ap.add_argument(
        "--lighton-dir",
        type=Path,
        default=LIGHTON_DIR,
    )
    args = ap.parse_args(argv)

    docling_dir: Path = args.docling_dir
    lighton_dir: Path = args.lighton_dir
    if not docling_dir.is_dir():
        print(f"Docling pages missing: {docling_dir}", file=sys.stderr)
        return 2

    docling_pages = {
        n: p
        for p in sorted(docling_dir.glob("page_*.md"))
        if (n := _page_num(p)) is not None
    }
    lighton_pages = {
        n: p
        for p in sorted(lighton_dir.glob("page_*.md"))
        if lighton_dir.is_dir() and (n := _page_num(p)) is not None
    }
    hints = _load_manifest_hints()
    all_nums = sorted(set(docling_pages) | set(lighton_pages) | {REFERENCE_TEST_PAGE})

    results = [
        diff_page(
            n,
            docling_path=docling_pages.get(n),
            lighton_path=lighton_pages.get(n),
            source_hint=hints.get(n, ""),
        )
        for n in all_nums
    ]
    summary = write_report(results, args.out)

    print("=== OCR diff summary ===")
    print(f"pages processed: {summary['processed']}")
    print(f"not yet processed: {summary['pending']}")
    print(f"section-number disagreements: {summary['section_disagreements']}")
    print(
        f"LightOn figure/caption extras vs Docling: {summary['figure_extras']}"
    )
    print(f"Docling clause-as-caption pages: {summary['clause_as_caption']}")
    print(
        f"reference (test p.{summary['reference_page']} / R94 source p.12): "
        f"section_disagreement={summary['reference_section_disagreement']}, "
        f"docling_clause_as_caption={summary['reference_clause_as_caption']}"
    )
    print(f"report: {summary['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
