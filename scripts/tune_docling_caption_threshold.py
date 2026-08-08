"""Try Docling layout caption knobs on OCR-diff clause-as-caption pages only.

Flagged pages (from ``scripts/ocr_diff_report.md``):
  - test p.4  — R94 Figure 3 / ``5.2.1.7`` TCFC 8 kN tagged ``caption``
  - test p.61 — R129 ``6.3.5.1`` tagged ``caption`` after a picture

Does **not** reprocess all 68 pages. Writes per-config page markdown under
``data/ocr_compare/figure_boundary_v1/docling_caption_tune/``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.caption_guard import (  # noqa: E402
    CLAUSE_IN_CAPTION_RE,
    find_caption_clause_violations,
)
from ingestion.parse import parse_pdf  # noqa: E402
from ingestion.vlm_figure_pass import export_docling_page_markdown  # noqa: E402

_CLAUSE_AS_CAPTION_MD_RE = __import__("re").compile(
    r"(?is)<!--\s*caption\s*-->\s*\n\s*(\d+(?:\.\d+){1,5}\.?[^\n]*)"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("tune_caption")

PDF = ROOT / "data/ocr_compare/figure_boundary_v1/test_figure_boundary.pdf"
OUT = ROOT / "data/ocr_compare/figure_boundary_v1/docling_caption_tune"
FLAGGED_PAGES = (4, 61)

# Config matrix: score_threshold is the only Docling OD confidence floor that
# applies to the caption class; presets swap the layout detector.
CONFIGS: list[dict] = [
    {"name": "baseline_0.30", "layout_score_threshold": 0.30, "layout_preset": None},
    {"name": "thresh_0.50", "layout_score_threshold": 0.50, "layout_preset": None},
    {"name": "thresh_0.70", "layout_score_threshold": 0.70, "layout_preset": None},
    {"name": "thresh_0.85", "layout_score_threshold": 0.85, "layout_preset": None},
    {
        "name": "egret_large_0.30",
        "layout_score_threshold": 0.30,
        "layout_preset": "layout_egret_large",
    },
    {
        "name": "egret_large_0.50",
        "layout_score_threshold": 0.50,
        "layout_preset": "layout_egret_large",
    },
]


def _run_one(cfg: dict) -> dict:
    name = cfg["name"]
    cfg_dir = OUT / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    page_results: list[dict] = []
    t0 = time.perf_counter()

    # One convert per page so we never touch the other 66 pages of the corpus.
    for page in FLAGGED_PAGES:
        logger.info("=== %s page %d ===", name, page)
        doc = parse_pdf(
            PDF,
            do_ocr=False,
            page_range=(page, page),
            layout_score_threshold=cfg["layout_score_threshold"],
            layout_preset=cfg["layout_preset"],
        )
        # With page_range=(N,N) Docling may renumber to page 1; scan both.
        md_orig = export_docling_page_markdown(doc, page)
        md_one = export_docling_page_markdown(doc, 1)
        md = md_orig if len(md_orig) >= len(md_one) else md_one
        out_md = cfg_dir / f"page_{page:03d}.md"
        out_md.write_text(md + ("\n" if md and not md.endswith("\n") else ""), encoding="utf-8")

        violations = find_caption_clause_violations(doc)
        md_hits = [
            m.group(1).strip()[:120] for m in _CLAUSE_AS_CAPTION_MD_RE.finditer(md)
        ]
        # Prefer markdown pattern (stable across page renumbering under page_range).
        fixed = len(md_hits) == 0 and not any(
            CLAUSE_IN_CAPTION_RE.search(v.text_preview) for v in violations
        )
        page_results.append(
            {
                "page": page,
                "chars": len(md),
                "violations": [v.to_dict() for v in violations],
                "md_clause_as_caption": md_hits,
                "fixed": fixed,
                "path": str(out_md.relative_to(ROOT)),
            }
        )
        logger.info(
            "page %d fixed=%s violations=%d md_hits=%s",
            page,
            page_results[-1]["fixed"],
            len(violations),
            md_hits,
        )

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "config": cfg,
        "elapsed_s": elapsed,
        "pages": page_results,
        "all_fixed": all(p["fixed"] for p in page_results),
        "unfixed_pages": [p["page"] for p in page_results if not p["fixed"]],
    }
    (cfg_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    if not PDF.is_file():
        logger.error("Missing test PDF: %s", PDF)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in CONFIGS:
        try:
            results.append(_run_one(cfg))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Config %s failed: %s", cfg["name"], exc)
            results.append({"config": cfg, "error": str(exc), "all_fixed": False})

    report = {
        "flagged_pages": list(FLAGGED_PAGES),
        "results": results,
        "configs_that_fixed_all": [r["config"]["name"] for r in results if r.get("all_fixed")],
        "pages_still_needing_vlm": sorted(
            {
                p
                for r in results
                if not r.get("all_fixed")
                for p in (r.get("unfixed_pages") or FLAGGED_PAGES)
            }
        ),
    }
    # Prefer configs that fixed everything; else note VLM fallback pages.
    if report["configs_that_fixed_all"]:
        report["recommendation"] = (
            "Use Docling config: " + ", ".join(report["configs_that_fixed_all"])
        )
    else:
        # Pick least-bad: fewest unfixed pages among successful runs.
        still = set(FLAGGED_PAGES)
        for r in results:
            if "unfixed_pages" in r:
                still &= set(r["unfixed_pages"])
        report["pages_still_needing_vlm"] = sorted(still) if still else list(FLAGGED_PAGES)
        report["recommendation"] = (
            "No Docling config cleared clause-as-caption on all flagged pages; "
            f"use LightOnOCR for pages {report['pages_still_needing_vlm']} only."
        )

    out_json = OUT / "tune_summary.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")
    print(report["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
