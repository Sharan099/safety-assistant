"""Confirm Stages 1–3 chunking/metadata across R94/R95/R16/R129.

Seeds VLM page caches from the Stage 1 figure-boundary OCR compare (when present),
runs the selective VLM figure pass on those pages, builds figure chunks, and
reports caption-mislabel / parent-linkage health. Does not replace Docling for
non-figure pages.

Usage::

    python scripts/confirm_stages_1_3.py
    python scripts/confirm_stages_1_3.py --upsert-figures
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docling_core.types.doc import DoclingDocument

from ingestion.chunk import chunk_document
from ingestion.vlm_figure_pass import (
    apply_vlm_figure_pass,
    build_figure_chunks_from_vlm,
    collect_figure_adjacent_pages,
    docling_caption_mislabels,
)

logger = logging.getLogger("confirm_stages")

REGS = {
    "UN-ECE-R94": {
        "pdf": ROOT / "data" / "pdfs" / "UN_R94.pdf",
        "docling": ROOT / "data" / "docling" / "UN_R94.docling.json",
        "stem": "UN_R94",
        "revision": "Rev.3",
    },
    "UN-ECE-R95": {
        "pdf": ROOT / "data" / "pdfs" / "UN_R95.pdf",
        "docling": ROOT / "data" / "docling" / "UN_R95.docling.json",
        "stem": "UN_R95",
        "revision": "Rev.3",
    },
    "UN-ECE-R16": {
        "pdf": ROOT / "data" / "pdfs" / "UN_R16.pdf",
        "docling": ROOT / "data" / "docling" / "UN_R16.docling.json",
        "stem": "UN_R16",
        "revision": "Rev.7",
    },
    "UN-ECE-R129": {
        "pdf": ROOT / "data" / "pdfs" / "UN_R129.pdf",
        "docling": ROOT / "data" / "docling" / "UN_R129.docling.json",
        "stem": "UN_R129",
        "revision": "Rev.3",
    },
}

MANIFEST = ROOT / "data" / "ocr_compare" / "figure_boundary_v1" / "manifest.json"
LIGHTON_DIR = ROOT / "data" / "ocr_compare" / "figure_boundary_v1" / "lightonocr_pages"
WORK = ROOT / "data" / "vlm_figure_pass"

# Known Stage 1 / Stage 2 ground-truth parent links (corrected reading order).
KNOWN_PARENTS = {
    ("UN-ECE-R94", 12, "Figure 3"): "5.2.1.6",
    ("UN-ECE-R94", 11, "Figure 1"): "5.2.1.2",
}


def load_doc(path: Path) -> DoclingDocument:
    if hasattr(DoclingDocument, "load_from_json"):
        return DoclingDocument.load_from_json(str(path))
    return DoclingDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def seed_caches_from_stage1() -> dict[str, list[int]]:
    """Copy Stage 1 LightOn page MD into per-PDF VLM cache paths."""
    if not MANIFEST.is_file() or not LIGHTON_DIR.is_dir():
        logger.warning("Stage 1 artifacts missing; skip cache seed")
        return {}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    seeded: dict[str, list[int]] = defaultdict(list)
    fig_roles = {
        "figure_adjacent",
        "r94_figure3_area",
        "audit_flagged",
        "docling_figure_page",
    }
    for page in manifest["pages"]:
        roles = set(page.get("roles") or [])
        if not (roles & fig_roles):
            continue
        reg = page["regulation_id"]
        meta = REGS.get(reg)
        if not meta:
            continue
        src = LIGHTON_DIR / f"page_{int(page['test_page']):03d}.md"
        if not src.is_file():
            continue
        dest_dir = WORK / "vlm_pages" / meta["stem"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"page_{int(page['source_page']):04d}.md"
        shutil.copy2(src, dest)
        seeded[reg].append(int(page["source_page"]))
    return {k: sorted(set(v)) for k, v in seeded.items()}


def clause_metadata_health(chunks: list) -> dict:
    """Share of clause chunks whose leading text matches section_number."""
    import re

    leading = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\.")
    checked = matched = 0
    for ch in chunks:
        if getattr(ch, "content_type", "") != "clause":
            continue
        m = leading.search(ch.text or "")
        if not m:
            continue
        checked += 1
        bare = (ch.section_number or "").split("/", 1)[-1].strip()
        explicit = m.group(1)
        if explicit == bare or bare.startswith(explicit + ".") or explicit.startswith(bare + "."):
            matched += 1
    return {
        "clause_checked": checked,
        "clause_matched": matched,
        "match_rate": round(matched / checked, 4) if checked else 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upsert-figures",
        action="store_true",
        help="Upsert emitted figure chunks into Qdrant (Stage 3 live)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    seeded = seed_caches_from_stage1()
    report: dict = {"seeded_pages": seeded, "regulations": {}}
    all_ok = True
    all_figure_chunks: list = []

    for reg, meta in REGS.items():
        entry: dict = {"ok": True, "issues": []}
        if not meta["docling"].is_file():
            entry["ok"] = False
            entry["issues"].append(f"missing docling {meta['docling']}")
            all_ok = False
            report["regulations"][reg] = entry
            continue
        if not meta["pdf"].is_file():
            entry["ok"] = False
            entry["issues"].append(f"missing pdf {meta['pdf']}")
            all_ok = False
            report["regulations"][reg] = entry
            continue

        doc = load_doc(meta["docling"])
        chunks = chunk_document(doc, regulation_id=reg, revision=meta["revision"])
        entry["n_chunks_pre_vlm"] = len(chunks)
        entry["clause_health"] = clause_metadata_health(chunks)
        if entry["clause_health"]["match_rate"] < 0.85:
            entry["ok"] = False
            entry["issues"].append(
                f"clause metadata match_rate "
                f"{entry['clause_health']['match_rate']} < 0.85"
            )

        adj = collect_figure_adjacent_pages(doc)
        cached = set(seeded.get(reg) or [])
        pages = sorted(set(adj) & cached) if cached else sorted(cached)
        # Always include known confirmation pages when cached.
        for (_r, pg, _lab) in KNOWN_PARENTS:
            if _r == reg and pg in cached:
                pages = sorted(set(pages) | {pg})

        entry["figure_adjacent_pages"] = len(adj)
        entry["vlm_pages"] = pages
        mis_before = {
            pg: docling_caption_mislabels(doc, pg) for pg in pages if docling_caption_mislabels(doc, pg)
        }
        entry["caption_mislabels_before"] = {str(k): v for k, v in mis_before.items()}

        if not pages:
            entry["issues"].append("no Stage 1 VLM cache pages for this regulation")
            # Still OK for metadata if clause health passed — Stage 2 needs cache.
            if not cached:
                entry["ok"] = False
            report["regulations"][reg] = entry
            if not entry["ok"]:
                all_ok = False
            continue

        vlm = apply_vlm_figure_pass(
            doc,
            meta["pdf"],
            pages=pages,
            work_dir=WORK,
        )
        figs = build_figure_chunks_from_vlm(
            vlm, doc, regulation_id=reg, revision=meta["revision"]
        )
        all_figure_chunks.extend(figs)
        entry["vlm_processed"] = vlm.pages_processed
        entry["vlm_corrected"] = vlm.pages_corrected
        entry["n_figure_chunks"] = len(figs)
        entry["figure_parents"] = [
            {
                "chunk_id": c.chunk_id,
                "section_number": c.section_number,
                "title": c.section_title,
                "page": c.page_number,
            }
            for c in figs
            if c.content_type == "figure"
        ]

        mis_after = {
            pg: docling_caption_mislabels(doc, pg) for pg in pages if docling_caption_mislabels(doc, pg)
        }
        entry["caption_mislabels_after"] = {str(k): v for k, v in mis_after.items()}
        if mis_after:
            entry["ok"] = False
            entry["issues"].append(f"caption mislabels remain after VLM: {mis_after}")

        # Known parent checks
        for (kr, page, label), expect in KNOWN_PARENTS.items():
            if kr != reg:
                continue
            hits = [
                c
                for c in figs
                if c.page_number == page and label.lower() in (c.section_title or "").lower()
            ]
            if not hits:
                # try text
                hits = [c for c in figs if c.page_number == page and label.lower() in c.text.lower()]
            if not hits:
                entry["ok"] = False
                entry["issues"].append(f"missing figure chunk {label} on page {page}")
            elif hits[0].section_number != expect:
                entry["ok"] = False
                entry["issues"].append(
                    f"{label} parent {hits[0].section_number!r} != expected {expect!r}"
                )
            else:
                entry.setdefault("known_parent_ok", []).append(
                    f"{label}→§{expect} ({hits[0].chunk_id})"
                )

        if not entry["ok"]:
            all_ok = False
        report["regulations"][reg] = entry
        print(
            f"{reg}: ok={entry['ok']} clauses_match={entry['clause_health']['match_rate']} "
            f"vlm_pages={len(pages)} figures={len(figs)} corrected={vlm.pages_corrected}"
        )
        for issue in entry["issues"]:
            print(f"  ! {issue}")

    if args.upsert_figures and all_figure_chunks:
        from ingestion.embed_upsert import upsert_chunks
        from ingestion.enrich import enrich_chunks

        # Never delete_existing — figure chunks are additive; wiping the
        # regulation would drop all clause/table points.
        n = upsert_chunks(enrich_chunks(all_figure_chunks), delete_existing=False)
        report["upserted_figures"] = n
        print(f"Upserted {n} figure chunks (additive)")

    out = WORK / "stages_1_3_confirmation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
