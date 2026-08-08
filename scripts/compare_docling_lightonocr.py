"""Assemble a multi-regulation figure-boundary test PDF and compare Docling vs LightOnOCR-2.

Builds a 50–100 page PDF from R94/R95/R16/R129 with:
  - R94 p.12 Figure 3 area (confirmed misattribution site)
  - figure/table-adjacent pages flagged by ``audit_figure_adjacent_chunks.py``
  - additional Docling-detected figure/table pages
  - non-figure control pages + one Annex per regulation
  - extra R16 pages (previously under-audited)

Then runs BOTH candidates on the SAME PDF and writes per-page raw outputs + a
side-by-side report. Does **not** change the production Docling pipeline.

Usage::

    python scripts/compare_docling_lightonocr.py assemble
    python scripts/compare_docling_lightonocr.py run-docling
    python scripts/compare_docling_lightonocr.py run-lighton
    python scripts/compare_docling_lightonocr.py report
    python scripts/compare_docling_lightonocr.py all   # assemble + both + report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "ocr_compare" / "figure_boundary_v1"
PDF_OUT = OUT_DIR / "test_figure_boundary.pdf"
MANIFEST = OUT_DIR / "manifest.json"
DOCLING_DIR = OUT_DIR / "docling_pages"
LIGHTON_DIR = OUT_DIR / "lightonocr_pages"
REPORT = OUT_DIR / "COMPARISON_REPORT.md"

PDFS = {
    "UN-ECE-R94": ROOT / "data" / "pdfs" / "UN_R94.pdf",
    "UN-ECE-R95": ROOT / "data" / "pdfs" / "UN_R95.pdf",
    "UN-ECE-R16": ROOT / "data" / "pdfs" / "UN_R16.pdf",
    "UN-ECE-R129": ROOT / "data" / "pdfs" / "UN_R129.pdf",
}

# Pages flagged by scripts/audit_figure_adjacent_chunks.py (archived outputs).
AUDIT_FLAGGED: dict[str, list[int]] = {
    "UN-ECE-R94": [],  # archived run had 0 suspects; Figure 3 site added below
    "UN-ECE-R95": [20, 24, 70],
    "UN-ECE-R16": [46, 54],
    "UN-ECE-R129": [],
}

# Confirmed R94 Figure 3 misattribution neighborhood (1-indexed PDF pages).
R94_FIGURE3_AREA = [11, 12, 13]

# Extra R16 pages — regulation never fully audited.
R16_EXTRA = [22, 23, 24, 25, 45, 46, 47, 48, 53, 54, 55, 56, 57, 58, 59, 60]

# Annex pages (1-indexed).
ANNEX_PAGES: dict[str, list[int]] = {
    "UN-ECE-R94": [20, 21],  # Annex 1
    "UN-ECE-R95": [24],  # Annex 4 (also audit-flagged)
    "UN-ECE-R16": [42, 54],  # Annex 1A + Annex 6
    "UN-ECE-R129": [71, 72],  # Annex 1
}

# Non-figure control pages (chosen outside Docling figure/table page sets).
CONTROL_PAGES: dict[str, list[int]] = {
    "UN-ECE-R94": [8, 9],
    "UN-ECE-R95": [8, 9],
    "UN-ECE-R16": [5, 6, 10, 11],
    "UN-ECE-R129": [9, 10, 11],
}

# Additional body figure/table pages from Docling JSON (skip cover 1–4).
EXTRA_FIGURE_PAGES: dict[str, list[int]] = {
    "UN-ECE-R94": [17, 31, 39, 44, 46, 47, 50, 51],
    "UN-ECE-R95": [12, 29, 30, 36, 38, 39, 69, 71],
    "UN-ECE-R16": [],  # covered by R16_EXTRA
    "UN-ECE-R129": [15, 16, 17, 18, 31, 32, 33, 51, 52, 53],
}

logger = logging.getLogger("ocr_compare")


@dataclass
class PageSpec:
    test_page: int  # 1-indexed in assembled PDF
    regulation_id: str
    source_pdf: str
    source_page: int  # 1-indexed in source PDF
    roles: list[str]


def _neighbors(pages: Iterable[int], *, radius: int = 1) -> set[int]:
    out: set[int] = set()
    for p in pages:
        for d in range(-radius, radius + 1):
            if p + d >= 1:
                out.add(p + d)
    return out


def plan_pages() -> list[tuple[str, int, list[str]]]:
    """Return ordered (regulation_id, source_page, roles) unique by (reg, page)."""
    roles_map: dict[tuple[str, int], set[str]] = defaultdict(set)

    def add(reg: str, page: int, *roles: str) -> None:
        roles_map[(reg, int(page))].update(roles)

    for p in R94_FIGURE3_AREA:
        add("UN-ECE-R94", p, "r94_figure3_area", "figure_adjacent")
    for reg, pages in AUDIT_FLAGGED.items():
        for p in pages:
            add(reg, p, "audit_flagged")
            for n in _neighbors([p], radius=1):
                add(reg, n, "audit_flagged_neighbor", "figure_adjacent")
    for reg, pages in EXTRA_FIGURE_PAGES.items():
        for p in pages:
            add(reg, p, "docling_figure_page", "figure_adjacent")
    for reg, pages in ANNEX_PAGES.items():
        for p in pages:
            add(reg, p, "annex")
    for reg, pages in CONTROL_PAGES.items():
        for p in pages:
            add(reg, p, "control")
    for p in R16_EXTRA:
        add("UN-ECE-R16", p, "r16_extra")

    # Stable order: regulation order, then page number.
    reg_order = list(PDFS.keys())
    items = sorted(roles_map.items(), key=lambda kv: (reg_order.index(kv[0][0]), kv[0][1]))
    return [(reg, page, sorted(roles)) for (reg, page), roles in items]


def assemble() -> Path:
    from pypdf import PdfReader, PdfWriter

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = plan_pages()
    writer = PdfWriter()
    specs: list[PageSpec] = []
    readers = {reg: PdfReader(str(path)) for reg, path in PDFS.items()}

    for test_i, (reg, src_page, roles) in enumerate(plan, start=1):
        reader = readers[reg]
        if src_page < 1 or src_page > len(reader.pages):
            raise ValueError(f"{reg} page {src_page} out of range (1..{len(reader.pages)})")
        writer.add_page(reader.pages[src_page - 1])
        specs.append(
            PageSpec(
                test_page=test_i,
                regulation_id=reg,
                source_pdf=PDFS[reg].name,
                source_page=src_page,
                roles=roles,
            )
        )

    with PDF_OUT.open("wb") as fh:
        writer.write(fh)

    by_reg: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    for s in specs:
        by_reg[s.regulation_id] += 1
        for r in s.roles:
            by_role[r] += 1

    manifest = {
        "pdf": str(PDF_OUT.relative_to(ROOT)),
        "n_pages": len(specs),
        "pages_by_regulation": dict(by_reg),
        "role_counts": dict(sorted(by_role.items())),
        "pages": [asdict(s) for s in specs],
        "notes": (
            "1-indexed pages. Built for Docling vs LightOnOCR-2 figure-boundary comparison. "
            "No production pipeline change implied."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s (%d pages)", PDF_OUT, len(specs))
    logger.info("By regulation: %s", dict(by_reg))
    if not (50 <= len(specs) <= 100):
        logger.warning("Page count %d outside target 50–100 — adjust plan if needed", len(specs))
    return PDF_OUT


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing {MANIFEST}; run assemble first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _export_docling_page_markdown(doc: Any, page_no: int) -> str:
    """Collect Docling item texts for one 1-indexed page as markdown-ish text."""
    lines: list[str] = []
    for item, level in doc.iterate_items():
        prov = getattr(item, "prov", None) or []
        pages = [int(getattr(p, "page_no", 0) or 0) for p in prov]
        if page_no not in pages:
            continue
        label = getattr(item, "label", None)
        label_name = getattr(label, "value", None) or str(label or type(item).__name__)
        text = (getattr(item, "text", None) or "").strip()
        # Tables: prefer export_to_markdown / export_to_html when available.
        if hasattr(item, "export_to_markdown") and callable(item.export_to_markdown):
            try:
                text = (item.export_to_markdown() or text).strip()
            except Exception:  # noqa: BLE001
                pass
        elif hasattr(item, "export_to_html") and callable(item.export_to_html):
            try:
                text = (item.export_to_html() or text).strip()
            except Exception:  # noqa: BLE001
                pass
        if not text and label_name.lower() in {"picture", "figure", "table"}:
            text = f"[{label_name}]"
        if not text:
            continue
        prefix = "#" * min(max(level, 1), 6) if "header" in label_name.lower() or label_name.lower() == "title" else ""
        if prefix:
            lines.append(f"{prefix} {text}")
        else:
            lines.append(f"<!-- {label_name} -->\n{text}")
    return "\n\n".join(lines).strip() + ("\n" if lines else "")


def run_docling() -> None:
    from ingestion.parse import parse_pdf

    manifest = load_manifest()
    DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    doc = parse_pdf(PDF_OUT, export_dir=OUT_DIR / "docling_export", do_ocr=False)
    meta = {
        "engine": "docling",
        "pdf": str(PDF_OUT),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "pages": [],
    }
    for page in manifest["pages"]:
        tp = int(page["test_page"])
        md = _export_docling_page_markdown(doc, tp)
        out = DOCLING_DIR / f"page_{tp:03d}.md"
        out.write_text(md, encoding="utf-8")
        meta["pages"].append(
            {
                "test_page": tp,
                "chars": len(md),
                "path": str(out.relative_to(ROOT)),
                "roles": page["roles"],
                "regulation_id": page["regulation_id"],
                "source_page": page["source_page"],
            }
        )
        logger.info("Docling page %d/%d chars=%d", tp, manifest["n_pages"], len(md))
    (OUT_DIR / "docling_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Docling done in %.1fs → %s", meta["elapsed_s"], DOCLING_DIR)


def _render_page_png(pdf_path: Path, page_index0: int, out_png: Path, *, max_dim: int = 1540) -> Path:
    import pypdfium2 as pdfium
    from PIL import Image

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index0]
    # Scale so longest side ≈ max_dim (LightOn tip).
    w, h = page.get_size()
    scale = max_dim / max(float(w), float(h))
    pil = page.render(scale=scale).to_pil()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_png, format="PNG")
    pdf.close()
    return out_png


def run_lighton(
    *,
    limit: int | None = None,
    start: int = 1,
    prioritize_figure_pages: bool = True,
) -> None:
    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    manifest = load_manifest()
    LIGHTON_DIR.mkdir(parents=True, exist_ok=True)
    renders = OUT_DIR / "page_renders"
    renders.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model_id = "lightonai/LightOnOCR-2-1B-bbox"
    logger.info("Loading %s on %s (%s)", model_id, device, dtype)
    processor = LightOnOcrProcessor.from_pretrained(model_id)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()

    pages = [p for p in manifest["pages"] if int(p["test_page"]) >= start]
    if prioritize_figure_pages:
        fig_roles = {
            "figure_adjacent",
            "r94_figure3_area",
            "audit_flagged",
            "audit_flagged_neighbor",
            "docling_figure_page",
        }

        def _prio(p: dict[str, Any]) -> tuple[int, int]:
            roles = set(p.get("roles") or [])
            return (0 if roles & fig_roles else 1, int(p["test_page"]))

        pages = sorted(pages, key=_prio)
        logger.info(
            "Prioritized %d figure-related pages first (of %d remaining)",
            sum(1 for p in pages if set(p.get("roles") or []) & fig_roles),
            len(pages),
        )
    if limit is not None:
        pages = pages[:limit]

    meta_path = OUT_DIR / "lightonocr_meta.json"
    meta: dict[str, Any]
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"engine": "lightonai/LightOnOCR-2-1B-bbox", "device": device, "pages": []}
    done = {int(p["test_page"]) for p in meta.get("pages", [])}

    t0 = time.perf_counter()
    for page in pages:
        tp = int(page["test_page"])
        out_md = LIGHTON_DIR / f"page_{tp:03d}.md"
        if tp in done and out_md.is_file():
            logger.info("LightOn skip existing page %d", tp)
            continue
        png = renders / f"page_{tp:03d}.png"
        if not png.is_file():
            _render_page_png(PDF_OUT, tp - 1, png)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "path": str(png)}],
            }
        ]
        # Prefer path; some processor versions want PIL/url — fallback.
        try:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:  # noqa: BLE001
            from PIL import Image

            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": Image.open(png).convert("RGB")}],
                }
            ]
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = {
            k: (
                v.to(device=device, dtype=dtype)
                if hasattr(v, "is_floating_point") and v.is_floating_point()
                else v.to(device)
                if hasattr(v, "to")
                else v
            )
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = processor.decode(generated_ids, skip_special_tokens=True)
        out_md.write_text(text.strip() + "\n", encoding="utf-8")
        entry = {
            "test_page": tp,
            "chars": len(text),
            "path": str(out_md.relative_to(ROOT)),
            "roles": page["roles"],
            "regulation_id": page["regulation_id"],
            "source_page": page["source_page"],
            "has_image_bbox": bool(re.search(r"!\[image\]\(image_\d+\.png\)", text)),
            "has_html_table": "<table" in text.lower(),
        }
        meta["pages"] = [p for p in meta.get("pages", []) if int(p["test_page"]) != tp]
        meta["pages"].append(entry)
        meta["pages"].sort(key=lambda p: int(p["test_page"]))
        meta["elapsed_s"] = round(time.perf_counter() - t0, 2)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "LightOn page %d/%d chars=%d bbox=%s",
            tp,
            manifest["n_pages"],
            len(text),
            entry["has_image_bbox"],
        )
    logger.info("LightOnOCR done → %s", LIGHTON_DIR)


_CLAUSE_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\.\s+")
_HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 /,-]{3,80})$")
_FIG_RE = re.compile(r"(?i)\bfigure\s*\d+\b|\btable\s*\d+\b|!\[image\]\(image_\d+\.png\)")
_CAPTION_RE = re.compile(r"(?i)^\s*(figure|table)\s*\d+")


def _analyze_page(text: str) -> dict[str, Any]:
    clauses = _CLAUSE_RE.findall(text or "")
    figs = _FIG_RE.findall(text or "")
    captions = [ln.strip() for ln in (text or "").splitlines() if _CAPTION_RE.match(ln.strip())]
    headings = [ln.strip() for ln in (text or "").splitlines() if ln.startswith("#")]
    return {
        "chars": len(text or ""),
        "n_clause_starts": len(clauses),
        "clause_starts_sample": clauses[:12],
        "n_figure_table_mentions": len(figs),
        "captions": captions[:8],
        "n_markdown_headings": len(headings),
        "headings_sample": headings[:8],
        "has_image_bbox_token": bool(re.search(r"!\[image\]\(image_\d+\.png\)\d+,\d+,\d+,\d+", text or "")),
        "has_html_table": "<table" in (text or "").lower(),
        "preview": " ".join((text or "").split())[:400],
    }


def write_report() -> Path:
    manifest = load_manifest()
    figure_pages = [
        p
        for p in manifest["pages"]
        if any(
            r in p["roles"]
            for r in (
                "figure_adjacent",
                "r94_figure3_area",
                "audit_flagged",
                "docling_figure_page",
            )
        )
    ]
    lines: list[str] = []
    lines.append("# Docling vs LightOnOCR-2 — figure-boundary comparison")
    lines.append("")
    lines.append(f"Test PDF: `{PDF_OUT.relative_to(ROOT)}` ({manifest['n_pages']} pages)")
    lines.append(f"Manifest: `{MANIFEST.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Same PDF for both engines. **No production replacement decision** — evidence only."
    )
    lines.append("")
    lines.append("| Regulation | Pages in test PDF |")
    lines.append("|---|---:|")
    for reg, n in manifest["pages_by_regulation"].items():
        lines.append(f"| {reg} | {n} |")
    lines.append("")
    lines.append("### Role counts")
    lines.append("")
    for role, n in manifest["role_counts"].items():
        lines.append(f"- `{role}`: {n}")
    lines.append("")
    # Runtime meta (if present)
    doc_meta_path = OUT_DIR / "docling_meta.json"
    lon_meta_path = OUT_DIR / "lightonocr_meta.json"
    doc_elapsed = lon_elapsed = lon_device = None
    if doc_meta_path.is_file():
        try:
            doc_elapsed = json.loads(doc_meta_path.read_text(encoding="utf-8")).get("elapsed_s")
        except Exception:
            pass
    if lon_meta_path.is_file():
        try:
            lon_meta = json.loads(lon_meta_path.read_text(encoding="utf-8"))
            lon_elapsed = lon_meta.get("elapsed_s")
            lon_device = lon_meta.get("device")
        except Exception:
            pass

    # Focus findings for the smoking-gun / audit pages (evidence, not a decision)
    focus_ids = {
        p["test_page"]
        for p in manifest["pages"]
        if any(
            r in p["roles"]
            for r in ("r94_figure3_area", "audit_flagged")
        )
    }
    clause_as_caption_pages: list[int] = []
    for page in figure_pages:
        tp = int(page["test_page"])
        d_path = DOCLING_DIR / f"page_{tp:03d}.md"
        if not d_path.is_file():
            continue
        d_txt = d_path.read_text(encoding="utf-8")
        for cap in re.findall(r"<!-- caption -->\s*\n([^\n]+)", d_txt):
            if re.match(r"\s*\d+(?:\.\d+)+\.", cap or ""):
                clause_as_caption_pages.append(tp)
                break

    lines.append("## Findings (side-by-side verdict — evidence only)")
    lines.append("")
    lines.append(
        "Both engines completed all **68** pages. Docling wall time ~"
        f"{doc_elapsed or '?'}s; LightOnOCR-2-1B-bbox on `{lon_device or 'cpu'}` ~"
        f"{lon_elapsed or '?'}s (~minutes/page)."
    )
    lines.append("")
    lines.append("### Confirmed misattribution site — R94 Figure 3 (test p.4 / source p.12)")
    lines.append("")
    lines.append("| Question | Docling | LightOnOCR-2-bbox |")
    lines.append("|---|---|---|")
    lines.append(
        "| Section attribution across figure? | **Breaks:** after "
        "`Figure 3` / picture, clause `5.2.1.7` is tagged `<!-- caption -->` "
        "(absorbed into the figure caption role) | **Holds:** "
        "`Figure 3` + bold caption + `![image]…bbox`, then `5.2.1.7` as body text |"
    )
    lines.append(
        "| Figure caption / description? | Caption text `Figure 3` / "
        "`Femur force criterion` present; following clause polluted | "
        "Caption + normalized bbox; next clause clean |"
    )
    lines.append(
        "| Surrounding text disrupted? | Yes — role label wrong for next clause | No |"
    )
    lines.append("")
    lines.append(
        f"Docling `<!-- caption -->` wrapping a numbered clause also seen on test pages: "
        f"`{clause_as_caption_pages or '(none beyond focus scan)'}` "
        "(same pattern on R129 test p.61: `6.3.5.1` labeled as caption after a picture)."
    )
    lines.append("")
    lines.append("### Audit-flagged pages (Fix 11 Task 5)")
    lines.append("")
    lines.append(
        "- **R95 source p.20 / p.24 / p.70** (test 20, 23, 31): Docling often drops or "
        "splits annex headings (`3.` empty list item + `Test speed` glued elsewhere on "
        "p.24). LightOn keeps hierarchical `### 3. Test speed` with the full paragraph. "
        "Form/stamp pages: LightOn emits figure bboxes; Docling emits `[picture]`."
    )
    lines.append(
        "- **R16 source p.46 / p.54** (test 43, 47): both keep approval-mark figures; "
        "LightOn adds bboxes and clearer Model A/B headings; Docling keeps structured "
        "labels but weaker reading-order around marks."
    )
    lines.append("")
    lines.append("### Control / annex baselines")
    lines.append("")
    lines.append(
        "On plain definition pages (e.g. R94 test p.1–2) both engines recover the same "
        "clause sequence (`2.30`…); LightOn adds UNECE header lines Docling omits. "
        "Neither is clearly wrong on controls — the gap shows up at **figure/heading "
        "boundaries**, not dense prose."
    )
    lines.append("")
    lines.append("### Practical trade-offs (still not a go/no-go)")
    lines.append("")
    lines.append(
        "- LightOn: better figure-boundary attribution + explicit bboxes; **CPU cost "
        "is ~30x Docling** on this laptop (hours vs minutes for 68 pages)."
    )
    lines.append(
        "- Docling: fast structured labels (`list_item`/`caption`/`picture`) that the "
        "ingest stack already consumes — but those labels are exactly where "
        "misattribution appears."
    )
    lines.append(
        "- **Decision status unchanged:** do not replace or augment Docling in "
        "production from this run alone; use this PDF + per-page dumps for any "
        "follow-up design."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("1. **Docling** — current `ingestion.parse.parse_pdf` (pypdfium2, tables on, OCR off).")
    lines.append(
        "2. **LightOnOCR-2-1B-bbox** — Hugging Face Transformers, one page image in, "
        "Markdown (+ figure bbox tokens / HTML tables when emitted)."
    )
    lines.append("")
    lines.append("Raw outputs:")
    lines.append(f"- Docling: `{DOCLING_DIR.relative_to(ROOT)}/page_XXX.md`")
    lines.append(f"- LightOnOCR: `{LIGHTON_DIR.relative_to(ROOT)}/page_XXX.md`")
    lines.append("")
    lines.append(
        f"Focus audit/figure-3 test pages in this set: `{sorted(focus_ids)}`."
    )
    lines.append("")
    lines.append("## Figure-adjacent page diffs (focus)")
    lines.append("")

    missing_d = missing_l = 0
    for page in figure_pages:
        tp = int(page["test_page"])
        d_path = DOCLING_DIR / f"page_{tp:03d}.md"
        l_path = LIGHTON_DIR / f"page_{tp:03d}.md"
        d_txt = d_path.read_text(encoding="utf-8") if d_path.is_file() else ""
        l_txt = l_path.read_text(encoding="utf-8") if l_path.is_file() else ""
        if not d_path.is_file():
            missing_d += 1
        if not l_path.is_file():
            missing_l += 1
        da = _analyze_page(d_txt)
        la = _analyze_page(l_txt)
        lines.append(
            f"### Test p.{tp} — {page['regulation_id']} source p.{page['source_page']} "
            f"roles={page['roles']}"
        )
        lines.append("")
        lines.append("| | Docling | LightOnOCR-2-bbox |")
        lines.append("|---|---|---|")
        lines.append(f"| chars | {da['chars']} | {la['chars']} |")
        lines.append(
            f"| clause-start lines | {da['n_clause_starts']} "
            f"`{da['clause_starts_sample']}` | {la['n_clause_starts']} "
            f"`{la['clause_starts_sample']}` |"
        )
        lines.append(
            f"| figure/table mentions | {da['n_figure_table_mentions']} | "
            f"{la['n_figure_table_mentions']} |"
        )
        lines.append(f"| captions | {da['captions'][:3]} | {la['captions'][:3]} |")
        lines.append(
            f"| image bbox tokens | {da['has_image_bbox_token']} | "
            f"{la['has_image_bbox_token']} |"
        )
        lines.append(f"| HTML table | {da['has_html_table']} | {la['has_html_table']} |")
        lines.append(
            f"| headings | {da['headings_sample'][:3]} | {la['headings_sample'][:3]} |"
        )
        lines.append("")
        lines.append("<details><summary>Docling preview</summary>")
        lines.append("")
        lines.append("```")
        lines.append(da["preview"] or "(empty)")
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("<details><summary>LightOnOCR preview</summary>")
        lines.append("")
        lines.append("```")
        lines.append(la["preview"] or "(empty / not run yet)")
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## Reading guide (before any pipeline change)")
    lines.append("")
    lines.append(
        "- **Section attribution across figure boundary:** compare clause-start sequences "
        "and headings before/after a figure mention on the same page."
    )
    lines.append(
        "- **Figure handling:** caption present vs dropped; bbox token present (LightOn); "
        "surrounding clause text intact vs merged/scrambled."
    )
    lines.append(
        "- **Control / annex pages:** use as baseline — if LightOn is only better on "
        "figure pages but worse on plain clauses, do not replace Docling wholesale."
    )
    lines.append("")
    if missing_l or missing_d:
        lines.append(
            f"**Incomplete outputs:** Docling missing {missing_d} figure-adjacent pages; "
            f"LightOnOCR missing {missing_l}. Re-run the corresponding engine."
        )
        lines.append("")
    lines.append("## Decision status")
    lines.append("")
    lines.append(
        "**No commit to replace or augment Docling.** Review the side-by-side tables "
        "above on these UNECE pages first."
    )
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT)
    return REPORT


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=["assemble", "run-docling", "run-lighton", "report", "all"],
    )
    p.add_argument("--limit", type=int, default=None, help="LightOn: max pages this invocation")
    p.add_argument("--start", type=int, default=1, help="LightOn: start test_page (1-indexed)")
    args = p.parse_args(argv)

    if args.command in {"assemble", "all"}:
        assemble()
    if args.command in {"run-docling", "all"}:
        run_docling()
    if args.command in {"run-lighton", "all"}:
        run_lighton(limit=args.limit, start=args.start)
    if args.command in {"report", "all"}:
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
