"""Parse regulation PDFs with Docling into DoclingDocument (+ JSON export)."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

# Windows often lacks MSVC (`cl`); torch inductor then crashes mid-layout.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument

logger = logging.getLogger(__name__)

# Layout OD confidence floor. Caption is one of the detected classes; raising this
# is the only Docling knob that can suppress low-confidence caption mislabels.
# Default matches Docling's TransformersObjectDetectionEngineOptions (0.3).
DEFAULT_LAYOUT_SCORE_THRESHOLD = 0.3


def _build_converter(
    *,
    do_ocr: bool = False,
    layout_score_threshold: float | None = None,
    layout_preset: str | None = None,
) -> DocumentConverter:
    """PDF converter tuned for born-digital UNECE regulation texts.

    Uses pypdfium2 + backend text extraction to avoid docling-parse
    ``std::bad_alloc`` spikes common on Windows with large regs.

    ``layout_score_threshold`` raises the layout object-detection confidence
    floor (including the ``caption`` class). ``layout_preset`` selects a
    Docling layout model preset (e.g. ``layout_egret_large``).
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.images_scale = 1.0
    opts.force_backend_text = not do_ocr  # OCR path needs rendered pages
    opts.layout_batch_size = 1
    opts.table_batch_size = 1
    opts.ocr_batch_size = 1
    opts.accelerator_options = AcceleratorOptions(num_threads=2, device="cpu")
    if hasattr(opts, "heading_hierarchy_options"):
        opts.heading_hierarchy_options.enabled = True

    if layout_preset:
        opts.layout_options = LayoutObjectDetectionOptions.from_preset(layout_preset)
    threshold = (
        DEFAULT_LAYOUT_SCORE_THRESHOLD
        if layout_score_threshold is None
        else float(layout_score_threshold)
    )
    try:
        opts.layout_options.engine_options.score_threshold = threshold
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not set layout score_threshold=%s: %s", threshold, exc)
    else:
        logger.info(
            "Docling layout score_threshold=%.2f preset=%s",
            threshold,
            layout_preset or "default",
        )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts,
                backend=PyPdfiumDocumentBackend,
            )
        },
    )


def parse_pdf(
    pdf_path: str | Path,
    *,
    export_dir: str | Path | None = None,
    do_ocr: bool = False,
    page_range: tuple[int, int] | None = None,
    layout_score_threshold: float | None = None,
    layout_preset: str | None = None,
) -> DoclingDocument:
    """Convert a regulation PDF into a DoclingDocument.

    Each element retains text, page_number, bounding_box (via ``prov``), and
    participates in the heading hierarchy exposed by ``iterate_items``.

    Optionally writes Docling JSON next to ``export_dir`` for inspection.
    """
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Parsing %s with Docling", path.name)
    kwargs: dict[str, Any] = {}
    if page_range is not None:
        kwargs["page_range"] = page_range
    result = _build_converter(
        do_ocr=do_ocr,
        layout_score_threshold=layout_score_threshold,
        layout_preset=layout_preset,
    ).convert(str(path), **kwargs)
    doc: DoclingDocument = result.document

    if export_dir is not None:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.docling.json"
        # Prefer native save when available; fall back to export_to_dict.
        if hasattr(doc, "save_as_json"):
            doc.save_as_json(out_path)
        else:
            out_path.write_text(
                json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        logger.info("Wrote Docling JSON → %s", out_path)

    return doc


def _element_summary(doc: DoclingDocument) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    with_bbox = 0
    with_page = 0
    headings: list[dict[str, Any]] = []

    for item, level in doc.iterate_items():
        label = getattr(item, "label", None)
        label_name = label.value if isinstance(label, DocItemLabel) else str(label or type(item).__name__)
        labels[label_name] += 1

        prov = getattr(item, "prov", None) or []
        if prov:
            with_page += 1
            if getattr(prov[0], "bbox", None) is not None:
                with_bbox += 1

        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE) or label_name in {
            "section_header",
            "title",
        }:
            text = (getattr(item, "text", None) or "").strip()
            page = prov[0].page_no if prov else None
            headings.append({"level": level, "page": page, "text": text[:120]})

    pages = getattr(doc, "pages", {}) or {}
    num_pages: Any
    if pages:
        num_pages = len(pages)
    elif callable(getattr(doc, "num_pages", None)):
        num_pages = doc.num_pages()
    else:
        num_pages = getattr(doc, "num_pages", None)

    return {
        "name": getattr(doc, "name", None) or "",
        "num_pages": num_pages,
        "label_counts": dict(labels.most_common()),
        "elements_with_page": with_page,
        "elements_with_bbox": with_bbox,
        "heading_count": len(headings),
        "headings_sample": headings[:15],
    }


def print_parse_summary(doc: DoclingDocument) -> dict[str, Any]:
    """Print (and return) a compact parse summary for humans / CI logs."""
    summary = _element_summary(doc)
    print("=" * 60)
    print("Docling parse summary")
    print("=" * 60)
    print(f"  name:              {summary['name']}")
    print(f"  pages:             {summary['num_pages']}")
    print(f"  elements w/ page:  {summary['elements_with_page']}")
    print(f"  elements w/ bbox:  {summary['elements_with_bbox']}")
    print(f"  headings:          {summary['heading_count']}")
    print("  label counts:")
    for label, count in summary["label_counts"].items():
        print(f"    - {label}: {count}")
    if summary["headings_sample"]:
        print("  heading sample (first 15):")
        for h in summary["headings_sample"]:
            print(f"    [L{h['level']} p{h['page']}] {h['text']}")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Parse a regulation PDF with Docling")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--export-dir", type=Path, default=Path("data/docling"))
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument(
        "--layout-score-threshold",
        type=float,
        default=None,
        help="Layout OD confidence floor (default 0.3); raises caption-class bar",
    )
    ap.add_argument(
        "--layout-preset",
        type=str,
        default=None,
        help="Docling layout preset, e.g. layout_egret_large",
    )
    ap.add_argument(
        "--page-range",
        type=str,
        default=None,
        help="Inclusive 1-indexed range START-END (e.g. 4-4)",
    )
    args = ap.parse_args()
    page_range = None
    if args.page_range:
        a, b = args.page_range.split("-", 1)
        page_range = (int(a), int(b))
    document = parse_pdf(
        args.pdf,
        export_dir=args.export_dir,
        do_ocr=args.ocr,
        page_range=page_range,
        layout_score_threshold=args.layout_score_threshold,
        layout_preset=args.layout_preset,
    )
    print_parse_summary(document)
