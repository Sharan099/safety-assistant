"""
docling_converter.py — Convert PDFs to Markdown using Docling (OCR for scans).

Outputs: output/markdown/<pdf_stem>.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DATA_DIR,
    DOCLING_FORCE_FULL_PAGE_OCR,
    DOCLING_OCR,
    INGEST_MANIFEST,
    MARKDOWN_DIR,
    OCR_ENGINE,
)

sys.stdout.reconfigure(line_buffering=True)


def p(msg: str) -> None:
    print(msg, flush=True)


def _detect_regulation(pdf_name: str) -> str:
    from data.hierarchical_chunker import detect_regulation_type

    return detect_regulation_type(pdf_name)


def _build_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = DOCLING_OCR
    pipeline_options.do_table_structure = True

    if DOCLING_OCR:
        ocr_set = False
        try:
            from docling.datamodel.pipeline_options import EasyOcrOptions

            pipeline_options.ocr_options = EasyOcrOptions(
                force_full_page_ocr=DOCLING_FORCE_FULL_PAGE_OCR,
                lang=["en"],
            )
            ocr_set = True
        except Exception:
            pass
        if not ocr_set:
            try:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions

                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=DOCLING_FORCE_FULL_PAGE_OCR,
                    lang=["eng"],
                )
                ocr_set = True
            except Exception:
                p("No OCR engine configured — Docling will use auto detection")

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _fallback_pymupdf_to_markdown(pdf_path: Path) -> str:
    """Fallback when Docling is unavailable."""
    import fitz

    doc = fitz.open(pdf_path)
    parts = [f"# {pdf_path.stem}\n"]
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"\n## Page {i + 1}\n\n{text}\n")
    doc.close()
    return "\n".join(parts)


def _pymupdf_page_markdown(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    parts = ["\n\n# Supplementary text layer (PyMuPDF)\n"]
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"\n## Page {i + 1}\n\n{text}\n")
    doc.close()
    return "\n".join(parts)


def convert_pdf(pdf_path: Path, converter=None) -> tuple[str, str]:
    """
    Returns (markdown_text, engine_used).
    Engine controlled by config.OCR_ENGINE: paddle | docling | pymupdf.
    """
    if OCR_ENGINE == "paddle":
        from data.paddle_ocr_converter import active_engine, convert_pdf_paddle

        md = convert_pdf_paddle(pdf_path)
        return md, active_engine()

    if OCR_ENGINE == "pymupdf":
        return _fallback_pymupdf_to_markdown(pdf_path), "pymupdf"

    # Docling path
    engine = "docling"
    try:
        conv = converter or _build_converter()
        result = conv.convert(str(pdf_path))
        md = result.document.export_to_markdown()
    except Exception as exc:
        p(f"  Docling failed ({exc}) — PyMuPDF fallback")
        return _fallback_pymupdf_to_markdown(pdf_path), "pymupdf"

    if os.getenv("PYMUPDF_SUPPLEMENT", "true").lower() == "true":
        md = md + _pymupdf_page_markdown(pdf_path)
        engine = "docling+pymupdf"
    return md, engine


def discover_pdfs(only_regs: list[str] | None = None) -> list[Path]:
    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    pdfs = [p for p in pdfs if p.is_file()]
    if not only_regs:
        return pdfs
    allowed = {x.upper() for x in only_regs}
    filtered = []
    for p in pdfs:
        if _detect_regulation(p.name).upper() in allowed:
            filtered.append(p)
    return filtered


def discover_root_pdfs() -> list[Path]:
    """PDFs directly under data/ only — excludes data/regulations/ and other subdirs."""
    return sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )


def convert_single_pdf(
    pdf_path: Path,
    *,
    force: bool = False,
    converter=None,
) -> dict:
    """Convert one PDF to markdown. Returns a manifest-style result dict."""
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    out_md = MARKDOWN_DIR / f"{stem}.md"
    rel_pdf = str(pdf_path.relative_to(DATA_DIR.parent))

    if out_md.exists() and not force:
        return {"status": "skipped", "pdf": rel_pdf, "markdown": out_md.name}

    t0 = time.perf_counter()
    md, engine = convert_pdf(pdf_path, converter=converter)
    if not md.strip():
        raise ValueError("empty markdown output")

    header = (
        f"---\n"
        f"source_pdf: {pdf_path.name}\n"
        f"regulation: {_detect_regulation(pdf_path.name)}\n"
        f"converter: {engine}\n"
        f"---\n\n"
    )
    out_md.write_text(header + md, encoding="utf-8")
    elapsed = round(time.perf_counter() - t0, 1)
    return {
        "status": "converted",
        "pdf": rel_pdf,
        "markdown": out_md.name,
        "chars": len(md),
        "engine": engine,
        "seconds": elapsed,
    }


def run(force: bool = False, only_regs: list[str] | None = None) -> dict:
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = discover_pdfs(only_regs=only_regs)
    p(f"Found {len(pdfs)} PDFs under {DATA_DIR}")

    converter = None
    if OCR_ENGINE == "paddle":
        p(f"OCR engine: PaddleOCR (low-DPI cache + batched pages)")
    elif OCR_ENGINE == "pymupdf":
        p("OCR engine: PyMuPDF text layer only")
    else:
        try:
            converter = _build_converter()
            p("OCR engine: Docling (OCR enabled for scans)")
        except Exception as exc:
            p(f"Docling init failed: {exc} — will use per-file fallback")

    manifest = {
        "converted": [],
        "skipped": [],
        "errors": [],
    }

    for idx, pdf_path in enumerate(pdfs, 1):
        stem = pdf_path.stem
        out_md = MARKDOWN_DIR / f"{stem}.md"
        rel_pdf = str(pdf_path.relative_to(DATA_DIR.parent))

        if out_md.exists() and not force:
            manifest["skipped"].append(rel_pdf)
            p(f"[{idx}/{len(pdfs)}] skip (exists) {out_md.name}")
            continue

        t0 = time.perf_counter()
        try:
            md, engine = convert_pdf(pdf_path, converter=converter)
            if not md.strip():
                raise ValueError("empty markdown output")

            header = (
                f"---\n"
                f"source_pdf: {pdf_path.name}\n"
                f"regulation: {_detect_regulation(pdf_path.name)}\n"
                f"converter: {engine}\n"
                f"---\n\n"
            )
            out_md.write_text(header + md, encoding="utf-8")
            elapsed = round(time.perf_counter() - t0, 1)
            manifest["converted"].append(
                {
                    "pdf": rel_pdf,
                    "markdown": out_md.name,
                    "chars": len(md),
                    "engine": engine,
                    "seconds": elapsed,
                }
            )
            p(f"[{idx}/{len(pdfs)}] OK {pdf_path.name} -> {out_md.name} ({elapsed}s, {engine})")
        except Exception as exc:
            manifest["errors"].append({"pdf": rel_pdf, "error": str(exc)})
            p(f"[{idx}/{len(pdfs)}] ERROR {pdf_path.name}: {exc}")

    INGEST_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    p(f"\nManifest: {INGEST_MANIFEST}")
    p(f"Converted: {len(manifest['converted'])}, skipped: {len(manifest['skipped'])}, errors: {len(manifest['errors'])}")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown via Docling")
    parser.add_argument("--force", action="store_true", help="Reconvert existing markdown files")
    args = parser.parse_args()
    run(force=args.force)
