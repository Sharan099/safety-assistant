"""
paddle_ocr_converter.py — Memory-friendly scanned-PDF OCR (PaddleOCR / PP-OCR).

Techniques to avoid OOM on CPU:
  1. Render pages at LOW DPI (config.OCR_DPI, default 150).
  2. Cache page images to disk (output/page_cache).
  3. Process pages in SMALL BATCHES with gc between batches.
  4. Skip OCR when embedded text layer is sufficient.

Engine selection (OCR_BACKEND env, default "auto"):
  - paddle: native PaddleOCR (Linux/Docker)
  - rapidocr: PP-OCR via ONNX (recommended on Windows when Paddle crashes)
  - auto: try Paddle, fall back to RapidOCR
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Skip slow model-host connectivity checks
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    MARKDOWN_DIR,
    OCR_BACKEND,
    OCR_BATCH_PAGES,
    OCR_DPI,
    OCR_FORCE_ALL,
    OCR_LANG,
    OCR_MIN_TEXT_CHARS,
    OCR_SKIP_TEXT_PAGES,
    PAGE_IMAGE_CACHE,
)

sys.stdout.reconfigure(line_buffering=True)

_PADDLE = None
_RAPID = None
_ACTIVE_ENGINE: str | None = None


def p(msg: str) -> None:
    print(msg, flush=True)


def _init_paddle() -> bool:
    global _PADDLE, _ACTIVE_ENGINE
    if _PADDLE is not None:
        return True
    try:
        from paddleocr import PaddleOCR

        lang = "en" if OCR_LANG in ("en", "eng") else OCR_LANG
        # PaddleOCR 3.7+ defaults to PP-OCRv6; older 2.x used explicit model names.
        init_attempts: list[dict] = [
            dict(
                lang=lang,
                ocr_version="PP-OCRv6",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
            dict(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv4_mobile_rec",
            ),
            dict(lang=lang),
        ]
        last_exc: Exception | None = None
        for kwargs in init_attempts:
            try:
                try:
                    _PADDLE = PaddleOCR(enable_mkldnn=False, **kwargs)
                except TypeError:
                    _PADDLE = PaddleOCR(**kwargs)
                version = kwargs.get("ocr_version", "PP-OCRv4")
                _ACTIVE_ENGINE = f"paddleocr-{version}"
                p(f"OCR engine: PaddleOCR ({version})")
                return True
            except Exception as exc:
                last_exc = exc
                _PADDLE = None
        if last_exc:
            raise last_exc
        return False
    except Exception as exc:
        p(f"PaddleOCR init failed: {exc}")
        return False


def _init_rapid() -> bool:
    global _RAPID, _ACTIVE_ENGINE
    if _RAPID is not None:
        return True
    try:
        from rapidocr_onnxruntime import RapidOCR

        _RAPID = RapidOCR()
        _ACTIVE_ENGINE = "rapidocr-ppocr"
        p("OCR engine: RapidOCR (PP-OCR ONNX — Paddle-compatible on Windows)")
        return True
    except Exception as exc:
        p(f"RapidOCR init failed: {exc}")
        return False


def _ensure_engine() -> str:
    global _ACTIVE_ENGINE
    if _ACTIVE_ENGINE:
        return _ACTIVE_ENGINE

    backend = OCR_BACKEND
    if backend == "rapidocr":
        if not _init_rapid():
            raise RuntimeError("RapidOCR unavailable")
        return _ACTIVE_ENGINE

    if backend == "paddle":
        if not _init_paddle():
            raise RuntimeError("PaddleOCR unavailable")
        return _ACTIVE_ENGINE

    # auto
    if _init_paddle():
        return _ACTIVE_ENGINE
    if _init_rapid():
        return _ACTIVE_ENGINE
    raise RuntimeError("No OCR engine available (install paddleocr or rapidocr-onnxruntime)")


def active_engine() -> str:
    return _ensure_engine()


def _render_page_to_cache(pdf_path: Path, page, page_idx: int, cache_dir: Path) -> Path:
    img_path = cache_dir / f"p{page_idx:04d}.png"
    if img_path.exists():
        return img_path
    import fitz

    zoom = OCR_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(img_path))
    del pix
    return img_path


def _ocr_paddle(img_path: Path) -> str:
    ocr = _PADDLE
    try:
        result = ocr.predict(str(img_path))  # type: ignore[union-attr]
    except AttributeError:
        result = ocr.ocr(str(img_path), cls=False)  # type: ignore[union-attr]

    lines: list[str] = []
    for page_res in result or []:
        if isinstance(page_res, dict):
            lines.extend(t for t in (page_res.get("rec_texts") or []) if t)
            continue
        if isinstance(page_res, list):
            for item in page_res:
                try:
                    txt = item[1][0]
                    if txt:
                        lines.append(txt)
                except (IndexError, TypeError):
                    continue
    return "\n".join(lines).strip()


def _ocr_rapid(img_path: Path) -> str:
    result, _ = _RAPID(str(img_path))  # type: ignore[misc]
    if not result:
        return ""
    return "\n".join(item[1] for item in result if len(item) > 1 and item[1]).strip()


def _ocr_image(img_path: Path) -> str:
    engine = _ensure_engine()
    if engine == "paddleocr":
        return _ocr_paddle(img_path)
    return _ocr_rapid(img_path)


def convert_pdf_paddle(pdf_path: Path) -> str:
    """Return markdown for a PDF using cached low-DPI images + batched OCR."""
    import fitz

    _ensure_engine()
    cache_dir = PAGE_IMAGE_CACHE / pdf_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    parts: list[str] = [f"# {pdf_path.stem}\n"]

    batch_start = 0
    while batch_start < n_pages:
        batch_end = min(n_pages, batch_start + OCR_BATCH_PAGES)
        for i in range(batch_start, batch_end):
            page = doc[i]
            embedded = page.get_text("text").strip()

            if (
                not OCR_FORCE_ALL
                and OCR_SKIP_TEXT_PAGES
                and len(embedded) >= OCR_MIN_TEXT_CHARS
            ):
                parts.append(f"\n## Page {i + 1}\n\n{embedded}\n")
                continue

            img_path = _render_page_to_cache(pdf_path, page, i, cache_dir)
            ocr_text = ""
            try:
                ocr_text = _ocr_image(img_path)
            except Exception as exc:
                p(f"  OCR failed page {i + 1}: {exc}")

            best = ocr_text if len(ocr_text) >= len(embedded) else embedded
            if best:
                parts.append(f"\n## Page {i + 1}\n\n{best}\n")

        p(f"  pages {batch_start + 1}-{batch_end}/{n_pages} ({active_engine()})")
        gc.collect()
        batch_start = batch_end

    doc.close()
    gc.collect()
    return "\n".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paddle/PP-OCR a PDF to markdown")
    parser.add_argument("pdf", help="Path to PDF")
    args = parser.parse_args()
    md = convert_pdf_paddle(Path(args.pdf))
    out = MARKDOWN_DIR / f"{Path(args.pdf).stem}.md"
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    p(f"Wrote {out} ({len(md)} chars, engine={active_engine()})")
