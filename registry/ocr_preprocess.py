"""OCR preprocessing for image-only scanned regulation PDFs."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import fitz
from loguru import logger

OCR_BACKEND = os.getenv("OCR_BACKEND", os.getenv("OCR_ENGINE", "auto")).lower()
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_BATCH_PAGES = int(os.getenv("OCR_BATCH_PAGES", "4"))
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "200"))
OCR_SKIP_TEXT_PAGES = os.getenv("OCR_SKIP_TEXT_PAGES", "true").lower() == "true"
OCR_FORCE_ALL = os.getenv("OCR_FORCE_ALL", "false").lower() == "true"

_PADDLE = None
_RAPID = None
_ACTIVE_ENGINE: str | None = None


def active_engine() -> str:
    _ensure_engine()
    return _ACTIVE_ENGINE or "unknown"


def _init_paddle() -> bool:
    global _PADDLE, _ACTIVE_ENGINE
    if _PADDLE is not None:
        return True
    try:
        from paddleocr import PaddleOCR

        lang = "en" if OCR_LANG in ("en", "eng") else OCR_LANG
        try:
            _PADDLE = PaddleOCR(
                lang=lang,
                ocr_version="PP-OCRv6",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        except TypeError:
            _PADDLE = PaddleOCR(lang=lang)
        _ACTIVE_ENGINE = "paddleocr"
        return True
    except Exception as exc:
        logger.warning(f"PaddleOCR init failed: {exc}")
        return False


def _init_rapid() -> bool:
    global _RAPID, _ACTIVE_ENGINE
    if _RAPID is not None:
        return True
    try:
        from rapidocr_onnxruntime import RapidOCR

        _RAPID = RapidOCR()
        _ACTIVE_ENGINE = "rapidocr-ppocr"
        return True
    except Exception as exc:
        logger.warning(f"RapidOCR init failed: {exc}")
        return False


def _ensure_engine() -> str:
    global _ACTIVE_ENGINE
    if _ACTIVE_ENGINE:
        return _ACTIVE_ENGINE

    if OCR_BACKEND in ("rapidocr", "paddle"):
        init = _init_rapid if OCR_BACKEND == "rapidocr" else _init_paddle
        if not init():
            raise RuntimeError(f"{OCR_BACKEND} OCR unavailable")
        return _ACTIVE_ENGINE or OCR_BACKEND

    if _init_paddle():
        return _ACTIVE_ENGINE or "paddleocr"
    if _init_rapid():
        return _ACTIVE_ENGINE or "rapidocr"
    raise RuntimeError("No OCR engine available (install paddleocr or rapidocr-onnxruntime)")


def _ocr_paddle(img_path: Path) -> str:
    try:
        result = _PADDLE.predict(str(img_path))  # type: ignore[union-attr]
    except AttributeError:
        result = _PADDLE.ocr(str(img_path), cls=False)  # type: ignore[union-attr]

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
    _ensure_engine()
    if (_ACTIVE_ENGINE or "").startswith("paddleocr"):
        return _ocr_paddle(img_path)
    return _ocr_rapid(img_path)


def merge_pdfs(sources: list[Path], dest: Path) -> Path:
    """Concatenate PDFs in order."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged = fitz.open()
    for src in sources:
        doc = fitz.open(src)
        merged.insert_pdf(doc)
        doc.close()
    merged.save(dest)
    merged.close()
    return dest


def ocr_pdf_to_text_layer(
    src: Path,
    dest: Path,
    *,
    page_cache_dir: Path | None = None,
) -> dict:
    """
    OCR a scanned PDF and write a new PDF with an extractable text layer per page.
    Returns stats dict with char counts and engine used.
    """
    _ensure_engine()
    cache_dir = (page_cache_dir or Path("data/ocr_cache")) / src.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_doc = fitz.open(src)
    out_doc = fitz.open()
    page_chars: list[int] = []
    zoom = OCR_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    batch_start = 0
    n_pages = len(src_doc)
    while batch_start < n_pages:
        batch_end = min(n_pages, batch_start + OCR_BATCH_PAGES)
        for i in range(batch_start, batch_end):
            src_page = src_doc[i]
            embedded = src_page.get_text("text").strip()

            if (
                not OCR_FORCE_ALL
                and OCR_SKIP_TEXT_PAGES
                and len(embedded) >= OCR_MIN_TEXT_CHARS
            ):
                page_text = embedded
            else:
                img_path = cache_dir / f"p{i:04d}.png"
                if not img_path.exists():
                    pix = src_page.get_pixmap(matrix=mat, alpha=False)
                    pix.save(str(img_path))
                    del pix
                page_text = _ocr_image(img_path)
                if len(page_text) < len(embedded):
                    page_text = embedded

            out_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
            if page_text.strip():
                rect = fitz.Rect(36, 36, src_page.rect.width - 36, src_page.rect.height - 36)
                out_page.insert_textbox(rect, page_text, fontsize=9, align=fitz.TEXT_ALIGN_LEFT)
            page_chars.append(len(page_text.strip()))

        logger.info(f"OCR pages {batch_start + 1}-{batch_end}/{n_pages} ({active_engine()})")
        gc.collect()
        batch_start = batch_end

    dest.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(dest)
    out_doc.close()
    src_doc.close()
    gc.collect()

    total_chars = sum(page_chars)
    return {
        "engine": active_engine(),
        "pages": n_pages,
        "chars_per_page": page_chars,
        "total_chars": total_chars,
        "output": str(dest),
    }
