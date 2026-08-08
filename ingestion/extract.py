"""Layout-aware extraction → unified per-page element schema.

Tool-agnostic Stage 1 contract. Every path (Docling, LightOnOCR, scanned OCR)
emits the same ``PageElement`` shape so chunking/retrieval never care which
parser produced the text.

Element types: heading | paragraph | table | figure | caption | footnote | formula
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ElementType = Literal[
    "heading",
    "paragraph",
    "table",
    "figure",
    "caption",
    "footnote",
    "formula",
]

ELEMENT_TYPES: frozenset[str] = frozenset(
    {"heading", "paragraph", "table", "figure", "caption", "footnote", "formula"}
)

# Same tibia-force / Figure 3 guard as caption_guard (≥3 numeric segments).
_CLAUSE_IN_CAPTION_RE = re.compile(
    r"(?m)(?<!\d)(\d+(?:\.\d+){2,})\.?(?=\s|[A-Za-z]|$)"
)
_LEADING_CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+\S")
_SCANNED_MIN_CHARS = 40  # below this + has figures → scanned candidate


class PageElement(BaseModel):
    """One layout element in reading order (tool-agnostic)."""

    element_id: str
    element_type: ElementType
    text: str = ""
    html: str | None = None
    markdown: str | None = None
    page_number: int
    coordinates: list[float] = Field(default_factory=list)  # [l, t, r, b]
    reading_order: int
    parent_id: str | None = None
    section_number: str | None = None
    section: str | None = None  # human title
    row_count: int | None = None
    col_count: int | None = None
    source: str = "docling"  # docling | lightonocr | scanned_ocr | pdfplumber
    confidence: float | None = None

    @property
    def bounding_box(self) -> list[float]:
        return self.coordinates


class PageExtract(BaseModel):
    """All elements on one page, sorted by reading_order."""

    page_number: int
    elements: list[PageElement] = Field(default_factory=list)
    is_scanned: bool = False
    source: str = "docling"
    flags: list[str] = Field(default_factory=list)


class DocumentExtract(BaseModel):
    """Full-document Stage 1 extract."""

    document_id: str
    regulation_id: str = ""
    revision: str = ""
    source_path: str = ""
    pages: list[PageExtract] = Field(default_factory=list)
    remediation_actions: list[str] = Field(default_factory=list)

    @property
    def elements(self) -> list[PageElement]:
        out: list[PageElement] = []
        for page in self.pages:
            out.extend(page.elements)
        return out


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def _label_name(item: Any) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "value", None) or str(label or type(item).__name__)


def _bbox_list(item: Any) -> list[float]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return []
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return []
    try:
        return [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]
    except Exception:  # noqa: BLE001
        return []


def _page_number(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page = getattr(prov[0], "page_no", None)
    return int(page) if page is not None else None


def _item_text(item: Any) -> str:
    return (getattr(item, "text", None) or "").strip()


def map_docling_label(label: Any, label_name: str) -> ElementType | None:
    """Map Docling labels → Stage 1 element_type. Returns None to skip."""
    from docling_core.types.doc import DocItemLabel

    name = (label_name or "").lower()
    if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE) or name in {
        "section_header",
        "title",
    }:
        return "heading"
    if label == DocItemLabel.TABLE or name == "table":
        return "table"
    if label in (DocItemLabel.PICTURE,) or name in {"picture", "figure", "chart"}:
        return "figure"
    if label == DocItemLabel.CAPTION or name == "caption":
        return "caption"
    if label == DocItemLabel.FOOTNOTE or name == "footnote":
        return "footnote"
    if label == DocItemLabel.FORMULA or name in {"formula", "equation"}:
        return "formula"
    if label in (
        DocItemLabel.TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.LIST_ITEM,
    ) or name in {"text", "paragraph", "list_item"}:
        return "paragraph"
    if name in {"page_header", "page_footer", "code"}:
        return None
    # Unknown prose-like → paragraph; skip empty unknowns.
    return "paragraph"


def _table_structure(item: Any, doc: Any) -> tuple[str, str | None, int, int]:
    """Return (markdown, html, row_count, col_count) for a table element."""
    md = ""
    html: str | None = None
    rows = 0
    cols = 0

    data = getattr(item, "data", None)
    if data is not None:
        rows = int(getattr(data, "num_rows", 0) or 0)
        cols = int(getattr(data, "num_cols", 0) or 0)
        grid = getattr(data, "grid", None)
        if (not rows or not cols) and grid:
            try:
                rows = len(grid)
                cols = len(grid[0]) if grid else 0
            except Exception:  # noqa: BLE001
                pass

    try:
        md = (item.export_to_markdown(doc=doc) or "").strip()
    except TypeError:
        try:
            md = (item.export_to_markdown() or "").strip()
        except Exception:  # noqa: BLE001
            md = ""
    except Exception:  # noqa: BLE001
        try:
            df = item.export_to_dataframe(doc=doc)
            md = df.to_markdown(index=False)
            rows = max(rows, int(getattr(df, "shape", (0, 0))[0] or 0))
            cols = max(cols, int(getattr(df, "shape", (0, 0))[1] or 0))
        except Exception:  # noqa: BLE001
            md = ""

    for html_fn in ("export_to_html", "export_to_html_string"):
        if hasattr(item, html_fn):
            try:
                fn = getattr(item, html_fn)
                try:
                    html = (fn(doc=doc) or "").strip() or None
                except TypeError:
                    html = (fn() or "").strip() or None
                if html:
                    break
            except Exception:  # noqa: BLE001
                continue

    # Infer grid size from markdown if Docling counts missing.
    if md and (rows <= 0 or cols <= 0):
        lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
        data_lines = [ln for ln in lines if not re.match(r"^\|[\s:|-]+\|$", ln.strip())]
        if data_lines:
            rows = max(rows, len(data_lines))
            cols = max(cols, data_lines[0].count("|") - 1)

    return md, html, rows, cols


def _crosscheck_table_pdfplumber(
    pdf_path: Path | None,
    page_number: int,
    *,
    confidence: float | None,
    low_confidence_threshold: float = 0.7,
) -> tuple[str | None, int, int] | None:
    """Optional pdfplumber table when Docling confidence is low / missing."""
    if pdf_path is None or not Path(pdf_path).is_file():
        return None
    if confidence is not None and confidence >= low_confidence_threshold:
        return None
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed; skip table cross-check")
        return None

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return None
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables() or []
            if not tables:
                return None
            # Pick the largest table on the page.
            best = max(tables, key=lambda t: (len(t) * max((len(r) for r in t), default=0)))
            rows = len(best)
            cols = max((len(r) for r in best), default=0)
            # Build markdown
            header = best[0] if best else []
            lines = [
                "| "
                + " | ".join("" if c is None else str(c).replace("\n", " ") for c in header)
                + " |"
            ]
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in best[1:]:
                lines.append(
                    "| "
                    + " | ".join("" if c is None else str(c).replace("\n", " ") for c in row)
                    + " |"
                )
            return "\n".join(lines), rows, cols
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber table cross-check failed p%s: %s", page_number, exc)
        return None


def extract_from_docling(
    doc: Any,
    *,
    document_id: str = "",
    regulation_id: str = "",
    revision: str = "",
    source_path: str = "",
    pdf_path: Path | str | None = None,
    source: str = "docling",
) -> DocumentExtract:
    """Convert a DoclingDocument into the unified Stage 1 element schema."""
    pdf = Path(pdf_path) if pdf_path else (Path(source_path) if source_path else None)
    doc_id = document_id or regulation_id or (pdf.stem if pdf else "document")

    by_page: dict[int, list[PageElement]] = {}
    reading = 0
    last_heading_id: str | None = None
    last_heading_num: str | None = None
    last_heading_title: str | None = None

    for item, _level in doc.iterate_items():
        page = _page_number(item)
        if page is None:
            continue
        label = getattr(item, "label", None)
        name = _label_name(item)
        etype = map_docling_label(label, name)
        if etype is None:
            continue

        text = _item_text(item)
        coords = _bbox_list(item)
        md: str | None = None
        html: str | None = None
        rows: int | None = None
        cols: int | None = None
        confidence = getattr(item, "confidence", None)
        if confidence is None:
            conf_obj = getattr(item, "prov", None) or []
            if conf_obj and hasattr(conf_obj[0], "score"):
                try:
                    confidence = float(conf_obj[0].score)
                except Exception:  # noqa: BLE001
                    confidence = None

        if etype == "table":
            md, html, r_count, c_count = _table_structure(item, doc)
            rows, cols = r_count, c_count
            # Cross-check low-confidence / empty structure with pdfplumber.
            if rows <= 0 or cols <= 0 or (confidence is not None and confidence < 0.7):
                alt = _crosscheck_table_pdfplumber(pdf, page, confidence=confidence)
                if alt is not None:
                    alt_md, alt_r, alt_c = alt
                    if rows <= 0 or cols <= 0:
                        md = alt_md or md
                        rows, cols = alt_r, alt_c
                        source = "pdfplumber"
            text = text or (md or "")
            if not md and text:
                md = text
        elif etype == "figure":
            text = text or f"[figure p{page}]"
        elif not text and etype not in {"figure", "table"}:
            continue

        reading += 1
        eid = _stable_id(doc_id, str(page), str(reading), etype, text[:64])

        section_number = last_heading_num
        section_title = last_heading_title
        parent_id = last_heading_id

        if etype == "heading":
            m = _LEADING_CLAUSE_RE.match(text)
            if m:
                section_number = m.group(1)
                section_title = text[m.end() :].strip() or text
            else:
                # "Annex 3" / free title
                section_number = section_number or ""
                section_title = text
            last_heading_id = eid
            last_heading_num = section_number or last_heading_num
            last_heading_title = section_title
            parent_id = None

        el = PageElement(
            element_id=eid,
            element_type=etype,
            text=text,
            html=html,
            markdown=md,
            page_number=page,
            coordinates=coords if coords else [0.0, 0.0, 0.0, 0.0],
            reading_order=reading,
            parent_id=parent_id,
            section_number=section_number,
            section=section_title,
            row_count=rows,
            col_count=cols,
            source=source,
            confidence=float(confidence) if confidence is not None else None,
        )
        by_page.setdefault(page, []).append(el)

    pages: list[PageExtract] = []
    for page_no in sorted(by_page):
        elems = by_page[page_no]
        text_len = sum(len(e.text) for e in elems if e.element_type == "paragraph")
        has_fig = any(e.element_type in {"figure", "table"} for e in elems)
        is_scanned = text_len < _SCANNED_MIN_CHARS and has_fig
        pages.append(
            PageExtract(
                page_number=page_no,
                elements=elems,
                is_scanned=is_scanned,
                source=source,
            )
        )

    return DocumentExtract(
        document_id=doc_id,
        regulation_id=regulation_id,
        revision=revision,
        source_path=str(pdf) if pdf else source_path,
        pages=pages,
    )


def remediate_clause_as_caption(doc: Any) -> list[str]:
    """Deterministic fix: caption/figure text that is a numbered clause → paragraph.

    Mirrors step 1 of ``apply_vlm_preference_to_page`` so unit tests and
    LightOnOCR-unavailable paths still clear the tibia-force regression.
    """
    from docling_core.types.doc import DocItemLabel

    from ingestion.vlm_figure_pass import _set_item_label_text

    actions: list[str] = []
    for item, _lvl in doc.iterate_items():
        label = getattr(item, "label", None)
        name = _label_name(item).lower()
        if label not in (DocItemLabel.CAPTION, DocItemLabel.PICTURE) and name not in {
            "caption",
            "picture",
            "figure",
        }:
            continue
        text = _item_text(item)
        if not text:
            continue
        if not _CLAUSE_IN_CAPTION_RE.search(text) and not _LEADING_CLAUSE_RE.match(text):
            continue
        _set_item_label_text(item, label=DocItemLabel.TEXT)
        actions.append(f"relabel_clause_caption_to_paragraph:{text.splitlines()[0][:80]}")
    return actions


def detect_scanned_pages(extract: DocumentExtract) -> list[int]:
    """Pages that look image-only / scanned."""
    return [p.page_number for p in extract.pages if p.is_scanned]


def elements_with_clause_in_caption_or_figure(
    extract: DocumentExtract,
) -> list[PageElement]:
    """Regression helper: caption/figure elements containing clause numbers."""
    bad: list[PageElement] = []
    for el in extract.elements:
        if el.element_type not in {"caption", "figure"}:
            continue
        if _CLAUSE_IN_CAPTION_RE.search(el.text or ""):
            bad.append(el)
    return bad


def assert_extract_invariants(extract: DocumentExtract) -> None:
    """Raise AssertionError if Stage 1 schema contract is violated."""
    bad = elements_with_clause_in_caption_or_figure(extract)
    if bad:
        preview = "; ".join(
            f"p{e.page_number}/{e.element_type}:{e.text[:60]!r}" for e in bad[:3]
        )
        raise AssertionError(f"clause-number in caption/figure: {preview}")

    for el in extract.elements:
        if el.page_number is None or int(el.page_number) < 1:
            raise AssertionError(f"missing page_number on {el.element_id}")
        if not el.coordinates or len(el.coordinates) < 4:
            raise AssertionError(f"missing coordinates on {el.element_id}")
        if el.reading_order is None or int(el.reading_order) < 1:
            raise AssertionError(f"missing reading_order on {el.element_id}")
        if el.element_type not in ELEMENT_TYPES:
            raise AssertionError(f"bad element_type {el.element_type}")

    for el in extract.elements:
        if el.element_type != "table":
            continue
        rows = int(el.row_count or 0)
        cols = int(el.col_count or 0)
        if rows <= 0 or cols <= 0:
            raise AssertionError(
                f"table {el.element_id} flattened/unstructured rows={rows} cols={cols}"
            )
        structured = (el.markdown or el.html or "").strip()
        if not structured:
            raise AssertionError(f"table {el.element_id} has no markdown/html structure")


def extract_pdf(
    pdf_path: str | Path,
    *,
    regulation_id: str = "",
    revision: str = "",
    document_id: str | None = None,
    export_dir: str | Path | None = None,
    do_ocr: bool = False,
    skip_auto_lighton: bool = False,
    apply_deterministic_caption_fix: bool = True,
    layout_score_threshold: float | None = None,
    layout_preset: str | None = None,
) -> DocumentExtract:
    """Full Stage 1: Docling → validate → LightOnOCR / scanned OCR → unified extract."""
    from ingestion.extraction_validator import (
        pages_needing_lighton,
        validate_extraction,
    )
    from ingestion.parse import parse_pdf

    path = Path(pdf_path).resolve()
    export_dir = Path(export_dir or os.getenv("DOCLING_EXPORT_DIR", "./data/docling"))
    doc_id = document_id or regulation_id or path.stem

    doc = parse_pdf(
        path,
        export_dir=export_dir,
        do_ocr=do_ocr,
        layout_score_threshold=layout_score_threshold,
        layout_preset=layout_preset,
    )

    validation = validate_extraction(doc)
    flags_path = export_dir / f"{path.stem}.extraction_flags.json"
    validation.save(flags_path)

    # Preliminary extract for scanned-page detection.
    prelim = extract_from_docling(
        doc,
        document_id=doc_id,
        regulation_id=regulation_id,
        revision=revision,
        source_path=str(path),
        pdf_path=path,
    )
    scanned = detect_scanned_pages(prelim)

    remediation: list[str] = []
    auto_lighton = (not skip_auto_lighton) and os.getenv(
        "EXTRACTION_AUTO_LIGHTON", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}

    pages_for_vlm: set[int] = set()
    if auto_lighton:
        pages_for_vlm.update(pages_needing_lighton(validation))
    pages_for_vlm.update(scanned)

    if pages_for_vlm:
        try:
            from ingestion.vlm_figure_pass import apply_vlm_figure_pass

            vlm_result = apply_vlm_figure_pass(
                doc,
                path,
                pages=sorted(pages_for_vlm),
            )
            remediation.append(
                f"lightonocr_pages={sorted(vlm_result.pages_processed)}"
            )
            if vlm_result.pages_corrected:
                remediation.append(
                    f"lightonocr_corrected={sorted(vlm_result.pages_corrected)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightOnOCR fallback failed: %s", exc)
            remediation.append(f"lightonocr_error:{exc}")

    # Scanned pages: re-parse with Docling OCR if still thin (Surya/Paddle optional).
    if scanned and not do_ocr:
        try:
            ocr_doc = parse_pdf(
                path,
                export_dir=None,
                do_ocr=True,
                page_range=(min(scanned), max(scanned)),
            )
            # Merge OCR page items by re-extracting OCR doc pages into flags.
            ocr_extract = extract_from_docling(
                ocr_doc,
                document_id=doc_id,
                regulation_id=regulation_id,
                revision=revision,
                source_path=str(path),
                pdf_path=path,
                source="scanned_ocr",
            )
            remediation.append(f"scanned_ocr_pages={scanned}")
            # Prefer OCR elements for scanned pages when denser.
            ocr_by_page = {p.page_number: p for p in ocr_extract.pages}
            for i, page in enumerate(prelim.pages):
                if page.page_number not in ocr_by_page:
                    continue
                ocr_page = ocr_by_page[page.page_number]
                if len(ocr_page.elements) >= len(page.elements):
                    prelim.pages[i] = ocr_page
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scanned OCR path failed: %s", exc)
            remediation.append(f"scanned_ocr_error:{exc}")
            # Optional Surya / PaddleOCR hooks (same schema if available).
            _try_external_ocr(path, scanned, remediation)

    if apply_deterministic_caption_fix:
        actions = remediate_clause_as_caption(doc)
        remediation.extend(actions)

    extract = extract_from_docling(
        doc,
        document_id=doc_id,
        regulation_id=regulation_id,
        revision=revision,
        source_path=str(path),
        pdf_path=path,
    )
    # Preserve scanned OCR page merges when we already swapped them in.
    if scanned and any(p.source == "scanned_ocr" for p in prelim.pages):
        by_page = {p.page_number: p for p in extract.pages}
        for page in prelim.pages:
            if page.source == "scanned_ocr":
                by_page[page.page_number] = page
        extract.pages = [by_page[k] for k in sorted(by_page)]

    extract.remediation_actions = remediation
    for page in extract.pages:
        page.flags = validation.triggers_for_page(page.page_number)
    return extract


def _try_external_ocr(
    pdf_path: Path,
    pages: Sequence[int],
    remediation: list[str],
) -> None:
    """Best-effort Surya/PaddleOCR import — records availability only for now."""
    for name in ("surya", "paddleocr"):
        try:
            __import__(name)
            remediation.append(f"external_ocr_available:{name}")
        except ImportError:
            continue
    if pages:
        remediation.append(
            "external_ocr_not_applied:install surya or paddleocr for alt scanned path"
        )


def load_docling_json(path: str | Path) -> Any:
    """Load a previously exported Docling JSON."""
    from docling_core.types.doc import DoclingDocument

    path = Path(path)
    if hasattr(DoclingDocument, "load_from_json"):
        return DoclingDocument.load_from_json(str(path))
    return DoclingDocument.model_validate_json(path.read_text(encoding="utf-8"))


def find_figure_parent_section(
    extract: DocumentExtract,
    *,
    page_number: int,
    figure_label: str = "Figure 3",
) -> str | None:
    """Infer parent section for a figure from nearby heading/paragraph context."""
    page = next((p for p in extract.pages if p.page_number == page_number), None)
    if page is None:
        return None
    label_l = figure_label.lower()
    last_section: str | None = None
    for el in sorted(page.elements, key=lambda e: e.reading_order):
        if el.section_number:
            last_section = el.section_number
        if el.element_type in {"heading", "paragraph"}:
            m = _LEADING_CLAUSE_RE.match(el.text)
            if m:
                last_section = m.group(1)
            if label_l in el.text.lower() and last_section:
                return last_section
        if el.element_type == "figure" and label_l in (el.text or "").lower():
            return el.section_number or last_section
        if el.element_type == "caption" and label_l in (el.text or "").lower():
            return el.section_number or last_section
    return last_section
