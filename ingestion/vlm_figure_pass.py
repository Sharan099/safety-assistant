"""Selective LightOnOCR-2 pass over figure/table-adjacent pages only.

Keep Docling for the bulk of straightforward clause text (verified correct when
metadata is right). Run a VLM **only** on pages flagged as figure/table-adjacent
by ``scripts/audit_figure_adjacent_chunks.py`` (or the same page-collection
logic). Prefer the VLM reading order / section attribution when it disagrees
with Docling — the Stage 1 failure mode where Docling tags the clause *after*
a figure as ``caption`` (e.g. R94 Figure 3 → ``5.2.1.7`` swallowed).

Licensing: page renders use **pypdfium2** (PDFium), never PyMuPDF/AGPL.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from docling_core.types.doc import DocItemLabel, DoclingDocument

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B-bbox"
DEFAULT_MAX_DIM = 1540

_CLAUSE_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\.\s+")
_CLAUSE_LINE_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>\S.*)$",
)
_BBOX_IMG_RE = re.compile(
    r"!\[image\]\(image_(?P<idx>\d+)\.png\)"
    r"(?P<x1>\d+),(?P<y1>\d+),(?P<x2>\d+),(?P<y2>\d+)",
)
_FIGURE_CAPTION_RE = re.compile(
    r"(?im)^\s*(Figure|Table)\s+(?P<num>\d+)\s*(?:[–—\-:.]?\s*(?P<title>.*))?$"
)
_LEADING_CLAUSE_ONLY_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+\S")


@dataclass
class FigureBox:
    """Normalized (0–1000) bbox from LightOnOCR-2-bbox plus caption / parent clause.

    ``parent_section_number`` comes from the **corrected VLM reading order**
    (clause that references the figure, else the immediately preceding clause) —
    never from Docling's mis-attributed caption role.
    """

    image_index: int
    bbox_norm: tuple[int, int, int, int]  # x1,y1,x2,y2 in 0–1000
    caption: str = ""
    figure_label: str = ""  # e.g. "Figure 3"
    parent_section_number: str = ""
    surrounding_text: str = ""


@dataclass
class PageDiscrepancy:
    page_number: int
    reason: str
    docling_clauses: list[str]
    vlm_clauses: list[str]
    docling_caption_mislabels: list[str]
    preferred: str = "vlm"
    detail: str = ""


@dataclass
class VlmPageResult:
    page_number: int
    markdown: str
    figures: list[FigureBox] = field(default_factory=list)
    clause_sequence: list[str] = field(default_factory=list)


@dataclass
class VlmFigurePassResult:
    """Outcome of a selective VLM pass (Docling document may be mutated)."""

    pages_input: list[int]
    pages_processed: list[int]
    pages_corrected: list[int]
    discrepancies: list[PageDiscrepancy]
    page_results: dict[int, VlmPageResult] = field(default_factory=dict)
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    elapsed_s: float = 0.0
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_input": self.pages_input,
            "pages_processed": self.pages_processed,
            "pages_corrected": self.pages_corrected,
            "discrepancies": [asdict(d) for d in self.discrepancies],
            "model_id": self.model_id,
            "device": self.device,
            "elapsed_s": self.elapsed_s,
            "log_path": self.log_path,
            "figures_by_page": {
                str(p): [asdict(f) for f in r.figures]
                for p, r in self.page_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Audit page list
# ---------------------------------------------------------------------------


def _label_name(item: Any) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "value", None) or str(label or type(item).__name__)


def _page_of(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page = getattr(prov[0], "page_no", None)
    return int(page) if page is not None else None


def _item_text(item: Any) -> str:
    return (getattr(item, "text", None) or "").strip()


def _is_figure_or_table_item(item: Any) -> bool:
    from docling_core.types.doc import TableItem

    if isinstance(item, TableItem):
        return True
    label = getattr(item, "label", None)
    if label in (
        DocItemLabel.TABLE,
        DocItemLabel.PICTURE,
        DocItemLabel.CAPTION,
        getattr(DocItemLabel, "CHART", None),
    ):
        return True
    name = _label_name(item).lower()
    return name in {"table", "picture", "caption", "figure", "chart"}


def collect_figure_adjacent_pages(
    doc: DoclingDocument,
    *,
    neighbor: bool = True,
) -> list[int]:
    """Pages that contain a figure/table/caption (+ optional next page).

    Matches the adjacency window used by ``audit_figure_adjacent_chunks``
    (clause chunks on a figure page or the page after it).
    """
    figure_pages: set[int] = set()
    for item, _lvl in doc.iterate_items():
        if not _is_figure_or_table_item(item):
            continue
        page = _page_of(item)
        if page is None:
            continue
        figure_pages.add(int(page))
        preview = _item_text(item).lower()
        if preview.startswith("figure") or "figure " in preview[:24]:
            figure_pages.add(int(page))

    pages = set(figure_pages)
    if neighbor:
        for p in list(figure_pages):
            pages.add(p + 1)
    return sorted(pages)


def load_audit_page_list(path: str | Path) -> list[int]:
    """Load page numbers from audit ``--json-out`` or a legacy text dump."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or raw.lstrip().startswith("{"):
        data = json.loads(raw)
        for key in (
            "figure_adjacent_pages",
            "pages",
            "suspect_pages",
            "figure_pages",
        ):
            if key in data and data[key]:
                return sorted({int(p) for p in data[key]})
        suspects = data.get("suspects") or []
        pages = {
            int(s["page_number"])
            for s in suspects
            if s.get("page_number") is not None
        }
        if pages:
            return sorted(pages)
        raise ValueError(f"No page list found in audit JSON: {path}")

    # Legacy stdout dump: ``page=20`` / ``page=24``
    found = {int(m.group(1)) for m in re.finditer(r"\bpage=(\d+)\b", raw)}
    if not found:
        raise ValueError(f"No page=N markers in audit text: {path}")
    return sorted(found)


# ---------------------------------------------------------------------------
# Render (pypdfium2 only) + LightOnOCR
# ---------------------------------------------------------------------------


def render_page_png(
    pdf_path: str | Path,
    page_number: int,
    out_png: str | Path,
    *,
    max_dim: int = DEFAULT_MAX_DIM,
) -> Path:
    """Render a 1-indexed PDF page to PNG via pypdfium2 (not PyMuPDF)."""
    import pypdfium2 as pdfium

    pdf_path = Path(pdf_path)
    out_png = Path(out_png)
    if page_number < 1:
        raise ValueError(f"page_number must be 1-indexed, got {page_number}")

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_number > len(pdf):
            raise IndexError(
                f"page {page_number} out of range for {pdf_path.name} ({len(pdf)} pages)"
            )
        page = pdf[page_number - 1]
        w, h = page.get_size()
        scale = max_dim / max(float(w), float(h))
        pil = page.render(scale=scale).to_pil()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_png, format="PNG")
    finally:
        pdf.close()
    return out_png


def parse_vlm_markdown(markdown: str, *, page_number: int) -> VlmPageResult:
    """Extract clause sequence + figures with parent linkage from LightOn markdown.

    Parent clause is resolved from **reading order**: the nearest preceding clause
    that mentions ``Figure N``, else the immediately preceding numbered clause.
    """
    text = markdown or ""
    clauses = _CLAUSE_RE.findall(text)
    lines = text.splitlines()

    # Walk once: track last clause + clause bodies; bind figures in order.
    last_clause: str | None = None
    clause_bodies: dict[str, str] = {}
    figure_ref_to_clause: dict[str, str] = {}  # "3" -> "5.2.1.6"
    current_clause_lines: list[str] = []
    current_clause_num: str | None = None
    pending_label = ""
    pending_caption_bits: list[str] = []
    figures: list[FigureBox] = []

    def _flush_clause() -> None:
        nonlocal current_clause_num, current_clause_lines
        if not current_clause_num:
            return
        body = "\n".join(current_clause_lines).strip()
        clause_bodies[current_clause_num] = body
        for m in re.finditer(r"(?i)\bFigures?\s+(\d+(?:\s*(?:and|,)\s*\d+)*)", body):
            nums = re.findall(r"\d+", m.group(1))
            for n in nums:
                figure_ref_to_clause.setdefault(n, current_clause_num)
        current_clause_lines = []

    for line in lines:
        stripped = line.strip()
        cm = _CLAUSE_LINE_RE.match(stripped)
        if cm:
            _flush_clause()
            current_clause_num = cm.group("num")
            last_clause = current_clause_num
            current_clause_lines = [stripped]
            pending_label = ""
            pending_caption_bits = []
            continue

        fig_m = _FIGURE_CAPTION_RE.match(stripped)
        if fig_m:
            _flush_clause()
            current_clause_num = None
            pending_label = f"{fig_m.group(1).title()} {fig_m.group('num')}"
            pending_caption_bits = [pending_label]
            title = (fig_m.groupdict().get("title") or "").strip(" .–—-:")
            if title:
                pending_caption_bits.append(title)
            continue

        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            title = stripped.strip("* ").strip()
            if title:
                pending_caption_bits.append(title)
            continue

        bm = _BBOX_IMG_RE.search(stripped)
        if bm:
            _flush_clause()
            current_clause_num = None
            label = pending_label
            caption = " — ".join(dict.fromkeys(pending_caption_bits)) if pending_caption_bits else ""
            fig_num = ""
            if label:
                nm = re.search(r"(\d+)", label)
                if nm:
                    fig_num = nm.group(1)
            parent = ""
            if fig_num and fig_num in figure_ref_to_clause:
                parent = figure_ref_to_clause[fig_num]
            elif last_clause:
                parent = last_clause
            surrounding = clause_bodies.get(parent, "") if parent else ""
            figures.append(
                FigureBox(
                    image_index=int(bm.group("idx")),
                    bbox_norm=(
                        int(bm.group("x1")),
                        int(bm.group("y1")),
                        int(bm.group("x2")),
                        int(bm.group("y2")),
                    ),
                    caption=caption or label,
                    figure_label=label or (f"Figure {fig_num}" if fig_num else ""),
                    parent_section_number=parent,
                    surrounding_text=surrounding,
                )
            )
            pending_label = ""
            pending_caption_bits = []
            continue

        if current_clause_num is not None and stripped:
            current_clause_lines.append(stripped)
        elif stripped and not stripped.startswith("E/ECE/") and not re.fullmatch(r"\d{1,3}", stripped):
            # Orphan prose between figures — attach to pending caption if any.
            if pending_label and not stripped.startswith("!["):
                pending_caption_bits.append(stripped)

    _flush_clause()

    return VlmPageResult(
        page_number=page_number,
        markdown=text,
        figures=figures,
        clause_sequence=list(clauses),
    )


def _docling_fallback_caption_and_surround(
    doc: DoclingDocument,
    page_number: int,
    *,
    figure_label: str,
) -> tuple[str, str]:
    """When VLM caption is empty, use Docling caption text + nearby paragraphs."""
    captions: list[str] = []
    prose: list[str] = []
    label_l = (figure_label or "").lower()
    for item, _lvl in doc.iterate_items():
        if _page_of(item) != page_number:
            continue
        text = _item_text(item)
        if not text:
            continue
        name = _label_name(item).lower()
        label = getattr(item, "label", None)
        if label == DocItemLabel.CAPTION or name == "caption":
            if label_l and label_l in text.lower():
                captions.insert(0, text)
            elif _FIGURE_CAPTION_RE.match(text) or not _LEADING_CLAUSE_ONLY_RE.match(text):
                captions.append(text)
            continue
        if label in (DocItemLabel.PICTURE,) or name in {"picture", "figure"}:
            continue
        if _LEADING_CLAUSE_ONLY_RE.match(text) or len(text) > 40:
            prose.append(text)
    caption = " — ".join(dict.fromkeys(captions)) if captions else figure_label
    # Prefer paragraph that mentions the figure label / "Figure".
    surround = ""
    for p in prose:
        if figure_label and figure_label.lower() in p.lower():
            surround = p
            break
        if re.search(r"(?i)\bfigure\s+\d+", p):
            surround = p
            break
    if not surround and prose:
        # Immediately surrounding: take last prose before we'd attach to figure.
        surround = prose[-1] if len(prose) == 1 else prose[-2] if len(prose) > 1 else prose[0]
    return caption, surround


def build_figure_chunks_from_vlm(
    vlm_result: VlmFigurePassResult,
    doc: DoclingDocument,
    *,
    regulation_id: str,
    revision: str,
) -> list:
    """Create ``content_type='figure'`` chunks for every Stage-2 figure.

    Links each figure to its parent clause ``section_number`` using the VLM
    corrected reading order. Falls back to Docling caption + surrounding
    paragraph when the VLM omits a caption/description.
    """
    from ingestion.chunk import _section_id, _stable_id
    from ingestion.models import Chunk

    chunks: list[Chunk] = []
    seen: set[str] = set()

    for page_number, page in sorted(vlm_result.page_results.items()):
        for fig in page.figures:
            caption = (fig.caption or "").strip()
            label = (fig.figure_label or "").strip()
            surround = (fig.surrounding_text or "").strip()
            parent = (fig.parent_section_number or "").strip()

            if not caption or not surround:
                fb_cap, fb_sur = _docling_fallback_caption_and_surround(
                    doc, page_number, figure_label=label or caption
                )
                if not caption:
                    caption = fb_cap
                if not surround:
                    surround = fb_sur
            if not label and caption:
                m = re.match(r"(?i)(Figure|Table)\s+(\d+)", caption)
                if m:
                    label = f"{m.group(1).title()} {m.group(2)}"
            if not label:
                label = f"Figure img{fig.image_index}"

            # Still no parent — try Docling surrounding clause mention.
            if not parent and surround:
                cm = _CLAUSE_LINE_RE.match(surround)
                if cm:
                    parent = cm.group("num")
            # Do NOT invent a parent from the first clause on the page — that
            # mis-links orphan leading images (e.g. figure continued from prior page).

            title = caption if caption and caption != label else label
            if caption and label and label.lower() not in caption.lower():
                title = f"{label} — {caption}"

            body_parts = [title]
            if surround and surround not in title:
                body_parts.append(surround)
            if parent:
                body_parts.append(f"Parent clause: §{parent}")
            if not surround and not caption:
                body_parts.append(
                    f"(Figure on page {page_number}; no caption recovered — "
                    "bbox indexed for retrieval anchoring.)"
                )
            text = "\n\n".join(body_parts).strip()

            section_number = parent or f"page-{page_number}"
            section_id = f"{regulation_id}::figure::{page_number}::{label}"
            if section_id in seen:
                section_id = f"{section_id}::img{fig.image_index}"
            seen.add(section_id)
            parent_section_id = (
                _section_id(regulation_id, parent, "root") if parent else None
            )
            chunk_id = _stable_id(
                regulation_id, revision, section_id, "figure", text[:200]
            )
            # Store LightOn normalized bbox as 0–1 floats for payload consumers.
            x1, y1, x2, y2 = fig.bbox_norm
            bbox = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    regulation_id=regulation_id,
                    revision=revision,
                    section_number=section_number,
                    section_title=title[:120],
                    page_number=page_number,
                    bounding_box=bbox,
                    content_type="figure",
                    parent_section_id=parent_section_id,
                    section_id=section_id,
                    heading_path=[
                        p
                        for p in (
                            f"§{section_number}" if section_number else "",
                            label,
                        )
                        if p
                    ],
                )
            )
            logger.info(
                "Figure chunk %s → parent §%s page=%s label=%s",
                chunk_id,
                section_number,
                page_number,
                label,
            )

    return chunks


class LightOnOcrRunner:
    """Lazy-loaded LightOnOCR-2-1B-bbox (one image in, markdown out)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._dtype = None

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        device = self.device
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        logger.info("Loading %s on %s (%s)", self.model_id, device, dtype)
        processor = LightOnOcrProcessor.from_pretrained(self.model_id)
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device)
        model.eval()
        self._torch = torch
        self._dtype = dtype
        self._processor = processor
        self._model = model

    def run_image(self, image_path: str | Path) -> str:
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        torch = self._torch
        dtype = self._dtype
        device = self.device
        png = str(image_path)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "path": png}],
            }
        ]
        try:
            inputs = self._processor.apply_chat_template(
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
                    "content": [
                        {
                            "type": "image",
                            "image": Image.open(png).convert("RGB"),
                        }
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
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
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Docling page view + disagreement
# ---------------------------------------------------------------------------


def export_docling_page_markdown(doc: DoclingDocument, page_number: int) -> str:
    """Markdown-ish dump of Docling items on one 1-indexed page (for diffs)."""
    lines: list[str] = []
    for item, level in doc.iterate_items():
        if _page_of(item) != page_number:
            continue
        label_name = _label_name(item)
        text = _item_text(item)
        if hasattr(item, "export_to_markdown") and callable(item.export_to_markdown):
            try:
                text = (item.export_to_markdown() or text).strip()
            except Exception:  # noqa: BLE001
                pass
        if not text and label_name.lower() in {"picture", "figure", "table"}:
            text = f"[{label_name}]"
        if not text:
            continue
        lines.append(f"<!-- {label_name} -->\n{text}")
        _ = level  # hierarchy unused in flat export
    return "\n\n".join(lines).strip()


def docling_caption_mislabels(doc: DoclingDocument, page_number: int) -> list[str]:
    """Captions whose text is actually a numbered UNECE clause (Stage 1 bug)."""
    from ingestion.caption_guard import find_caption_clause_violations

    return [
        v.text_preview
        for v in find_caption_clause_violations(doc, pages=[page_number])
    ]


def docling_clause_sequence(doc: DoclingDocument, page_number: int) -> list[str]:
    """Clause ids visible in Docling item text on a page (incl. mislabeled captions)."""
    md = export_docling_page_markdown(doc, page_number)
    return _CLAUSE_RE.findall(md)


def reading_order_disagrees(
    *,
    docling_clauses: Sequence[str],
    vlm_clauses: Sequence[str],
    caption_mislabels: Sequence[str],
) -> tuple[bool, str]:
    """Return (disagrees, reason) for prefer-VLM decision."""
    if caption_mislabels:
        return True, "docling_caption_is_clause"
    if not vlm_clauses and not docling_clauses:
        return False, ""
    if list(docling_clauses) != list(vlm_clauses):
        # Prefer VLM when it recovers clauses Docling dropped into captions,
        # or when order differs. Identical empty → no disagreement.
        if not docling_clauses and vlm_clauses:
            return True, "vlm_has_clauses_docling_missing"
        if docling_clauses and not vlm_clauses:
            # Don't overwrite a healthy Docling page with empty VLM OCR.
            return False, "vlm_empty_keep_docling"
        return True, "clause_sequence_mismatch"
    return False, ""


def _set_item_label_text(item: Any, *, label: DocItemLabel, text: str | None = None) -> None:
    """Best-effort mutate Docling item label/text (pydantic models vary by version)."""
    if text is not None:
        try:
            item.text = text
        except Exception:  # noqa: BLE001
            try:
                object.__setattr__(item, "text", text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not set item.text: %s", exc)
    try:
        item.label = label
    except Exception:  # noqa: BLE001
        try:
            object.__setattr__(item, "label", label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set item.label: %s", exc)


def apply_vlm_preference_to_page(
    doc: DoclingDocument,
    page_number: int,
    vlm: VlmPageResult,
) -> list[str]:
    """Prefer VLM for this page's section attribution; return actions taken.

    Directly fixes the Figure 3 class: Docling ``caption`` wrapping ``5.2.1.7…``
    is relabeled to ``text`` so ``chunk_document`` no longer skips it.
    When VLM clause order still differs, rewrite page text/list items from the
    VLM markdown clause segments (reading order).
    """
    actions: list[str] = []

    # 1) Caption → text for clause-looking captions (confirmed Stage 1 bug).
    for item, _lvl in doc.iterate_items():
        if _page_of(item) != page_number:
            continue
        label = getattr(item, "label", None)
        if label != DocItemLabel.CAPTION and _label_name(item).lower() != "caption":
            continue
        text = _item_text(item)
        if not _LEADING_CLAUSE_ONLY_RE.match(text):
            continue
        _set_item_label_text(item, label=DocItemLabel.TEXT)
        actions.append(f"relabel_caption_to_text:{text.splitlines()[0][:80]}")

    # 2) If sequences still disagree, push VLM clause bodies onto matching items /
    #    append missing clause text onto the last text-like item on the page.
    doc_after = docling_clause_sequence(doc, page_number)
    if list(doc_after) == list(vlm.clause_sequence):
        return actions

    vlm_segments = _split_vlm_clause_segments(vlm.markdown)
    if not vlm_segments:
        return actions

    text_items: list[Any] = []
    for item, _lvl in doc.iterate_items():
        if _page_of(item) != page_number:
            continue
        label = getattr(item, "label", None)
        name = _label_name(item).lower()
        if label in (
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.LIST_ITEM,
        ) or name in {"text", "paragraph", "list_item"}:
            text_items.append(item)
        elif label == DocItemLabel.CAPTION or name == "caption":
            # Already attempted relabel; include if now text-like clause.
            if _LEADING_CLAUSE_ONLY_RE.match(_item_text(item)):
                text_items.append(item)

    # Map VLM segments onto clause-starting Docling items; leftover segments
    # appended to the last item (preserves page provenance).
    used_items: set[int] = set()
    for num, seg in vlm_segments:
        target = None
        for idx, item in enumerate(text_items):
            if idx in used_items:
                continue
            t = _item_text(item)
            m = _CLAUSE_LINE_RE.match(t)
            if m and m.group("num") == num:
                target = (idx, item)
                break
        if target is None:
            # Prefer empty/short items or create by overwriting a non-clause filler.
            for idx, item in enumerate(text_items):
                if idx in used_items:
                    continue
                if not _CLAUSE_LINE_RE.match(_item_text(item)):
                    target = (idx, item)
                    break
        if target is None and text_items:
            target = (len(text_items) - 1, text_items[-1])
            # Append rather than overwrite last.
            idx, item = target
            merged = (_item_text(item) + "\n\n" + seg).strip()
            _set_item_label_text(item, label=DocItemLabel.TEXT, text=merged)
            actions.append(f"append_vlm_clause:{num}")
            used_items.add(idx)
            continue
        if target is None:
            actions.append(f"orphan_vlm_clause_no_item:{num}")
            continue
        idx, item = target
        _set_item_label_text(item, label=DocItemLabel.TEXT, text=seg)
        actions.append(f"rewrite_item_from_vlm:{num}")
        used_items.add(idx)

    return actions


def _split_vlm_clause_segments(markdown: str) -> list[tuple[str, str]]:
    """Split VLM markdown into ``(clause_num, full_segment)`` in reading order."""
    raw = markdown or ""
    matches = list(_CLAUSE_RE.finditer(raw))
    if not matches:
        return []
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        seg = raw[start:end].strip()
        # Drop trailing page number / UNECE headers noise lines that are alone.
        lines = [
            ln
            for ln in seg.splitlines()
            if ln.strip()
            and not re.fullmatch(r"\d{1,3}", ln.strip())
            and not ln.strip().startswith("E/ECE/")
        ]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            out.append((m.group(1), cleaned))
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def apply_vlm_figure_pass(
    doc: DoclingDocument,
    pdf_path: str | Path,
    *,
    pages: Iterable[int] | None = None,
    audit_json: str | Path | None = None,
    model_id: str | None = None,
    device: str | None = None,
    work_dir: str | Path | None = None,
    max_dim: int = DEFAULT_MAX_DIM,
    runner: LightOnOcrRunner | None = None,
    dry_run: bool = False,
) -> VlmFigurePassResult:
    """Run LightOnOCR on figure-adjacent pages and prefer VLM on disagreements.

    Mutates ``doc`` in place when not ``dry_run``. Writes a JSON discrepancy log
    under ``work_dir``.
    """
    pdf_path = Path(pdf_path)
    model_id = model_id or os.getenv("VLM_FIGURE_MODEL", DEFAULT_MODEL_ID)
    work_dir = Path(
        work_dir
        or os.getenv("VLM_FIGURE_WORK_DIR", "data/vlm_figure_pass")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    renders = work_dir / "renders" / pdf_path.stem
    renders.mkdir(parents=True, exist_ok=True)
    md_dir = work_dir / "vlm_pages" / pdf_path.stem
    md_dir.mkdir(parents=True, exist_ok=True)

    if pages is not None:
        page_list = sorted({int(p) for p in pages})
    elif audit_json is not None:
        page_list = load_audit_page_list(audit_json)
    else:
        page_list = collect_figure_adjacent_pages(doc)

    t0 = time.perf_counter()
    ocr = runner or LightOnOcrRunner(model_id=model_id, device=device)
    discrepancies: list[PageDiscrepancy] = []
    corrected: list[int] = []
    processed: list[int] = []
    page_results: dict[int, VlmPageResult] = {}

    for page_number in page_list:
        png = renders / f"page_{page_number:04d}.png"
        if not png.is_file():
            render_page_png(pdf_path, page_number, png, max_dim=max_dim)

        cached_md = md_dir / f"page_{page_number:04d}.md"
        if cached_md.is_file():
            markdown = cached_md.read_text(encoding="utf-8")
            logger.info("VLM cache hit page %s", page_number)
        else:
            if dry_run:
                logger.info("dry_run: skip OCR page %s", page_number)
                continue
            markdown = ocr.run_image(png)
            cached_md.write_text(markdown + "\n", encoding="utf-8")

        vlm = parse_vlm_markdown(markdown, page_number=page_number)
        page_results[page_number] = vlm
        processed.append(page_number)

        cap_bad = docling_caption_mislabels(doc, page_number)
        doc_clauses = docling_clause_sequence(doc, page_number)
        disagrees, reason = reading_order_disagrees(
            docling_clauses=doc_clauses,
            vlm_clauses=vlm.clause_sequence,
            caption_mislabels=cap_bad,
        )
        if not disagrees:
            logger.info(
                "VLM page %s agrees with Docling (clauses=%s figures=%d)",
                page_number,
                doc_clauses[:8],
                len(vlm.figures),
            )
            continue

        disc = PageDiscrepancy(
            page_number=page_number,
            reason=reason,
            docling_clauses=list(doc_clauses),
            vlm_clauses=list(vlm.clause_sequence),
            docling_caption_mislabels=list(cap_bad),
            preferred="vlm",
            detail=(
                f"figures={len(vlm.figures)}; "
                + "; ".join(
                    f"{f.caption or 'fig'}@{f.bbox_norm}" for f in vlm.figures[:4]
                )
            ),
        )
        discrepancies.append(disc)
        logger.warning(
            "VLM/Docling disagreement page %s (%s): docling=%s vlm=%s mislabels=%s",
            page_number,
            reason,
            doc_clauses,
            vlm.clause_sequence,
            cap_bad,
        )

        if dry_run:
            continue
        actions = apply_vlm_preference_to_page(doc, page_number, vlm)
        disc.detail = (disc.detail + " | actions=" + ",".join(actions)).strip(" |")
        corrected.append(page_number)
        logger.info("Preferred VLM on page %s: %s", page_number, actions)

    elapsed = round(time.perf_counter() - t0, 2)
    result = VlmFigurePassResult(
        pages_input=page_list,
        pages_processed=processed,
        pages_corrected=corrected,
        discrepancies=discrepancies,
        page_results=page_results,
        model_id=model_id,
        device=getattr(ocr, "device", device or "cpu"),
        elapsed_s=elapsed,
    )
    log_path = work_dir / f"{pdf_path.stem}_vlm_figure_pass.json"
    log_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result.log_path = str(log_path)
    logger.info(
        "VLM figure pass done: processed=%d corrected=%d discrepancies=%d → %s",
        len(processed),
        len(corrected),
        len(discrepancies),
        log_path,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ingestion.parse import parse_pdf

    p = argparse.ArgumentParser(
        description="Selective LightOnOCR pass on figure/table-adjacent pages"
    )
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--docling", type=Path, default=None, help="Existing Docling JSON")
    p.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Output of audit_figure_adjacent_chunks.py --json-out",
    )
    p.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated 1-indexed pages (overrides audit/auto)",
    )
    p.add_argument("--export-dir", type=Path, default=Path("data/docling"))
    p.add_argument("--work-dir", type=Path, default=Path("data/vlm_figure_pass"))
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--write-docling",
        type=Path,
        default=None,
        help="If set, write corrected Docling JSON here",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.docling and args.docling.is_file():
        if hasattr(DoclingDocument, "load_from_json"):
            doc = DoclingDocument.load_from_json(str(args.docling))
        else:
            doc = DoclingDocument.model_validate(
                json.loads(args.docling.read_text(encoding="utf-8"))
            )
    else:
        doc = parse_pdf(args.pdf, export_dir=args.export_dir)

    pages = None
    if args.pages:
        pages = [int(x.strip()) for x in args.pages.split(",") if x.strip()]

    result = apply_vlm_figure_pass(
        doc,
        args.pdf,
        pages=pages,
        audit_json=args.audit_json,
        model_id=args.model,
        work_dir=args.work_dir,
        dry_run=args.dry_run,
    )
    print(
        f"processed={len(result.pages_processed)} "
        f"corrected={len(result.pages_corrected)} "
        f"discrepancies={len(result.discrepancies)} "
        f"log={result.log_path}"
    )
    for d in result.discrepancies:
        print(
            f"  page={d.page_number} reason={d.reason} "
            f"docling={d.docling_clauses} vlm={d.vlm_clauses}"
        )

    if args.write_docling and not args.dry_run:
        out = args.write_docling
        out.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(doc, "save_as_json"):
            doc.save_as_json(out)
        else:
            out.write_text(
                json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"Wrote corrected Docling JSON → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
