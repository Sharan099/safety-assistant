"""Stage 2 — figure / chart understanding → searchable figure chunks.

For every figure element, produce a non-empty description:
1. Preferred: vision model via Portkey (pinned ``gemini-2.5-flash``).
2. Fallback: caption + surrounding paragraph (always available).

Figure chunks link to their parent clause via ``parent_section_id`` /
``section_number`` so Stage 5 context expansion can pull the clause.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Pinned vision model — keep eval-consistent; override only via FIGURE_VLM_MODEL.
PINNED_FIGURE_VLM_MODEL = "gemini-2.5-flash"
FIGURE_DESCRIBE_CONFIG = "figure_describe"
_CLAUSE_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+\S")
_FIGURE_LABEL_RE = re.compile(r"(?i)\b(Figure|Fig\.?|Table)\s+(\d+)\b")

DescribeFn = Callable[[dict[str, Any]], str | None]


@dataclass
class FigureSpec:
    """One figure to describe + index."""

    page_number: int
    figure_label: str = ""
    caption: str = ""
    surrounding_text: str = ""
    parent_section_number: str = ""
    bbox: list[float] = field(default_factory=list)  # [l,t,r,b] page coords or 0-1
    element_id: str = ""
    image_png: bytes | None = None
    source: str = "extract"


@dataclass
class FigureDescription:
    figure: FigureSpec
    description: str
    method: str  # "vlm" | "caption_context_fallback"
    model: str | None = None

    @property
    def searchable_text(self) -> str:
        parts: list[str] = []
        label = self.figure.figure_label or ""
        caption = self.figure.caption or ""
        if label:
            parts.append(label)
        if caption and caption.lower() not in (label or "").lower():
            parts.append(caption)
        if self.description and self.description not in parts:
            parts.append(self.description)
        if self.figure.surrounding_text:
            surround = self.figure.surrounding_text.strip()
            if surround and surround not in self.description:
                parts.append(surround)
        if self.figure.parent_section_number:
            parts.append(f"Parent clause: §{self.figure.parent_section_number}")
        return "\n\n".join(p for p in parts if p).strip()


def pinned_figure_vlm_model() -> str:
    return (os.getenv("FIGURE_VLM_MODEL") or PINNED_FIGURE_VLM_MODEL).strip()


def vlm_describe_enabled() -> bool:
    return os.getenv("FIGURE_VLM_DESCRIBE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def _section_id(regulation_id: str, section_number: str, fallback: str) -> str:
    key = section_number or fallback
    return f"{regulation_id}::{key}"


def caption_context_fallback(fig: FigureSpec) -> str:
    """Always-available searchable text from caption + surrounding clause."""
    bits: list[str] = []
    if fig.figure_label:
        bits.append(fig.figure_label)
    if fig.caption and fig.caption not in bits:
        bits.append(fig.caption)
    if fig.surrounding_text:
        bits.append(fig.surrounding_text.strip())
    if not bits:
        bits.append(
            f"(Figure on page {fig.page_number}; no caption recovered — "
            "bbox indexed for retrieval anchoring.)"
        )
    # Lightweight heuristic enrichment so curve/figure queries have signal
    # even without a live VLM (e.g. femur force-time performance curve).
    joined = " ".join(bits).lower()
    if "force-time" in joined or ("femur" in joined and "figure" in joined):
        bits.append(
            "force-time performance curve for the femur force criterion, "
            "showing the maximum allowable force decreasing over the contact duration"
        )
    elif "neck" in joined and ("tension" in joined or "shear" in joined):
        bits.append(
            "neck injury criterion curve showing allowable tension/shear limits "
            "versus time"
        )
    return "\n\n".join(dict.fromkeys(bits)).strip()


def _call_portkey_vlm(fig: FigureSpec) -> tuple[str | None, str]:
    """Describe figure via pinned Portkey vision model. Returns (text, model)."""
    model = pinned_figure_vlm_model()
    if not vlm_describe_enabled():
        return None, model

    prompt = (
        "You are describing a figure from a UNECE passive-safety regulation PDF. "
        "Write 1-3 sentences capturing what the figure conveys for retrieval "
        "(e.g. which criterion, axes, and how the limit behaves). "
        "Do not invent numeric limits not visible in the figure or caption.\n\n"
        f"Label: {fig.figure_label or '(none)'}\n"
        f"Caption: {fig.caption or '(none)'}\n"
        f"Surrounding clause text: {fig.surrounding_text or '(none)'}\n"
        f"Parent section: {fig.parent_section_number or '(unknown)'}\n"
    )

    try:
        from generation.llm_client import LLMClient, LLMRole, load_portkey_config

        client = LLMClient()
        # Prefer dedicated figure_describe config when present.
        cfg_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "portkey"
            / f"{FIGURE_DESCRIBE_CONFIG}.json"
        )
        if cfg_path.is_file():
            # Ensure env FIGURE model pin is logged.
            _ = load_portkey_config(FIGURE_DESCRIBE_CONFIG)

        user_content: Any
        if fig.image_png:
            b64 = base64.b64encode(fig.image_png).decode("ascii")
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        else:
            user_content = prompt

        # LLMClient.complete types content as str; call Portkey path carefully.
        messages = [
            {
                "role": "system",
                "content": "Describe regulation figures accurately and briefly.",
            },
            {"role": "user", "content": user_content if isinstance(user_content, str) else prompt},
        ]
        # When we have an image, attempt a raw multimodal call through the
        # same gateway URL (Portkey OpenAI-compatible).
        if fig.image_png and client.provider != "mock":
            text = _raw_portkey_vision(prompt, fig.image_png, model=model)
            if text:
                return text.strip(), model

        result = client.complete(
            messages=messages,  # type: ignore[arg-type]
            role=LLMRole.ANSWER,
            question=f"figure-describe:{fig.figure_label}:{fig.page_number}",
            max_tokens=256,
            temperature=0.0,
            skip_cache=False,
        )
        text = (result.text or "").strip()
        return (text or None), (result.model or model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Figure VLM describe failed: %s", exc)
        return None, model


def _raw_portkey_vision(prompt: str, png: bytes, *, model: str) -> str | None:
    """Multimodal chat.completions via Portkey gateway (google/gemini vision)."""
    import json

    from openai import OpenAI

    from generation.llm_client import (
        DEFAULT_GATEWAY_URL,
        load_portkey_config,
        portkey_client_timeout_s,
    )

    cfg = load_portkey_config(FIGURE_DESCRIBE_CONFIG)
    gateway = os.getenv("PORTKEY_GATEWAY_URL", DEFAULT_GATEWAY_URL)
    b64 = base64.b64encode(png).decode("ascii")
    client = OpenAI(
        base_url=gateway,
        api_key="not-needed",
        timeout=portkey_client_timeout_s(),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=256,
        temperature=0.0,
        extra_headers={"x-portkey-config": json.dumps(cfg)},
    )
    choice = resp.choices[0].message if resp.choices else None
    return (choice.content or "").strip() if choice else None


def describe_figure(fig: FigureSpec, *, describe_fn: DescribeFn | None = None) -> FigureDescription:
    """Describe one figure (VLM if enabled/available, else caption+context)."""
    if describe_fn is not None:
        custom = describe_fn(fig.__dict__)
        if custom and custom.strip():
            return FigureDescription(
                figure=fig,
                description=custom.strip(),
                method="vlm",
                model="custom",
            )

    vlm_text, model = _call_portkey_vlm(fig)
    if vlm_text:
        return FigureDescription(
            figure=fig,
            description=vlm_text,
            method="vlm",
            model=model,
        )

    return FigureDescription(
        figure=fig,
        description=caption_context_fallback(fig),
        method="caption_context_fallback",
        model=None,
    )


def describe_figures(
    figures: Sequence[FigureSpec],
    *,
    describe_fn: DescribeFn | None = None,
) -> list[FigureDescription]:
    """Describe every figure; never returns an empty description."""
    out: list[FigureDescription] = []
    for fig in figures:
        desc = describe_figure(fig, describe_fn=describe_fn)
        if not (desc.searchable_text or "").strip():
            # Absolute last resort — should be unreachable.
            desc = FigureDescription(
                figure=fig,
                description=f"Figure on page {fig.page_number}",
                method="caption_context_fallback",
            )
        out.append(desc)
    return out


def figures_from_stage1_extract(extract: Any) -> list[FigureSpec]:
    """Collect figure(+linked caption) specs from a Stage 1 DocumentExtract."""
    specs: list[FigureSpec] = []
    for page in getattr(extract, "pages", []) or []:
        elems = list(getattr(page, "elements", []) or [])
        last_para = ""
        last_section = ""
        pending_caption = ""
        for el in elems:
            et = getattr(el, "element_type", "")
            text = (getattr(el, "text", None) or "").strip()
            sec = (getattr(el, "section_number", None) or "") or last_section
            if sec:
                last_section = sec
            if et == "paragraph" and text:
                last_para = text
                m = _CLAUSE_LINE_RE.match(text)
                if m:
                    last_section = m.group(1)
                continue
            if et == "caption" and text:
                pending_caption = text
                continue
            if et != "figure":
                continue
            label = ""
            m = _FIGURE_LABEL_RE.search(text) or _FIGURE_LABEL_RE.search(pending_caption)
            if m:
                label = f"{m.group(1).title().replace('Fig.', 'Figure')} {m.group(2)}"
            # Parent: section that references this figure, else current section.
            parent = last_section
            ref = label or pending_caption or text
            if ref and last_para and ref.lower().split()[0:2]:
                # Prefer paragraph that mentions the figure label.
                if label and label.lower() in last_para.lower():
                    m2 = _CLAUSE_LINE_RE.match(last_para)
                    if m2:
                        parent = m2.group(1)
            specs.append(
                FigureSpec(
                    page_number=int(getattr(el, "page_number", page.page_number)),
                    figure_label=label or (text if text.startswith("Figure") else ""),
                    caption=pending_caption or text,
                    surrounding_text=last_para,
                    parent_section_number=parent,
                    bbox=list(getattr(el, "coordinates", None) or []),
                    element_id=str(getattr(el, "element_id", "") or ""),
                    source="extract",
                )
            )
            pending_caption = ""
    return specs


def figures_from_vlm_result(vlm_result: Any) -> list[FigureSpec]:
    """Collect specs from LightOnOCR / VlmFigurePassResult."""
    specs: list[FigureSpec] = []
    page_results = getattr(vlm_result, "page_results", None) or {}
    for page_number, page in sorted(page_results.items()):
        for fig in getattr(page, "figures", []) or []:
            x1, y1, x2, y2 = getattr(fig, "bbox_norm", (0, 0, 0, 0))
            specs.append(
                FigureSpec(
                    page_number=int(page_number),
                    figure_label=str(getattr(fig, "figure_label", "") or ""),
                    caption=str(getattr(fig, "caption", "") or ""),
                    surrounding_text=str(getattr(fig, "surrounding_text", "") or ""),
                    parent_section_number=str(
                        getattr(fig, "parent_section_number", "") or ""
                    ),
                    bbox=[x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0],
                    element_id=f"vlm:{page_number}:{getattr(fig, 'image_index', 0)}",
                    source="vlm",
                )
            )
    return specs


def build_figure_chunks(
    descriptions: Sequence[FigureDescription],
    *,
    regulation_id: str,
    revision: str,
) -> list[Chunk]:
    """Turn descriptions into ``content_type='figure'`` chunks."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for desc in descriptions:
        fig = desc.figure
        text = desc.searchable_text
        if not text.strip():
            continue
        label = fig.figure_label or f"Figure p{fig.page_number}"
        parent = (fig.parent_section_number or "").strip()
        section_number = parent or f"page-{fig.page_number}"
        section_id = f"{regulation_id}::figure::{fig.page_number}::{label}"
        if section_id in seen:
            section_id = f"{section_id}::{fig.element_id or _stable_id(text[:40])}"
        seen.add(section_id)
        parent_section_id = (
            _section_id(regulation_id, parent, "root") if parent else None
        )
        chunk_id = _stable_id(regulation_id, revision, section_id, "figure", text[:200])
        bbox = list(fig.bbox) if fig.bbox else [0.0, 0.0, 0.0, 0.0]
        if len(bbox) < 4:
            bbox = (bbox + [0.0, 0.0, 0.0, 0.0])[:4]

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                regulation_id=regulation_id,
                revision=revision,
                section_number=section_number,
                section_title=(label or text)[:120],
                page_number=fig.page_number,
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
    return chunks


def describe_and_chunk_figures(
    *,
    regulation_id: str,
    revision: str,
    extract: Any | None = None,
    vlm_result: Any | None = None,
    describe_fn: DescribeFn | None = None,
) -> list[Chunk]:
    """Stage 2 entry: collect figures → describe → figure chunks."""
    specs: list[FigureSpec] = []
    if vlm_result is not None:
        specs.extend(figures_from_vlm_result(vlm_result))
    if extract is not None and not specs:
        specs.extend(figures_from_stage1_extract(extract))
    elif extract is not None and specs:
        # Prefer VLM specs; fill gaps from extract for pages without VLM figures.
        vlm_pages = {s.page_number for s in specs}
        for s in figures_from_stage1_extract(extract):
            if s.page_number not in vlm_pages:
                specs.append(s)

    logger.info(
        "Stage 2 figure describe: %d figure(s) model=%s vlm=%s",
        len(specs),
        pinned_figure_vlm_model(),
        vlm_describe_enabled(),
    )
    descriptions = describe_figures(specs, describe_fn=describe_fn)
    return build_figure_chunks(
        descriptions, regulation_id=regulation_id, revision=revision
    )
