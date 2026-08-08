"""Structure-aware chunking over DoclingDocument (clause / table atomic units)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from ingestion.models import Chunk, ContentType

logger = logging.getLogger(__name__)

# UNECE-style clause numbers: "5.2.1", "2.1.", "Annex 3", "Appendix 1"
_SECTION_NUM_RE = re.compile(
    r"^(?:"
    r"(?P<num>\d+(?:\.\d+)*)\.?"  # 5.2.1 / 5.2.1.
    r"|(?P<annex>(?:Annex|Appendix|Schedule)\s+[A-Za-z0-9]+)"
    r")"
    r"(?:\s*[–—\-:.]?\s*|\s+)?"
    r"(?P<title>.*)$",
    re.IGNORECASE,
)

_ANNEX_ONLY_RE = re.compile(
    r"^(?P<annex>(?:Annex|Appendix|Schedule)\s+[A-Za-z0-9]+)\s*$",
    re.IGNORECASE,
)

_CLAUSE_LINE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.\s+\S")
# Leading clause id for body / list items (title = remainder of first line).
_CLAUSE_START_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>\S.*)$",
    re.DOTALL,
)
# Mid-document clause boundaries (Docling sometimes emits many clauses in one TextItem).
_INLINE_CLAUSE_START_RE = re.compile(
    r"(?m)^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>\S[^\n]*)"
)


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def _bbox_list(item: Any) -> list[float]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return []
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return []
    return [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]


def _page_number(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page = getattr(prov[0], "page_no", None)
    return int(page) if page is not None else None


def _parse_heading(text: str) -> tuple[str, str]:
    """Split heading into (section_number, section_title)."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "", ""
    m = _SECTION_NUM_RE.match(cleaned)
    if not m:
        return "", cleaned
    num = (m.group("num") or m.group("annex") or "").strip().rstrip(".")
    title = (m.group("title") or "").strip(" .–—-:")
    return num, title or ("" if _ANNEX_ONLY_RE.match(cleaned) else cleaned)


def _is_annex_number(section_number: str) -> bool:
    s = (section_number or "").strip().lower().split("/", 1)[0]
    return s.startswith(("annex ", "appendix ", "schedule "))


def _is_numbered_section(section_number: str) -> bool:
    """True for real clause numbers or Annex/Appendix ids — not Preamble."""
    s = (section_number or "").strip()
    if not s or s.lower() == "preamble":
        return False
    if _is_annex_number(s):
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*$", s))


def _parent_section_number(section_number: str) -> str | None:
    if not section_number:
        return None
    if "/" in section_number:
        left, right = section_number.split("/", 1)
        if _is_annex_number(left):
            if "." in right:
                return f"{left}/{right.rsplit('.', 1)[0]}"
            return left
    if _is_annex_number(section_number):
        return None
    if "." in section_number:
        return section_number.rsplit(".", 1)[0]
    return None


def _section_id(regulation_id: str, section_number: str, fallback: str) -> str:
    key = section_number or fallback
    return f"{regulation_id}::{key}"


def _item_text(item: Any) -> str:
    return (getattr(item, "text", None) or "").strip()


def _nearest_numbered_from_stack(
    heading_stack: dict[int, tuple[str, str]],
    *,
    below_level: int | None = None,
) -> tuple[str, str] | None:
    levels = sorted(heading_stack, reverse=True)
    for lv in levels:
        if below_level is not None and lv >= below_level:
            continue
        num, title = heading_stack[lv]
        if _is_numbered_section(num):
            return num, title
    return None


def _resolve_section_context(
    *,
    open_sec: _OpenSection | None,
    heading_stack: dict[int, tuple[str, str]],
    last_numbered: tuple[str, str] | None,
) -> tuple[str, str, list[str]]:
    """Pick the nearest real section number for tables / unnumbered leaves."""
    if open_sec and _is_numbered_section(open_sec.section_number):
        return (
            open_sec.section_number,
            open_sec.section_title,
            list(open_sec.heading_path),
        )
    stacked = _nearest_numbered_from_stack(heading_stack)
    if stacked:
        num, title = stacked
        path = [" ".join(x for x in heading_stack[lv] if x).strip() for lv in sorted(heading_stack)]
        return num, title, path
    if last_numbered and _is_numbered_section(last_numbered[0]):
        return (
            last_numbered[0],
            last_numbered[1],
            [f"{last_numbered[0]} {last_numbered[1]}".strip()],
        )
    if open_sec:
        return (
            open_sec.section_number,
            open_sec.section_title,
            list(open_sec.heading_path),
        )
    return "", "", []


def _table_markdown(table: TableItem, doc: DoclingDocument) -> str:
    """Export table as markdown; keep caption attached."""
    caption_parts: list[str] = []
    captions = getattr(table, "captions", None) or []
    for cap_ref in captions:
        cap_item = cap_ref
        if hasattr(cap_ref, "resolve"):
            try:
                cap_item = cap_ref.resolve(doc)
            except Exception:  # noqa: BLE001
                cap_item = cap_ref
        cap_text = _item_text(cap_item)
        if cap_text:
            caption_parts.append(cap_text)

    if not caption_parts:
        for attr in ("caption_text", "caption"):
            val = getattr(table, attr, None)
            if callable(val):
                try:
                    val = val(doc=doc) if attr == "caption_text" else val()
                except TypeError:
                    val = val()
            if isinstance(val, str) and val.strip():
                caption_parts.append(val.strip())

    try:
        md = table.export_to_markdown(doc=doc)
    except TypeError:
        md = table.export_to_markdown()
    except Exception:  # noqa: BLE001
        try:
            df = table.export_to_dataframe(doc=doc)
            md = df.to_markdown(index=False)
        except Exception:  # noqa: BLE001
            md = ""

    md = (md or "").strip()
    caption = " ".join(caption_parts).strip()
    if caption and md:
        return f"{caption}\n\n{md}"
    return caption or md


def _is_heading(item: Any) -> bool:
    if isinstance(item, (TitleItem, SectionHeaderItem)):
        return True
    label = getattr(item, "label", None)
    return label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE)


def _heading_level(item: Any, walk_level: int) -> int:
    if isinstance(item, TitleItem):
        return 0
    if isinstance(item, SectionHeaderItem):
        return int(getattr(item, "level", walk_level) or walk_level)
    return walk_level


@dataclass
class _OpenSection:
    """Accumulates body text under the current clause heading."""

    section_number: str
    section_title: str
    section_id: str
    parent_section_id: str | None
    heading_path: list[str]
    level: int
    page_number: int | None
    bounding_box: list[float]
    body_parts: list[str] = field(default_factory=list)
    body_pages: list[int] = field(default_factory=list)
    body_bboxes: list[list[float]] = field(default_factory=list)


def _flush_section(
    open_sec: _OpenSection | None,
    *,
    regulation_id: str,
    revision: str,
    chunks: list[Chunk],
) -> None:
    if open_sec is None:
        return
    body = "\n\n".join(p for p in open_sec.body_parts if p).strip()
    if not body:
        return

    page = open_sec.page_number
    if page is None and open_sec.body_pages:
        page = open_sec.body_pages[0]

    bbox = open_sec.bounding_box
    if not bbox and open_sec.body_bboxes:
        bbox = open_sec.body_bboxes[0]

    heading_line = " ".join(
        x for x in (open_sec.section_number, open_sec.section_title) if x
    ).strip()
    text = f"{heading_line}\n\n{body}" if heading_line else body

    chunk_id = _stable_id(regulation_id, revision, open_sec.section_id, "clause", text[:200])
    chunks.append(
        Chunk(
            chunk_id=chunk_id,
            text=text,
            regulation_id=regulation_id,
            revision=revision,
            section_number=open_sec.section_number,
            section_title=open_sec.section_title,
            page_number=page,
            bounding_box=bbox,
            content_type="clause",
            parent_section_id=open_sec.parent_section_id,
            section_id=open_sec.section_id,
            heading_path=list(open_sec.heading_path),
        )
    )


def _emit_table_chunk(
    table: TableItem,
    doc: DoclingDocument,
    *,
    regulation_id: str,
    revision: str,
    open_sec: _OpenSection | None,
    heading_stack: dict[int, tuple[str, str]],
    last_numbered: tuple[str, str] | None,
    chunks: list[Chunk],
) -> None:
    md = _table_markdown(table, doc)
    if not md:
        return

    section_number, section_title, heading_path = _resolve_section_context(
        open_sec=open_sec,
        heading_stack=heading_stack,
        last_numbered=last_numbered,
    )
    # Prefer open_sec.section_id when it already matches the resolved number
    if open_sec and open_sec.section_number == section_number:
        parent_section_id = open_sec.section_id
        heading_path = list(open_sec.heading_path) or heading_path
        section_title = open_sec.section_title or section_title
    else:
        parent_section_id = (
            _section_id(regulation_id, section_number, "root") if section_number else None
        )

    # Keep caption visible in text; do not replace inherited clause section_number.
    caption_note = ""
    captions = getattr(table, "captions", None) or []
    if not captions:
        # First line of md is often caption when we prefixed it
        first = md.split("\n", 1)[0].strip()
        if first and not first.startswith("|"):
            caption_note = first

    page = _page_number(table)
    bbox = _bbox_list(table)
    table_key = getattr(table, "self_ref", None) or md[:80]
    section_id = f"{regulation_id}::table::{_stable_id(str(table_key))}"
    chunk_id = _stable_id(regulation_id, revision, section_id, "table")

    chunks.append(
        Chunk(
            chunk_id=chunk_id,
            text=md,
            regulation_id=regulation_id,
            revision=revision,
            section_number=section_number,
            section_title=section_title or caption_note,
            page_number=page,
            bounding_box=bbox,
            content_type="table",
            parent_section_id=parent_section_id,
            section_id=section_id,
            heading_path=heading_path,
        )
    )


def _maybe_track_clause_line(text: str, last_numbered: tuple[str, str] | None) -> tuple[str, str] | None:
    m = _CLAUSE_LINE_RE.match(text.strip())
    if not m:
        return last_numbered
    return m.group("num"), last_numbered[1] if last_numbered else ""


def _clause_start(text: str) -> tuple[str, str] | None:
    """If ``text`` begins a numbered UNECE clause, return (section_number, title)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    m = _CLAUSE_START_RE.match(cleaned)
    if not m:
        return None
    num = m.group("num").strip()
    title = " ".join((m.group("title") or "").split())
    # Keep title short for section_title metadata (full text stays in body).
    if len(title) > 80:
        title = title[:77].rstrip() + "…"
    return num, title


def iter_inline_clause_segments(text: str) -> list[tuple[str, str, str]]:
    """Split a multi-clause text block into ``(section_number, title, segment)``.

    Docling occasionally emits one giant TextItem that starts at e.g. ``6.2.2``
    and continues through ``6.2.5.3`` Emergency locking retractors. Without a
    split, the ELR requirements are buried under the wrong ``section_number``
    and truncated by context budgets — retrieval then falls through to other
    regs' boilerplate.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    matches = list(_INLINE_CLAUSE_START_RE.finditer(raw))
    if len(matches) <= 1:
        return []
    nums = [m.group("num") for m in matches]
    # Split when sibling clauses appear (6.2.2 vs 6.2.5), many clause starts,
    # or the block is oversized (mega TextItem burying ELR under Buckle).
    depth3 = {".".join(n.split(".")[:3]) for n in nums}
    oversized = len(raw.split()) >= 400
    if len(depth3) < 2 and len(matches) < 3 and not oversized:
        return []
    out: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        seg = raw[start:end].strip()
        if not seg:
            continue
        title = " ".join((m.group("title") or "").split())
        if len(title) > 80:
            title = title[:77].rstrip() + "…"
        out.append((m.group("num").strip(), title, seg))
    return out if len(out) > 1 else []

def _apply_annex_prefix(
    section_number: str,
    heading_stack: dict[int, tuple[str, str]],
) -> str:
    if not section_number or not re.match(r"^\d+(?:\.\d+)*$", section_number):
        return section_number
    annex_ancestor = None
    for lv in sorted(heading_stack):
        cand = (heading_stack[lv][0] or "").split("/", 1)[0]
        if _is_annex_number(cand):
            annex_ancestor = cand
    if annex_ancestor:
        return f"{annex_ancestor}/{section_number}"
    return section_number


def _new_open_section(
    *,
    regulation_id: str,
    section_number: str,
    section_title: str,
    level: int,
    heading_stack: dict[int, tuple[str, str]],
    item: Any,
) -> _OpenSection:
    parent_num = _parent_section_number(section_number)
    if parent_num is None and heading_stack:
        shallower = [k for k in heading_stack if k < level]
        if shallower:
            cand = heading_stack[max(shallower)][0] or None
            if cand and _is_numbered_section(cand):
                parent_num = cand
    parent_section_id = (
        _section_id(regulation_id, parent_num, "root") if parent_num else None
    )
    sid = _section_id(
        regulation_id,
        section_number,
        fallback=_stable_id(section_number or section_title or "sec"),
    )
    path_stack = dict(heading_stack)
    path_stack[level] = (section_number, section_title)
    heading_path = [
        " ".join(x for x in path_stack[lv] if x).strip() for lv in sorted(path_stack)
    ]
    return _OpenSection(
        section_number=section_number,
        section_title=section_title,
        section_id=sid,
        parent_section_id=parent_section_id,
        heading_path=heading_path,
        level=level,
        page_number=_page_number(item),
        bounding_box=_bbox_list(item),
    )


def chunk_document(
    doc: DoclingDocument,
    *,
    regulation_id: str,
    revision: str,
) -> list[Chunk]:
    """Structure-aware chunking — one chunk per clause/sub-clause; tables atomic.

    Uses Docling's heading hierarchy from ``iterate_items`` (not a fixed-size
    splitter). Each child chunk stores ``parent_section_id`` linking to its
    enclosing section.

    Numbered clause lines in body/list text (``5.2.6. …``) open a new section even
    when Docling did not emit a ``SectionHeaderItem``. Unnumbered headings (e.g.
    figure titles) never inherit a prior clause number into a new open section —
    figures/tables/captions do not reset heading state.
    """
    chunks: list[Chunk] = []
    heading_stack: dict[int, tuple[str, str]] = {}
    open_sec: _OpenSection | None = None
    last_numbered: tuple[str, str] | None = None
    seen_numbered_heading = False

    for item, walk_level in doc.iterate_items():
        if _is_heading(item):
            raw = _item_text(item)
            level = _heading_level(item, walk_level)
            section_number, section_title = _parse_heading(raw)

            # Docling often emits Annex as two L1 headers: "Annex 11" then subtitle.
            if (
                not section_number
                and open_sec is not None
                and _is_annex_number(open_sec.section_number)
                and not open_sec.body_parts
                and not (open_sec.section_title or "").strip()
            ):
                open_sec.section_title = section_title or raw
                heading_stack[open_sec.level] = (open_sec.section_number, open_sec.section_title)
                open_sec.heading_path = [
                    " ".join(x for x in heading_stack[lv] if x).strip() for lv in sorted(heading_stack)
                ]
                open_sec.section_id = _section_id(
                    regulation_id, open_sec.section_number, fallback=_stable_id(raw)
                )
                last_numbered = (open_sec.section_number, open_sec.section_title)
                continue

            # Unnumbered heading (figure title, caption-like header): never invent a
            # clause id by inheritance — that pulls following clauses into the wrong
            # section (e.g. "Femur force criterion" → 5.2.1.6 swallowing 5.2.6).
            if not section_number:
                if open_sec is not None:
                    title_line = (section_title or raw).strip()
                    if title_line:
                        if not (open_sec.section_title or "").strip():
                            open_sec.section_title = title_line[:80]
                        open_sec.body_parts.append(title_line)
                        page = _page_number(item)
                        if page is not None:
                            open_sec.body_pages.append(page)
                        bbox = _bbox_list(item)
                        if bbox:
                            open_sec.body_bboxes.append(bbox)
                    continue
                # No open section yet — preamble leaf without a number.
                section_number = "Preamble" if not seen_numbered_heading else ""
                section_title = (section_title or raw)[:60]
                if not section_number:
                    continue

            # Close previous clause before switching numbered headings.
            _flush_section(open_sec, regulation_id=regulation_id, revision=revision, chunks=chunks)
            open_sec = None

            for k in [k for k in list(heading_stack) if k >= level]:
                heading_stack.pop(k, None)

            if not section_title:
                section_title = raw

            section_number = _apply_annex_prefix(section_number, heading_stack)

            if _is_numbered_section(section_number):
                seen_numbered_heading = True
                last_numbered = (section_number, section_title)

            heading_stack[level] = (section_number, section_title)
            open_sec = _new_open_section(
                regulation_id=regulation_id,
                section_number=section_number,
                section_title=section_title,
                level=level,
                heading_stack=heading_stack,
                item=item,
            )
            continue

        if isinstance(item, TableItem) or getattr(item, "label", None) == DocItemLabel.TABLE:
            # Tables inherit current heading; they must not flush/reset it.
            _emit_table_chunk(
                item,  # type: ignore[arg-type]
                doc,
                regulation_id=regulation_id,
                revision=revision,
                open_sec=open_sec,
                heading_stack=heading_stack,
                last_numbered=last_numbered,
                chunks=chunks,
            )
            continue

        label = getattr(item, "label", None)
        if label in (
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.PICTURE,
            DocItemLabel.CAPTION,
        ):
            # Figures / captions never change current heading state.
            continue

        text = _item_text(item)
        if not text:
            continue

        last_numbered = _maybe_track_clause_line(text, last_numbered) or last_numbered

        clause = _clause_start(text)
        if clause:
            segments = iter_inline_clause_segments(text)
            # One Docling item spanning 6.2.2…6.2.5.x → emit one chunk per clause.
            to_emit: list[tuple[str, str, str]] = (
                segments if len(segments) > 1 else [(clause[0], clause[1], text)]
            )
            for num_raw, title, seg in to_emit:
                num = _apply_annex_prefix(num_raw, heading_stack)
                if open_sec is None or open_sec.section_number != num:
                    _flush_section(
                        open_sec,
                        regulation_id=regulation_id,
                        revision=revision,
                        chunks=chunks,
                    )
                    seen_numbered_heading = True
                    last_numbered = (num, title)
                    body_level = max(heading_stack) + 1 if heading_stack else walk_level
                    heading_stack[body_level] = (num, title)
                    open_sec = _new_open_section(
                        regulation_id=regulation_id,
                        section_number=num,
                        section_title=title,
                        level=body_level,
                        heading_stack=heading_stack,
                        item=item,
                    )
                open_sec.body_parts.append(seg)
                page = _page_number(item)
                if page is not None:
                    open_sec.body_pages.append(page)
                bbox = _bbox_list(item)
                if bbox:
                    open_sec.body_bboxes.append(bbox)
            continue

        if open_sec is None:
            # Preamble before first numbered clause heading.
            title = " ".join(text.split())[:60]
            open_sec = _OpenSection(
                section_number="Preamble",
                section_title=title,
                section_id=_section_id(regulation_id, "Preamble", "preamble"),
                parent_section_id=None,
                heading_path=["Preamble"],
                level=0,
                page_number=_page_number(item),
                bounding_box=_bbox_list(item),
            )

        if isinstance(item, (TextItem, ListItem)) or label in (
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.LIST_ITEM,
            DocItemLabel.FORMULA,
            DocItemLabel.CODE,
        ):
            open_sec.body_parts.append(text)
            page = _page_number(item)
            if page is not None:
                open_sec.body_pages.append(page)
            bbox = _bbox_list(item)
            if bbox:
                open_sec.body_bboxes.append(bbox)

    _flush_section(open_sec, regulation_id=regulation_id, revision=revision, chunks=chunks)
    chunks = finalize_chunks(chunks, regulation_id=regulation_id)
    logger.info(
        "Chunked %s %s → %d chunks (%d tables)",
        regulation_id,
        revision,
        len(chunks),
        sum(1 for c in chunks if c.content_type == "table"),
    )
    return chunks


def finalize_chunks(
    chunks: list[Chunk],
    *,
    regulation_id: str,
) -> list[Chunk]:
    """Enforce metadata contract: non-empty section_number + contract ids.

    The historical 24.8%-null ``section_number`` regression is guarded here —
    any chunk that would ship without a section id is assigned a deterministic
    page-scoped fallback rather than null/empty.

    Also drops exact ``chunk_id`` duplicates (same content hashed twice) so
    Qdrant UUIDv5 upserts cannot silently collapse the reported count.
    """
    out: list[Chunk] = []
    seen_ids: set[str] = set()
    for i, ch in enumerate(chunks):
        section_number = (ch.section_number or "").strip()
        if not section_number:
            page = ch.page_number if ch.page_number is not None else 0
            section_number = f"page-{page}-chunk-{i + 1}"
        document_id = ch.document_id or regulation_id or ch.regulation_id
        element_id = ch.element_id or ch.section_id or ch.chunk_id
        # Disambiguate rare hash collisions on distinct texts.
        chunk_id = ch.chunk_id
        if chunk_id in seen_ids:
            chunk_id = f"{chunk_id}-{i + 1}"
        seen_ids.add(chunk_id)
        out.append(
            ch.model_copy(
                update={
                    "chunk_id": chunk_id,
                    "section_number": section_number,
                    "document_id": document_id,
                    "element_id": element_id,
                }
            )
        )
    return out


def parent_id_index(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    """Map ``section_id`` / ``element_id`` → chunk for parent resolution checks."""
    index: dict[str, Chunk] = {}
    for ch in chunks:
        if ch.section_id:
            index[ch.section_id] = ch
        if ch.element_id:
            index[ch.element_id] = ch
        # Also index by regulation::section_number for parent_section_id lookup.
        if ch.section_number:
            index[_section_id(ch.regulation_id, ch.section_number, ch.chunk_id)] = ch
    return index


def assert_parent_links_resolve(chunks: Sequence[Chunk]) -> None:
    """Raise if any child ``parent_section_id`` does not resolve to a known id.

    Parents may be structural (section_id of a clause) even when that exact
    parent chunk was merged into children — we accept resolution against any
    chunk's section_id, element_id, or a sibling sharing the same parent key
    as a prefix of a known section_number.
    """
    index = parent_id_index(list(chunks))
    known_section_numbers = {
        (c.section_number or "").strip() for c in chunks if (c.section_number or "").strip()
    }
    for ch in chunks:
        parent = (ch.parent_section_id or "").strip()
        if not parent:
            continue
        if parent in index:
            continue
        # parent_section_id is often ``UN-ECE-R94::5.2`` while children are ``5.2.1``
        suffix = parent.split("::", 1)[-1]
        if suffix in known_section_numbers:
            continue
        if any(
            sn == suffix or sn.startswith(suffix + ".")
            for sn in known_section_numbers
            if sn
        ):
            continue
        raise AssertionError(
            f"parent_id {parent!r} on chunk {ch.chunk_id} does not resolve"
        )
