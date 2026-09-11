"""Structural chunking — CLAUDE.md §7.

Units follow the section tree, never a fixed window:

- one chunk per clause/definition; consecutive *tiny* sibling clauses (same
  parent) are merged so a citation still names an exact clause range;
- oversized clauses are split on paragraph/sentence boundaries, each part
  keeping the same citation label plus a part index;
- every table becomes its own TABLE chunk carrying caption, headers and rows
  (headers repeated in every row-group split) — a cell is never indexed alone;
- each chunk starts with a one-line context header ("UN R94 Rev.4 › 5
  Specifications › 5.2.1.8") so both lexical and dense retrieval see the
  hierarchy (contextual chunking, cheap and deterministic).

Token counts are a word-based estimate (words × 1.3); the exact tokenizer
differs per embedding model and this is documented as an estimate.

Chunk IDs are deterministic: uuid5(version_id, ordinal, chunk_sha256).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field

from safety_assistant.ingestion.normalize.structure import NormalizedDocument, NormSection
from safety_assistant.ingestion.parse.contract import ParsedTable

CHUNKER_VERSION = "2.0.1"
TARGET_MIN_TOKENS = 120
TARGET_MAX_TOKENS = 500
TABLE_ROWS_PER_CHUNK = 25
_TOKENS_PER_WORD = 1.3

_CHUNK_NAMESPACE = uuid.UUID("6f1c9c2e-3d2b-4a0e-9d8e-3a3f7f8a1c11")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKENS_PER_WORD) + 1


def chunker_config_hash() -> str:
    cfg = {
        "chunker": CHUNKER_VERSION,
        "min": TARGET_MIN_TOKENS,
        "max": TARGET_MAX_TOKENS,
        "table_rows": TABLE_ROWS_PER_CHUNK,
        "tpw": _TOKENS_PER_WORD,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ChunkDraft:
    ordinal: int
    section_path: str
    chunk_type: str  # TEXT | TABLE | DEFINITION
    content: str
    token_count: int
    page_start: int | None
    page_end: int | None
    citation_label: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def chunk_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def deterministic_id(self, version_id: uuid.UUID) -> uuid.UUID:
        return uuid.uuid5(_CHUNK_NAMESPACE, f"{version_id}:{self.ordinal}:{self.chunk_sha256}")


@dataclass(frozen=True)
class CitationContext:
    """What a citation label needs: 'UN R94 Rev.4'."""

    regulation_key: str
    version_label: str

    @property
    def prefix(self) -> str:
        return f"{self.regulation_key.replace('-', ' ')} {self.version_label.split(' ')[0]}"


def _breadcrumb(section: NormSection, by_path: dict[str, NormSection]) -> list[str]:
    crumbs: list[str] = []
    node: NormSection | None = section
    while node is not None:
        label = node.section_number.rstrip(".") if node.section_number else (node.title or node.path)
        if node.title and node.section_number:
            label = f"{label} {node.title}"
        crumbs.append(label)
        node = by_path.get(node.parent_path) if node.parent_path else None
    if section.annex and (not crumbs or not crumbs[-1].lower().startswith("annex")):
        crumbs.append(section.annex)
    return list(reversed(crumbs))


def _label(
    ctx: CitationContext, section: NormSection, page_start: int | None, page_end: int | None, extra: str = ""
) -> str:
    where = ""
    if section.section_number:
        where = f"§{section.section_number.rstrip('.')}"
        if section.annex:
            where = f"{section.annex} {where}"
    elif section.annex:
        where = section.annex
    elif section.title:
        where = section.title
    pages = ""
    if page_start:
        pages = f" (p. {page_start})" if not page_end or page_end == page_start else f" (pp. {page_start}–{page_end})"
    return f"{ctx.prefix} {where}{extra}{pages}".strip()


def _split_long(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    buf: list[str] = []
    for unit in _SENTENCE_SPLIT_RE.split(text):
        if buf and estimate_tokens(" ".join(buf) + " " + unit) > max_tokens:
            parts.append(" ".join(buf))
            buf = []
        buf.append(unit)
    if buf:
        parts.append(" ".join(buf))
    return parts


def chunk_document(doc: NormalizedDocument, tables: list[ParsedTable], ctx: CitationContext) -> list[ChunkDraft]:
    by_path = doc.by_path()
    drafts: list[ChunkDraft] = []
    ordinal = 0

    def emit(
        section: NormSection,
        content: str,
        chunk_type: str,
        label: str,
        page_start: int | None,
        page_end: int | None,
        meta: dict[str, object],
    ) -> None:  # noqa: E501
        nonlocal ordinal
        header = f"{ctx.prefix} › " + " › ".join(_breadcrumb(section, by_path))
        body = f"{header}\n{content}"
        drafts.append(
            ChunkDraft(
                ordinal=ordinal,
                section_path=section.path,
                chunk_type=chunk_type,
                content=body,
                token_count=estimate_tokens(body),
                page_start=page_start,
                page_end=page_end,
                citation_label=label,
                metadata=meta,
            )
        )
        ordinal += 1

    # --- text chunks, merging tiny consecutive siblings -------------------------------
    pending: list[NormSection] = []

    def flush_pending() -> None:
        if not pending:
            return
        first, last = pending[0], pending[-1]
        text = "\n".join(_section_text(s) for s in pending)
        extra = "" if first is last else f"–{last.section_number.rstrip('.')}" if last.section_number else ""
        label = _label(ctx, first, first.page_start, last.page_end, extra)
        kind = "DEFINITION" if all(s.kind == "DEFINITION" for s in pending) else "TEXT"
        emit(
            first,
            text,
            kind,
            label,
            first.page_start,
            last.page_end,
            {"merged_paths": [s.path for s in pending], "normative": first.normative},
        )
        pending.clear()

    for section in doc.sections:
        text = _section_text(section)
        if not text.strip():
            continue
        if section.kind == "FRONT_MATTER":
            # Cover + contents: keep one chunk so document-level queries ("which revision?") resolve.
            for i, part in enumerate(_split_long(text, TARGET_MAX_TOKENS)[:1]):
                emit(
                    section,
                    part,
                    "TEXT",
                    _label(ctx, section, section.page_start, section.page_end, f" part {i + 1}"),
                    section.page_start,
                    section.page_end,
                    {"normative": None},
                )
            continue
        tokens = estimate_tokens(text)
        if tokens < TARGET_MIN_TOKENS and section.depth > 1:
            if pending and (pending[-1].parent_path != section.parent_path or pending[-1].kind != section.kind):
                flush_pending()
            pending.append(section)
            if sum(estimate_tokens(_section_text(s)) for s in pending) >= TARGET_MIN_TOKENS:
                flush_pending()
            continue
        flush_pending()
        parts = _split_long(text, TARGET_MAX_TOKENS)
        for i, part in enumerate(parts):
            extra = f" part {i + 1}/{len(parts)}" if len(parts) > 1 else ""
            emit(
                section,
                part,
                "DEFINITION" if section.kind == "DEFINITION" else "TEXT",
                _label(ctx, section, section.page_start, section.page_end, extra),
                section.page_start,
                section.page_end,
                {"normative": section.normative, "part": i + 1 if len(parts) > 1 else None},
            )
    flush_pending()

    # --- table chunks --------------------------------------------------------------------
    for t in tables:
        if not _table_is_indexable(t):
            continue  # nothing usable extracted; the page text still carries whatever PyMuPDF read
        section = _section_for_page(doc, t.page_number)
        headers = t.headers or []
        header_line = " | ".join(h or "" for h in headers) if headers else ""
        for g in range(0, len(t.rows), TABLE_ROWS_PER_CHUNK):
            rows = t.rows[g : g + TABLE_ROWS_PER_CHUNK]
            lines = [f"Table {t.table_index + 1} on page {t.page_number}"]
            if header_line:
                lines.append(header_line)
                lines.append(" | ".join("---" for _ in headers))
            lines.extend(" | ".join((c or "").replace("\n", " ") for c in r) for r in rows)
            part = f" rows {g + 1}–{g + len(rows)}" if len(t.rows) > TABLE_ROWS_PER_CHUNK else ""
            label = _label(ctx, section, t.page_number, t.page_number, f" Table {t.table_index + 1}{part}")
            emit(
                section,
                "\n".join(lines),
                "TABLE",
                label,
                t.page_number,
                t.page_number,
                {"table_index": t.table_index, "quality_score": t.quality_score, "has_header": bool(header_line)},
            )
    return drafts


def _section_text(section: NormSection) -> str:
    """Clause number + title kept inline so a merged chunk still names each clause exactly."""
    head = ""
    if section.section_number:
        head = section.section_number + (f" {section.title}" if section.title else "")
    elif section.title and section.kind == "DEFINITION":
        head = section.title
    return f"{head}\n{section.content}" if head else section.content


def _table_is_indexable(t: ParsedTable) -> bool:
    """Drawings/approval marks are often detected as 'tables' with near-empty cells."""
    cells = [c for r in t.rows for c in r if c and c.strip()]
    return len(t.rows) >= 2 and max((len(r) for r in t.rows), default=0) >= 2 and sum(len(c) for c in cells) >= 40


def _section_for_page(doc: NormalizedDocument, page: int) -> NormSection:
    best = doc.sections[0]
    for s in doc.sections:
        if s.page_start <= page and s.kind != "FRONT_MATTER":
            best = s
        if s.page_start > page:
            break
    return best
