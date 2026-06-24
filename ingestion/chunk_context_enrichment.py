"""
Uniform contextual annotation for every indexed chunk.

Prepends a structured CONTEXT block (regulation, section, topic summary) so dense
and BM25 retrieval can match short numeric / procedural queries consistently.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.core.document_registry import get_document_meta

_NUMERIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|km/h|kph|mm|cm|m/s|g|daN|kN|°|deg|ms|s)\b",
    re.I,
)
_CLAUSE_INLINE_RE = re.compile(r"\b(\d+(?:\.\d+)+)\b")


def _regulation_display(regulation: str) -> str:
    meta = get_document_meta(regulation)
    return meta.display_name or regulation.replace("_", " ")


def _section_label(clause_number: str | None, section_title: str, heading_path: str) -> str:
    if clause_number:
        return f"§{clause_number.rstrip('.')}"
    if section_title and not section_title.lower().startswith("page "):
        return section_title.strip()[:120]
    if heading_path:
        parts = [p for p in heading_path.split(">") if p.strip()]
        if parts:
            return parts[-1].strip()[:120]
    return "general provisions"


def _topic_summary(
    *,
    regulation: str,
    clause_number: str | None,
    section_title: str,
    body: str,
    test_type: str | None,
) -> str:
    meta = get_document_meta(regulation)
    bits: list[str] = [meta.full_title or meta.display_name]

    tt = test_type or meta.impact_mode or "general"
    if tt not in ("general", ""):
        bits.append(f"{tt.replace('_', ' ')} test context")

    title_low = (section_title or "").lower()
    body_low = body[:800].lower()

    if _NUMERIC_RE.search(body):
        bits.append("contains numeric thresholds, speeds, dimensions, or loads")
    if any(k in title_low + body_low for k in ("annex", "appendix")):
        bits.append("annex / appendix material")
    if any(k in title_low + body_low for k in ("definition", "means", "interpretation")):
        bits.append("definitions and terminology")
    if any(k in title_low + body_low for k in ("approval", "type approval", "mark")):
        bits.append("type approval and marking")
    if any(k in title_low + body_low for k in ("barrier", "dummy", "worldsid", "es-2", "odb")):
        bits.append("test device / barrier / dummy specifications")
    if any(k in title_low + body_low for k in ("procedure", "test method", "measurement")):
        bits.append("test procedure and measurement")
    if regulation in ("UN_R14", "UN_R16") and (clause_number or "").startswith("6.4"):
        bits.append("belt anchorage strength test configuration")

    if clause_number:
        bits.append(f"clause {clause_number.rstrip('.')}")
    elif m := _CLAUSE_INLINE_RE.search(section_title or ""):
        bits.append(f"clause {m.group(1)}")

    return "; ".join(dict.fromkeys(bits))


def build_chunk_context_meta(
    *,
    regulation: str,
    clause_number: str | None,
    section_title: str,
    heading_path: str,
    body: str,
    test_type: str | None = None,
    authority_tier: str | None = None,
) -> dict[str, Any]:
    meta = get_document_meta(regulation)
    section = _section_label(clause_number, section_title, heading_path)
    summary = _topic_summary(
        regulation=regulation,
        clause_number=clause_number,
        section_title=section_title,
        body=body,
        test_type=test_type,
    )
    return {
        "context_regulation": regulation,
        "context_section": section,
        "context_summary": summary,
        "context_display": _regulation_display(regulation),
        "authority_tier": authority_tier or meta.authority_tier,
    }


def format_context_header(ctx: dict[str, Any]) -> str:
    lines = [
        "CONTEXT:",
        f"  Regulation: {ctx.get('context_display', ctx.get('context_regulation', ''))}",
        f"  Section: {ctx.get('context_section', 'n/a')}",
        f"  Covers: {ctx.get('context_summary', '')}",
    ]
    tier = ctx.get("authority_tier")
    if tier:
        lines.append(f"  Authority: {tier}")
    return "\n".join(lines) + "\n---\n"


def prepend_context_to_chunk_text(
    text: str,
    ctx: dict[str, Any],
    *,
    applicability_header: str | None = None,
) -> str:
    """Prepend CONTEXT (and optional APPLICABILITY) once per chunk body."""
    if "CONTEXT:" in text[:250]:
        return text
    parts: list[str] = []
    if applicability_header:
        parts.append(applicability_header.rstrip())
    parts.append(format_context_header(ctx).rstrip())
    parts.append(text.lstrip())
    return "\n".join(parts) + ("\n" if not text.endswith("\n") else "")
