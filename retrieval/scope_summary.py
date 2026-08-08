"""SCOPE_SUMMARY pipeline — hard-reg lock + deterministic section fetch + limits.

Fixes \"Summarize the scope of UN R94\" answering from R95:

1. Hard ``regulation_id`` filter — never scroll other corpora.
2. Deterministically fetch Scope, Definitions, and configured key sections
   (``config/scope_summary_sections.json``), not similarity top-k alone.
3. Structured summary: scope/applicability, key injury criteria (from Fix 22
   ``data/limits/*.json`` — single source of truth), test configuration,
   homologation obligations — each cited.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "scope_summary_sections.json"

SECTION_SCOPE = "scope"
SECTION_DEFINITIONS = "definitions"
SECTION_INJURY = "injury_criteria"
SECTION_TEST = "test_configuration"
SECTION_HOMOLOGATION = "homologation"

SCOPE_SYSTEM_PROMPT = """\
You are a UNECE passive-safety regulation assistant writing a STRUCTURED
SUMMARY of ONE named regulation.

Reply with a single JSON object:
{
  "answer_segments": [
    {
      "text": "<one factual claim; no §/page/citation chips>",
      "citation_chunk_id": "<exact chunk_id from a provided passage>",
      "category_id": "scope" | "definitions" | "test_configuration" | "homologation"
    }
  ]
}

STRICT RULES:
1. Use ONLY the provided passages for THIS regulation. Never mention or cite
   another regulation.
2. Do NOT invent injury-criteria numeric limits — those are inserted by the
   backend from the verified limits table.
3. category_id must be one of: scope, definitions, test_configuration, homologation.
4. Prefer one clear claim per segment with a valid citation_chunk_id.
5. If a category has no passages, omit it (backend notes the gap).
6. If nothing is supported, return {"answer_segments": []}.
"""

SCOPE_USER_INSTRUCTION = """\
SCOPE SUMMARY — structured deliverable for a single named regulation:
- Emit segments for scope / definitions / test_configuration / homologation only.
- Do not emit numeric injury limits (backend fills those from the verified table).
- Cite only chunk_ids from the provided passages for this regulation.
"""


@dataclass
class ScopeSummarySpec:
    regulation_id: str
    label: str
    scope_section: str
    definitions_section: str
    key_requirement_sections: list[str]
    test_configuration_sections: list[str]
    homologation_sections: list[str]


@dataclass
class ScopeExpansion:
    question: str
    regulation_id: str | None
    spec: ScopeSummarySpec | None
    config_path: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "regulation_id": self.regulation_id,
            "label": self.spec.label if self.spec else None,
            "config_path": self.config_path,
        }


@dataclass
class RoleChunk:
    role: str
    chunk: Any


@dataclass
class ScopeRetrievalResult:
    chunks: list[Any]
    expansion: ScopeExpansion
    by_role: dict[str, list[Any]] = field(default_factory=dict)
    chunk_role: dict[str, str] = field(default_factory=dict)
    limits: list[Any] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "regulation_id": self.expansion.regulation_id,
            "roles": {k: len(v) for k, v in self.by_role.items()},
            "chunk_role": dict(self.chunk_role),
            "n_limits": len(self.limits),
            "missing_roles": list(self.missing_roles),
            "n_chunks": len(self.chunks),
            "regulation_ids": sorted(
                {
                    (getattr(c, "regulation_id", None) or "").strip()
                    for c in self.chunks
                    if getattr(c, "regulation_id", None)
                }
            ),
        }


_CONFIG_CACHE: tuple[float, dict[str, ScopeSummarySpec], Path] | None = None


def scope_summary_config_path() -> Path:
    raw = (os.getenv("SCOPE_SUMMARY_SECTIONS_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG_PATH


def load_scope_summary_specs(
    *, path: Path | None = None, force: bool = False
) -> dict[str, ScopeSummarySpec]:
    global _CONFIG_CACHE
    cfg = path or scope_summary_config_path()
    try:
        mtime = cfg.stat().st_mtime
    except OSError:
        logger.warning("scope_summary_sections config missing: %s", cfg)
        return {}
    if (
        not force
        and _CONFIG_CACHE is not None
        and _CONFIG_CACHE[0] == mtime
        and _CONFIG_CACHE[2] == cfg
    ):
        return dict(_CONFIG_CACHE[1])

    data = json.loads(cfg.read_text(encoding="utf-8"))
    specs: dict[str, ScopeSummarySpec] = {}
    for rid, row in (data.get("regulations") or {}).items():
        rid = str(rid).strip()
        if not rid or not isinstance(row, dict):
            continue
        specs[rid] = ScopeSummarySpec(
            regulation_id=rid,
            label=str(row.get("label") or rid).strip(),
            scope_section=str(row.get("scope_section") or "1").strip(),
            definitions_section=str(row.get("definitions_section") or "2").strip(),
            key_requirement_sections=[
                str(s).strip()
                for s in (row.get("key_requirement_sections") or [])
                if str(s).strip()
            ],
            test_configuration_sections=[
                str(s).strip()
                for s in (row.get("test_configuration_sections") or [])
                if str(s).strip()
            ],
            homologation_sections=[
                str(s).strip()
                for s in (row.get("homologation_sections") or [])
                if str(s).strip()
            ],
        )
    _CONFIG_CACHE = (mtime, specs, cfg)
    return dict(specs)


def expand_scope_summary_query(
    question: str,
    *,
    regulation_id: str | None = None,
) -> ScopeExpansion:
    """Resolve the single named regulation for a scope-summary ask."""
    from retrieval.enumerative import detect_named_regulation, resolve_hard_regulation_filter

    q = (question or "").strip()
    rid = (
        (regulation_id or "").strip()
        or resolve_hard_regulation_filter(q)
        or detect_named_regulation(q)
    )
    specs = load_scope_summary_specs()
    spec = specs.get(rid) if rid else None
    if rid and spec is None:
        # Fallback skeleton from legacy scope/definitions maps.
        from retrieval.retrieve import load_definitions_sections, load_scope_sections

        scope_map = load_scope_sections()
        def_map = load_definitions_sections()
        spec = ScopeSummarySpec(
            regulation_id=rid,
            label=rid.replace("UN-ECE-", "UN "),
            scope_section=scope_map.get(rid, "1"),
            definitions_section=def_map.get(rid, "2"),
            key_requirement_sections=["5"],
            test_configuration_sections=[],
            homologation_sections=["3", "4"],
        )
    return ScopeExpansion(
        question=q,
        regulation_id=rid,
        spec=spec,
        config_path=str(scope_summary_config_path()),
    )


def _fetch_sections_for_reg(
    *,
    regulation_id: str,
    section_numbers: Sequence[str],
    client: Any,
    collection: str | None,
    role: str,
) -> list[Any]:
    from qdrant_client import models as qm

    from retrieval.retrieve import (
        DEFAULT_COLLECTION,
        _payload_to_chunk,
        get_qdrant_client,
    )

    if not section_numbers:
        return []
    collection = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    client = client or get_qdrant_client()
    out: list[Any] = []
    seen: set[str] = set()
    for sec in section_numbers:
        must = [
            qm.FieldCondition(
                key="regulation_id", match=qm.MatchValue(value=regulation_id)
            ),
            qm.FieldCondition(
                key="section_number", match=qm.MatchValue(value=sec)
            ),
        ]
        try:
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=qm.Filter(must=must),
                limit=6,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scope_summary scroll failed reg=%s sec=%s: %s",
                regulation_id,
                sec,
                exc,
            )
            continue
        for pt in points or []:
            payload = dict(pt.payload or {})
            # Hard lock — never accept a foreign regulation_id.
            if (payload.get("regulation_id") or "").strip() != regulation_id:
                continue
            chunk = _payload_to_chunk(payload, score=1.0)
            cid = chunk.chunk_id or ""
            if cid and cid not in seen:
                seen.add(cid)
                out.append(chunk)
    logger.info(
        "scope_summary role=%s reg=%s sections=%s chunks=%d",
        role,
        regulation_id,
        list(section_numbers),
        len(out),
    )
    return out


def _load_limits_for_reg(regulation_id: str) -> list[Any]:
    from ingestion.extract_limits import load_limits_table, seed_known_limits

    table = load_limits_table(regulation_id)
    if table is None:
        seed_known_limits()
        table = load_limits_table(regulation_id)
    if table is None:
        return []
    return list(table.limits or [])


def retrieve_scope_summary(
    query: str,
    *,
    regulation_id: str | None = None,
    client: object | None = None,
    collection: str | None = None,
    expansion: ScopeExpansion | None = None,
) -> ScopeRetrievalResult:
    """Deterministic per-section fetch for one regulation — no cross-reg hybrid."""
    from retrieval.retrieve import fetch_chunks_by_ids

    expansion = expansion or expand_scope_summary_query(
        query, regulation_id=regulation_id
    )
    rid = expansion.regulation_id
    if not rid or expansion.spec is None:
        logger.warning(
            "scope_summary refused: no named regulation in %r", query[:120]
        )
        return ScopeRetrievalResult(
            chunks=[],
            expansion=expansion,
            missing_roles=[
                SECTION_SCOPE,
                SECTION_DEFINITIONS,
                SECTION_INJURY,
                SECTION_TEST,
                SECTION_HOMOLOGATION,
            ],
        )

    spec = expansion.spec
    by_role: dict[str, list[Any]] = {}
    chunk_role: dict[str, str] = {}
    merged: list[Any] = []
    seen: set[str] = set()

    def _add(role: str, chunks: Sequence[Any]) -> None:
        kept = []
        for c in chunks:
            # Belt-and-suspenders hard filter.
            if (getattr(c, "regulation_id", None) or "").strip() != rid:
                continue
            cid = getattr(c, "chunk_id", None) or ""
            kept.append(c)
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(c)
                chunk_role[cid] = role
        by_role[role] = kept

    _add(
        SECTION_SCOPE,
        _fetch_sections_for_reg(
            regulation_id=rid,
            section_numbers=[spec.scope_section],
            client=client,
            collection=collection,
            role=SECTION_SCOPE,
        ),
    )
    _add(
        SECTION_DEFINITIONS,
        _fetch_sections_for_reg(
            regulation_id=rid,
            section_numbers=[spec.definitions_section],
            client=client,
            collection=collection,
            role=SECTION_DEFINITIONS,
        ),
    )
    _add(
        "key_requirements",
        _fetch_sections_for_reg(
            regulation_id=rid,
            section_numbers=spec.key_requirement_sections,
            client=client,
            collection=collection,
            role="key_requirements",
        ),
    )
    _add(
        SECTION_TEST,
        _fetch_sections_for_reg(
            regulation_id=rid,
            section_numbers=spec.test_configuration_sections,
            client=client,
            collection=collection,
            role=SECTION_TEST,
        ),
    )
    _add(
        SECTION_HOMOLOGATION,
        _fetch_sections_for_reg(
            regulation_id=rid,
            section_numbers=spec.homologation_sections,
            client=client,
            collection=collection,
            role=SECTION_HOMOLOGATION,
        ),
    )

    limits = _load_limits_for_reg(rid)
    # Attach limit source chunks (same regulation only).
    limit_ids = [
        str(getattr(row, "source_chunk_id", "") or "").strip()
        for row in limits
        if str(getattr(row, "source_chunk_id", "") or "").strip()
    ]
    if limit_ids:
        try:
            limit_chunks = fetch_chunks_by_ids(
                limit_ids, client=client, collection=collection  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("scope_summary limit chunk fetch failed: %s", exc)
            limit_chunks = []
        # Also fetch by section_number when chunk id missing/missed.
        sec_nums = [
            str(getattr(row, "section_number", "") or "").strip()
            for row in limits
            if str(getattr(row, "section_number", "") or "").strip()
        ]
        if sec_nums:
            limit_chunks = list(limit_chunks) + _fetch_sections_for_reg(
                regulation_id=rid,
                section_numbers=list(dict.fromkeys(sec_nums)),
                client=client,
                collection=collection,
                role=SECTION_INJURY,
            )
        _add(SECTION_INJURY, limit_chunks)
    else:
        by_role[SECTION_INJURY] = []

    missing = [
        role
        for role in (
            SECTION_SCOPE,
            SECTION_DEFINITIONS,
            SECTION_TEST,
            SECTION_HOMOLOGATION,
        )
        if not by_role.get(role)
    ]
    if not limits:
        missing.append(SECTION_INJURY)

    # Final hard filter on merged list.
    merged = [
        c
        for c in merged
        if (getattr(c, "regulation_id", None) or "").strip() == rid
    ]

    result = ScopeRetrievalResult(
        chunks=merged,
        expansion=expansion,
        by_role=by_role,
        chunk_role=chunk_role,
        limits=limits,
        missing_roles=missing,
    )
    logger.info(
        "scope_summary retrieved reg=%s chunks=%d roles=%s missing=%s limits=%d",
        rid,
        len(merged),
        {k: len(v) for k, v in by_role.items()},
        missing,
        len(limits),
    )
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.optimizations["scope_summary"] = True
            tr.optimizations["scope_summary_retrieval"] = result.to_public_dict()
    except Exception:  # noqa: BLE001
        pass
    return result


def format_scope_summary_context(result: ScopeRetrievalResult) -> str:
    parts: list[str] = [
        f"Regulation (HARD FILTER): {result.expansion.regulation_id}",
        "Do not cite any other regulation.",
    ]
    idx = 0
    for role in (
        SECTION_SCOPE,
        SECTION_DEFINITIONS,
        "key_requirements",
        SECTION_TEST,
        SECTION_HOMOLOGATION,
        SECTION_INJURY,
    ):
        chunks = result.by_role.get(role) or []
        parts.append(f"## Role `{role}`")
        if not chunks:
            parts.append("(No passages for this role.)")
            continue
        for c in chunks:
            idx += 1
            if hasattr(c, "context_block"):
                parts.append(f"category_id={role}\n{c.context_block(index=idx)}")
            else:
                parts.append(
                    f"category_id={role}\n[passage {idx}] "
                    f"chunk_id={getattr(c, 'chunk_id', '')}\n"
                    f"{getattr(c, 'text', '')}"
                )
    if result.limits:
        parts.append(
            "## Verified limits table (do NOT restate numbers — backend inserts them)\n"
            + ", ".join(
                f"{getattr(r, 'criterion_name', '?')} "
                f"{getattr(r, 'operator', '<=')}{getattr(r, 'limit_value', '?')}"
                f"{getattr(r, 'unit', '')}"
                for r in result.limits
            )
        )
    return "\n\n".join(parts)


def format_limits_section(
    limits: Sequence[Any],
    chunks_by_id: dict[str, Any],
    *,
    regulation_id: str,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Render injury criteria from the verified limits table (single source of truth)."""
    if not limits:
        return (
            "_No verified limits table found for this regulation — "
            "injury-criteria numbers unavailable._",
            [],
        )
    lines: list[str] = []
    sources: list[Any] = []
    seen: set[str] = set()
    for row in limits:
        name = getattr(row, "criterion_name", "") or "criterion"
        op = getattr(row, "operator", "<=") or "<="
        val = getattr(row, "limit_value", None)
        unit = (getattr(row, "unit", None) or "").strip()
        sec = (getattr(row, "section_number", None) or "").strip()
        cid = (getattr(row, "source_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid) if cid else None
        if chunk is None and sec:
            # Prefer any same-reg chunk with matching section.
            chunk = next(
                (
                    c
                    for c in chunks_by_id.values()
                    if (getattr(c, "section_number", "") or "") == sec
                    and (getattr(c, "regulation_id", "") or "") == regulation_id
                ),
                None,
            )
        unit_s = f" {unit}" if unit else ""
        body = f"{name}: {op} {val}{unit_s}".strip()
        if chunk is not None:
            chip = chunk.citation_tag()
            lines.append(f"- {body} {chip}")
            cid2 = chunk.chunk_id or ""
            if cid2 and cid2 not in seen:
                seen.add(cid2)
                sources.append(to_source(chunk))
        elif sec:
            lines.append(f"- {body} [{regulation_id} §{sec}]")
        else:
            lines.append(f"- {body}")
    return "\n".join(lines), sources


def render_scope_summary(
    *,
    result: ScopeRetrievalResult,
    segments: Sequence[Any] | None = None,
    chunks: Sequence[Any] | None = None,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Assemble structured summary; injury criteria always from limits table."""
    from generation.answer import _CITATION_CHIP_RE

    rid = result.expansion.regulation_id or "?"
    label = result.expansion.spec.label if result.expansion.spec else rid
    pool = list(chunks) if chunks is not None else list(result.chunks)
    by_id = {getattr(c, "chunk_id", ""): c for c in pool if getattr(c, "chunk_id", None)}

    buckets: dict[str, list[str]] = {
        SECTION_SCOPE: [],
        SECTION_DEFINITIONS: [],
        SECTION_TEST: [],
        SECTION_HOMOLOGATION: [],
    }
    sources: list[Any] = []
    seen: set[str] = set()

    def _take_source(chunk: Any) -> None:
        cid = getattr(chunk, "chunk_id", "") or ""
        if cid and cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))

    for seg in segments or []:
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        if (getattr(chunk, "regulation_id", None) or "").strip() != rid:
            continue  # hard reject foreign cites
        role = (
            (getattr(seg, "category_id", None) or "").strip()
            or result.chunk_role.get(cid, SECTION_SCOPE)
        )
        if role not in buckets:
            if role == SECTION_INJURY:
                continue  # backend owns injury numbers
            role = SECTION_SCOPE
        body = _CITATION_CHIP_RE.sub("", (getattr(seg, "text", None) or "")).strip()
        buckets[role].append(f"{body} {chunk.citation_tag()}".strip())
        _take_source(chunk)

    # Extractive fill for empty roles from retrieved chunks.
    role_labels = {
        SECTION_SCOPE: "Scope / applicability",
        SECTION_DEFINITIONS: "Definitions (selected)",
        SECTION_TEST: "Test configuration",
        SECTION_HOMOLOGATION: "Homologation obligations",
    }
    for role, heading in role_labels.items():
        if buckets[role]:
            continue
        for c in result.by_role.get(role) or []:
            if (getattr(c, "regulation_id", None) or "").strip() != rid:
                continue
            text = " ".join((getattr(c, "text", None) or "").split())
            if not text:
                continue
            words = text.split()
            excerpt = " ".join(words[:50]) + ("…" if len(words) > 50 else "")
            buckets[role].append(f"{excerpt} {c.citation_tag()}")
            _take_source(c)
            break

    injury_text, injury_sources = format_limits_section(
        result.limits,
        by_id,
        regulation_id=rid,
        to_source=to_source,
    )
    for s in injury_sources:
        cid = getattr(s, "chunk_id", "") or ""
        if cid and cid not in seen:
            seen.add(cid)
            sources.append(s)

    blocks = [f"Structured summary — {label}", ""]
    for role, heading in role_labels.items():
        blocks.append(f"## {heading}")
        items = buckets.get(role) or []
        if items:
            blocks.extend(items)
        else:
            blocks.append(
                f"_No indexed content retrieved for this section of {label}._"
            )
        blocks.append("")

    blocks.append("## Key injury criteria (verified limits table)")
    blocks.append(injury_text)
    blocks.append("")

    if result.missing_roles:
        blocks.append("### Incomplete sections")
        blocks.append(
            "No indexed content for: "
            + ", ".join(result.missing_roles)
            + ". Summary may be incomplete."
        )

    # Absolute guarantee: no foreign regulation ids in prose from our chips
    # (chips are backend-built from filtered chunks).
    return "\n".join(blocks).strip(), sources


def scope_summary_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "scope_summary_answer",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "answer_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "citation_chunk_id": {"type": "string"},
                                "category_id": {
                                    "type": "string",
                                    "enum": [
                                        SECTION_SCOPE,
                                        SECTION_DEFINITIONS,
                                        SECTION_TEST,
                                        SECTION_HOMOLOGATION,
                                    ],
                                },
                            },
                            "required": ["text", "citation_chunk_id", "category_id"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["answer_segments"],
                "additionalProperties": False,
            },
        },
    }


def rebuild_scope_result(
    question: str,
    chunks: Sequence[Any],
    *,
    meta: dict[str, Any] | None = None,
    regulation_id: str | None = None,
) -> ScopeRetrievalResult:
    expansion = expand_scope_summary_query(question, regulation_id=regulation_id)
    rid = expansion.regulation_id
    meta = meta or {}
    chunk_role = {
        str(k): str(v)
        for k, v in (meta.get("chunk_role") or {}).items()
        if str(k).strip()
    }
    # Drop foreign chunks.
    filtered = [
        c
        for c in chunks
        if not rid or (getattr(c, "regulation_id", None) or "").strip() == rid
    ]
    by_role: dict[str, list[Any]] = {}
    for c in filtered:
        cid = getattr(c, "chunk_id", "") or ""
        role = chunk_role.get(cid, SECTION_SCOPE)
        by_role.setdefault(role, []).append(c)
    limits = _load_limits_for_reg(rid) if rid else []
    missing = [str(x) for x in (meta.get("missing_roles") or [])]
    return ScopeRetrievalResult(
        chunks=filtered,
        expansion=expansion,
        by_role=by_role,
        chunk_role=chunk_role,
        limits=limits,
        missing_roles=missing,
    )


def assert_single_regulation(chunks: Sequence[Any], regulation_id: str) -> None:
    """Raise if any chunk leaks outside the hard-filtered regulation."""
    bad = [
        (getattr(c, "chunk_id", ""), getattr(c, "regulation_id", ""))
        for c in chunks
        if (getattr(c, "regulation_id", None) or "").strip() != regulation_id
    ]
    if bad:
        raise AssertionError(
            f"scope_summary hard-filter leak: expected only {regulation_id}, got {bad}"
        )
