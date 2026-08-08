"""RETEST_SCOPE pipeline — modification / extension-of-approval clauses + honesty.

BMW-critical ask: \"After changing X, do we need to retest?\"

1. Retrieve extension-of-approval / modification clauses (not random hybrid hits).
2. Reason about the described change against those clauses.
3. CRITICAL: frame as informational governing-clause analysis for the homologation
   authority — NEVER as an authoritative retest decision. Over-confidence is a liability.
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
DEFAULT_CONFIG_PATH = ROOT / "config" / "retest_modification_sections.json"

# Shown in answer prose AND as AnswerResponse.mode_disclaimer / UI banner.
RETEST_DISCLAIMER = (
    "Informational only — not an authoritative retest or type-approval decision. "
    "The passages below are what the indexed regulation text says about modifications "
    "of this type. Verify any retest / extension-of-approval conclusion with your "
    "homologation authority before acting. Over-confidence here is a liability."
)

RETEST_DISCLAIMER_TITLE = "Homologation guidance — verify with your authority"

RETEST_SYSTEM_PROMPT = """\
You are a UNECE passive-safety assistant helping an engineer FIND governing
modification / extension-of-approval clauses. You do NOT issue retest decisions.

Reply with a single JSON object:
{
  "answer_segments": [
    {
      "text": "<what the cited clause says about modifications / further testing; no chips>",
      "citation_chunk_id": "<exact chunk_id>",
      "claim_kind": "REGULATORY_FACT" | "ENGINEERING_INFERENCE"
    }
  ]
}

STRICT RULES:
1. Prefer REGULATORY_FACT: quote/paraphrase only what the modification /
   extension-of-approval passages state.
2. ENGINEERING_INFERENCE may relate the user's change to those clauses, but MUST
   not sound like a final retest order (no \"you must retest\" / \"no retest needed\"
   as a definitive ruling).
3. If the indexed text does not address this change type, say so — do not invent
   a retest matrix.
4. Never present engineering judgment as if it were a cited approval decision.
5. The backend adds a mandatory disclaimer; your text must stay consistent with it.
"""

RETEST_USER_INSTRUCTION = """\
RETEST / MODIFICATION ANALYSIS — informational only:
- Cite governing modification / extension-of-approval clauses.
- Relate the described change to those clauses as analysis, not as a decision.
- Never conclude \"retest required\" or \"retest not required\" as authority.
- If silent, state that the indexed text does not settle the question.
"""


@dataclass
class ChangeSpec:
    id: str
    aliases: list[str]
    likely_regulations: list[str]
    change_kind: str


@dataclass
class RetestExpansion:
    question: str
    changes: list[ChangeSpec]
    named_regulation_id: str | None
    target_regulation_ids: list[str]
    config_path: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "changes": [
                {"id": c.id, "kind": c.change_kind, "likely": c.likely_regulations}
                for c in self.changes
            ],
            "named_regulation_id": self.named_regulation_id,
            "target_regulation_ids": list(self.target_regulation_ids),
            "config_path": self.config_path,
        }


@dataclass
class RetestRetrievalResult:
    chunks: list[Any]
    expansion: RetestExpansion
    by_regulation: dict[str, list[Any]] = field(default_factory=dict)
    missing_regs: list[str] = field(default_factory=list)
    found_modification_language: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "expansion": self.expansion.to_public_dict(),
            "regs": {k: len(v) for k, v in self.by_regulation.items()},
            "missing_regs": list(self.missing_regs),
            "found_modification_language": self.found_modification_language,
            "n_chunks": len(self.chunks),
        }


_CONFIG_CACHE: tuple[float, dict[str, Any], Path] | None = None

_MOD_LANG_RE = re.compile(
    r"(?ix)\b("
    r"modification|extension\s+of\s+approval|further\s+test|"
    r"shall\s+be\s+notified|type\s+approval|amended\s+approval"
    r")\b"
)


def retest_config_path() -> Path:
    raw = (os.getenv("RETEST_MODIFICATION_SECTIONS_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG_PATH


def load_retest_config(*, path: Path | None = None, force: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    cfg = path or retest_config_path()
    try:
        mtime = cfg.stat().st_mtime
    except OSError:
        logger.warning("retest_modification_sections missing: %s", cfg)
        return {"regulations": {}, "change_aliases": {}}
    if (
        not force
        and _CONFIG_CACHE is not None
        and _CONFIG_CACHE[0] == mtime
        and _CONFIG_CACHE[2] == cfg
    ):
        return dict(_CONFIG_CACHE[1])
    data = json.loads(cfg.read_text(encoding="utf-8"))
    _CONFIG_CACHE = (mtime, data, cfg)
    return dict(data)


def match_changes(question: str, *, config: dict[str, Any] | None = None) -> list[ChangeSpec]:
    cfg = config or load_retest_config()
    q = (question or "").lower()
    out: list[ChangeSpec] = []
    for cid, row in (cfg.get("change_aliases") or {}).items():
        aliases = [str(a).lower() for a in (row.get("aliases") or []) if str(a).strip()]
        if any(a in q for a in aliases):
            out.append(
                ChangeSpec(
                    id=str(cid),
                    aliases=aliases,
                    likely_regulations=[
                        str(r).strip()
                        for r in (row.get("likely_regulations") or [])
                        if str(r).strip()
                    ],
                    change_kind=str(row.get("change_kind") or "other").strip(),
                )
            )
    # Mass increase: numeric kg after change/adding.
    if re.search(r"(?i)(?:add(?:ing)?|increase|extra)\s+\d+\s*kg|\b\d+\s*kg\b", question or ""):
        if not any(c.id == "mass_increase" for c in out):
            out.append(
                ChangeSpec(
                    id="mass_increase",
                    aliases=["kg"],
                    likely_regulations=["UN-ECE-R94", "UN-ECE-R95"],
                    change_kind="mass",
                )
            )
    return out


def expand_retest_query(question: str) -> RetestExpansion:
    from retrieval.enumerative import detect_named_regulation, resolve_hard_regulation_filter
    from retrieval.retrieve import indexed_regulation_ids

    q = (question or "").strip()
    cfg = load_retest_config()
    named = resolve_hard_regulation_filter(q) or detect_named_regulation(q)
    changes = match_changes(q, config=cfg)
    indexed = indexed_regulation_ids()
    targets: list[str] = []
    if named:
        targets = [named]
    else:
        for ch in changes:
            for rid in ch.likely_regulations:
                if rid in indexed and rid not in targets:
                    targets.append(rid)
        if not targets:
            # Default passive-safety pair when change is unspecified.
            for rid in ("UN-ECE-R94", "UN-ECE-R95"):
                if rid in indexed:
                    targets.append(rid)
    return RetestExpansion(
        question=q,
        changes=changes,
        named_regulation_id=named,
        target_regulation_ids=targets,
        config_path=str(retest_config_path()),
    )


def _fetch_mod_sections(
    *,
    regulation_id: str,
    section_numbers: Sequence[str],
    client: Any,
    collection: str | None,
) -> list[Any]:
    from qdrant_client import models as qm

    from retrieval.retrieve import (
        DEFAULT_COLLECTION,
        _payload_to_chunk,
        get_qdrant_client,
    )

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
            logger.warning("retest scroll failed %s §%s: %s", regulation_id, sec, exc)
            continue
        for pt in points or []:
            payload = dict(pt.payload or {})
            if (payload.get("regulation_id") or "").strip() != regulation_id:
                continue
            chunk = _payload_to_chunk(payload, score=1.0)
            cid = chunk.chunk_id or ""
            if cid and cid not in seen:
                seen.add(cid)
                out.append(chunk)
    return out


def retest_result_from_chunks(
    question: str,
    chunks: Sequence[Any],
    *,
    expansion: RetestExpansion | None = None,
) -> RetestRetrievalResult:
    """Rebuild a structured result from already-retrieved modification chunks."""
    expansion = expansion or expand_retest_query(question)
    by_reg: dict[str, list[Any]] = {}
    for c in chunks:
        rid = (getattr(c, "regulation_id", None) or "").strip() or "unknown"
        by_reg.setdefault(rid, []).append(c)
    ordered: dict[str, list[Any]] = {
        rid: list(by_reg.get(rid) or []) for rid in expansion.target_regulation_ids
    }
    for rid, cs in by_reg.items():
        if rid not in ordered:
            ordered[rid] = list(cs)
    found = False
    for c in chunks:
        blob = f"{getattr(c, 'text', '') or ''} {getattr(c, 'section_title', '') or ''}"
        if _MOD_LANG_RE.search(blob):
            found = True
            break
    missing = [
        rid
        for rid in expansion.target_regulation_ids
        if not ordered.get(rid)
    ]
    return RetestRetrievalResult(
        chunks=list(chunks),
        expansion=expansion,
        by_regulation=ordered,
        missing_regs=missing,
        found_modification_language=found,
    )


def retrieve_retest_scope(
    query: str,
    *,
    client: object | None = None,
    embedder: object | None = None,
    collection: str | None = None,
    expansion: RetestExpansion | None = None,
) -> RetestRetrievalResult:
    """Fetch modification / extension-of-approval clauses for target regs."""
    from retrieval.retrieve import hybrid_search

    expansion = expansion or expand_retest_query(query)
    cfg = load_retest_config()
    regs_cfg = cfg.get("regulations") or {}
    by_reg: dict[str, list[Any]] = {}
    missing: list[str] = []
    merged: list[Any] = []
    seen: set[str] = set()
    found_mod = False

    for rid in expansion.target_regulation_ids:
        row = regs_cfg.get(rid) or {}
        sections = [str(s).strip() for s in (row.get("modification_sections") or []) if str(s).strip()]
        queries = [str(q).strip() for q in (row.get("retrieval_queries") or []) if str(q).strip()]
        hits: list[Any] = []
        if sections:
            hits.extend(
                _fetch_mod_sections(
                    regulation_id=rid,
                    section_numbers=sections,
                    client=client,
                    collection=collection,
                )
            )
        # Hybrid probes for modification language (same reg only).
        for sq in (queries or ["modification extension of approval"])[:4]:
            try:
                batch = hybrid_search(
                    sq,
                    top_k=8,
                    regulation_id=rid,
                    client=client,  # type: ignore[arg-type]
                    embedder=embedder,  # type: ignore[arg-type]
                    collection=collection,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("retest hybrid failed %s: %s", rid, exc)
                batch = []
            for c in batch:
                if (getattr(c, "regulation_id", None) or "").strip() != rid:
                    continue
                text = getattr(c, "text", "") or ""
                if _MOD_LANG_RE.search(text) or not hits:
                    hits.append(c)
        # Prefer chunks that actually mention modification/extension.
        preferred = [
            c for c in hits if _MOD_LANG_RE.search(getattr(c, "text", "") or "")
        ]
        chosen = (preferred or hits)[:6]
        if preferred:
            found_mod = True
        if not chosen:
            missing.append(rid)
        by_reg[rid] = chosen
        for c in chosen:
            cid = getattr(c, "chunk_id", "") or ""
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(c)

    result = RetestRetrievalResult(
        chunks=merged,
        expansion=expansion,
        by_regulation=by_reg,
        missing_regs=missing,
        found_modification_language=found_mod,
    )
    logger.info(
        "retest_scope changes=%s targets=%s chunks=%d mod_lang=%s missing=%s",
        [c.id for c in expansion.changes],
        expansion.target_regulation_ids,
        len(merged),
        found_mod,
        missing,
    )
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.optimizations["retest_scope"] = True
            tr.optimizations["retest_retrieval"] = result.to_public_dict()
            tr.optimizations["mode_disclaimer"] = RETEST_DISCLAIMER
            tr.optimizations["mode_disclaimer_title"] = RETEST_DISCLAIMER_TITLE
    except Exception:  # noqa: BLE001
        pass
    return result


def format_retest_context(result: RetestRetrievalResult) -> str:
    parts = [
        f"MANDATORY FRAMING: {RETEST_DISCLAIMER}",
        f"Described changes: "
        + (
            ", ".join(f"{c.id} ({c.change_kind})" for c in result.expansion.changes)
            or "(parse from question)"
        ),
        f"Question: {result.expansion.question}",
    ]
    idx = 0
    for rid, chunks in result.by_regulation.items():
        parts.append(f"## Modification / extension clauses — `{rid}`")
        if not chunks:
            parts.append(
                "(No modification/extension-of-approval clause retrieved for this "
                "regulation in the index.)"
            )
            continue
        for c in chunks:
            idx += 1
            if hasattr(c, "context_block"):
                parts.append(c.context_block(index=idx))
            else:
                parts.append(
                    f"[passage {idx}] chunk_id={getattr(c, 'chunk_id', '')}\n"
                    f"{getattr(c, 'text', '')}"
                )
    if not result.found_modification_language:
        parts.append(
            "## Note\nIndexed passages may not contain clear modification/"
            "extension-of-approval language — do not invent a retest matrix."
        )
    return "\n\n".join(parts)


def retest_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "retest_scope_answer",
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
                                "claim_kind": {
                                    "type": "string",
                                    "enum": ["REGULATORY_FACT", "ENGINEERING_INFERENCE"],
                                },
                            },
                            "required": ["text", "citation_chunk_id", "claim_kind"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["answer_segments"],
                "additionalProperties": False,
            },
        },
    }


def render_retest_answer(
    segments: Sequence[Any],
    chunks_by_id: dict[str, Any],
    *,
    result: RetestRetrievalResult,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Assemble answer with mandatory disclaimer + non-decisive framing."""
    from generation.answer import _CITATION_CHIP_RE
    from retrieval.design_implication import CLAIM_FACT, CLAIM_INFERENCE, normalize_claim_kind

    fact_lines: list[str] = []
    inf_lines: list[str] = []
    sources: list[Any] = []
    seen: set[str] = set()

    for seg in segments:
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        body = _CITATION_CHIP_RE.sub("", (getattr(seg, "text", None) or "")).strip()
        # Strip overconfident decision language from model text.
        body = re.sub(
            r"(?i)\b(you must|must|shall)\s+re-?test\b",
            "the text discusses further testing",
            body,
        )
        body = re.sub(
            r"(?i)\bno\s+re-?test\s+(?:is\s+)?(?:needed|required)\b",
            "the indexed text does not by itself settle whether retesting is needed",
            body,
        )
        chip = chunk.citation_tag()
        kind = normalize_claim_kind(getattr(seg, "claim_kind", None))
        if kind == CLAIM_INFERENCE:
            inf_lines.append(f"- [Analysis] {body} {chip}")
        else:
            fact_lines.append(f"- [Governing clause] {body} {chip}")
        if cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))

    # Extractive fill from modification-language chunks if LLM empty.
    if not fact_lines and not inf_lines:
        for c in result.chunks[:8]:
            text = " ".join((getattr(c, "text", "") or "").split())
            if not text:
                continue
            if not _MOD_LANG_RE.search(text):
                continue
            words = text.split()
            excerpt = " ".join(words[:45]) + ("…" if len(words) > 45 else "")
            fact_lines.append(f"- [Governing clause] {excerpt} {c.citation_tag()}")
            cid = getattr(c, "chunk_id", "") or ""
            if cid and cid not in seen:
                seen.add(cid)
                sources.append(to_source(c))

    changes = (
        ", ".join(c.id.replace("_", " ") for c in result.expansion.changes)
        or "the described modification"
    )
    blocks = [
        f"**{RETEST_DISCLAIMER_TITLE}**",
        "",
        RETEST_DISCLAIMER,
        "",
        f"### Change under review",
        changes,
        f"Question: {result.expansion.question}",
        "",
        "### What the indexed regulation says (governing clauses)",
    ]
    if fact_lines:
        blocks.extend(fact_lines)
    else:
        blocks.append(
            "_No clear modification / extension-of-approval clause was found in the "
            "indexed text for the target regulation(s). The system cannot settle "
            "retest scope from the current index — consult the full regulation and "
            "your homologation authority._"
        )
    blocks.append("")
    blocks.append("### Analysis (not a decision)")
    if inf_lines:
        blocks.extend(inf_lines)
    else:
        blocks.append(
            "- Relating this change to the clauses above requires engineering and "
            "type-approval judgment beyond what the indexed text alone decides."
        )
    blocks.append("")
    blocks.append("### Bottom line")
    blocks.append(
        "This output is **not** a determination that retesting is or is not required. "
        "Use the cited clauses as the starting point for discussion with your "
        "homologation authority."
    )
    if result.missing_regs:
        blocks.append("")
        blocks.append(
            "No modification clauses retrieved for: "
            + ", ".join(result.missing_regs)
            + "."
        )
    return "\n".join(blocks).strip(), sources


def extractive_retest_fallback(
    result: RetestRetrievalResult,
    *,
    to_source: Any,
) -> tuple[str, list[Any]]:
    by_id = {
        getattr(c, "chunk_id", ""): c
        for c in result.chunks
        if getattr(c, "chunk_id", None)
    }
    return render_retest_answer([], by_id, result=result, to_source=to_source)
