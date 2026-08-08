"""DESIGN_IMPLICATION pipeline — component expansion, multi-reg retrieve, per-claim grounding.

1. Expand the design element to regulatory concepts BEFORE retrieval
   (``config/design_components.json`` — maintainable, engineer-reviewed).
2. Retrieve across relevant regulations (top-k per reg), scoped by expanded concepts.
3. Synthesize as a list of claims each tied to a chunk_id; validity = every kept claim
   is grounded (not all-or-nothing on the whole multi-clause answer).
4. Separate REGULATORY_FACT from ENGINEERING_INFERENCE in rendered output.
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
DEFAULT_CONFIG_PATH = ROOT / "config" / "design_components.json"
DEFAULT_PER_REG_TOP_K = 3

CLAIM_FACT = "REGULATORY_FACT"
CLAIM_INFERENCE = "ENGINEERING_INFERENCE"
_VALID_KINDS = frozenset({CLAIM_FACT, CLAIM_INFERENCE})

DESIGN_SYSTEM_PROMPT = """\
You are a UNECE passive-safety regulation assistant answering DESIGN IMPLICATION
questions (which requirements affect a component / design decision).

Reply with a single JSON object:
{
  "answer_segments": [
    {
      "text": "<one claim; no section/page/citation markup>",
      "citation_chunk_id": "<exact chunk_id from a provided passage>",
      "claim_kind": "REGULATORY_FACT" | "ENGINEERING_INFERENCE"
    }
  ]
}

STRICT RULES:
1. Answer ONLY from the provided context passages. Do not invent clauses.
2. Each segment is ONE claim with EXACTLY ONE citation_chunk_id from the passages.
3. claim_kind MUST be set on every segment:
   - REGULATORY_FACT: what the regulation text literally requires / states.
   - ENGINEERING_INFERENCE: an engineering implication you draw from that fact
     (e.g. how it constrains a named component). Never present inference as if
     it were regulation text.
4. Prefer several REGULATORY_FACT claims covering distinct requirements across
   regulations, each with its own citation. Add ENGINEERING_INFERENCE segments
   only when they help the engineer; every inference must cite the fact chunk
   it rests on.
5. Do NOT write section numbers, page numbers, or citation chips in "text".
6. If a passage does not support a claim, omit that claim. If nothing is
   supported, return {"answer_segments": []}.
7. Multi-clause answers are expected and correct: no single clause needs to
   answer the whole design question — each claim must be grounded individually.
"""

DESIGN_USER_INSTRUCTION = """\
DESIGN IMPLICATION — produce a multi-claim synthesis:
- Each answer_segment = one claim + one citation_chunk_id + claim_kind.
- REGULATORY_FACT segments state what the indexed text requires.
- ENGINEERING_INFERENCE segments state design implications and MUST be labelled
  as inference (claim_kind=ENGINEERING_INFERENCE); never as a cited requirement.
- Cover distinct requirements across the retrieved regulations when present.
- The answer is valid when EACH claim is grounded — the whole question need not
  be answered by any single clause.
"""


@dataclass
class ComponentSpec:
    id: str
    label: str
    aliases: list[str]
    concepts: list[str]
    likely_regulations: list[str]
    retrieval_queries: list[str]
    notes: str = ""


@dataclass
class DesignExpansion:
    question: str
    component: ComponentSpec | None
    concepts: list[str] = field(default_factory=list)
    likely_regulations: list[str] = field(default_factory=list)
    named_regulation_id: str | None = None
    subqueries: list[str] = field(default_factory=list)
    config_path: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component.id if self.component else None,
            "component_label": self.component.label if self.component else None,
            "concepts": list(self.concepts),
            "likely_regulations": list(self.likely_regulations),
            "named_regulation_id": self.named_regulation_id,
            "subqueries": list(self.subqueries),
            "config_path": self.config_path,
        }


@dataclass
class DesignRetrievalResult:
    chunks: list[Any]
    covered: list[str]
    missing: list[str]
    expansion: DesignExpansion
    per_regulation: dict[str, list[Any]] = field(default_factory=dict)


_CONFIG_CACHE: tuple[float, list[ComponentSpec], Path] | None = None


def design_components_path() -> Path:
    raw = (os.getenv("DESIGN_COMPONENTS_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG_PATH


def load_design_components(*, path: Path | None = None, force: bool = False) -> list[ComponentSpec]:
    """Load component → concept map (cached by mtime)."""
    global _CONFIG_CACHE
    cfg = path or design_components_path()
    try:
        mtime = cfg.stat().st_mtime
    except OSError:
        logger.warning("design_components config missing: %s", cfg)
        return []
    if (
        not force
        and _CONFIG_CACHE is not None
        and _CONFIG_CACHE[0] == mtime
        and _CONFIG_CACHE[2] == cfg
    ):
        return list(_CONFIG_CACHE[1])

    data = json.loads(cfg.read_text(encoding="utf-8"))
    meta = dict(data.get("_meta") or {})
    if meta.get("production_approved") is False:
        logger.info(
            "design_components loaded (agent_reviewed, production_approved=false) path=%s",
            cfg,
        )
    specs: list[ComponentSpec] = []
    for row in data.get("components") or []:
        aliases = [str(a).strip().lower() for a in (row.get("aliases") or []) if str(a).strip()]
        specs.append(
            ComponentSpec(
                id=str(row.get("id") or "").strip(),
                label=str(row.get("label") or row.get("id") or "").strip(),
                aliases=aliases,
                concepts=[str(c).strip() for c in (row.get("concepts") or []) if str(c).strip()],
                likely_regulations=[
                    str(r).strip()
                    for r in (row.get("likely_regulations") or [])
                    if str(r).strip()
                ],
                retrieval_queries=[
                    str(q).strip()
                    for q in (row.get("retrieval_queries") or [])
                    if str(q).strip()
                ],
                notes=str(row.get("notes") or "").strip(),
            )
        )
    _CONFIG_CACHE = (mtime, specs, cfg)
    return list(specs)


def match_component(question: str, *, specs: Sequence[ComponentSpec] | None = None) -> ComponentSpec | None:
    """Longest-alias wins so 'driver's seat' beats bare 'seat' if both exist."""
    q = (question or "").lower()
    if not q:
        return None
    best: ComponentSpec | None = None
    best_len = -1
    for spec in specs or load_design_components():
        for alias in spec.aliases:
            if not alias:
                continue
            # Word-ish boundary for short aliases; substring OK for hyphenated.
            if len(alias) <= 4:
                if not re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
                    continue
            elif alias not in q:
                continue
            if len(alias) > best_len:
                best = spec
                best_len = len(alias)
    return best


def expand_design_query(question: str) -> DesignExpansion:
    """Map design element → regulatory concepts + subqueries BEFORE retrieval."""
    from retrieval.enumerative import detect_named_regulation, resolve_hard_regulation_filter

    q = (question or "").strip()
    cfg_path = str(design_components_path())
    specs = load_design_components()
    component = match_component(q, specs=specs)
    named = resolve_hard_regulation_filter(q) or detect_named_regulation(q)

    if component is None:
        # Soft fallback: vehicle_design_general if "design" / "requirements affect"
        for spec in specs:
            if spec.id == "vehicle_design_general":
                component = spec
                break

    concepts = list(component.concepts) if component else []
    likely = list(component.likely_regulations) if component else []
    subqueries: list[str] = []
    if component:
        subqueries.extend(component.retrieval_queries)
        # Always include a concept-joined probe.
        if concepts:
            subqueries.append(" ".join(concepts[:4]))
        subqueries.append(f"{component.label} requirements {q}")
    if not subqueries:
        subqueries = [q]

    # Dedupe while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for sq in subqueries:
        key = sq.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(sq.strip())

    expansion = DesignExpansion(
        question=q,
        component=component,
        concepts=concepts,
        likely_regulations=likely,
        named_regulation_id=named,
        subqueries=uniq,
        config_path=cfg_path,
    )
    logger.info(
        "design_expansion component=%s concepts=%d likely=%s named=%s subqueries=%d",
        component.id if component else None,
        len(concepts),
        likely,
        named,
        len(uniq),
    )
    return expansion


def per_regulation_design_top_k() -> int:
    try:
        return max(1, int((os.getenv("DESIGN_PER_REG_TOP_K") or str(DEFAULT_PER_REG_TOP_K)).strip()))
    except ValueError:
        return DEFAULT_PER_REG_TOP_K


def _target_regulation_ids(expansion: DesignExpansion, *, indexed: Sequence[str]) -> list[str]:
    indexed_set = {r for r in indexed if r}
    if expansion.named_regulation_id and expansion.named_regulation_id in indexed_set:
        return [expansion.named_regulation_id]
    # Prefer likely regs only (Fix 13) — surveying every indexed reg × subqueries
    # balloons tokens/latency. Missing preferred regs are reported via ``missing``.
    preferred = [r for r in expansion.likely_regulations if r in indexed_set]
    if preferred:
        return preferred
    return sorted(indexed_set)


def retrieve_design_implication(
    query: str,
    *,
    top_k_per_reg: int | None = None,
    llm: object | None = None,
    rewrite_result: object | None = None,
    do_rerank: bool = True,
    client: object | None = None,
    embedder: object | None = None,
    collection: str | None = None,
    expansion: DesignExpansion | None = None,
) -> DesignRetrievalResult:
    """Broad multi-reg retrieval scoped by expanded design concepts."""
    from retrieval.multi_regulation import (
        MultiRegulationResult,
        chunk_matches_topic,
        topic_terms,
    )
    from retrieval.retrieve import get_indexed_regulations, hybrid_search, retrieve

    expansion = expansion or expand_design_query(query)
    per_k = top_k_per_reg or per_regulation_design_top_k()
    regs = [
        r.regulation_id
        for r in get_indexed_regulations(
            client=client,  # type: ignore[arg-type]
            collection=collection,
        )
    ]
    regs = _target_regulation_ids(expansion, indexed=regs)
    if not regs:
        logger.warning("design_implication retrieve: no indexed regulations")
        return DesignRetrievalResult(
            chunks=[], covered=[], missing=[], expansion=expansion
        )

    # Topic terms from concepts + original question (not just surface "B-pillar").
    topic_blob = " ".join(
        [expansion.question]
        + list(expansion.concepts)
        + ([expansion.component.label] if expansion.component else [])
    )
    terms = topic_terms(topic_blob)
    # Soften: also allow concept keywords as terms.
    for concept in expansion.concepts:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", concept.lower()):
            if tok not in terms and tok not in {
                "the",
                "and",
                "for",
                "under",
                "with",
                "from",
            }:
                terms.append(tok)

    covered: list[str] = []
    missing: list[str] = []
    merged: list = []
    per_reg: dict[str, list] = {}
    seen_ids: set[str] = set()

    subqueries = list(expansion.subqueries) or [query]

    for rid in regs:
        # Prefer concept subqueries; fall back to condensed hybrid retrieve.
        hits: list = []
        for sq in subqueries[:4]:
            try:
                batch = hybrid_search(
                    sq,
                    top_k=max(per_k * 2, 10),
                    regulation_id=rid,
                    client=client,  # type: ignore[arg-type]
                    embedder=embedder,  # type: ignore[arg-type]
                    collection=collection,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("design hybrid_search failed reg=%s: %s", rid, exc)
                batch = []
            for c in batch:
                cid = getattr(c, "chunk_id", None) or ""
                if cid and cid in seen_ids:
                    continue
                hits.append(c)
        if not hits:
            # Full retrieve path as fallback (rerank) for this corpus.
            try:
                hits = retrieve(
                    subqueries[0],
                    top_k=per_k,
                    regulation_id=rid,
                    llm=llm,
                    rewrite=False,
                    rewrite_result=rewrite_result,
                    do_rerank=do_rerank,
                    small_to_big=False,
                    client=client,  # type: ignore[arg-type]
                    embedder=embedder,  # type: ignore[arg-type]
                    collection=collection,
                    history=None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("design retrieve fallback failed reg=%s: %s", rid, exc)
                hits = []

        # Soft topic filter — keep concept-relevant; if none, keep top per_k
        # so we still have grounding material (design concepts may be abstract).
        relevant = [c for c in hits if chunk_matches_topic(c, terms)]
        chosen = (relevant or hits)[:per_k]
        per_reg[rid] = list(chosen)
        if relevant:
            covered.append(rid)
        else:
            # Named-reg only ask: still "covered" if we got any hits.
            if expansion.named_regulation_id == rid and chosen:
                covered.append(rid)
            else:
                missing.append(rid)
        for c in chosen:
            cid = getattr(c, "chunk_id", None) or ""
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(c)

    # Prefer likely regulations first in the merged list.
    prefer = set(expansion.likely_regulations)
    if prefer and not expansion.named_regulation_id:
        merged.sort(
            key=lambda c: (
                0 if (getattr(c, "regulation_id", "") or "") in prefer else 1,
                -float(getattr(c, "score", 0.0) or 0.0),
            )
        )

    logger.info(
        "design_implication retrieved regs_covered=%s missing=%s chunks=%d component=%s",
        covered,
        missing,
        len(merged),
        expansion.component.id if expansion.component else None,
    )
    # Record multi-reg coverage on trace for answer augmentation.
    try:
        from retrieval.multi_regulation import record_multi_regulation_on_trace

        record_multi_regulation_on_trace(
            MultiRegulationResult(
                chunks=merged,
                covered=covered,
                missing=missing,
            )
        )
    except Exception:  # noqa: BLE001
        pass

    return DesignRetrievalResult(
        chunks=merged,
        covered=covered,
        missing=missing,
        expansion=expansion,
        per_regulation=per_reg,
    )


# --- Per-claim grounding + FACT / INFERENCE rendering -----------------------


def normalize_claim_kind(raw: str | None) -> str:
    k = (raw or "").strip().upper().replace(" ", "_")
    if k in {"FACT", "REGULATORY", "REQUIREMENT", "REG"}:
        return CLAIM_FACT
    if k in {"INFERENCE", "ENGINEERING", "IMPLICATION", "DESIGN_INFERENCE"}:
        return CLAIM_INFERENCE
    if k in _VALID_KINDS:
        return k
    return CLAIM_FACT  # safer default: treat as fact so engineers scrutinize citations


def keep_grounded_design_segments(
    segments: Sequence[Any],
    allowed_ids: set[str],
) -> tuple[list[Any], list[str]]:
    """Per-claim grounding: keep segments with valid chunk_ids; drop the rest.

    Unlike all-or-nothing validate_segment_chunk_ids, a multi-clause design
    answer remains valid when every *kept* claim is grounded — even if some
    model claims cited bad ids.
    """
    kept: list[Any] = []
    dropped: list[str] = []
    for seg in segments:
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        if cid and cid in allowed_ids:
            # Normalize claim_kind onto the segment when present as attribute.
            kind = normalize_claim_kind(getattr(seg, "claim_kind", None))
            if hasattr(seg, "claim_kind"):
                try:
                    seg.claim_kind = kind  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            kept.append(seg)
        else:
            dropped.append(cid or "(empty)")
    return kept, dropped


def render_design_answer(
    segments: Sequence[Any],
    chunks_by_id: dict[str, Any],
    *,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Render FACT vs INFERENCE labels; chips from chunk metadata only."""
    from generation.answer import _CITATION_CHIP_RE

    fact_parts: list[str] = []
    inf_parts: list[str] = []
    sources: list[Any] = []
    seen: set[str] = set()

    for seg in segments:
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        chip = chunk.citation_tag()
        body = (getattr(seg, "text", None) or "").strip()
        body = _CITATION_CHIP_RE.sub("", body).strip()
        kind = normalize_claim_kind(getattr(seg, "claim_kind", None))
        if kind == CLAIM_INFERENCE:
            line = f"[Engineering inference] {body} {chip}".strip()
            inf_parts.append(line)
        else:
            line = f"[Regulatory fact] {body} {chip}".strip()
            fact_parts.append(line)
        if cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))

    blocks: list[str] = []
    if fact_parts:
        blocks.append("Regulatory facts\n" + "\n\n".join(fact_parts))
    if inf_parts:
        blocks.append("Engineering inferences\n" + "\n\n".join(inf_parts))
    return "\n\n".join(blocks).strip(), sources


def design_answer_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "design_implication_answer",
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
                                    "enum": [CLAIM_FACT, CLAIM_INFERENCE],
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


def extractive_design_fallback(
    *,
    question: str,
    chunks: Sequence[Any],
    expansion: DesignExpansion | None = None,
    to_source: Any,
    max_facts: int = 8,
) -> tuple[str, list[Any]]:
    """Deterministic multi-claim sketch when the LLM abstains — FACT-only."""
    from generation.answer import AnswerSegment

    expansion = expansion or expand_design_query(question)
    label = expansion.component.label if expansion.component else "this design element"
    segs: list[AnswerSegment] = []
    for c in chunks[:max_facts]:
        cid = (getattr(c, "chunk_id", None) or "").strip()
        if not cid:
            continue
        rid = getattr(c, "regulation_id", "") or ""
        sec = getattr(c, "section_number", "") or ""
        snippet = (getattr(c, "text", None) or "").strip()
        # One short sentence from the chunk — truncate hard.
        words = snippet.split()
        excerpt = " ".join(words[:40]) + ("…" if len(words) > 40 else "")
        text = (
            f"{rid} §{sec} addresses requirements relevant to {label}: {excerpt}"
            if sec
            else f"{rid} addresses requirements relevant to {label}: {excerpt}"
        )
        segs.append(
            AnswerSegment(
                text=text,
                citation_chunk_id=cid,
                claim_kind=CLAIM_FACT,
            )
        )
    if not segs:
        return "", []
    by_id = {c.chunk_id: c for c in chunks if getattr(c, "chunk_id", None)}
    return render_design_answer(segs, by_id, to_source=to_source)
