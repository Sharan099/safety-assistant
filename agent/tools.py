"""Grounded agent tools — every factual output carries citations from retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from agent.citations import enforce_grounded_text, extract_citations
from agent.regs import is_indexed, normalize_regulation
from generation.answer import SourceChunk, to_source
from generation.llm_client import LLMClient
from retrieval.retrieve import RetrievedChunk, format_context, format_indexed_regulations_label, retrieve

logger = logging.getLogger(__name__)


def _not_indexed_msg(reg: str) -> str:
    live = format_indexed_regulations_label()
    return (
        f"Regulation {reg} is not in the local index "
        f"(available: {live}). "
        "Ingest or upload the PDF first, or compare against an indexed regulation."
    )


def _dedupe_sources(chunks: Sequence[RetrievedChunk]) -> list[SourceChunk]:
    seen: set[str] = set()
    out: list[SourceChunk] = []
    for c in chunks:
        key = c.chunk_id or c.citation_tag()
        if key in seen:
            continue
        seen.add(key)
        out.append(to_source(c))
    return out


def tool_retrieve(
    regulation: str | None,
    query: str,
    *,
    top_k: int | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Retrieve cited passages for a regulation + query."""
    query = (query or "").strip()
    rid = normalize_regulation(regulation)
    if rid and not is_indexed(rid):
        return {
            "ok": False,
            "not_found": True,
            "regulation_id": rid,
            "query": query,
            "text": _not_indexed_msg(rid),
            "citations": [],
            "chunk_ids": [],
            "sources": [],
            "chunks": [],
        }
    if not query:
        return {
            "ok": False,
            "not_found": True,
            "text": "Empty retrieve query.",
            "citations": [],
            "chunk_ids": [],
            "sources": [],
            "chunks": [],
        }

    chunks = retrieve(query, top_k=top_k, regulation_id=rid, llm=llm)
    sources = _dedupe_sources(chunks)
    citations = [s.citation for s in sources if s.citation]
    if not chunks:
        return {
            "ok": False,
            "not_found": True,
            "regulation_id": rid,
            "query": query,
            "text": (
                f"No passages found for query={query!r}"
                + (f" in {rid}" if rid else "")
                + "."
            ),
            "citations": [],
            "chunk_ids": [],
            "sources": [],
            "chunks": [],
        }

    # Extractive grounded summary: bullet each passage with its own citation (no free prose).
    lines = [f"### Retrieved: {rid or 'all regs'} — {query}", ""]
    for s in sources[:8]:
        preview = " ".join((s.text or "").split())[:280]
        lines.append(f"- {preview} {s.citation}")
    text = "\n".join(lines)
    text, _ = enforce_grounded_text(text, allowed=citations)
    return {
        "ok": True,
        "not_found": False,
        "regulation_id": rid,
        "query": query,
        "text": text,
        "citations": citations,
        "chunk_ids": [s.chunk_id for s in sources],
        "sources": [s.model_dump() for s in sources],
        "chunks": chunks,
    }


def tool_lookup_table(
    regulation: str,
    criterion: str,
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Prefer table/criterion chunks for a named injury criterion or limit."""
    rid = normalize_regulation(regulation)
    criterion = (criterion or "").strip()
    query = f"{criterion} limit table performance criteria requirements"
    result = tool_retrieve(rid, query, top_k=8, llm=llm)
    if not result.get("ok"):
        return result

    chunks: list[RetrievedChunk] = list(result.get("chunks") or [])
    # Prefer table content_type, then passages mentioning the criterion.
    crit_re = re.compile(re.escape(criterion), re.I) if criterion else None

    def score(c: RetrievedChunk) -> tuple[int, float]:
        text = c.text or c.enriched_text or ""
        table_bonus = 2 if (c.content_type or "").lower() == "table" else 0
        hit = 1 if (crit_re and crit_re.search(text)) else 0
        return (table_bonus + hit, float(c.score or 0.0))

    ranked = sorted(chunks, key=score, reverse=True)
    sources = _dedupe_sources(ranked[:5])
    citations = [s.citation for s in sources if s.citation]
    lines = [f"### Table / criterion lookup: {rid} — {criterion}", ""]
    for s in sources:
        body = " ".join((s.text or "").split())[:400]
        lines.append(f"| {s.section_number or '?'} | {body} | {s.citation} |")
    header = "| Section | Excerpt | Citation |\n|---|---|---|"
    text = "\n".join([lines[0], "", header, *lines[2:]])
    text, _ = enforce_grounded_text(text, allowed=citations)
    return {
        "ok": True,
        "not_found": False,
        "regulation_id": rid,
        "criterion": criterion,
        "text": text,
        "table_markdown": text,
        "citations": citations,
        "chunk_ids": [s.chunk_id for s in sources],
        "sources": [s.model_dump() for s in sources],
        "chunks": ranked[:5],
    }


def tool_compare_regulations(
    reg_a: str,
    reg_b: str,
    topic: str,
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Pull clauses for the same topic from two regulations → cited comparison table."""
    topic = (topic or "").strip()
    a = normalize_regulation(reg_a)
    b = normalize_regulation(reg_b)
    left = tool_retrieve(a, topic, llm=llm)
    right = tool_retrieve(b, topic, llm=llm)

    all_chunks: list[RetrievedChunk] = []
    all_chunks.extend(left.get("chunks") or [])
    all_chunks.extend(right.get("chunks") or [])
    sources = _dedupe_sources(all_chunks)
    citations = [s.citation for s in sources if s.citation]

    def _cell(result: dict[str, Any]) -> str:
        if result.get("not_found") or not result.get("ok"):
            msg = (result.get("text") or "Not found.").replace("\n", " ").strip()
            return f"Note: {msg[:280]}"
        for line in (result.get("text") or "").splitlines():
            line = line.strip()
            if line.startswith("- "):
                return line[2:]
        # Fallback: first non-header line with a citation
        for line in (result.get("text") or "").splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            return line[:400]
        return "Note: no excerpt."

    rows = [
        f"| Regulation | {topic} (cited excerpt) |",
        "|---|---|",
        f"| {a or reg_a} | {_cell(left)} |",
        f"| {b or reg_b} | {_cell(right)} |",
    ]
    table = "\n".join(rows)

    # Optional synthesis row — only from evidence, via extractive notes
    notes: list[str] = []
    if left.get("ok") and right.get("ok"):
        notes.append(
            f"Note: both {a} and {b} returned indexed passages for {topic!r}; "
            "compare numeric limits only where each cell cites a clause."
        )
    elif left.get("ok") and not right.get("ok"):
        notes.append(
            f"Note: {a} has indexed evidence; {b} does not — ingest {b} before closing the gap."
        )
    elif right.get("ok") and not left.get("ok"):
        notes.append(
            f"Note: {b} has indexed evidence; {a} does not — ingest {a} before closing the gap."
        )
    else:
        notes.append("Note: neither regulation returned grounded passages for this topic.")

    text = f"### Compliance comparison: {topic}\n\n{table}\n\n" + "\n".join(notes)
    # Notes are meta (no numbers) — citation gate on table cells already embeds cites
    return {
        "ok": bool(left.get("ok") or right.get("ok")),
        "not_found": not (left.get("ok") or right.get("ok")),
        "reg_a": a,
        "reg_b": b,
        "topic": topic,
        "text": text,
        "table_markdown": table,
        "citations": citations,
        "chunk_ids": [s.chunk_id for s in sources],
        "sources": [s.model_dump() for s in sources],
        "chunks": all_chunks,
        "side_a": {k: left[k] for k in ("ok", "not_found", "text", "citations") if k in left},
        "side_b": {k: right[k] for k in ("ok", "not_found", "text", "citations") if k in right},
    }


def tool_draft_report(
    sections: Sequence[dict[str, Any]] | Sequence[str],
    *,
    title: str = "Engineering memo",
    llm: LLMClient | None = None,
    evidence_sources: Sequence[SourceChunk] | None = None,
    evidence_chunks: Sequence[RetrievedChunk] | None = None,
) -> dict[str, Any]:
    """Draft a structured, fully-cited memo from provided section briefs + evidence only."""
    client = llm or LLMClient()
    # Normalize section briefs
    briefs: list[dict[str, str]] = []
    for s in sections or []:
        if isinstance(s, str):
            briefs.append({"heading": s, "focus": s})
        elif isinstance(s, dict):
            briefs.append(
                {
                    "heading": str(s.get("heading") or s.get("title") or "Section"),
                    "focus": str(s.get("focus") or s.get("query") or s.get("heading") or ""),
                }
            )

    chunks: list[RetrievedChunk] = list(evidence_chunks or [])
    sources = list(evidence_sources or [])
    if not sources and chunks:
        sources = _dedupe_sources(chunks)

    # If no evidence yet, retrieve per section focus (grounding first)
    if not chunks:
        for b in briefs:
            r = tool_retrieve(None, b["focus"] or b["heading"], llm=client)
            chunks.extend(r.get("chunks") or [])
        sources = _dedupe_sources(chunks)

    citations = [s.citation for s in sources if s.citation]
    from retrieval.context_budget import apply_context_budget

    budgeted, _stats = apply_context_budget(chunks, question=task or "agent synthesize")
    context = format_context(budgeted) if budgeted else "(no evidence)"

    section_list = "\n".join(f"- {b['heading']}: {b['focus']}" for b in briefs) or "- Findings"

    system = """\
You draft a short engineering memo from UNECE regulation evidence ONLY.
Rules:
1. Use ONLY facts present in the Context passages.
2. Every factual sentence MUST end with a citation of the form [regulation_id §section, p.page].
3. If evidence is missing for a section, write exactly: Insufficient evidence in indexed regulations. [no citation]
4. Do not invent limits, clause numbers, or pages.
5. Structure: Title, then ## headings for each requested section, then ## Sources.
"""
    user = (
        f"Title: {title}\n\nRequested sections:\n{section_list}\n\n"
        f"Context passages:\n{context}\n\n"
        "Write the memo now."
    )

    if client.provider == "mock" or not chunks:
        # Deterministic extractive memo — never free-hallucinate in mock.
        lines = [f"# {title}", ""]
        for b in briefs:
            lines.append(f"## {b['heading']}")
            # Pick best matching source
            focus = (b["focus"] or b["heading"]).lower()
            picked = None
            for s in sources:
                blob = f"{s.section_title} {s.text}".lower()
                if any(tok in blob for tok in focus.split() if len(tok) > 3):
                    picked = s
                    break
            if picked is None and sources:
                picked = sources[0]
            if picked:
                preview = " ".join((picked.text or "").split())[:320]
                lines.append(f"{preview} {picked.citation}")
            else:
                lines.append("Insufficient evidence in indexed regulations.")
            lines.append("")
        lines.append("## Sources")
        for s in sources[:10]:
            lines.append(f"- {s.citation} — {s.section_title or s.section_number}")
        text = "\n".join(lines)
    else:
        result = client.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            role="answer",
            question=title,
            chunk_ids=[c.chunk_id for c in chunks],
        )
        text = result.text

    text, checks = enforce_grounded_text(text, allowed=citations)
    return {
        "ok": bool(sources),
        "not_found": not bool(sources),
        "text": text,
        "report_markdown": text,
        "citations": citations or extract_citations(text),
        "chunk_ids": [s.chunk_id for s in sources],
        "sources": [s.model_dump() for s in sources],
        "chunks": chunks,
        "claim_checks": [c.model_dump() for c in checks],
        "title": title,
    }


TOOL_SPECS = {
    "retrieve": {
        "description": "Retrieve cited passages for a regulation + query.",
        "args": ["regulation", "query"],
    },
    "compare_regulations": {
        "description": "Compare the same topic across two regulations with a cited table.",
        "args": ["reg_a", "reg_b", "topic"],
    },
    "lookup_table": {
        "description": "Look up a criterion/limit table inside a regulation.",
        "args": ["regulation", "criterion"],
    },
    "draft_report": {
        "description": "Draft a fully-cited engineering memo from section briefs + evidence.",
        "args": ["sections", "title"],
    },
    "design_implication": {
        "description": "Layer 3–5: component→concept expansion + multi-reg design synthesis.",
        "args": ["query"],
    },
    "applicability": {
        "description": "Layer 3–5: survey Scope of every indexed regulation for a vehicle.",
        "args": ["query"],
    },
    "checklist_gen": {
        "description": "Layer 3–5: per-category homologation / test-prep checklist.",
        "args": ["query"],
    },
    "retest_scope": {
        "description": "Layer 3–5: change-impact clauses (informational — verify with authority).",
        "args": ["query"],
    },
}


def _answer_as_tool_result(query: str, *, llm: LLMClient | None = None) -> dict[str, Any]:
    """Run the hybrid chat pipeline and reshape into an agent tool result."""
    from generation.answer import answer_question

    ans = answer_question(
        query,
        llm=llm or LLMClient(),
        skip_answer_cache=True,
    )
    sources = list(ans.sources or [])
    citations = [s.citation for s in sources if s.citation]
    text = ans.answer or ""
    if ans.mode_disclaimer:
        text = f"{ans.mode_disclaimer_title or 'Note'}: {ans.mode_disclaimer}\n\n{text}"
    return {
        "ok": not ans.not_found,
        "not_found": ans.not_found,
        "text": text,
        "citations": citations,
        "chunk_ids": [s.chunk_id for s in sources if s.chunk_id],
        "sources": [s.model_dump() for s in sources],
        "chunks": [],
        "query_intent": ans.query_intent,
        "execution_layer": ans.execution_layer,
        "multi_step": ans.multi_step,
        "mode_disclaimer": ans.mode_disclaimer,
        "mode_disclaimer_title": ans.mode_disclaimer_title,
        "failure_kind": ans.failure_kind,
    }


def tool_design_implication(query: str, *, llm: LLMClient | None = None) -> dict[str, Any]:
    return _answer_as_tool_result(query, llm=llm)


def tool_applicability(query: str, *, llm: LLMClient | None = None) -> dict[str, Any]:
    return _answer_as_tool_result(query, llm=llm)


def tool_checklist_gen(query: str, *, llm: LLMClient | None = None) -> dict[str, Any]:
    return _answer_as_tool_result(query, llm=llm)


def tool_retest_scope(query: str, *, llm: LLMClient | None = None) -> dict[str, Any]:
    """Change-impact analysis — always informational; never an authoritative decision."""
    return _answer_as_tool_result(query, llm=llm)
