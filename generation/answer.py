"""Grounded Q&A: retrieve → structured segments → backend-rendered citations."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field, ValidationError

from generation.llm_client import LLMClient, LLMResult
from generation.prompt_cache import prompt_cache_lookup
from observability.context import reset_current_trace, set_current_trace
from observability.trace import QueryTrace, new_trace
from retrieval.context_budget import apply_context_budget
from retrieval.retrieve import RetrievedChunk, format_context, retrieve

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
GROUNDEDNESS_VIOLATIONS_PATH = ROOT / "groundedness_violations.jsonl"

# Exactly two content-miss failure states, plus numeric_hallucination (reject
# wrong pass/fail that altered the user's measured figures).
FAILURE_RETRIEVAL_MISS = "retrieval_miss"
FAILURE_GROUNDING_REJECTED = "grounding_rejected"
FAILURE_NUMERIC_HALLUCINATION = "numeric_hallucination"

UNVERIFIED_CITATION = "I couldn't verify this citation"  # legacy alias; prefer grounding_rejected_message

SYSTEM_PROMPT = """\
You are a UNECE passive-safety regulation assistant.

You MUST reply with a single JSON object matching this schema:
{
  "answer_segments": [
    { "text": "<factual claim without any section/page/citation markup>",
      "citation_chunk_id": "<exact chunk_id from a provided passage>" }
  ]
}

STRICT RULES:
1. Answer ONLY from the provided context passages. Do not use outside knowledge.
2. Each factual claim is one answer_segment that cites EXACTLY ONE retrieved
   chunk via citation_chunk_id (the chunk_id value shown on the passage).
3. Do NOT write section numbers, page numbers, regulation citation chips, or
   bracketed citations in "text". The backend attaches citations from metadata.
4. citation_chunk_id MUST be copied verbatim from a provided chunk_id.
   Never invent ids or use section numbers as ids.
5. If the answer is not supported by the context, return:
   {"answer_segments": []}
6. Prefer concise, precise answers. Preserve limits and units from the text.
7. If the question asks for the regulation's objective, scope, or purpose, answer
   in ONE short factual sentence from the Scope article passage only — do not
   hedge or synthesize multi-paragraph background.
8. VALUE-VS-LIMIT: when the user gives a measured number and asks pass/fail /
   comply / satisfy against a regulation, you MUST (a) cite the injury-criterion
   LIMIT clause (performance criteria / shall not exceed / ≤), (b) state the
   limit, (c) compare the measured value to it, (d) say PASS or FAIL. Never treat
   ISO 6487 / CFC / channel filtering / calibration as the injury limit. If the
   limit clause is missing from context, return {"answer_segments": []}.
   Treat HIC / HIC15 as the same family as HPC (Head Performance Criterion) when
   the passages state an HPC limit — answer with the HPC figure and cite that chunk.
9. ENUMERATIVE: when the user asks to list/every/summarize all requirements,
   cover each distinct requirement found in the passages with its own segment
   and citation — do not collapse multiple requirements into a single vague claim.
10. NEGATIVE / ABSENCE CLAIMS: Do NOT assert that something is absent, unrelated,
   not covered, or that "no relationship / no requirement exists" unless a
   provided passage EXPLICITLY states that absence or non-applicability.
   If the passages are merely silent on the asked relationship or topic, return
   {"answer_segments": []} so the backend can say the content was not found /
   not addressed — never invent a confident negative as if it were a regulation fact.
11. MULTI-REGULATION SURVEY: when the question asks which / which regulations /
   across regulations / in general (no single named regulation), address EACH
   regulation that has supporting passages with its own answer_segment and
   citation. Do not collapse to a single regulation. Do not invent absence
   claims for other regulations — the backend states which indexed regulations
   lacked relevant content.
12. COMPLIANCE / "does it comply": Prefer the deterministic Fix 22 path
   (overall verdict + exact measured/limit numbers; LLM only phrases around
   them). If emitting segments instead, you MUST end with an explicit verdict —
   PASS, FAIL, or state that compliance cannot be determined from the indexed
   passages. Never answer with only a measurement procedure or definition and
   no conclusion. Never invent or alter numeric limits.
13. PARTIAL ANSWERS: If the passages support a partial but defensible answer
   (e.g. the limit is present under a synonymous name, or isolation-resistance
   minima appear in the cited clauses), return those grounded segments — do not
   abstain solely because the wording differs slightly from the question.
   Do NOT invent synonym mappings for criterion names that never appear in the
   passages (e.g. do not treat an unknown "Soft Tissue Criterion" as VC/HPC).
"""

COMPLIANCE_VERDICT_USER_INSTRUCTION = """\
COMPLIANCE QUESTION — required conclusion (Fix 22 deterministic structure):
The user asks whether the vehicle complies / passes. Prefer the backend
deterministic path (overall verdict + measured/limit lines). If you must emit
answer_segments, each must preserve exact numbers from the passages and end
with an explicit verdict (PASS or FAIL) grounded in a requirement clause,
OR state that compliance cannot be determined from the indexed passages.
Do not stop at describing a test/measurement procedure without a conclusion.
Do not invent or alter numeric limits — phrase only around verified values.
"""

MULTI_REGULATION_USER_INSTRUCTION = """\
MULTI-REGULATION SURVEY — Context spans multiple regulations on purpose.
Emit a separate answer_segment (with citation_chunk_id) for EACH regulation that
has relevant passages. Name the regulation in the text (e.g. UN R94, UN R95).
Do not omit a regulation that appears in the passages. Do not claim that a
regulation lacks content — the backend adds that separately.
"""


RETRY_REMINDER = """\
IMPORTANT: Your previous response cited a citation_chunk_id that was NOT in the
retrieved chunk_id set. Return JSON again. Every citation_chunk_id MUST be one of:
{allowed}
If you cannot cite a valid id, return {{"answer_segments": []}}.
"""

ANSWER_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_answer",
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
                                "description": (
                                    "REGULATORY_FACT or ENGINEERING_INFERENCE "
                                    "(required for design-implication answers)"
                                ),
                            },
                        },
                        "required": ["text", "citation_chunk_id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["answer_segments"],
            "additionalProperties": False,
        },
    },
}

# Inline chip written by the backend (never by the model).
_CITATION_CHIP_RE = re.compile(
    r"\[([^\]]+?)\s*§([^,\]]+)\s*,\s*p\.(\d+|\?)\]"
)
_SECTION_MARK_RE = re.compile(r"§\s*([0-9A-Za-z./_-]+)")


class AnswerSegment(BaseModel):
    text: str = ""
    citation_chunk_id: str = ""
    # DESIGN_IMPLICATION: REGULATORY_FACT | ENGINEERING_INFERENCE (optional elsewhere).
    claim_kind: str = ""
    # CHECKLIST_GEN: category id from config/checklist_categories.json.
    category_id: str = ""


class StructuredAnswer(BaseModel):
    answer_segments: list[AnswerSegment] = Field(default_factory=list)


class SourceChunk(BaseModel):
    """Source chunk returned to the caller (page + bbox for grounding UI)."""

    chunk_id: str
    regulation_id: str = ""
    revision: str = ""
    section_number: str = ""
    section_title: str = ""
    page_number: int | None = None
    bounding_box: list[float] = Field(default_factory=list)
    content_type: str = "clause"
    text: str = ""
    score: float = 0.0
    citation: str = ""


class AnswerResponse(BaseModel):
    """Structured answer + sources (Phase 1 contract)."""

    question: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    # Portkey target index for the answer-role completion (0 = primary).
    target_index: int | None = None
    answering_provider_was_fallback: bool = False
    cached: bool = False
    answer_cached: bool = False
    served_from_cache: bool = False
    cache_hit_kind: str = ""
    not_found: bool = False
    # Exclusive failure taxonomy:
    # None | retrieval_miss | grounding_rejected | numeric_hallucination
    failure_kind: str | None = None
    trace_id: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str = ""
    condensed_question: str = ""
    condensation_applied: bool = False
    # Deterministic compliance payload (numbers as separate fields for the UI).
    compliance: dict[str, Any] | None = None
    # Intent + mode-specific honesty banner (e.g. RETEST_SCOPE).
    query_intent: str | None = None
    mode_disclaimer: str | None = None
    mode_disclaimer_title: str | None = None
    # Hybrid router: fast (Layer 1–2) vs multi_step (Layer 3–5).
    execution_layer: str | None = None
    multi_step: bool = False

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


def retrieval_miss_message(indexed: Sequence[Any] | None = None) -> str:
    """RETRIEVAL_MISS — no usable chunks for this question."""
    from retrieval.retrieve import (
        IndexedRegulation,
        format_indexed_regulations_label,
        get_indexed_regulations,
    )

    if indexed is None:
        rows: list[IndexedRegulation] = get_indexed_regulations()
    else:
        rows = list(indexed)  # type: ignore[arg-type]
    labels = format_indexed_regulations_label(rows)
    return (
        f"I couldn't find relevant content on this in the indexed regulations ({labels}). "
        "Try rephrasing, specifying a clause number, or upload the relevant regulation "
        "if it isn't listed."
    )


def grounding_rejected_message(
    closest: RetrievedChunk | SourceChunk | None = None,
) -> str:
    """GROUNDING_REJECTED — chunks existed but claims could not be grounded."""
    if closest is None:
        pointer = "(no citation available)"
    elif isinstance(closest, SourceChunk):
        pointer = (closest.citation or closest.chunk_id or "").strip() or "(no citation available)"
    else:
        pointer = (closest.citation_tag() or closest.chunk_id or "").strip() or (
            "(no citation available)"
        )
    return (
        "I found related content but couldn't produce a confidently grounded answer — "
        f"here's the closest relevant section I found: {pointer}, "
        "you may want to check it directly."
    )


def not_found_in_regulations_message(
    indexed: Sequence[Any] | None = None,
) -> str:
    """Backward-compatible alias for ``retrieval_miss_message``."""
    return retrieval_miss_message(indexed=indexed)


def closest_retrieved_chunk(
    chunks: Sequence[RetrievedChunk],
    *,
    question: str = "",
) -> RetrievedChunk | None:
    """Highest-scoring retrieved chunk (stable tie-break on chunk_id).

    For enumerative topic asks (e.g. doors), prefer a topic-matching chunk so a
    grounding-rejected fallback never cites an unrelated preamble / foreign leaf.
    """
    if not chunks:
        return None
    if question:
        try:
            from retrieval.enumerative import (
                chunk_matches_enumerative_topic,
                extract_enumerative_topic,
                is_enumerative_query,
            )

            if is_enumerative_query(question):
                topic = extract_enumerative_topic(question)
                topical = [
                    c for c in chunks if chunk_matches_enumerative_topic(c, topic)
                ]
                if topical:
                    return max(
                        topical,
                        key=lambda c: (float(c.score or 0.0), c.chunk_id or ""),
                    )
        except Exception:  # noqa: BLE001
            pass
    return max(
        chunks,
        key=lambda c: (float(c.score or 0.0), c.chunk_id or ""),
    )


def to_source(chunk: RetrievedChunk) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk.chunk_id,
        regulation_id=chunk.regulation_id,
        revision=chunk.revision,
        section_number=chunk.section_number,
        section_title=chunk.section_title,
        page_number=chunk.page_number,
        bounding_box=list(chunk.bounding_box),
        content_type=chunk.content_type,
        text=chunk.text,
        score=chunk.score,
        citation=chunk.citation_tag(),
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _is_insufficient(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return (
        a.startswith("the provided context does not contain enough information")
        or "does not contain enough information" in a
        or a.startswith("i couldn't find relevant content on this")
        or a.startswith("i could not find this in the indexed regulations")
        or a.startswith("i found related content but couldn't produce")
        or a.startswith("i want to make sure i use your exact figures")
        or a == UNVERIFIED_CITATION.lower()
    )


def _failure_kind_from_answer(answer: str) -> str | None:
    """Infer exclusive failure_kind from cached/legacy answer text."""
    a = (answer or "").strip().lower()
    if a.startswith("i want to make sure i use your exact figures"):
        return FAILURE_NUMERIC_HALLUCINATION
    if a.startswith("i found related content but couldn't produce"):
        return FAILURE_GROUNDING_REJECTED
    if a == UNVERIFIED_CITATION.lower():
        return FAILURE_GROUNDING_REJECTED
    if (
        a.startswith("i couldn't find relevant content on this")
        or a.startswith("i could not find this in the indexed regulations")
    ):
        return FAILURE_RETRIEVAL_MISS
    return None


def _citations_for_trace(sources: list[SourceChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": s.chunk_id,
            "regulation_id": s.regulation_id,
            "section_number": s.section_number,
            "page_number": s.page_number,
            "citation": s.citation,
            "score": s.score,
        }
        for s in sources
    ]


def parse_structured_answer(raw: str) -> StructuredAnswer:
    """Parse LLM JSON into answer segments."""
    text = _strip_fences(raw)
    data = json.loads(text)
    if isinstance(data, list):
        data = {"answer_segments": data}
    return StructuredAnswer.model_validate(data)


def validate_segment_chunk_ids(
    structured: StructuredAnswer,
    allowed_ids: set[str],
) -> list[str]:
    """Return invalid citation_chunk_id values (empty list if all ok)."""
    bad: list[str] = []
    for seg in structured.answer_segments:
        cid = (seg.citation_chunk_id or "").strip()
        if not cid or cid not in allowed_ids:
            bad.append(cid or "(empty)")
    return bad


def keep_grounded_answer_segments(
    segments: Sequence[AnswerSegment],
    allowed_ids: set[str],
) -> tuple[list[AnswerSegment], list[str]]:
    """Per-claim keep: valid citation_chunk_id stays; bad ids are dropped.

    Mirrors design/comparison grounding — a multi-claim answer is valid when
    each *kept* claim is grounded, not only when every model claim is perfect.
    """
    kept: list[AnswerSegment] = []
    dropped: list[str] = []
    for seg in segments:
        cid = (seg.citation_chunk_id or "").strip()
        if cid and cid in allowed_ids and (seg.text or "").strip():
            kept.append(seg)
        else:
            dropped.append(cid or "(empty)")
    return kept, dropped


_EXTRACT_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_QUESTION_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "how",
    "does",
    "do",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "under",
    "with",
    "from",
    "used",
    "requirement",
    "requirements",
    "regulation",
    "un",
    "ece",
}


def _question_content_tokens(question: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (question or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _QUESTION_STOPWORDS}


def _extractive_factual_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, list[SourceChunk]]:
    """Quote the best-matching sentence(s) when the LLM abstains but context fits.

    Intentionally narrow: only known over-rejection patterns (ATD/Hybrid III,
    isolation-resistance minima, HIC↔HPC). Broad extractive quoting reintroduced
    fabrication on hallucination_probe cases that must stay declined.
    """
    tokens = _question_content_tokens(question)
    if not tokens or not chunks:
        return "", []

    q_low = (question or "").lower()
    blob = " ".join((c.text or c.enriched_text or "") for c in chunks).lower()

    require_any: set[str] = set()
    # Gate: only run for patterns we audited as false declines.
    if re.search(r"anthropomorphic|atd|\bdummy\b", q_low) and "hybrid" in blob:
        tokens.update({"hybrid", "iii", "dummy", "percentile"})
        require_any = {"hybrid"}
    elif re.search(r"isolation|resistance", q_low) and "100" in blob and "isolat" in blob:
        tokens.update({"isolation", "resistance", "ohm", "ω", "100"})
        require_any = {"isolation", "100", "ohm", "ω"}
    elif re.search(r"\bhic", q_low) and (
        "hpc" in blob or "1,000" in blob or "1000" in blob
    ):
        # Simple HIC/HPC lookups only — not fabricated amendment/table probes.
        if re.search(
            r"(?ix)amendment|table\s+\d|confirm\s+that|impose|"
            r"annex\s+\d+|clause\s+\d|sets?\s+a\s+maximum|q10",
            q_low,
        ):
            return "", []
        if not re.search(r"(?ix)\b(what|which|limit)\b", q_low):
            return "", []
        tokens.update({"hpc", "head", "performance", "criterion", "1000", "1,000"})
        require_any = {"hpc", "1000", "1,000", "head"}
    else:
        return "", []

    scored: list[tuple[float, str, RetrievedChunk]] = []
    for chunk in chunks:
        text = (chunk.text or chunk.enriched_text or "").strip()
        if not text:
            continue
        for sent in _EXTRACT_SENTENCE_RE.split(text):
            sent = " ".join(sent.split()).strip()
            if len(sent) < 40:
                continue
            low = sent.lower()
            if require_any and not any(t in low for t in require_any):
                continue
            hit = sum(1 for t in tokens if t in low)
            if hit < 2:
                continue
            bonus = 0.0
            if re.search(r"\b(shall|must|not exceed|minimum|maximum|hybrid)\b", low):
                bonus += 1.0
            if "hybrid iii" in low:
                bonus += 3.0
            if re.search(r"\d", sent):
                bonus += 0.5
            scored.append((hit + bonus, sent, chunk))

    if not scored:
        return "", []

    scored.sort(key=lambda row: (-row[0], -len(row[1])))
    parts: list[str] = []
    sources: list[SourceChunk] = []
    seen_cid: set[str] = set()
    seen_sent: set[str] = set()
    for _score, sent, chunk in scored:
        key = sent.lower()
        if key in seen_sent:
            continue
        seen_sent.add(key)
        chip = chunk.citation_tag()
        parts.append(f"{sent} {chip}".strip())
        cid = chunk.chunk_id or ""
        if cid and cid not in seen_cid:
            seen_cid.add(cid)
            sources.append(to_source(chunk))
        if len(parts) >= 2:
            break

    if not parts:
        return "", []
    caveat = (
        "Extracted from the retrieved regulation text "
        "(model abstained; content was present in context):"
    )
    return f"{caveat}\n\n" + "\n\n".join(parts), sources


def render_answer_from_segments(
    structured: StructuredAnswer,
    chunks_by_id: dict[str, RetrievedChunk],
) -> tuple[str, list[SourceChunk]]:
    """Build final prose + chips from chunk metadata (never trust model section numbers)."""
    parts: list[str] = []
    sources: list[SourceChunk] = []
    seen: set[str] = set()
    for seg in structured.answer_segments:
        cid = (seg.citation_chunk_id or "").strip()
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            raise ValueError(f"missing chunk metadata for {cid!r}")
        chip = chunk.citation_tag()
        body = (seg.text or "").strip()
        # Strip any citation-like markup the model may have smuggled into text.
        body = _CITATION_CHIP_RE.sub("", body).strip()
        parts.append(f"{body} {chip}".strip() if body else chip)
        if cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))
    return "\n\n".join(parts).strip(), sources


def log_groundedness_violation(
    *,
    question: str,
    model_output: str,
    retrieved_chunk_ids: Sequence[str],
    invalid_ids: Sequence[str],
    path: Path | None = None,
) -> None:
    """Append a regression record for unverifiable citation_chunk_id values."""
    out = path or GROUNDEDNESS_VIOLATIONS_PATH
    rec = {
        "question": question,
        "model_output": model_output,
        "retrieved_chunk_ids": list(retrieved_chunk_ids),
        "invalid_citation_chunk_ids": list(invalid_ids),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.warning(
        "groundedness violation logged → %s invalid=%s",
        out,
        list(invalid_ids),
    )


def assert_prose_sections_match_citations(
    answer: str,
    sources: Sequence[SourceChunk],
) -> None:
    """Fail if any §section in the answer disagrees with cited chunk metadata."""
    by_citation = {s.citation: s for s in sources}
    cited_sections = {
        (s.section_number or "").strip() for s in sources if (s.section_number or "").strip()
    }

    chip_spans: list[tuple[int, int]] = []
    for match in _CITATION_CHIP_RE.finditer(answer or ""):
        chip_spans.append((match.start(), match.end()))
        chip = match.group(0)
        prose_sec = match.group(2).strip()
        src = by_citation.get(chip)
        if src is None:
            # Match by section+page when whitespace differs slightly.
            page_raw = match.group(3).strip()
            src = next(
                (
                    s
                    for s in sources
                    if (s.section_number or "").strip() == prose_sec
                    and (
                        page_raw == "?"
                        or (s.page_number is not None and str(s.page_number) == page_raw)
                    )
                ),
                None,
            )
        if src is None:
            raise AssertionError(
                f"Citation chip {chip!r} has no matching source among "
                f"{[s.citation for s in sources]}"
            )
        actual = (src.section_number or "").strip()
        if prose_sec != actual:
            raise AssertionError(
                f"Prose section {prose_sec!r} in {chip!r} does not match "
                f"cited chunk {src.chunk_id!r} section_number={actual!r}"
            )

    # Free-standing §marks outside chips (chips already validated above).
    for match in _SECTION_MARK_RE.finditer(answer or ""):
        if any(start <= match.start() < end for start, end in chip_spans):
            continue
        prose_sec = match.group(1).strip()
        if prose_sec not in cited_sections:
            raise AssertionError(
                f"Prose section mark §{prose_sec} is not among cited chunk "
                f"section_numbers {sorted(cited_sections)}"
            )


def _call_structured_llm(
    client: LLMClient,
    *,
    question: str,
    context: str,
    chunk_ids: Sequence[str],
    system: str,
    user: str,
    skip_cache: bool = False,
    response_format: dict[str, Any] | None = None,
) -> LLMResult:
    from generation.llm_client import answer_temperature

    return client.complete(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        role="answer",
        question=question,
        chunk_ids=list(chunk_ids),
        response_format=response_format if response_format is not None else ANSWER_RESPONSE_FORMAT,
        skip_cache=skip_cache,
        temperature=answer_temperature(),
    )


def _resolve_comparison_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """Side-by-side cross-reg synthesis with per-claim grounding (design-style).

    An answer spanning two regulations is valid when each claim cites its own
    retrieved chunk — not when the whole answer maps to one chunk.
    """
    from retrieval.comparison import (
        COMPARISON_SYSTEM_PROMPT,
        COMPARISON_USER_INSTRUCTION,
        detect_named_regulations,
    )
    from retrieval.design_implication import keep_grounded_design_segments

    trace.optimizations["comparison_mode"] = True
    named = detect_named_regulations(question)
    trace.optimizations["comparison_regulations"] = named

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    context = format_context(chunks)
    closest = closest_retrieved_chunk(list(chunks), question=question)
    regs_line = ", ".join(named) if named else "(named regulations in the question)"
    user_prompt = (
        f"Question: {question}\n\n"
        f"Named regulations to compare: {regs_line}\n\n"
        f"Context passages (retrieved per regulation):\n{context}\n\n"
        f"{COMPARISON_USER_INSTRUCTION}\n\n"
        "Return JSON with answer_segments now. "
        "Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    def _grounding_fail(result: LLMResult) -> tuple[str, list[SourceChunk], LLMResult, str]:
        msg = grounding_rejected_message(closest)
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, result, FAILURE_GROUNDING_REJECTED

    result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=COMPARISON_SYSTEM_PROMPT,
        user=user_prompt,
        response_format=ANSWER_RESPONSE_FORMAT,
    )
    trace.add_llm(
        role="answer",
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached=result.cached,
        provider=result.served_provider or result.provider,
        cache_status=result.cache_status,
        cost_usd=result.cost_usd,
        target_index=result.target_index,
        latency_ms=result.latency_ms,
        retry_attempts=result.retry_attempts,
    )

    raw = result.text
    structured: StructuredAnswer | None = None
    try:
        structured = parse_structured_answer(raw)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        structured = None
        log_groundedness_violation(
            question=question,
            model_output=raw,
            retrieved_chunk_ids=list(allowed_ids),
            invalid_ids=[f"parse_error:{exc}"],
        )

    kept: list[AnswerSegment] = []
    dropped_ids: list[str] = []
    if structured is not None:
        kept, dropped_ids = keep_grounded_design_segments(
            structured.answer_segments, allowed
        )
        if dropped_ids:
            trace.optimizations["comparison_dropped_ungrounded_claims"] = len(dropped_ids)
            log_groundedness_violation(
                question=question,
                model_output=raw,
                retrieved_chunk_ids=list(allowed_ids),
                invalid_ids=[f"comparison_drop:{d}" for d in dropped_ids],
            )

    if not kept:
        reminder = RETRY_REMINDER.format(allowed=", ".join(allowed_ids))
        result = _call_structured_llm(
            client,
            question=question + "\n#retry",
            context=context,
            chunk_ids=allowed_ids,
            system=COMPARISON_SYSTEM_PROMPT,
            user=f"{user_prompt}\n\n{reminder}",
            response_format=ANSWER_RESPONSE_FORMAT,
            skip_cache=True,
        )
        trace.add_llm(
            role="answer",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached=result.cached,
            provider=result.served_provider or result.provider,
            cache_status=result.cache_status,
            cost_usd=result.cost_usd,
            target_index=result.target_index,
            latency_ms=result.latency_ms,
            retry_attempts=result.retry_attempts,
        )
        raw = result.text
        try:
            structured = parse_structured_answer(raw)
            kept, dropped_ids = keep_grounded_design_segments(
                structured.answer_segments, allowed
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            kept = []

    if not kept:
        return _grounding_fail(result)

    # Ensure we cite ≥2 regulations when context supports it (coverage signal).
    cited_regs = {
        (by_id[s.citation_chunk_id].regulation_id or "")
        for s in kept
        if s.citation_chunk_id in by_id
    }
    cited_regs.discard("")
    trace.optimizations["comparison_cited_regulations"] = sorted(cited_regs)

    # Render like the standard structured path (chips from metadata only).
    structured_kept = StructuredAnswer(answer_segments=list(kept))
    text, sources = render_answer_from_segments(structured_kept, by_id)

    if not text.strip():
        return _grounding_fail(result)
    return text, sources, result, None


def _resolve_design_implication_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """Multi-claim design synthesis with per-claim grounding + FACT/INFERENCE labels.

    Unlike the standard path, one bad citation_chunk_id does not reject the whole
    answer — only ungrounded claims are dropped. The answer is valid when each
    remaining claim cites a retrieved clause.
    """
    from retrieval.design_implication import (
        DESIGN_SYSTEM_PROMPT,
        DESIGN_USER_INSTRUCTION,
        design_answer_response_format,
        expand_design_query,
        extractive_design_fallback,
        keep_grounded_design_segments,
        render_design_answer,
    )

    trace.optimizations["design_implication"] = True
    expansion = expand_design_query(question)
    trace.optimizations["design_expansion"] = expansion.to_public_dict()

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    context = format_context(chunks)
    closest = closest_retrieved_chunk(list(chunks), question=question)
    concepts_line = ", ".join(expansion.concepts[:8]) if expansion.concepts else "(none)"
    user_prompt = (
        f"Question: {question}\n\n"
        f"Design component: "
        f"{(expansion.component.label if expansion.component else 'unknown')}\n"
        f"Expanded regulatory concepts: {concepts_line}\n\n"
        f"Context passages:\n{context}\n\n"
        f"{DESIGN_USER_INSTRUCTION}\n\n"
        "Return JSON with answer_segments now (include claim_kind on every segment). "
        "Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    def _grounding_fail(result: LLMResult) -> tuple[str, list[SourceChunk], LLMResult, str]:
        msg = grounding_rejected_message(closest)
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, result, FAILURE_GROUNDING_REJECTED

    fmt = design_answer_response_format()
    result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=DESIGN_SYSTEM_PROMPT,
        user=user_prompt,
        response_format=fmt,
    )
    trace.add_llm(
        role="answer",
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached=result.cached,
        provider=result.served_provider or result.provider,
        cache_status=result.cache_status,
        cost_usd=result.cost_usd,
        target_index=result.target_index,
        latency_ms=result.latency_ms,
        retry_attempts=result.retry_attempts,
    )

    raw = result.text
    structured: StructuredAnswer | None = None
    try:
        structured = parse_structured_answer(raw)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        structured = None
        log_groundedness_violation(
            question=question,
            model_output=raw,
            retrieved_chunk_ids=list(allowed_ids),
            invalid_ids=[f"parse_error:{exc}"],
        )

    kept: list[AnswerSegment] = []
    dropped_ids: list[str] = []
    if structured is not None:
        kept, dropped_ids = keep_grounded_design_segments(
            structured.answer_segments, allowed
        )
        if dropped_ids:
            trace.optimizations["design_dropped_ungrounded_claims"] = len(dropped_ids)
            log_groundedness_violation(
                question=question,
                model_output=raw,
                retrieved_chunk_ids=list(allowed_ids),
                invalid_ids=[f"design_drop:{d}" for d in dropped_ids],
            )

    # Retry only when zero claims survive — partial grounding is success.
    if not kept:
        reminder = RETRY_REMINDER.format(allowed=", ".join(allowed_ids))
        result = _call_structured_llm(
            client,
            question=question + "\n#retry",
            context=context,
            chunk_ids=allowed_ids,
            system=DESIGN_SYSTEM_PROMPT,
            user=f"{user_prompt}\n\n{reminder}",
            response_format=fmt,
            skip_cache=True,
        )
        trace.add_llm(
            role="answer",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached=result.cached,
            provider=result.served_provider or result.provider,
            cache_status=result.cache_status,
            cost_usd=result.cost_usd,
            target_index=result.target_index,
            latency_ms=result.latency_ms,
            retry_attempts=result.retry_attempts,
        )
        raw = result.text
        try:
            structured = parse_structured_answer(raw)
            kept, dropped_ids = keep_grounded_design_segments(
                structured.answer_segments, allowed
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            kept = []

    if not kept:
        # Extractive FACT-only fallback — still multi-clause grounded synthesis.
        if chunks:
            det_text, det_sources = extractive_design_fallback(
                question=question,
                chunks=list(chunks),
                expansion=expansion,
                to_source=to_source,
            )
            if det_text:
                trace.optimizations["design_extractive_fallback"] = True
                det_text, det_sources = _augment_multi_regulation_answer(
                    det_text,
                    det_sources,
                    chunks=list(chunks),
                    trace=trace,
                )
                return det_text, det_sources, result, None
        return _grounding_fail(result)

    answer_text, cited = render_design_answer(kept, by_id, to_source=to_source)
    answer_text, cited = _augment_multi_regulation_answer(
        answer_text,
        cited,
        chunks=list(chunks),
        trace=trace,
    )
    n_fact = sum(
        1
        for s in kept
        if (getattr(s, "claim_kind", "") or "").upper().startswith("REGULATORY")
        or not getattr(s, "claim_kind", "")
    )
    n_inf = sum(
        1
        for s in kept
        if "INFERENCE" in (getattr(s, "claim_kind", "") or "").upper()
    )
    trace.optimizations["design_fact_claims"] = n_fact
    trace.optimizations["design_inference_claims"] = n_inf
    return answer_text, cited, result, None


def _resolve_checklist_gen_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """Structured homologation checklist with per-category citations + gaps."""
    from retrieval.checklist import (
        CHECKLIST_SYSTEM_PROMPT,
        CHECKLIST_USER_INSTRUCTION,
        checklist_answer_response_format,
        extractive_checklist_fallback,
        format_checklist_context,
        keep_grounded_checklist_segments,
        rebuild_checklist_result,
        render_checklist_answer,
    )

    trace.optimizations["checklist_gen"] = True
    meta = trace.optimizations.get("checklist_retrieval")
    if not isinstance(meta, dict):
        meta = (trace.retrieval_log or {}).get("checklist_retrieval") or {}
    result = rebuild_checklist_result(question, chunks, meta=meta if isinstance(meta, dict) else {})
    trace.optimizations["checklist_retrieval"] = result.to_public_dict()

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    allowed_cats = {c.id for c in result.expansion.categories}
    context = format_checklist_context(result, chunks=chunks)
    closest = closest_retrieved_chunk(list(chunks), question=question)
    cat_list = ", ".join(
        f"{c.id} ({c.label})" for c in result.expansion.categories
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Regulation: {result.expansion.regulation_id or 'indexed'}\n"
        f"Allowed category_id values: {cat_list}\n\n"
        f"Context passages (grouped by category):\n{context}\n\n"
        f"{CHECKLIST_USER_INSTRUCTION}\n\n"
        "Return JSON with answer_segments now (include category_id on every segment). "
        "Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    def _grounding_fail(result_llm: LLMResult) -> tuple[str, list[SourceChunk], LLMResult, str]:
        msg = grounding_rejected_message(closest)
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, result_llm, FAILURE_GROUNDING_REJECTED

    fmt = checklist_answer_response_format()
    llm_result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=CHECKLIST_SYSTEM_PROMPT,
        user=user_prompt,
        response_format=fmt,
    )
    trace.add_llm(
        role="answer",
        model=llm_result.model,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cached=llm_result.cached,
        provider=llm_result.served_provider or llm_result.provider,
        cache_status=llm_result.cache_status,
        cost_usd=llm_result.cost_usd,
        target_index=llm_result.target_index,
        latency_ms=llm_result.latency_ms,
        retry_attempts=llm_result.retry_attempts,
    )

    raw = llm_result.text
    structured: StructuredAnswer | None = None
    try:
        structured = parse_structured_answer(raw)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        structured = None
        log_groundedness_violation(
            question=question,
            model_output=raw,
            retrieved_chunk_ids=list(allowed_ids),
            invalid_ids=[f"parse_error:{exc}"],
        )

    kept: list[AnswerSegment] = []
    dropped_ids: list[str] = []
    if structured is not None:
        kept, dropped_ids = keep_grounded_checklist_segments(
            structured.answer_segments,
            allowed,
            allowed_categories=allowed_cats,
        )
        if dropped_ids:
            trace.optimizations["checklist_dropped_ungrounded"] = len(dropped_ids)
            log_groundedness_violation(
                question=question,
                model_output=raw,
                retrieved_chunk_ids=list(allowed_ids),
                invalid_ids=[f"checklist_drop:{d}" for d in dropped_ids],
            )

    if not kept:
        reminder = RETRY_REMINDER.format(allowed=", ".join(allowed_ids))
        llm_result = _call_structured_llm(
            client,
            question=question + "\n#retry",
            context=context,
            chunk_ids=allowed_ids,
            system=CHECKLIST_SYSTEM_PROMPT,
            user=f"{user_prompt}\n\n{reminder}",
            response_format=fmt,
            skip_cache=True,
        )
        trace.add_llm(
            role="answer",
            model=llm_result.model,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
            cached=llm_result.cached,
            provider=llm_result.served_provider or llm_result.provider,
            cache_status=llm_result.cache_status,
            cost_usd=llm_result.cost_usd,
            target_index=llm_result.target_index,
            latency_ms=llm_result.latency_ms,
            retry_attempts=llm_result.retry_attempts,
        )
        raw = llm_result.text
        try:
            structured = parse_structured_answer(raw)
            kept, dropped_ids = keep_grounded_checklist_segments(
                structured.answer_segments,
                allowed,
                allowed_categories=allowed_cats,
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            kept = []

    if not kept:
        det_text, det_sources = extractive_checklist_fallback(
            result=result,
            chunks=list(chunks),
            to_source=to_source,
        )
        if det_text:
            trace.optimizations["checklist_extractive_fallback"] = True
            return det_text, det_sources, llm_result, None
        if not chunks:
            return _grounding_fail(llm_result)
        # Still render empty-category notes even with zero items.
        det_text, det_sources = render_checklist_answer(
            [],
            by_id,
            expansion=result.expansion,
            covered=result.covered_categories,
            missing=result.missing_categories or [c.id for c in result.expansion.categories],
            chunk_category=result.chunk_category,
            to_source=to_source,
        )
        if det_text:
            trace.optimizations["checklist_empty_categories_only"] = True
            return det_text, det_sources, llm_result, None
        return _grounding_fail(llm_result)

    # Fill category_id from chunk map when the model omitted it.
    for seg in kept:
        if not (getattr(seg, "category_id", None) or "").strip():
            cid = (seg.citation_chunk_id or "").strip()
            seg.category_id = result.chunk_category.get(cid, "")

    answer_text, cited = render_checklist_answer(
        kept,
        by_id,
        expansion=result.expansion,
        covered=result.covered_categories,
        missing=result.missing_categories,
        chunk_category=result.chunk_category,
        to_source=to_source,
    )
    trace.optimizations["checklist_items"] = len(kept)
    trace.optimizations["checklist_missing_categories"] = list(result.missing_categories)
    return answer_text, cited, llm_result, None


def _resolve_scope_summary_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
    routed: Any | None = None,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """Structured single-reg summary; injury numbers from verified limits table."""
    from retrieval.scope_summary import (
        SCOPE_SYSTEM_PROMPT,
        SCOPE_USER_INSTRUCTION,
        format_scope_summary_context,
        rebuild_scope_result,
        render_scope_summary,
        scope_summary_response_format,
    )
    from retrieval.checklist import keep_grounded_checklist_segments

    trace.optimizations["scope_summary"] = True
    meta = trace.optimizations.get("scope_summary_retrieval")
    if not isinstance(meta, dict):
        meta = (trace.retrieval_log or {}).get("scope_summary_retrieval") or {}
    named = None
    if routed is not None:
        named = getattr(routed, "regulation_id", None)
    result = rebuild_scope_result(
        question,
        chunks,
        meta=meta if isinstance(meta, dict) else {},
        regulation_id=named,
    )
    rid = result.expansion.regulation_id
    if rid:
        # Hard filter again at answer time.
        chunks = [
            c for c in chunks if (c.regulation_id or "").strip() == rid
        ]
        result = rebuild_scope_result(
            question,
            chunks,
            meta=meta if isinstance(meta, dict) else {},
            regulation_id=rid,
        )
    trace.optimizations["scope_summary_retrieval"] = result.to_public_dict()

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    context = format_scope_summary_context(result)
    closest = closest_retrieved_chunk(list(chunks), question=question)

    # Prefer deterministic assembly (limits table is source of truth for numbers).
    # Still try LLM for scope/test/homologation phrasing when passages exist.
    user_prompt = (
        f"Question: {question}\n\n"
        f"Named regulation (ONLY): {rid}\n\n"
        f"Context passages:\n{context}\n\n"
        f"{SCOPE_USER_INSTRUCTION}\n\n"
        "Return JSON with answer_segments now. Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    def _grounding_fail(result_llm: LLMResult) -> tuple[str, list[SourceChunk], LLMResult, str]:
        msg = grounding_rejected_message(closest)
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, result_llm, FAILURE_GROUNDING_REJECTED

    fmt = scope_summary_response_format()
    llm_result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=SCOPE_SYSTEM_PROMPT,
        user=user_prompt,
        response_format=fmt,
    )
    trace.add_llm(
        role="answer",
        model=llm_result.model,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cached=llm_result.cached,
        provider=llm_result.served_provider or llm_result.provider,
        cache_status=llm_result.cache_status,
        cost_usd=llm_result.cost_usd,
        target_index=llm_result.target_index,
        latency_ms=llm_result.latency_ms,
        retry_attempts=llm_result.retry_attempts,
    )

    kept: list[AnswerSegment] = []
    try:
        structured = parse_structured_answer(llm_result.text)
        kept, dropped = keep_grounded_checklist_segments(
            structured.answer_segments, allowed
        )
        # Drop any segment citing a foreign regulation chunk.
        if rid:
            kept = [
                s
                for s in kept
                if (by_id.get(s.citation_chunk_id) is not None)
                and (by_id[s.citation_chunk_id].regulation_id or "").strip() == rid
            ]
        if dropped:
            trace.optimizations["scope_summary_dropped"] = len(dropped)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        kept = []

    answer_text, cited = render_scope_summary(
        result=result,
        segments=kept,
        chunks=chunks,
        to_source=to_source,
    )
    if not answer_text:
        if not chunks:
            return _grounding_fail(llm_result)
        # Still render limits-only skeleton.
        answer_text, cited = render_scope_summary(
            result=result,
            segments=[],
            chunks=chunks,
            to_source=to_source,
        )
    if not answer_text:
        return _grounding_fail(llm_result)

    # Final leak check on sources.
    if rid:
        cited = [s for s in cited if (s.regulation_id or "").strip() == rid]
    trace.optimizations["scope_summary_limits"] = len(result.limits)
    return answer_text, cited, llm_result, None


def _resolve_applicability_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """One APPLIES / DOES_NOT_APPLY / CANNOT_DETERMINE verdict per indexed reg."""
    from retrieval.applicability import (
        APPLICABILITY_SYSTEM_PROMPT,
        APPLICABILITY_USER_INSTRUCTION,
        applicability_response_format,
        extractive_applicability_fallback,
        format_applicability_context,
        merge_llm_with_heuristics,
        rebuild_applicability_result,
        render_applicability_answer,
        retrieve_applicability,
    )

    trace.optimizations["applicability"] = True
    meta = trace.optimizations.get("applicability_retrieval")
    if not isinstance(meta, dict):
        meta = (trace.retrieval_log or {}).get("applicability_retrieval") or {}

    # Prefer live re-fetch of all scopes so answer never depends on a trimmed
    # single-reg hybrid set (the X3 EV failure mode).
    try:
        live = retrieve_applicability(question)
        result = live
        chunks = list(live.chunks) or list(chunks)
        trace.optimizations["applicability_retrieval"] = live.to_public_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("applicability live retrieve failed, rebuild: %s", exc)
        result = rebuild_applicability_result(
            question, chunks, meta=meta if isinstance(meta, dict) else {}
        )

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    context = format_applicability_context(result)
    closest = closest_retrieved_chunk(list(chunks), question=question)

    user_prompt = (
        f"Question: {question}\n\n"
        f"{context}\n\n"
        f"{APPLICABILITY_USER_INSTRUCTION}\n\n"
        "Return JSON with answer_segments now (one per indexed regulation). "
        "Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    def _grounding_fail(result_llm: LLMResult) -> tuple[str, list[SourceChunk], LLMResult, str]:
        # Still emit heuristic board — never a single-reg ungrounded fallback.
        det_text, det_sources = extractive_applicability_fallback(
            result, to_source=to_source
        )
        if det_text:
            trace.optimizations["applicability_heuristic_fallback"] = True
            return det_text, det_sources, result_llm, None
        msg = grounding_rejected_message(closest)
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, result_llm, FAILURE_GROUNDING_REJECTED

    fmt = applicability_response_format()
    llm_result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=APPLICABILITY_SYSTEM_PROMPT,
        user=user_prompt,
        response_format=fmt,
    )
    trace.add_llm(
        role="answer",
        model=llm_result.model,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cached=llm_result.cached,
        provider=llm_result.served_provider or llm_result.provider,
        cache_status=llm_result.cache_status,
        cost_usd=llm_result.cost_usd,
        target_index=llm_result.target_index,
        latency_ms=llm_result.latency_ms,
        retry_attempts=llm_result.retry_attempts,
    )

    segments: list[AnswerSegment] = []
    try:
        structured = parse_structured_answer(llm_result.text)
        for seg in structured.answer_segments:
            cid = (seg.citation_chunk_id or "").strip()
            if cid and cid in allowed:
                segments.append(seg)
            elif not cid and seg.category_id:
                # Allow verdict without cite — merge will attach scope chunk.
                segments.append(seg)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        segments = []

    decisions = merge_llm_with_heuristics(segments, result, allowed)
    # Force passenger-EV / X3 case: R94+R95 must not both be dropped to unknown
    # when heuristics say APPLIES — already in heuristics; merge keeps them.

    answer_text, cited = render_applicability_answer(
        decisions,
        vehicle=result.expansion.vehicle,
        to_source=to_source,
    )
    if not answer_text:
        return _grounding_fail(llm_result)

    applies = [d.regulation_id for d in decisions if d.verdict == "APPLIES"]
    trace.optimizations["applicability_applies"] = applies
    trace.optimizations["applicability_decisions"] = [
        {"regulation_id": d.regulation_id, "verdict": d.verdict} for d in decisions
    ]
    # Reject single-reg-only answers when ≥2 regs indexed.
    indexed_n = len(result.expansion.indexed_regulation_ids)
    if indexed_n >= 2 and len(decisions) < indexed_n:
        answer_text, cited = extractive_applicability_fallback(
            result, to_source=to_source
        )
        trace.optimizations["applicability_forced_full_survey"] = True
    return answer_text, cited, llm_result, None


def _resolve_retest_scope_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """Modification-clause analysis with mandatory non-authoritative framing."""
    from retrieval.design_implication import keep_grounded_design_segments
    from retrieval.retest_scope import (
        RETEST_DISCLAIMER,
        RETEST_DISCLAIMER_TITLE,
        RETEST_SYSTEM_PROMPT,
        RETEST_USER_INSTRUCTION,
        expand_retest_query,
        extractive_retest_fallback,
        format_retest_context,
        render_retest_answer,
        retest_result_from_chunks,
        retrieve_retest_scope,
        retest_response_format,
    )

    trace.optimizations["retest_scope"] = True
    trace.optimizations["mode_disclaimer"] = RETEST_DISCLAIMER
    trace.optimizations["mode_disclaimer_title"] = RETEST_DISCLAIMER_TITLE

    expansion = expand_retest_query(question)
    incoming = [c for c in chunks if getattr(c, "chunk_id", None)]
    if incoming:
        # Prefer chunks already gathered by the RETEST_SCOPE retrieve path.
        result = retest_result_from_chunks(
            question, incoming, expansion=expansion
        )
        chunks = list(result.chunks)
        trace.optimizations["retest_retrieval"] = result.to_public_dict()
    else:
        try:
            result = retrieve_retest_scope(question, expansion=expansion)
            chunks = list(result.chunks)
            trace.optimizations["retest_retrieval"] = result.to_public_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("retest live retrieve failed: %s", exc)
            result = retest_result_from_chunks(
                question, [], expansion=expansion
            )

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    context = format_retest_context(result)
    closest = closest_retrieved_chunk(list(chunks), question=question)

    user_prompt = (
        f"Question: {question}\n\n"
        f"{context}\n\n"
        f"{RETEST_USER_INSTRUCTION}\n\n"
        "Return JSON with answer_segments now. Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    fmt = retest_response_format()
    llm_result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=RETEST_SYSTEM_PROMPT,
        user=user_prompt,
        response_format=fmt,
    )
    trace.add_llm(
        role="answer",
        model=llm_result.model,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cached=llm_result.cached,
        provider=llm_result.served_provider or llm_result.provider,
        cache_status=llm_result.cache_status,
        cost_usd=llm_result.cost_usd,
        target_index=llm_result.target_index,
        latency_ms=llm_result.latency_ms,
        retry_attempts=llm_result.retry_attempts,
    )

    kept: list[AnswerSegment] = []
    try:
        structured = parse_structured_answer(llm_result.text)
        kept, _dropped = keep_grounded_design_segments(
            structured.answer_segments, allowed
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        kept = []

    if not kept:
        answer_text, cited = extractive_retest_fallback(result, to_source=to_source)
        if answer_text:
            trace.optimizations["retest_extractive_fallback"] = True
            return answer_text, cited, llm_result, None
        # Still emit disclaimer-only skeleton.
        answer_text, cited = render_retest_answer(
            [], by_id, result=result, to_source=to_source
        )
        if answer_text:
            return answer_text, cited, llm_result, None
        msg = grounding_rejected_message(closest)
        # Prepend disclaimer even on soft failure.
        msg = f"**{RETEST_DISCLAIMER_TITLE}**\n\n{RETEST_DISCLAIMER}\n\n{msg}"
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, llm_result, FAILURE_GROUNDING_REJECTED

    answer_text, cited = render_retest_answer(
        kept, by_id, result=result, to_source=to_source
    )
    return answer_text, cited, llm_result, None


def _resolve_structured_answer(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient,
    trace: QueryTrace,
    routed: Any | None = None,
) -> tuple[str, list[SourceChunk], LLMResult, str | None]:
    """Call LLM, validate chunk ids (retry once), render from metadata.

    Returns ``(answer_text, cited_sources, last_llm_result, failure_kind)``.
    ``failure_kind`` is None on success, otherwise exactly one of
    ``retrieval_miss`` / ``grounding_rejected``. With non-empty ``chunks``,
    failures are always ``grounding_rejected`` (never retrieval_miss).
    """
    from retrieval.value_limit import (
        VALUE_VS_LIMIT_USER_INSTRUCTION,
        is_value_vs_limit_query,
    )
    from retrieval.enumerative import is_enumerative_query
    from retrieval.multi_regulation import (
        is_plural_regulation_query,
    )
    from generation.compliance import has_compliance_intent
    from retrieval.pipelines import intent_prompt_extra
    from retrieval.router import QueryIntent
    from retrieval.checklist import is_checklist_pipeline_query

    intent = getattr(routed, "intent", None) if routed is not None else None

    # Structured limits table (Fix 22 presentation) — before LLM prose paths.
    try:
        from retrieval.limits_aggregation import (
            is_limits_aggregation_query,
            render_limits_aggregation_answer,
        )

        if is_limits_aggregation_query(question):
            from generation.llm_client import LLMResult

            text, sources = render_limits_aggregation_answer(
                question,
                chunks=chunks,
                routed=routed,
                to_source=to_source,
            )
            trace.optimizations["limits_aggregation"] = True
            trace.optimizations["limits_aggregation_rows"] = text.count("\n| ") - 1
            mock = LLMResult(
                text=text,
                model=getattr(client, "large_model", None) or "limits-table",
                provider=getattr(client, "provider", "mock") or "mock",
                role="answer",
            )
            return text, sources, mock, None if text.strip() else "retrieval_miss"
    except Exception as exc:  # noqa: BLE001
        logger.warning("limits_aggregation path failed: %s", exc)

    # Layer 3–5 multi-step resolvers — only for hybrid multi-step intents.
    from retrieval.hybrid_layers import should_use_multi_step_layer
    from retrieval.comparison import is_comparison_mode_query, detect_named_regulations

    # Pairwise / multi-named-reg comparison — before single-reg FACTUAL path.
    if is_comparison_mode_query(question) and len(detect_named_regulations(question)) >= 2:
        return _resolve_comparison_answer(
            question=question,
            chunks=chunks,
            client=client,
            trace=trace,
        )

    if should_use_multi_step_layer(routed) and intent == QueryIntent.DESIGN_IMPLICATION:
        return _resolve_design_implication_answer(
            question=question,
            chunks=chunks,
            client=client,
            trace=trace,
        )
    if should_use_multi_step_layer(routed) and intent == QueryIntent.CHECKLIST_GEN and is_checklist_pipeline_query(question):
        return _resolve_checklist_gen_answer(
            question=question,
            chunks=chunks,
            client=client,
            trace=trace,
        )
    if intent == QueryIntent.SCOPE_SUMMARY:
        return _resolve_scope_summary_answer(
            question=question,
            chunks=chunks,
            client=client,
            trace=trace,
            routed=routed,
        )
    if should_use_multi_step_layer(routed) and intent == QueryIntent.APPLICABILITY:
        return _resolve_applicability_answer(
            question=question,
            chunks=chunks,
            client=client,
            trace=trace,
        )
    if should_use_multi_step_layer(routed) and intent == QueryIntent.RETEST_SCOPE:
        return _resolve_retest_scope_answer(
            question=question,
            chunks=chunks,
            client=client,
            trace=trace,
        )

    allowed_ids = [c.chunk_id for c in chunks if c.chunk_id]
    allowed = set(allowed_ids)
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    context = format_context(chunks)
    closest = closest_retrieved_chunk(list(chunks), question=question)
    value_vs_limit = is_value_vs_limit_query(question)
    enumerative = is_enumerative_query(question)
    plural_regs = is_plural_regulation_query(question)
    compliance_intent = has_compliance_intent(question)
    if intent == QueryIntent.CHECKLIST_GEN:
        enumerative = True
    if intent == QueryIntent.APPLICABILITY:
        plural_regs = True
    if intent == QueryIntent.COMPLIANCE_CHECK:
        compliance_intent = True
    if value_vs_limit:
        trace.optimizations["value_vs_limit"] = True
    if enumerative:
        trace.optimizations["enumerative"] = True
    if plural_regs:
        trace.optimizations["multi_regulation"] = True
    extra = intent_prompt_extra(routed)
    if value_vs_limit:
        extra += f"{VALUE_VS_LIMIT_USER_INSTRUCTION}\n\n"
    if enumerative:
        extra += (
            "ENUMERATIVE QUESTION — list each distinct requirement found in the "
            "passages as its own answer_segment with its own citation_chunk_id. "
            "Prefer coverage across multiple passages over a single summary sentence.\n\n"
        )
    if plural_regs:
        extra += f"{MULTI_REGULATION_USER_INSTRUCTION}\n\n"
    if compliance_intent:
        extra += f"{COMPLIANCE_VERDICT_USER_INSTRUCTION}\n\n"
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        + extra
        + "Return JSON with answer_segments now. "
        "Allowed citation_chunk_id values: "
        + ", ".join(allowed_ids)
    )

    def _grounding_fail(result: LLMResult) -> tuple[str, list[SourceChunk], LLMResult, str]:
        msg = grounding_rejected_message(closest)
        cited = [to_source(closest)] if closest is not None else []
        return msg, cited, result, FAILURE_GROUNDING_REJECTED

    result = _call_structured_llm(
        client,
        question=question,
        context=context,
        chunk_ids=allowed_ids,
        system=SYSTEM_PROMPT,
        user=user_prompt,
    )
    trace.add_llm(
        role="answer",
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached=result.cached,
        provider=result.served_provider or result.provider,
        cache_status=result.cache_status,
        cost_usd=result.cost_usd,
        target_index=result.target_index,
        latency_ms=result.latency_ms,
        retry_attempts=result.retry_attempts,
    )

    raw = result.text
    structured: StructuredAnswer | None = None
    invalid: list[str] = []
    try:
        structured = parse_structured_answer(raw)
        invalid = validate_segment_chunk_ids(structured, allowed)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        invalid = [f"parse_error:{exc}"]
        structured = None

    if invalid:
        log_groundedness_violation(
            question=question,
            model_output=raw,
            retrieved_chunk_ids=list(allowed_ids),
            invalid_ids=invalid,
        )
        reminder = RETRY_REMINDER.format(allowed=", ".join(allowed_ids))
        retry_user = f"{user_prompt}\n\n{reminder}"
        result = _call_structured_llm(
            client,
            question=question + "\n#retry",
            context=context,
            chunk_ids=allowed_ids,
            system=SYSTEM_PROMPT,
            user=retry_user,
            skip_cache=True,
        )
        trace.add_llm(
            role="answer",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached=result.cached,
            provider=result.served_provider or result.provider,
            cache_status=result.cache_status,
            cost_usd=result.cost_usd,
            target_index=result.target_index,
            latency_ms=result.latency_ms,
            retry_attempts=result.retry_attempts,
        )
        raw = result.text
        try:
            structured = parse_structured_answer(raw)
            invalid = validate_segment_chunk_ids(structured, allowed)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            invalid = [f"parse_error:{exc}"]
            structured = None

        if structured is None:
            log_groundedness_violation(
                question=question,
                model_output=raw,
                retrieved_chunk_ids=list(allowed_ids),
                invalid_ids=invalid or ["(unknown)"],
            )
            det_text, det_sources = _extractive_factual_answer(
                question=question, chunks=list(chunks)
            )
            if det_text:
                trace.optimizations["factual_extractive_fallback"] = True
                return det_text, det_sources, result, None
            return _grounding_fail(result)

        # Per-claim keep: drop bad ids, keep grounded claims (do not all-or-nothing).
        kept, dropped = keep_grounded_answer_segments(
            structured.answer_segments, allowed
        )
        if dropped:
            trace.optimizations["dropped_ungrounded_claims"] = len(dropped)
            log_groundedness_violation(
                question=question,
                model_output=raw,
                retrieved_chunk_ids=list(allowed_ids),
                invalid_ids=[f"partial_drop:{d}" for d in dropped],
            )
        structured = StructuredAnswer(answer_segments=list(kept))

    assert structured is not None
    if not structured.answer_segments:
        # Context was retrieved; model abstained → try extractive / specialized
        # fallbacks before hard grounding_rejected (avoids false declines when
        # the answer is already in the passages).
        if plural_regs and chunks:
            det_text, det_sources = _deterministic_multi_regulation_answer(
                question=question,
                chunks=list(chunks),
                trace=trace,
            )
            if det_text:
                return det_text, det_sources, result, None
        if enumerative and chunks:
            det_text, det_sources = _extractive_enumerative_answer(
                question=question,
                chunks=list(chunks),
            )
            if det_text:
                trace.optimizations["enumerative_extractive_fallback"] = True
                return det_text, det_sources, result, None
        det_text, det_sources = _extractive_factual_answer(
            question=question, chunks=list(chunks)
        )
        if det_text:
            trace.optimizations["factual_extractive_fallback"] = True
            return det_text, det_sources, result, None
        return _grounding_fail(result)
    # Layer 2: drop confident negative/absence claims the cited chunk does not
    # actually state (chunk_id validity alone is not enough).
    from generation.semantic_grounding import (
        audit_segments_semantic,
        drop_unsupported_audit_segments,
        filter_unsupported_negative_segments,
        log_semantic_disagreements,
    )

    kept, dropped = filter_unsupported_negative_segments(
        structured.answer_segments, by_id
    )
    if dropped:
        trace.optimizations["dropped_unsupported_negative_claims"] = len(dropped)
        log_groundedness_violation(
            question=question,
            model_output=raw,
            retrieved_chunk_ids=list(allowed_ids),
            invalid_ids=[f"unsupported_negative:{c[:80]}" for c in dropped],
        )
        structured = StructuredAnswer(answer_segments=list(kept))

    # Optional cheap LLM sample (eval / SEMANTIC_GROUNDEDNESS_LLM=1).
    audit = audit_segments_semantic(
        question=question,
        segments=structured.answer_segments,
        chunks_by_id=by_id,
        llm=client,
        use_llm=None,  # honor env
    )
    if audit.disagreements:
        log_semantic_disagreements(audit.disagreements)
        trace.optimizations["semantic_groundedness_disagreements"] = len(audit.disagreements)
    if audit.results:
        unsupported_n = sum(1 for r in audit.results if not r.supported)
        if unsupported_n:
            trace.optimizations["semantic_groundedness_unsupported"] = unsupported_n
            # Fail-closed: drop claims the semantic judge rejected or could not confirm.
            kept_audit, dropped_audit = drop_unsupported_audit_segments(
                structured.answer_segments, audit
            )
            if dropped_audit:
                trace.optimizations["dropped_semantic_unsupported_claims"] = len(
                    dropped_audit
                )
                log_groundedness_violation(
                    question=question,
                    model_output=raw,
                    retrieved_chunk_ids=list(allowed_ids),
                    invalid_ids=[
                        f"semantic_unsupported:{c[:80]}" for c in dropped_audit
                    ],
                )
                structured = StructuredAnswer(answer_segments=list(kept_audit))

    if not structured.answer_segments:
        if plural_regs and chunks:
            det_text, det_sources = _deterministic_multi_regulation_answer(
                question=question,
                chunks=list(chunks),
                trace=trace,
            )
            if det_text:
                return det_text, det_sources, result, None
        if enumerative and chunks:
            det_text, det_sources = _extractive_enumerative_answer(
                question=question,
                chunks=list(chunks),
            )
            if det_text:
                trace.optimizations["enumerative_extractive_fallback"] = True
                return det_text, det_sources, result, None
        det_text, det_sources = _extractive_factual_answer(
            question=question, chunks=list(chunks)
        )
        if det_text:
            trace.optimizations["factual_extractive_fallback"] = True
            return det_text, det_sources, result, None
        return _grounding_fail(result)

    answer_text, cited = render_answer_from_segments(structured, by_id)
    if plural_regs:
        answer_text, cited = _augment_multi_regulation_answer(
            answer_text,
            cited,
            chunks=list(chunks),
            trace=trace,
        )
    return answer_text, cited, result, None


def _extractive_enumerative_answer(
    *,
    question: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[SourceChunk]]:
    """List topic requirement sentences from retrieved chunks (no LLM).

    Used when the structured LLM abstains / fails citation checks on an
    enumerative ask — still returns grounded R95 door clauses instead of a
    grounding-rejected message pointing at the wrong regulation.
    """
    from retrieval.enumerative import (
        chunk_matches_enumerative_topic,
        extract_enumerative_topic,
    )

    topic = extract_enumerative_topic(question)
    topical = [c for c in chunks if chunk_matches_enumerative_topic(c, topic)]
    pool = topical or list(chunks)
    parts: list[str] = []
    sources: list[SourceChunk] = []
    seen: set[str] = set()
    for chunk in pool:
        text = " ".join((chunk.text or "").split())
        if not text:
            continue
        # Prefer requirement-like sentences that mention the topic.
        sentences = re.split(r"(?<=[.:;])\s+", text)
        picked: list[str] = []
        for sent in sentences:
            s = sent.strip()
            if len(s) < 40:
                continue
            low = s.lower()
            if topic and not chunk_matches_enumerative_topic(
                type("T", (), {"text": s, "section_title": "", "section_number": ""})(),
                topic,
            ):
                continue
            if not re.search(r"(?i)\bshall\b|\bmust\b|\bshall\s+not\b|\brequired\b", low):
                continue
            picked.append(s)
            if len(picked) >= 2:
                break
        if not picked:
            # Fall back to a short excerpt that still mentions the topic.
            if topic and any(t in text.lower() for t in (topic, topic.rstrip("s"))):
                picked = [text[:280] + ("…" if len(text) > 280 else "")]
            else:
                continue
        chip = chunk.citation_tag()
        for p in picked:
            body = _CITATION_CHIP_RE.sub("", p).strip()
            parts.append(f"{body} {chip}".strip())
        cid = chunk.chunk_id or ""
        if cid and cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))
        if len(parts) >= 6:
            break
    if not parts:
        return "", []
    header = f"Requirements related to {topic or 'this topic'} from the indexed passages:"
    return f"{header}\n\n" + "\n\n".join(parts), sources


def _multi_reg_coverage_from_trace(
    trace: QueryTrace,
    chunks: Sequence[RetrievedChunk],
) -> tuple[list[str], list[str]]:
    covered = [
        str(x)
        for x in (trace.optimizations.get("multi_regulation_covered") or [])
        if str(x).strip()
    ]
    missing = [
        str(x)
        for x in (trace.optimizations.get("multi_regulation_missing") or [])
        if str(x).strip()
    ]
    if not covered and chunks:
        covered = list(
            dict.fromkeys(
                str(c.regulation_id).strip()
                for c in chunks
                if str(getattr(c, "regulation_id", "") or "").strip()
            )
        )
    return covered, missing


def _deterministic_multi_regulation_answer(
    *,
    question: str,
    chunks: list[RetrievedChunk],
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk]]:
    """Per-regulation sketch + explicit missing list (no LLM)."""
    from retrieval.multi_regulation import format_coverage_summary, short_regulation_label

    covered, missing = _multi_reg_coverage_from_trace(trace, chunks)
    parts: list[str] = []
    sources: list[SourceChunk] = []
    seen: set[str] = set()
    for rid in covered:
        chunk = next((c for c in chunks if (c.regulation_id or "") == rid), None)
        if chunk is None:
            continue
        label = short_regulation_label(rid)
        chip = chunk.citation_tag()
        parts.append(
            f"{label} includes relevant provisions on this topic. {chip}".strip()
        )
        cid = chunk.chunk_id or ""
        if cid and cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))
    summary = format_coverage_summary(covered=covered, missing=missing)
    body = "\n\n".join(parts)
    text = f"{summary}\n\n{body}".strip() if body else summary
    if question and not text:
        return "", []
    return text, sources


def _augment_multi_regulation_answer(
    answer_text: str,
    cited: list[SourceChunk],
    *,
    chunks: list[RetrievedChunk],
    trace: QueryTrace,
) -> tuple[str, list[SourceChunk]]:
    """Ensure every covered reg is cited and missing regs are stated explicitly."""
    from retrieval.multi_regulation import format_coverage_summary, short_regulation_label

    covered, missing = _multi_reg_coverage_from_trace(trace, chunks)
    sources = list(cited)
    seen = {s.chunk_id for s in sources if s.chunk_id}
    cited_regs = {(s.regulation_id or "").strip() for s in sources}
    extra_parts: list[str] = []
    for rid in covered:
        if rid in cited_regs:
            continue
        chunk = next((c for c in chunks if (c.regulation_id or "") == rid), None)
        if chunk is None:
            continue
        label = short_regulation_label(rid)
        chip = chunk.citation_tag()
        extra_parts.append(
            f"{label} also includes relevant provisions on this topic. {chip}".strip()
        )
        cid = chunk.chunk_id or ""
        if cid and cid not in seen:
            seen.add(cid)
            sources.append(to_source(chunk))
            cited_regs.add(rid)

    summary = format_coverage_summary(covered=covered, missing=missing)
    body = (answer_text or "").strip()
    if extra_parts:
        body = (
            (body + "\n\n" + "\n\n".join(extra_parts)).strip()
            if body
            else "\n\n".join(extra_parts)
        )
    # Always lead with the deterministic covered/missing summary.
    if summary:
        if not body:
            body = summary
        elif summary.lower() not in body.lower():
            body = f"{summary}\n\n{body}".strip()
    return body, sources

def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    regulation_id: str | None = None,
    llm: LLMClient | None = None,
    chunks: Sequence[RetrievedChunk] | None = None,
    skip_answer_cache: bool = False,
    conversation_id: str | None = None,
    persist_turn: bool = True,
) -> AnswerResponse:
    """Retrieve (unless ``chunks`` given), call the LLM, return structured JSON-ready result.

    Condensation (if any) uses conversation history; the answer LLM call stays
    stateless on the standalone condensed question + retrieved context only.
    """
    question = (question or "").strip()
    trace = new_trace(question, regulation_id=regulation_id)
    token = set_current_trace(trace)
    try:
        return _answer_question_traced(
            question,
            trace=trace,
            top_k=top_k,
            regulation_id=regulation_id,
            llm=llm,
            chunks=chunks,
            skip_answer_cache=skip_answer_cache,
            conversation_id=conversation_id,
            persist_turn=persist_turn,
        )
    except Exception as exc:  # noqa: BLE001
        trace.error = str(exc)
        trace.finalize()
        raise
    finally:
        reset_current_trace(token)


def _answer_question_traced(
    question: str,
    *,
    trace: QueryTrace,
    top_k: int | None,
    regulation_id: str | None,
    llm: LLMClient | None,
    chunks: Sequence[RetrievedChunk] | None,
    skip_answer_cache: bool,
    conversation_id: str | None,
    persist_turn: bool,
) -> AnswerResponse:
    from api.conversations import (
        append_turn,
        get_turns,
        inject_vehicle_context,
        new_conversation_id,
        update_vehicle_context_from_question,
    )
    # Optional local pipeline cache (ANSWER_CACHE=1). Portkey simple cache is primary.
    from cache.response_cache import lookup, store
    from retrieval.rewrite import rewrite_query

    cid = (conversation_id or "").strip() or new_conversation_id()

    if not question:
        trace.not_found = True
        trace.optimizations["failure_kind"] = FAILURE_RETRIEVAL_MISS
        payload = trace.finalize()
        return AnswerResponse(
            question="",
            answer=retrieval_miss_message(),
            sources=[],
            not_found=True,
            failure_kind=FAILURE_RETRIEVAL_MISS,
            trace_id=trace.trace_id,
            metrics=payload,
            conversation_id=cid,
        )

    history = get_turns(cid)
    # Carry platform / category / powertrain across multi-turn BMW workflows.
    vehicle_ctx = update_vehicle_context_from_question(cid, question)
    if vehicle_ctx.has_any():
        trace.optimizations["vehicle_context"] = vehicle_ctx.to_public_dict()
    client = llm or LLMClient()

    def _with_conv(resp: AnswerResponse) -> AnswerResponse:
        resp.conversation_id = cid
        if persist_turn and question and resp.answer:
            append_turn(cid, question, resp.answer)
        return resp

    def _intent_fields(
        *,
        routed_obj: Any | None = None,
        cached: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from retrieval.hybrid_layers import layer_public_dict

        intent: str | None = None
        if routed_obj is not None:
            intent = getattr(getattr(routed_obj, "intent", None), "value", None)
        if not intent and cached:
            intent = cached.get("query_intent") or None
        if not intent:
            qi = trace.optimizations.get("query_intent")
            if isinstance(qi, dict):
                intent = qi.get("intent")
        disclaimer = trace.optimizations.get("mode_disclaimer")
        title = trace.optimizations.get("mode_disclaimer_title")
        if cached:
            disclaimer = disclaimer or cached.get("mode_disclaimer")
            title = title or cached.get("mode_disclaimer_title")
        layer_info = layer_public_dict(routed_obj)
        if cached and cached.get("execution_layer"):
            layer_info["execution_layer"] = cached.get("execution_layer")
            layer_info["multi_step"] = bool(cached.get("multi_step"))
        return {
            "query_intent": intent,
            "mode_disclaimer": disclaimer,
            "mode_disclaimer_title": title,
            "execution_layer": layer_info.get("execution_layer"),
            "multi_step": bool(layer_info.get("multi_step")),
        }

    def _from_cache(
        cached: dict[str, Any],
        *,
        condensed: str,
        condensation_applied: bool,
    ) -> AnswerResponse:
        hit_kind = str(cached.get("cache_hit") or "exact")
        trace.answer_cached = True
        trace.served_from_cache = True
        trace.cache_hit_kind = hit_kind
        trace.provider = str(cached.get("provider") or "cache")
        trace.model = str(cached.get("model") or "")
        sources_raw = cached.get("sources") or []
        sources = [SourceChunk.model_validate(s) for s in sources_raw]
        trace.chunk_ids = [s.chunk_id for s in sources]
        trace.citations = _citations_for_trace(sources)
        failure_kind = cached.get("failure_kind") or _failure_kind_from_answer(
            str(cached.get("answer") or "")
        )
        not_found = bool(cached.get("not_found")) or bool(failure_kind) or _is_insufficient(
            str(cached.get("answer"))
        )
        if not_found and not failure_kind:
            failure_kind = FAILURE_RETRIEVAL_MISS if not sources else FAILURE_GROUNDING_REJECTED
        trace.not_found = not_found
        if failure_kind:
            trace.optimizations["failure_kind"] = failure_kind
        if not_found:
            trace.faithfulness_passed = False
        metrics = trace.finalize()
        return _with_conv(
            AnswerResponse(
                question=question,
                answer=str(cached["answer"]),
                sources=sources,
                model=trace.model,
                provider=trace.provider,
                target_index=cached.get("target_index"),
                answering_provider_was_fallback=bool(
                    cached.get("answering_provider_was_fallback") or False
                ),
                cached=True,
                answer_cached=True,
                served_from_cache=True,
                cache_hit_kind=hit_kind,
                not_found=not_found,
                failure_kind=failure_kind if not_found else None,
                trace_id=trace.trace_id,
                metrics=metrics,
                condensed_question=condensed,
                condensation_applied=condensation_applied,
                conversation_id=cid,
                **_intent_fields(cached=cached),
            )
        )

    # Exact/semantic cache BEFORE condensation when there is no prior history.
    if not skip_answer_cache and not history:
        early = lookup(question, regulation_id=regulation_id, allow_semantic=True)
        if early and early.get("answer"):
            return _from_cache(early, condensed=question, condensation_applied=False)

    rewritten = rewrite_query(question, llm=client, history=history)
    condensed_raw = (rewritten.condensed or question).strip()
    condensed = inject_vehicle_context(condensed_raw, vehicle_ctx)
    condensation_applied = bool(rewritten.condensation_applied) or (
        condensed != condensed_raw
    )
    logger.info(
        "answer condensation applied=%s original=%r condensed=%r conversation_id=%s",
        condensation_applied,
        question,
        condensed,
        cid,
    )

    # Classify once — drives retrieval strategy, budget, grounding, and layer.
    from retrieval.hybrid_layers import execution_layer_for, layer_public_dict
    from retrieval.router import QueryIntent, classify_query

    routed = classify_query(
        question,
        condensed=condensed,
        llm=client,
        log=True,
    )
    layer_info = layer_public_dict(routed)
    trace.optimizations["query_intent"] = routed.to_public_dict()
    trace.optimizations["execution_layer"] = layer_info
    logger.info(
        "query_router intent=%s layer=%s source=%s strategy=%s budget=%s",
        routed.intent.value,
        layer_info.get("execution_layer"),
        routed.source,
        routed.pipeline.retrieval_strategy.value,
        routed.pipeline.budget_mode,
    )
    _ = execution_layer_for(routed)  # explicit for readability / future gates

    # Cache on the standalone question so follow-ups don't collide across chats.
    if not skip_answer_cache:
        cached = lookup(condensed, regulation_id=regulation_id, allow_semantic=True)
        if cached and cached.get("answer"):
            resp = _from_cache(
                cached, condensed=condensed, condensation_applied=condensation_applied
            )
            resp.condensed_question = condensed
            resp.condensation_applied = condensation_applied
            return resp

    if chunks is None:
        # History already applied above — retrieve on standalone question only.
        chunks = retrieve(
            condensed,
            top_k=top_k,
            regulation_id=regulation_id or routed.regulation_id,
            llm=client,
            history=None,
            # Already condensed/rewritten above — do not LLM-rewrite a second time
            # (that was a major source of non-deterministic chunk sets).
            rewrite=False,
            rewrite_result=rewritten,
            routed=routed,
        )
    else:
        chunks = list(chunks)

    # Belt-and-suspenders: enforce budget even when caller supplied ``chunks``.
    chunks, budget_stats = apply_context_budget(
        chunks, question=condensed, routed=routed
    )
    trace.context_chunks_to_llm = int(budget_stats.get("context_chunks_to_llm") or 0)
    trace.context_tokens_est = int(budget_stats.get("context_tokens_est") or 0)
    trace.context_budget_mode = str(budget_stats.get("mode") or "")
    if budget_stats.get("budget_exceeded_incoming"):
        trace.optimizations["context_budget_trimmed"] = True

    all_sources = [to_source(c) for c in chunks]
    trace.chunk_ids = [c.chunk_id for c in chunks]
    trace.citations = _citations_for_trace(all_sources)

    # --- Deterministic compliance path (replaces prompt-only Fix 14) --------
    from generation.compliance import (
        align_compliance_sections_with_sources,
        evaluate_compliance,
        is_compliance_check_query,
        phrase_compliance_with_llm,
        sources_for_compliance,
    )

    compliance_routed = routed.intent == QueryIntent.COMPLIANCE_CHECK
    if compliance_routed or is_compliance_check_query(condensed) or is_compliance_check_query(
        question
    ):
        compliance = evaluate_compliance(
            condensed or question,
            regulation_id=regulation_id,
        )
        if compliance is not None and (
            compliance.criteria or compliance.overall_verdict == "CANNOT_DETERMINE"
        ):
            trace.optimizations["compliance_deterministic"] = True
            trace.optimizations["compliance_overall"] = compliance.overall_verdict
            # Optional LLM phrasing only — numbers stay fixed from Python.
            if compliance.overall_verdict != "CANNOT_DETERMINE" and compliance.criteria:
                try:
                    compliance = phrase_compliance_with_llm(
                        compliance, llm=client, question=condensed or question
                    )
                    if compliance.phrasing_model or compliance.phrasing_provider:
                        trace.add_llm(
                            role="answer",
                            model=compliance.phrasing_model,
                            input_tokens=0,
                            output_tokens=0,
                            provider=compliance.phrasing_provider,
                            target_index=compliance.phrasing_target_index,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("compliance phrasing skipped: %s", exc)
            sources = sources_for_compliance(compliance, chunks)
            if not sources and chunks:
                # Prefer electrolyte containment chunks when present.
                prefer = [
                    c
                    for c in chunks
                    if re.search(
                        r"(?i)electrolyte\s+leakage|passenger\s+compart",
                        (c.text or "") + " " + (c.section_number or ""),
                    )
                ]
                sources = [to_source((prefer or chunks)[0])]
            compliance = align_compliance_sections_with_sources(compliance, sources)
            trace.citations = _citations_for_trace(sources)
            trace.faithfulness_passed = True
            trace.not_found = False
            metrics = trace.finalize()
            if compliance.phrasing_model or compliance.phrasing_provider:
                ans_model = compliance.phrasing_model
                ans_provider = compliance.phrasing_provider
                ans_idx = compliance.phrasing_target_index
                ans_fb = bool(compliance.phrasing_was_fallback)
            else:
                # Pure deterministic verdict — no generative model answered.
                ans_provider = "deterministic_compliance"
                ans_model = "rules_engine"
                ans_idx = 0
                ans_fb = False
            payload = {
                "question": condensed,
                "answer": compliance.answer_text,
                "sources": [s.model_dump() for s in sources],
                "model": ans_model,
                "provider": ans_provider,
                "target_index": ans_idx,
                "answering_provider_was_fallback": ans_fb,
                "not_found": False,
                "failure_kind": None,
                "compliance": compliance.to_public_dict(),
                "input_tokens": int(metrics.get("input_tokens") or 0),
                "output_tokens": int(metrics.get("output_tokens") or 0),
                "cost_usd": float(metrics.get("cost_usd") or 0.0),
                "trace_id": trace.trace_id,
            }
            if not skip_answer_cache:
                store(condensed, payload, regulation_id=regulation_id)
            logger.info(
                "compliance deterministic overall=%s criteria=%d provider=%s model=%s "
                "fallback=%s trace=%s",
                compliance.overall_verdict,
                len(compliance.criteria),
                ans_provider,
                ans_model,
                ans_fb,
                trace.trace_id,
            )
            return _with_conv(
                AnswerResponse(
                    question=question,
                    answer=compliance.answer_text,
                    sources=sources,
                    model=ans_model,
                    provider=ans_provider,
                    target_index=ans_idx,
                    answering_provider_was_fallback=ans_fb,
                    not_found=False,
                    failure_kind=None,
                    trace_id=trace.trace_id,
                    metrics=metrics,
                    condensed_question=condensed,
                    condensation_applied=condensation_applied,
                    conversation_id=cid,
                    compliance=compliance.to_public_dict(),
                )
            )

    if not chunks:
        # RETRIEVAL_MISS — exclusive: no chunks → this message only.
        answer_text = retrieval_miss_message()
        failure_kind = FAILURE_RETRIEVAL_MISS
        if routed.intent == QueryIntent.RETEST_SCOPE:
            from retrieval.retest_scope import RETEST_DISCLAIMER, RETEST_DISCLAIMER_TITLE

            answer_text = (
                f"**{RETEST_DISCLAIMER_TITLE}**\n\n{RETEST_DISCLAIMER}\n\n{answer_text}"
            )
            trace.optimizations["mode_disclaimer"] = RETEST_DISCLAIMER
            trace.optimizations["mode_disclaimer_title"] = RETEST_DISCLAIMER_TITLE
        trace.not_found = True
        trace.faithfulness_passed = False
        trace.optimizations["failure_kind"] = failure_kind
        intent_fields = _intent_fields(routed_obj=routed)
        cache_payload = {
            "question": condensed,
            "answer": answer_text,
            "sources": [],
            "model": "",
            "provider": client.provider,
            "not_found": True,
            "failure_kind": failure_kind,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            **intent_fields,
        }
        if not skip_answer_cache:
            store(condensed, cache_payload, regulation_id=regulation_id)
        metrics = trace.finalize()
        return _with_conv(
            AnswerResponse(
                question=question,
                answer=answer_text,
                sources=[],
                provider=client.provider,
                not_found=True,
                failure_kind=failure_kind,
                trace_id=trace.trace_id,
                metrics=metrics,
                condensed_question=condensed,
                condensation_applied=condensation_applied,
                conversation_id=cid,
                **intent_fields,
            )
        )

    cache_hit, saved = prompt_cache_lookup(SYSTEM_PROMPT)
    if cache_hit:
        trace.prompt_cache_hit = True
        trace.prompt_cache_tokens_saved = saved
        trace.optimizations["prompt_cache"] = True

    # Extract user-stated measurements BEFORE the LLM call (numeric fidelity guard).
    from generation.numeric_guard import (
        NUMERIC_CONFIRM_MESSAGE as _NUMERIC_CONFIRM,
        check_numeric_fidelity,
        extract_user_numbers,
        log_numeric_hallucination,
    )

    user_numbers = extract_user_numbers(condensed) or extract_user_numbers(question)
    if user_numbers:
        trace.optimizations["user_measured_numbers"] = [
            {"raw": u.raw, "unit": u.unit} for u in user_numbers
        ]
        logger.info(
            "numeric_guard extracted user_numbers=%s question=%r",
            [(u.raw, u.unit) for u in user_numbers],
            condensed[:120],
        )

    # Answer generation: condensed question + context only (no chat history).
    answer_text, cited_sources, result, failure_kind = _resolve_structured_answer(
        question=condensed,
        chunks=chunks,
        client=client,
        trace=trace,
        routed=routed,
    )
    trace.provider = result.provider
    not_found = failure_kind is not None

    if failure_kind == FAILURE_GROUNDING_REJECTED:
        # Keep the closest retrieved section so the UI can open it; do not mix
        # with a fabricated multi-claim answer.
        if cited_sources:
            sources = cited_sources[:1]
        else:
            closest = closest_retrieved_chunk(chunks, question=question)
            sources = [to_source(closest)] if closest is not None else []
        trace.not_found = True
        trace.faithfulness_passed = False
        trace.optimizations["failure_kind"] = FAILURE_GROUNDING_REJECTED
        trace.citations = _citations_for_trace(sources)
    elif failure_kind == FAILURE_RETRIEVAL_MISS:
        sources = []
        trace.not_found = True
        trace.faithfulness_passed = False
        trace.optimizations["failure_kind"] = FAILURE_RETRIEVAL_MISS
    else:
        sources = cited_sources if cited_sources else []
        trace.citations = _citations_for_trace(sources)
        try:
            # Design-implication answers use backend FACT/INFERENCE labels + chips;
            # per-claim grounding already validated — do not all-or-nothing reject.
            if (
                not trace.optimizations.get("design_implication")
                and not trace.optimizations.get("checklist_gen")
                and not trace.optimizations.get("scope_summary")
                and not trace.optimizations.get("applicability")
                and not trace.optimizations.get("retest_scope")
            ):
                assert_prose_sections_match_citations(answer_text, sources)
        except AssertionError as exc:
            logger.error("citation integrity failed after render: %s", exc)
            log_groundedness_violation(
                question=question,
                model_output=result.text,
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
                invalid_ids=[f"integrity:{exc}"],
            )
            closest = closest_retrieved_chunk(chunks, question=question)
            answer_text = grounding_rejected_message(closest)
            sources = [to_source(closest)] if closest is not None else []
            failure_kind = FAILURE_GROUNDING_REJECTED
            not_found = True
            trace.not_found = True
            trace.faithfulness_passed = False
            trace.optimizations["failure_kind"] = FAILURE_GROUNDING_REJECTED
            trace.citations = _citations_for_trace(sources)

    # Numeric fidelity: reject answers that alter the user's measured figures.
    if failure_kind is None and answer_text:
        fidelity = check_numeric_fidelity(condensed or question, answer_text)
        if fidelity.rejected:
            log_numeric_hallucination(
                question=condensed or question,
                answer=answer_text,
                result=fidelity,
                trace_id=trace.trace_id,
                model=result.model,
                provider=result.served_provider or result.provider,
            )
            answer_text = _NUMERIC_CONFIRM
            sources = []
            failure_kind = FAILURE_NUMERIC_HALLUCINATION
            not_found = True
            trace.not_found = True
            trace.faithfulness_passed = False
            trace.optimizations["failure_kind"] = FAILURE_NUMERIC_HALLUCINATION
            trace.optimizations["numeric_hallucination"] = {
                "offending_token": fidelity.offending_token,
                "reason": fidelity.reason,
            }
            trace.citations = []

    # Invariant: success XOR exactly one failure_kind.
    if not_found:
        assert failure_kind in {
            FAILURE_RETRIEVAL_MISS,
            FAILURE_GROUNDING_REJECTED,
            FAILURE_NUMERIC_HALLUCINATION,
        }
    else:
        assert failure_kind is None
        assert answer_text and not _is_insufficient(answer_text)

    logger.info(
        "answer provider=%s model=%s cache=%s cost=%.6f sources=%d failure=%s trace=%s",
        result.served_provider or result.provider,
        result.model,
        result.cache_status or ("HIT" if result.cached else "MISS"),
        result.cost_usd,
        len(sources),
        failure_kind or "ok",
        trace.trace_id,
    )
    metrics = trace.finalize()
    intent_fields = _intent_fields(routed_obj=routed)
    payload = {
        "question": condensed,
        "answer": answer_text,
        "sources": [s.model_dump() for s in sources],
        "model": result.model,
        "provider": result.served_provider or result.provider,
        "target_index": result.target_index,
        "answering_provider_was_fallback": bool(result.was_fallback),
        "not_found": not_found,
        "failure_kind": failure_kind,
        "input_tokens": int(metrics.get("input_tokens") or 0),
        "output_tokens": int(metrics.get("output_tokens") or 0),
        "embedding_tokens": int(metrics.get("embedding_tokens") or 0),
        "cost_usd": float(metrics.get("cost_usd") or 0.0),
        "trace_id": trace.trace_id,
        **intent_fields,
    }
    if not skip_answer_cache:
        store(condensed, payload, regulation_id=regulation_id)
        # Also index the raw first-turn phrasing when no condensation ran.
        if not condensation_applied and condensed != question:
            store(question, payload, regulation_id=regulation_id)

    return _with_conv(
        AnswerResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            model=result.model,
            provider=result.served_provider or result.provider,
            target_index=result.target_index,
            answering_provider_was_fallback=bool(result.was_fallback),
            cached=result.cached,
            not_found=not_found,
            failure_kind=failure_kind,
            trace_id=trace.trace_id,
            metrics=metrics,
            condensed_question=condensed,
            condensation_applied=condensation_applied,
            conversation_id=cid,
            **intent_fields,
        )
    )


def answer_as_dict(question: str, **kwargs: Any) -> dict[str, Any]:
    """Convenience: plain dict for JSON serialization."""
    return answer_question(question, **kwargs).model_dump()
