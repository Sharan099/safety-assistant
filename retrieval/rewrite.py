"""Query rewrite: follow-up condensation + local acronym expand + optional split."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from api.conversations import Turn, format_history_for_prompt
from generation.llm_client import LLMClient, LLMRole
from retrieval.acronyms import expand_acronyms

logger = logging.getLogger(__name__)

CONDENSE_SYSTEM = """\
You rewrite a follow-up message into a standalone UNECE passive-safety question.

Given recent conversation turns and a new user message, output ONLY the rewritten
question as plain text (no JSON, no quotes, no preamble).

Rules:
- Resolve pronouns and implicit references (it, that, the rear seat, what about…).
- Carry forward the regulation, impact type, and criterion context from prior turns
  when the follow-up depends on them.
- Keep clause numbers, limits, units, and technical terms.
- Do not answer the question — only rewrite it.
- One sentence preferred.
"""

# LLM only splits already-expanded queries — acronym expansion is local.
REWRITE_SYSTEM = """\
You rewrite engineer questions for UNECE passive-safety regulation retrieval.

Acronyms are ALREADY expanded in the input (e.g. "VC (Viscous Criterion)").
Do not change those expansions. Return ONLY valid JSON (no markdown):
{
  "expanded": "the input question unchanged or lightly cleaned",
  "subqueries": ["sub-question 1", "sub-question 2"]
}

Rules:
- If the question has multiple independent parts (and/or; commas; "also"),
  split into focused subqueries; otherwise subqueries = [expanded].
- Keep clause numbers, limits, units, and parenthetical expansions.
- Max 4 subqueries.
"""


class RewriteResult(BaseModel):
    original: str
    condensed: str = ""
    expanded: str = ""
    subqueries: list[str] = Field(default_factory=list)
    condensation_applied: bool = False


def _heuristic_split(expanded: str) -> list[str]:
    """Split multi-part questions on common separators."""
    parts = re.split(
        r"\?\s+(?=[A-Z])|\s+;\s+|\s+—\s+|\band also\b|\bas well as\b|\band what\b",
        expanded,
        flags=re.IGNORECASE,
    )
    cleaned = [
        p.strip(" ?") + ("?" if not p.strip().endswith("?") else "")
        for p in parts
        if p.strip()
    ]
    if len(cleaned) <= 1:
        return [expanded.strip()]
    return cleaned[:4]


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _heuristic_condense(question: str, history: Sequence[Turn]) -> str:
    """Deterministic fallback when the small model is unavailable."""
    q = (question or "").strip()
    if not history:
        return q
    prior = history[-1].question.strip()
    # Pull regulation / criterion / vehicle tokens from prior turns.
    ctx_bits: list[str] = []
    blob = " ".join(t.question for t in history)
    for pat, label in (
        (r"\bUN[- ]?ECE[- ]?R\s*94\b|\bR94\b|\bRegulation No\.?\s*94\b", "UN-ECE-R94"),
        (r"\bUN[- ]?ECE[- ]?R\s*95\b|\bR95\b|\bRegulation No\.?\s*95\b", "UN-ECE-R95"),
        (r"\bUN[- ]?ECE[- ]?R\s*16\b|\bR16\b", "UN-ECE-R16"),
        (r"\bHPC\b|\bHead Performance\b", "HPC"),
        (r"\bHIC\b", "HIC"),
        (r"\bThCC\b|\bThorax Compression\b", "ThCC"),
        (r"\bRDC\b|\bRib Deflection\b", "RDC"),
        (r"\bPSPF\b|\bPubic Symphysis\b", "PSPF"),
        (r"\bVC\b|\bViscous Criterion\b", "VC"),
        (r"\bfrontal\b", "frontal impact"),
        (r"\blateral\b|\bside impact\b", "side impact"),
        (r"\brear seat\b", "rear seat"),
        (r"(?i)\bBMW\s+X\d+\b", None),  # filled from match
        (r"(?i)\b(?:M1|N1|M2|N2)\b", None),
        (r"(?i)\b(?:BEV|EV|electric|hybrid|PHEV|ICE)\b", None),
    ):
        m = re.search(pat, blob, re.I)
        if not m:
            continue
        bit = label if label else m.group(0)
        if bit.lower() not in q.lower() and bit not in ctx_bits:
            ctx_bits.append(bit)
    low = q.lower()
    if re.match(r"^(what about|and the|how about|and what about)\b", low):
        rest = re.sub(
            r"^(what about|and the|how about|and what about)\s+",
            "",
            q,
            flags=re.I,
        ).strip(" ?")
        rest = re.sub(r"^(the|a|an)\s+", "", rest, flags=re.I).strip()
        base = f"What is the {rest}" if rest else q
        if ctx_bits:
            return f"{base} under {' / '.join(ctx_bits)}?"
        if prior:
            return f"{base} (following: {prior})?"
        return base if base.endswith("?") else base + "?"
    if ctx_bits and not any(b.lower() in low for b in ctx_bits):
        return f"{q.rstrip(' ?')} ({', '.join(ctx_bits)})?"
    return q


def condense_followup(
    question: str,
    history: Sequence[Turn],
    *,
    llm: LLMClient | None = None,
    use_llm: bool | None = None,
) -> tuple[str, bool]:
    """Rewrite a follow-up into a standalone question using the small model.

    Returns ``(condensed, applied)``. Skips the LLM entirely when ``history`` is empty.
    """
    question = (question or "").strip()
    if not question:
        return "", False
    if not history:
        return question, False

    client = llm or LLMClient()
    if use_llm is None:
        use_llm = client.provider != "mock"

    if not use_llm:
        condensed = _heuristic_condense(question, history)
        logger.info(
            "condensation heuristic original=%r condensed=%r",
            question,
            condensed,
        )
        return condensed, True

    user = (
        "Recent conversation:\n"
        f"{format_history_for_prompt(history)}\n\n"
        f"New user message:\n{question}\n\n"
        "Standalone rewritten question:"
    )
    result = client.complete(
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content": user},
        ],
        role=LLMRole.REWRITE,
        question=f"condense:{question}",
        chunk_ids=[],
        max_tokens=256,
        temperature=0.0,
        seed=42,
        skip_cache=True,
    )
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.add_llm(
                role="rewrite",
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
    except Exception:  # noqa: BLE001
        pass

    condensed = (result.text or "").strip()
    condensed = condensed.strip("\"'`")
    if condensed.lower().startswith("standalone"):
        condensed = re.sub(r"(?i)^standalone(?:\s+rewritten)?\s+question:\s*", "", condensed).strip()
    if not condensed or len(condensed) < 3:
        condensed = _heuristic_condense(question, history)
    logger.info(
        "condensation llm original=%r condensed=%r model=%s",
        question,
        condensed,
        result.model,
    )
    return condensed, True


def rewrite_query(
    question: str,
    *,
    llm: LLMClient | None = None,
    use_llm: bool | None = None,
    history: Sequence[Turn] | None = None,
) -> RewriteResult:
    """Condense follow-ups (if history), expand acronyms, optionally LLM-split.

    ``original`` is the raw user message. ``condensed`` is used for retrieval
    (equals original on turn one). Answer generation should stay stateless on
    the condensed standalone question + retrieved context only.
    """
    question = (question or "").strip()
    if not question:
        return RewriteResult(original="", condensed="", expanded="", subqueries=[])

    condensed, applied = condense_followup(
        question,
        list(history or []),
        llm=llm,
        use_llm=use_llm,
    )
    # Deterministic local expand — never an LLM call.
    expanded = expand_acronyms(condensed)

    client = llm or LLMClient()
    if use_llm is None:
        use_llm = client.provider != "mock"

    # Subquery split: heuristics are deterministic. Optional LLM split via
    # RETRIEVAL_REWRITE_LLM=1 (even at temperature=0, providers can still drift).
    # Never LLM-split enumerative list/every asks — that strips the cue and
    # turns coverage queries into narrow "what are the requirements…" lookups.
    llm_split = (os.getenv("RETRIEVAL_REWRITE_LLM") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        from retrieval.enumerative import is_enumerative_query

        if is_enumerative_query(question) or is_enumerative_query(condensed):
            if llm_split:
                logger.info(
                    "rewrite skip llm-split for enumerative question=%r",
                    question[:120],
                )
            llm_split = False
    except Exception:  # noqa: BLE001
        pass

    if use_llm and llm_split:
        result = client.rewrite(expanded, system=REWRITE_SYSTEM)
        try:
            from observability.context import get_current_trace

            tr = get_current_trace()
            if tr is not None:
                tr.add_llm(
                    role="rewrite",
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
                tr.optimizations["model_routing"] = {
                    "rewrite_model": result.model,
                    "answer_model": client.large_model,
                    "rewrite_provider": result.served_provider or result.provider,
                }
        except Exception:  # noqa: BLE001
            pass
        data = _parse_llm_json(result.text)
        if data:
            llm_expanded = str(data.get("expanded") or "").strip() or expanded
            llm_expanded = expand_acronyms(llm_expanded)
            subs = data.get("subqueries") or []
            subqueries = [
                expand_acronyms(str(s).strip()) for s in subs if str(s).strip()
            ]
            if not subqueries:
                subqueries = _heuristic_split(llm_expanded)
            out = RewriteResult(
                original=question,
                condensed=condensed,
                expanded=llm_expanded,
                subqueries=subqueries[:4],
                condensation_applied=applied,
            )
            logger.info(
                "rewrite llm-split original=%r condensed=%r subqueries=%d",
                question,
                condensed,
                len(out.subqueries),
            )
            return out
        logger.warning("rewrite LLM JSON parse failed; falling back to heuristics")

    subqueries = _heuristic_split(expanded)
    out = RewriteResult(
        original=question,
        condensed=condensed,
        expanded=expanded,
        subqueries=subqueries,
        condensation_applied=applied,
    )
    logger.info(
        "rewrite heuristic original=%r condensed=%r subqueries=%d",
        question,
        condensed,
        len(out.subqueries),
    )
    return out
