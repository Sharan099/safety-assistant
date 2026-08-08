"""Hard caps on retrieved context sent to the answer LLM.

Fix 13 — standard single-fact / factual / compliance: ≤5 chunks / ~3000 tokens
(hard ceiling; env can only lower). Wider budgets are explicit for enumerative,
multi-regulation, multi-criterion, and Layer 4–5 intent pipelines
(design / checklist / scope / applicability / retest).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Standard single-fact budget (Fix 13 hard ceiling).
STANDARD_MAX_CHUNKS = 5
STANDARD_MAX_TOKENS = 3000

# Intent budget modes that inherit the Fix 13 hard ceiling.
STANDARD_BUDGET_MODES = frozenset({"standard", "factual_lookup", "compliance_check"})

# Enumerative / checklist budget — intentional exception to the standard cap.
# Aligned with ENUM_RERANK_TOP_K (default 20): broader coverage for list/every asks.
ENUM_MAX_CHUNKS = 20
ENUM_MAX_TOKENS = 8000

# Cap for a single small-to-big expanded parent; above this keep the leaf.
DEFAULT_MAX_PARENT_EXPAND_TOKENS = 800


def approx_tokens(text: str) -> int:
    """Fast whitespace token proxy (good enough for budget guards)."""
    t = (text or "").strip()
    if not t:
        return 0
    return max(1, len(t.split()))


def is_enumerative_query(question: str) -> bool:
    """Delegate to the dedicated regex classifier (no LLM)."""
    from retrieval.enumerative import is_enumerative_query as _enum

    return _enum(question)


def context_budgets(
    question: str,
    *,
    routed: Any | None = None,
) -> tuple[int, int, str]:
    """Return ``(max_chunks, max_tokens, mode)`` for this question.

    When ``routed`` (a ``RoutedQuery``) is provided, budgets come solely from that
    intent's ``PipelineConfig`` — not from a shared heuristic table.
    """
    if routed is not None:
        from retrieval.enumerative import is_enumerative_query as _is_enum_q
        from retrieval.limits_aggregation import (
            is_limits_aggregation_query as _is_limits_agg,
        )

        # Fix 15: enumerative / limits-aggregation asks keep the broad budget even
        # when the router labeled the intent FACTUAL_LOOKUP (narrow 5/3k).
        if _is_limits_agg(question) or _is_enum_q(question):
            chunks = int(os.getenv("ENUM_CONTEXT_MAX_CHUNKS", str(ENUM_MAX_CHUNKS)))
            tokens = int(os.getenv("ENUM_CONTEXT_MAX_TOKENS", str(ENUM_MAX_TOKENS)))
            mode = (
                "limits_aggregation"
                if _is_limits_agg(question)
                else "enumerative"
            )
            return max(1, chunks), max(500, tokens), mode

        from retrieval.router import budgets_for_intent

        return budgets_for_intent(routed)

    if is_enumerative_query(question):
        chunks = int(os.getenv("ENUM_CONTEXT_MAX_CHUNKS", str(ENUM_MAX_CHUNKS)))
        tokens = int(os.getenv("ENUM_CONTEXT_MAX_TOKENS", str(ENUM_MAX_TOKENS)))
        return max(1, chunks), max(500, tokens), "enumerative"

    from retrieval.multi_regulation import is_plural_regulation_query, per_regulation_top_k

    if is_plural_regulation_query(question):
        per_k = per_regulation_top_k()
        # Room for a few chunks from each indexed regulation.
        try:
            from retrieval.retrieve import indexed_regulation_ids

            n_regs = max(1, len(indexed_regulation_ids()))
        except Exception:  # noqa: BLE001
            n_regs = 4
        chunks = max(
            int(os.getenv("CONTEXT_MAX_CHUNKS", str(STANDARD_MAX_CHUNKS))),
            per_k * n_regs,
            int(os.getenv("MULTI_REG_CONTEXT_MAX_CHUNKS", str(per_k * n_regs))),
        )
        tokens = max(
            int(os.getenv("CONTEXT_MAX_TOKENS", str(STANDARD_MAX_TOKENS))),
            int(os.getenv("MULTI_REG_CONTEXT_MAX_TOKENS", "8000")),
        )
        return max(1, chunks), max(500, tokens), "multi_regulation"

    from retrieval.multi_criterion import is_multi_criterion_query, list_named_criteria, per_criterion_top_k

    if is_multi_criterion_query(question):
        n = len(list_named_criteria(question))
        per_k = per_criterion_top_k()
        # Enough room for top-k per criterion after merge/dedupe.
        chunks = max(
            int(os.getenv("CONTEXT_MAX_CHUNKS", str(STANDARD_MAX_CHUNKS))),
            per_k * n,
            int(os.getenv("MULTI_CRITERION_CONTEXT_MAX_CHUNKS", str(per_k * n))),
        )
        tokens = max(
            int(os.getenv("CONTEXT_MAX_TOKENS", str(STANDARD_MAX_TOKENS))),
            int(os.getenv("MULTI_CRITERION_CONTEXT_MAX_TOKENS", "6000")),
        )
        return max(1, chunks), max(500, tokens), "multi_criterion"

    # Hard ceiling: CONTEXT_MAX_* may only lower the standard budget, not raise it.
    chunks = min(
        STANDARD_MAX_CHUNKS,
        max(1, int(os.getenv("CONTEXT_MAX_CHUNKS", str(STANDARD_MAX_CHUNKS)))),
    )
    tokens = min(
        STANDARD_MAX_TOKENS,
        max(500, int(os.getenv("CONTEXT_MAX_TOKENS", str(STANDARD_MAX_TOKENS)))),
    )
    return chunks, tokens, "standard"


def max_parent_expand_tokens() -> int:
    try:
        return max(
            100,
            int(
                (
                    os.getenv("MAX_PARENT_EXPAND_TOKENS")
                    or str(DEFAULT_MAX_PARENT_EXPAND_TOKENS)
                ).strip()
            ),
        )
    except ValueError:
        return DEFAULT_MAX_PARENT_EXPAND_TOKENS


def estimate_context_tokens(chunks: Sequence[Any]) -> int:
    total = 0
    for c in chunks:
        if hasattr(c, "context_block"):
            total += approx_tokens(c.context_block(index=1))
        else:
            total += approx_tokens(getattr(c, "text", "") or "")
    return total


_LIMIT_WINDOW_RE = re.compile(
    r"(?is)("
    r"(?:performance\s+criteria|shall\s+not\s+exceed|less\s+than\s+or\s+equal|"
    r"Rib\s+Deflection\s+Criterion|\bRDC\b|"
    r"Head\s+Performance\s+Criterion|\bHPC\b|\bHIC\b|"
    r"Viscous\s+Criterion|\bVC\b|"
    r"Pubic\s+Symphysis\s+Peak\s+Force|\bPSPF\b|"
    r"Thorax\s+Compression\s+Criterion|\bThCC\b|"
    r"fuel[- ]?feed|leakage\s+shall\s+not\s+exceed|\bg\s*/\s*min\b"
    r").{0,800})"
)


def excerpt_for_value_vs_limit(text: str, *, max_tokens: int) -> str:
    """Keep injury-criteria windows when a Specifications chunk is oversized."""
    text = (text or "").strip()
    if not text:
        return text
    if approx_tokens(text) <= max_tokens:
        return text
    windows = [m.group(1).strip() for m in _LIMIT_WINDOW_RE.finditer(text)]
    if windows:
        # Deduplicate overlapping captures while preserving order.
        seen: set[str] = set()
        parts: list[str] = []
        for w in windows:
            key = w[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(w)
        focused = "\n\n".join(parts)
        words = focused.split()
        if words:
            if len(words) > max_tokens:
                return " ".join(words[:max_tokens]) + " …"
            return focused
    words = text.split()
    return " ".join(words[:max_tokens]) + " …"


def excerpt_for_query_terms(text: str, question: str, *, max_tokens: int) -> str:
    """Keep windows around query keywords inside mega-chunks (e.g. buried R16 §6.2.5).

    When Docling collapsed many clauses into one chunk, leading-prose truncation
    would drop the matching clause. Prefer term-centered windows instead.
    """
    text = (text or "").strip()
    if not text:
        return text
    if approx_tokens(text) <= max_tokens:
        return text
    terms = [
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question or "")
        if t.lower()
        not in {
            "the",
            "and",
            "for",
            "under",
            "with",
            "from",
            "what",
            "which",
            "does",
            "need",
            "that",
            "this",
            "when",
            "after",
            "about",
            "into",
            "regulation",
            "requirements",
        }
    ]
    # Prefer longer / more specific terms first.
    terms = sorted(set(terms), key=len, reverse=True)[:12]
    low = text.lower()
    windows: list[str] = []
    words = text.split()
    # Approximate char span ~ 5 chars/token for windowing.
    half = max(80, (max_tokens // 2) * 5)
    for term in terms:
        idx = low.find(term.lower())
        if idx < 0:
            continue
        # Snap to nearest preceding numbered-clause start when present so we
        # do not keep hundreds of tokens of leading buckle/preamble padding.
        clause_starts = [
            m.start()
            for m in re.finditer(r"(?m)^\d+(?:\.\d+)*\.\s+", text[: idx + 1])
        ]
        if clause_starts:
            start = clause_starts[-1]
        else:
            start = max(0, idx - half)
            while start > 0 and text[start] not in " \n\t":
                start -= 1
        end = min(len(text), idx + len(term) + half)
        while end < len(text) and text[end - 1] not in " \n\t":
            end += 1
        windows.append(text[start:end].strip())
        if len(windows) >= 3:
            break
    if windows:
        focused = "\n\n…\n\n".join(windows)
        w = focused.split()
        if len(w) > max_tokens:
            return " ".join(w[:max_tokens]) + " …"
        return focused
    return " ".join(words[:max_tokens]) + " …"


def apply_context_budget(
    chunks: Sequence[Any],
    *,
    question: str,
    warn: bool = True,
    routed: Any | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Trim to chunk + token budgets. Returns ``(trimmed, stats)``.

    Logs a warning whenever the *incoming* set already exceeds the budget so
    regressions are visible before a cost review. Pass ``routed`` to use the
    intent-specific ``PipelineConfig`` instead of heuristic modes.
    """
    from retrieval.multi_criterion import is_multi_criterion_query
    from retrieval.value_limit import is_value_vs_limit_query

    max_chunks, max_tokens, mode = context_budgets(question, routed=routed)
    value_vs_limit = is_value_vs_limit_query(question) or is_multi_criterion_query(question)
    if routed is not None:
        try:
            from retrieval.router import QueryIntent

            intent = getattr(routed, "intent", None)
            if intent == QueryIntent.COMPLIANCE_CHECK:
                value_vs_limit = True
        except Exception:  # noqa: BLE001
            pass
    incoming = list(chunks)

    # For pass/fail questions, prefer injury-criteria excerpts before token trim
    # so §5 Specifications does not keep only the leading test-procedure prose.
    # Also excerpt mega-chunks toward query terms (buried ELR under mis-tagged §6.2.2).
    if incoming:
        focused: list[Any] = []
        per_chunk_cap = max(400, max_tokens // max(1, min(len(incoming), max_chunks)))
        for c in incoming:
            text = (getattr(c, "text", None) or "").strip()
            if approx_tokens(text) <= per_chunk_cap:
                focused.append(c)
                continue
            if value_vs_limit:
                clipped = excerpt_for_value_vs_limit(text, max_tokens=per_chunk_cap)
            else:
                clipped = excerpt_for_query_terms(
                    text, question, max_tokens=per_chunk_cap
                )
            if hasattr(c, "model_copy"):
                focused.append(c.model_copy(update={"text": clipped, "enriched_text": ""}))
            else:
                try:
                    c.text = clipped  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
                focused.append(c)
        incoming = focused

    in_tokens = estimate_context_tokens(incoming)
    over = len(incoming) > max_chunks or in_tokens > max_tokens
    if over and warn:
        logger.warning(
            "context_budget_exceeded mode=%s incoming_chunks=%d incoming_tokens≈%d "
            "cap_chunks=%d cap_tokens=%d — trimming before LLM",
            mode,
            len(incoming),
            in_tokens,
            max_chunks,
            max_tokens,
        )

    trimmed = incoming[:max_chunks]
    # Drop from the end until under token budget (keep highest-ranked first).
    while len(trimmed) > 1 and estimate_context_tokens(trimmed) > max_tokens:
        trimmed.pop()
    # If a single remaining chunk is still over budget, truncate its text in place
    # for the LLM path only when it exposes a mutable ``text`` field.
    if trimmed and estimate_context_tokens(trimmed) > max_tokens:
        only = trimmed[0]
        text = (getattr(only, "text", None) or "").strip()
        clipped = (
            excerpt_for_value_vs_limit(text, max_tokens=max_tokens)
            if value_vs_limit
            else excerpt_for_query_terms(text, question, max_tokens=max_tokens)
        )
        if hasattr(only, "model_copy"):
            trimmed[0] = only.model_copy(update={"text": clipped, "enriched_text": ""})
        elif hasattr(only, "text"):
            only.text = clipped  # type: ignore[attr-defined]
        logger.warning(
            "context_budget: truncated single oversized chunk %s to ~%d tokens",
            getattr(only, "chunk_id", "?"),
            max_tokens,
        )

    out_tokens = estimate_context_tokens(trimmed)
    stats = {
        "mode": mode,
        "max_chunks": max_chunks,
        "max_tokens": max_tokens,
        "incoming_chunks": len(incoming),
        "incoming_tokens": in_tokens,
        "context_chunks_to_llm": len(trimmed),
        "context_tokens_est": out_tokens,
        "trimmed": over or len(trimmed) != len(incoming),
        "budget_exceeded_incoming": over,
        "value_vs_limit": value_vs_limit,
    }
    logger.info(
        "context_to_llm chunks=%d tokens≈%d mode=%s (incoming %d / ≈%d)",
        stats["context_chunks_to_llm"],
        stats["context_tokens_est"],
        mode,
        len(incoming),
        in_tokens,
    )
    return trimmed, stats
