"""Enumerative / broad-recall query classifier (regex only — no LLM).

Intentional exception to the standard top-5 / ~3k context path: list/every/
summarize-all style asks need broader coverage.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Post-rerank breadth for enumerative queries (vs standard RERANK_TOP_K=5).
DEFAULT_ENUM_RERANK_TOP_K = 20
# Hybrid recall floor when enumerative (must be ≥ rerank depth).
DEFAULT_ENUM_HYBRID_TOP_K = 40

# Explicit cues — keep strict so single-fact queries stay on the narrow path.
_ENUM_RE = re.compile(
    r"(?ix)"
    r"("
    r"\blist\s+(?:all|every|each)\b"
    r"|\blist\s+the\s+(?:requirements|criteria|tests|steps|items|checks|provisions)\b"
    # "every requirement related to …" / "all requirements related to …"
    # (with or without a leading "list")
    r"|\b(?:list\s+)?every\s+requirement\s+related\s+to\b"
    r"|\b(?:list\s+)?all\s+(?:of\s+)?(?:the\s+)?requirements?\s+related\s+to\b"
    r"|\bevery\s+requirement\s+(?:concerning|regarding|about|for)\b"
    r"|\ball\s+requirements?\s+(?:concerning|regarding|about|for)\b"
    r"|\bsummarize\s+all\b"
    r"|\bsummarise\s+all\b"
    r"|\benumerate\b"
    r"|\bchecklists?\b"
    r"|\ball\s+(?:of\s+)?(?:the\s+)?(?:requirements|criteria|tests|steps|items|checks|provisions)\b"
    r"|\bevery\s+(?:requirement|criterion|test|step|item|check|provision)\b"
    r"|\bwhat\s+are\s+all\s+(?:of\s+)?(?:the\s+)?(?:requirements|criteria|tests|steps|items|checks|provisions)\b"
    r"|\bwhat\s+are\s+(?:all\s+)?(?:the\s+)?(?:requirements|criteria|tests|steps)\s+for\b"
    r"|\bprepare(?:ing)?\s+a\s+(?:vehicle|test)\b"
    r")"
)

# Topic after "related to X" / "concerning X" for coverage-biased retrieval.
_TOPIC_RELATED_RE = re.compile(
    r"(?ix)"
    r"(?:related\s+to|concerning|regarding|about|for)\s+"
    r"(?P<topic>[a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9\-]*){0,3})"
    r"(?=\s+in\s+|\s+under\s+|\s+for\s+UN|\s*\?|$|,)"
)

# High-value enumerative topics that must not be crowded out by preamble/admin text.
_TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "door": ("door", "doors"),
    "doors": ("door", "doors"),
    "seat belt": ("seat-belt", "seat belt", "safety-belt", "safety belt", "belt"),
    "seatbelt": ("seat-belt", "seat belt", "safety-belt", "belt"),
    "reess": ("reess", "electrical energy storage"),
}

_NAMED_REG_RE = re.compile(
    r"(?ix)\b(?:UN[- ]?(?:ECE[- ]?)?)?R\s*(\d{2,3})\b"
    r"|\bRegulation\s+No\.?\s*(\d{2,3})\b"
)

# Cross-regulation intents — do NOT hard-filter to a single regulation_id.
_COMPARATIVE_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bcompar(?:e|ison|ing)\b"
    r"|\bversus\b"
    r"|\bvs\.?\b"
    r"|\bdiffer(?:ence|ences|ent)?\b"
    r"|\brelat(?:e|es|ed|ion|ionship|ive|ively)\b"
    r"|\brelevant\b"
    r"|\bin\s+contrast\b"
    r"|\bas\s+opposed\s+to\b"
    r"|\bsimilarit(?:y|ies)\s+between\b"
    r"|\bcross[- ]reg"
    r")"
)


@dataclass(frozen=True)
class EnumerativeClassification:
    is_enumerative: bool
    rerank_top_k: int
    hybrid_top_k: int
    named_regulation_id: str | None
    reason: str = ""


def is_enumerative_query(question: str) -> bool:
    """True for list-all / every / summarize-all style asks — not single-fact lookups."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(_ENUM_RE.search(q))


def is_comparative_query(question: str) -> bool:
    """True when the ask intentionally spans / contrasts multiple regulations.

    Uses comparative cues (compare / differ / vs / relate / relevant). When the
    question also names ≥2 regulations, prefer :func:`retrieval.comparison.is_comparison_mode_query`.
    """
    q = (question or "").strip()
    if not q:
        return False
    try:
        from retrieval.comparison import is_comparison_mode_query

        if is_comparison_mode_query(q):
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool(_COMPARATIVE_RE.search(q))


def detect_named_regulation(question: str) -> str | None:
    """If the query names a single UNECE regulation, return canonical ``UN-ECE-Rxx``."""
    q = (question or "").strip()
    if not q:
        return None
    nums: list[str] = []
    for m in _NAMED_REG_RE.finditer(q):
        n = m.group(1) or m.group(2)
        if n:
            nums.append(n.lstrip("0") or n)
    uniq = list(dict.fromkeys(nums))
    if len(uniq) != 1:
        return None
    try:
        from agent.regs import normalize_regulation

        return normalize_regulation(f"R{uniq[0]}")
    except Exception:  # noqa: BLE001
        return f"UN-ECE-R{uniq[0]}"


# Domain-exclusive topics → regulation when the user does not name one.
# Narrow on purpose: bare "seat belt" can be design/crash cross-reg; ELR/retractor
# is R16-owned and must not fall through to R94/R95 boilerplate.
_R16_TOPIC_RE = re.compile(
    r"(?ix)\b("
    r"emergency\s*-?\s*locking\s+retractor|"
    r"emergency\s+locking|"
    r"\bELR\b|"
    r"retractors?|"
    r"non[\s-]?locking\s+retractor|"
    r"manually\s+unlocking\s+retractor|"
    r"automatically\s+locking\s+retractor|"
    r"safety[\s-]?belt\s+reminder|"
    r"\bSBR\b|"
    r"belt\s+adjustment\s+device\s+for\s+height"
    r")\b"
)


def infer_regulation_from_topic(question: str) -> str | None:
    """Map unambiguous domain topics to a regulation when none is named."""
    q = (question or "").strip()
    if not q:
        return None
    if detect_named_regulation(q) is not None:
        return None
    if is_comparative_query(q):
        return None
    try:
        from retrieval.multi_regulation import is_plural_regulation_query

        if is_plural_regulation_query(q):
            return None
    except Exception:  # noqa: BLE001
        pass
    if _R16_TOPIC_RE.search(q):
        return "UN-ECE-R16"
    return None


def resolve_hard_regulation_filter(question: str) -> str | None:
    """Hard Qdrant ``regulation_id`` filter for single-named-reg queries.

    Returns canonical ``UN-ECE-Rxx`` when the query names exactly one regulation
    and is not comparative / plural-scope. Comparative and "which regulations"
    survey asks return ``None`` so retrieval may span corpora intentionally.

    Also applies narrow topic→regulation inference (e.g. ELR/retractor → R16)
    so seat-belt concepts are not answered from R95 boilerplate.
    """
    if is_comparative_query(question):
        return None
    try:
        from retrieval.multi_regulation import is_plural_regulation_query

        if is_plural_regulation_query(question):
            return None
    except Exception:  # noqa: BLE001
        pass
    named = detect_named_regulation(question)
    if named:
        return named
    return infer_regulation_from_topic(question)


def enum_rerank_top_k() -> int:
    try:
        return max(5, int((os.getenv("ENUM_RERANK_TOP_K") or str(DEFAULT_ENUM_RERANK_TOP_K)).strip()))
    except ValueError:
        return DEFAULT_ENUM_RERANK_TOP_K


def enum_hybrid_top_k() -> int:
    try:
        return max(
            enum_rerank_top_k(),
            int((os.getenv("ENUM_HYBRID_TOP_K") or str(DEFAULT_ENUM_HYBRID_TOP_K)).strip()),
        )
    except ValueError:
        return max(enum_rerank_top_k(), DEFAULT_ENUM_HYBRID_TOP_K)


def extract_enumerative_topic(question: str) -> str | None:
    """Return a normalized topic token (e.g. ``doors``) when the ask scopes to one."""
    q = (question or "").strip()
    if not q:
        return None
    m = _TOPIC_RELATED_RE.search(q)
    if not m:
        # Bare "door requirements" / "doors in UN R95"
        m2 = re.search(r"(?i)\b(doors?|seat[\s-]?belts?|reess)\b", q)
        return m2.group(1).lower().replace("-", " ") if m2 else None
    topic = re.sub(r"\s+", " ", (m.group("topic") or "").strip().lower())
    # Strip trailing regulation crumbs / prepositions accidentally captured.
    topic = re.sub(
        r"\b(un|ece|r\d{2,3}|regulation|in|under|for|of|the|a|an)\b",
        "",
        topic,
    ).strip()
    topic = re.sub(r"\s+", " ", topic).strip(" -")
    return topic or None

def topic_match_terms(topic: str | None) -> tuple[str, ...]:
    if not topic:
        return ()
    key = topic.lower().strip()
    if key in _TOPIC_ALIASES:
        return _TOPIC_ALIASES[key]
    # Singular/plural door
    if key.rstrip("s") == "door" or key == "door":
        return _TOPIC_ALIASES["doors"]
    return (key, key.rstrip("s") if key.endswith("s") and len(key) > 3 else key)


def topic_focused_subquery(question: str) -> str | None:
    """Extra hybrid subquery so scattered topic clauses are not lost to preamble."""
    topic = extract_enumerative_topic(question)
    terms = topic_match_terms(topic)
    if not terms:
        return None
    primary = terms[0]
    return (
        f"{primary} {' '.join(terms)} requirement shall must closed locked "
        f"opening latch during the test"
    )


def chunk_matches_enumerative_topic(chunk: object, topic: str | None) -> bool:
    terms = topic_match_terms(topic)
    if not terms:
        return False
    blob = " ".join(
        str(getattr(chunk, attr, "") or "")
        for attr in ("text", "section_title", "section_number")
    ).lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", blob) for t in terms)


def bias_chunks_for_enumerative_topic(
    chunks: list,
    *,
    question: str,
) -> list:
    """Prefer topic-bearing requirement clauses over preamble/admin text."""
    topic = extract_enumerative_topic(question)
    if not topic or not chunks:
        return list(chunks)
    scored: list[tuple[float, int, object]] = []
    for i, chunk in enumerate(chunks):
        base = float(getattr(chunk, "score", 0.0) or 0.0)
        boost = 0.0
        if chunk_matches_enumerative_topic(chunk, topic):
            boost += 2.5
            text = str(getattr(chunk, "text", "") or "").lower()
            if re.search(r"\bshall\b|\bmust\b|\bshall\s+not\b", text):
                boost += 0.75
            # Prefer performance / post-impact requirement trees over definitions.
            sec = str(getattr(chunk, "section_number", "") or "")
            if re.match(r"(?i)^5\.", sec.strip()):
                boost += 0.5
            if re.match(r"(?i)^annex\s*4", sec.strip()):
                boost += 0.35
        else:
            # Demote preamble / application / admin without the topic.
            sec = str(getattr(chunk, "section_number", "") or "")
            if re.match(r"(?i)^(preamble|3|4|6|annex\s*[12])$", sec.strip()):
                boost -= 2.0
            elif re.match(r"(?i)^2\.", sec.strip()):
                boost -= 0.5
        scored.append((base + boost, i, chunk))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for fused, _, chunk in scored:
        if hasattr(chunk, "model_copy"):
            try:
                out.append(chunk.model_copy(update={"score": fused}))
                continue
            except Exception:  # noqa: BLE001
                pass
        try:
            setattr(chunk, "score", fused)
        except Exception:  # noqa: BLE001
            pass
        out.append(chunk)
    logger.info(
        "enumerative topic bias topic=%r top=%s",
        topic,
        [
            (
                getattr(c, "section_number", None),
                (getattr(c, "chunk_id", None) or "")[:12],
                chunk_matches_enumerative_topic(c, topic),
            )
            for c in out[:6]
        ],
    )
    return out


def classify_enumerative(question: str) -> EnumerativeClassification:
    """Regex classifier + breadth knobs for the enumerative retrieve path."""
    fired = is_enumerative_query(question)
    logger.info(
        "enumerative_classifier fired=%s query=%r",
        fired,
        (question or "")[:160],
    )
    if not fired:
        return EnumerativeClassification(
            is_enumerative=False,
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "5") or "5"),
            hybrid_top_k=int(os.getenv("HYBRID_TOP_K", "30") or "30"),
            named_regulation_id=detect_named_regulation(question),
            reason="",
        )
    m = _ENUM_RE.search(question or "")
    reason = m.group(0) if m else "enumerative"
    named = detect_named_regulation(question)
    cls = EnumerativeClassification(
        is_enumerative=True,
        rerank_top_k=enum_rerank_top_k(),
        hybrid_top_k=enum_hybrid_top_k(),
        named_regulation_id=named,
        reason=reason.strip(),
    )
    logger.info(
        "enumerative_classified cue=%r topic=%r rerank_k=%d hybrid_k=%d named_reg=%s",
        cls.reason,
        extract_enumerative_topic(question),
        cls.rerank_top_k,
        cls.hybrid_top_k,
        cls.named_regulation_id,
    )
    return cls
