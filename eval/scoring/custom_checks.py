"""Exact / regex hard gates — no LLM judge (numeric safety is zero-tolerance).

Applies to ``numeric_safety``, ``compliance_check``, and ``cross_regulation``
as pass/fail gates alongside any RAGAS scores those cases also receive.

Also exports ``regulation_match_check`` — shared asked-vs-retrieved regulation
guard used by ``hallucination_probe`` and ``out_of_scope`` scorers.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from generation.answer import AnswerSegment, StructuredAnswer, validate_segment_chunk_ids
from generation.numeric_guard import (
    UserNumber,
    _number_appears_verbatim,
    extract_user_numbers,
    normalize_number_token,
)

# Categories that must pass these checks as hard gates.
CUSTOM_HARD_GATE_CATEGORIES = frozenset(
    {
        "numeric_safety",
        "compliance_check",
        "cross_regulation",
    }
)

# Digits (optional sign + optional decimal) used when mining must_not / answer text.
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w./])([+-]?\d+(?:[.,]\d+)?)(?![\w.])")

# must_not items that are ONLY a number (optional unit) — safe to promote to
# banned_numbers. Prose phrases like "measured 42 mm" keep phrase-only checks so
# a legitimate limit token "42" / "42.0" is not globally banned.
_PURE_NUMBER_OR_UNIT_RE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:\s*[A-Za-z%]+(?:\s*/\s*[A-Za-z]+)?)?$"
)

# Explicit comparative / multi-reg survey phrasing → allow cross-reg citations.
_COMPARATIVE_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bcompare\b|\bversus\b|\bvs\.?\b|\bdiffer(?:ence|s|ent)?\b"
    r"|\brelat(?:e|es|ed|ion|ionship|ive|ively)\b|\brelevant\b|\bboth\b"
    r"|\bacross\s+regulations?\b"
    r"|\bwhich\s+regulations?\b|\bin\s+general\b|\bmulti[- ]?reg"
    r"|\bhow\s+does\s+un\s*r\d+\s+relat"
    r")"
)

_REG_MENTION_RE = re.compile(
    r"(?ix)\b(?:UN(?:-ECE)?[-\s]*)?R\s*(\d+)\b|\bUN-ECE-R(\d+)\b"
)


def _normalize_reg_id(raw: str | None) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    upper = text.upper().replace(" ", "")
    m = re.search(r"(?:UN-?ECE-?)?R0*(\d+)", upper)
    if m:
        return f"UN-ECE-R{int(m.group(1))}"
    return text


def _extract_number_strings(text: str) -> list[str]:
    """All digit tokens in ``text`` (normalized), preserving order / uniqueness."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _NUMBER_TOKEN_RE.finditer(text or ""):
        norm = normalize_number_token(m.group(1))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _number_token_in_text(answer: str, token: str) -> bool:
    """True if ``token`` appears as an exact number token in ``answer``.

    Splits ``answer`` into number-like tokens (optional sign + digits + one
    optional decimal) and requires exact equality after normalization.
    ``42`` does not match ``42.5`` or ``42.0``; ``1`` does not match ``10`` /
    ``100``.
    """
    if not token or not answer:
        return False
    target = normalize_number_token(token)
    if not target:
        return False
    for m in _NUMBER_TOKEN_RE.finditer(answer):
        if normalize_number_token(m.group(1)) == target:
            return True
    return False


def _forbidden_phrase_in_answer(answer: str, phrase: str) -> bool:
    """True if ``phrase`` occurs in ``answer`` with numbers as whole tokens.

    Prevents ``measured 0.9`` from matching inside ``measured 0.95``.
    Non-numeric text still matches case-insensitively as a literal.
    """
    if not phrase or not answer:
        return False
    parts: list[str] = []
    last = 0
    for m in _NUMBER_TOKEN_RE.finditer(phrase):
        parts.append(re.escape(phrase[last : m.start()]))
        num = re.escape(m.group(1))
        parts.append(rf"(?<![\d.]){num}(?![\d.])")
        last = m.end()
    parts.append(re.escape(phrase[last:]))
    return re.search("".join(parts), answer, flags=re.IGNORECASE) is not None


def numeric_verbatim_check(
    question: str,
    answer: str,
    *,
    must_not_contain: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pass iff every user-stated number appears verbatim and banned numbers do not.

    Uses :func:`generation.numeric_guard.extract_user_numbers` (measurement-aware
    regex) plus whole-token presence checks — pass/fail only, no graded score.
    """
    user_numbers: list[UserNumber] = extract_user_numbers(question or "")
    answer = answer or ""

    missing: list[str] = []
    for u in user_numbers:
        if not _number_appears_verbatim(answer, u):
            label = u.raw + (f" {u.unit}" if u.unit else "")
            missing.append(label.strip())

    banned_raw = [str(s).strip() for s in (must_not_contain or []) if str(s).strip()]
    banned_numbers: list[str] = []
    for item in banned_raw:
        # Only mine global banned digits from pure number/unit items (e.g. "3 g/min").
        # Prose bans stay phrase-only so limits like "42" / "1000.0" are not flagged.
        if _PURE_NUMBER_OR_UNIT_RE.match(item):
            banned_numbers.extend(_extract_number_strings(item))
    # De-dupe while preserving order
    seen_b: set[str] = set()
    banned_unique: list[str] = []
    for n in banned_numbers:
        if n not in seen_b:
            seen_b.add(n)
            banned_unique.append(n)

    # Do not treat a banned token as a hit when it is exactly a required user number
    # that must appear (e.g. user 35 vs banned "3 g/min" → only flag "3").
    user_norms = {u.normalized for u in user_numbers}
    forbidden_hits: list[str] = []
    for banned in banned_unique:
        if banned in user_norms:
            # Still flag if the banned *string* form is a digit-corruption of a user
            # number and appears as its own token (35 → 3).
            if any(
                banned != u.normalized
                and (
                    u.normalized.replace(".", "").startswith(banned.replace(".", ""))
                    or u.normalized.replace(".", "").endswith(banned.replace(".", ""))
                )
                and _number_token_in_text(answer, banned)
                for u in user_numbers
            ):
                forbidden_hits.append(banned)
            continue
        if _number_token_in_text(answer, banned):
            forbidden_hits.append(banned)

    # Flag full must_not_contain phrases when present (number-token-aware).
    substring_hits = [s for s in banned_raw if _forbidden_phrase_in_answer(answer, s)]

    passed = not missing and not forbidden_hits and not substring_hits
    return {
        "pass": passed,
        "check": "numeric_verbatim",
        "user_numbers": [
            {"raw": u.raw, "normalized": u.normalized, "unit": u.unit}
            for u in user_numbers
        ],
        "missing_verbatim": missing,
        "banned_numbers": banned_unique,
        "forbidden_number_hits": forbidden_hits,
        "forbidden_substring_hits": substring_hits,
    }


def citation_grounding_check(
    answer_citations: Sequence[str] | None,
    retrieved_chunk_ids: Sequence[str] | None,
) -> dict[str, Any]:
    """Pass iff every cited chunk_id was in the retrieved set (Fix 4 validator).

    Reuses :func:`generation.answer.validate_segment_chunk_ids` — does not reimplement.
    """
    cited = [str(c).strip() for c in (answer_citations or []) if str(c).strip()]
    allowed = {str(c).strip() for c in (retrieved_chunk_ids or []) if str(c).strip()}

    structured = StructuredAnswer(
        answer_segments=[
            AnswerSegment(text="", citation_chunk_id=cid) for cid in cited
        ]
    )
    invalid = validate_segment_chunk_ids(structured, allowed)
    # Empty citation list: vacuously pass for grounding (nothing claimed).
    # Callers that require ≥1 citation should assert separately.
    return {
        "pass": len(invalid) == 0,
        "check": "citation_grounding",
        "cited_chunk_ids": cited,
        "retrieved_chunk_ids": sorted(allowed),
        "invalid_citation_ids": invalid,
    }


def _is_comparative_question(question: str | None) -> bool:
    q = question or ""
    if _COMPARATIVE_RE.search(q):
        return True
    regs = set()
    for m in _REG_MENTION_RE.finditer(q):
        n = m.group(1) or m.group(2)
        if n:
            regs.add(int(n))
    return len(regs) >= 2


def _citation_regulation_id(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        # Bare chunk id — regulation unknown from id alone.
        return ""
    if isinstance(item, dict):
        return _normalize_reg_id(
            item.get("regulation_id") or item.get("regulation_scope") or ""
        )
    return _normalize_reg_id(getattr(item, "regulation_id", None) or "")


def _citation_chunk_id(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("chunk_id") or item.get("citation_chunk_id") or "").strip()
    return str(
        getattr(item, "chunk_id", None) or getattr(item, "citation_chunk_id", None) or ""
    ).strip()


def cross_regulation_check(
    answer_citations: Sequence[Any] | None,
    regulation_scope: str | None,
    *,
    question: str | None = None,
) -> dict[str, Any]:
    """Pass iff citations stay inside ``regulation_scope`` (unless comparative Q).

    ``answer_citations`` may be chunk-id strings or objects/dicts with
    ``regulation_id`` (preferred). When only bare ids are supplied and scope is
    set, this check cannot verify regulation ownership and returns
    ``pass=True`` with ``skipped_reason`` — callers should pass source metadata.
    """
    scope = _normalize_reg_id(regulation_scope)
    comparative = _is_comparative_question(question)

    if not scope:
        return {
            "pass": True,
            "check": "cross_regulation",
            "regulation_scope": None,
            "comparative": comparative,
            "foreign_citations": [],
            "skipped_reason": "no_regulation_scope",
        }

    if comparative:
        return {
            "pass": True,
            "check": "cross_regulation",
            "regulation_scope": scope,
            "comparative": True,
            "foreign_citations": [],
            "skipped_reason": "comparative_question",
        }

    foreign: list[dict[str, str]] = []
    checked = 0
    for item in answer_citations or []:
        rid = _citation_regulation_id(item)
        cid = _citation_chunk_id(item)
        if not rid:
            continue
        checked += 1
        if rid != scope:
            foreign.append({"chunk_id": cid, "regulation_id": rid})

    if checked == 0 and list(answer_citations or []):
        # Fail-closed: citations without regulation_id are ambiguous — cannot
        # verify scope match, so decline rather than assume ALLOW.
        return {
            "pass": False,
            "check": "cross_regulation",
            "regulation_scope": scope,
            "comparative": False,
            "foreign_citations": [],
            "skipped_reason": "citations_lack_regulation_id_fail_closed",
        }

    return {
        "pass": len(foreign) == 0,
        "check": "cross_regulation",
        "regulation_scope": scope,
        "comparative": False,
        "foreign_citations": foreign,
        "skipped_reason": None,
    }


# Indexed UNECE corpus (must stay aligned with retrieval/api catalog).
INDEXED_REGULATION_IDS = frozenset(
    {"UN-ECE-R94", "UN-ECE-R95", "UN-ECE-R16", "UN-ECE-R129"}
)

_FOREIGN_CORPUS_RE = re.compile(
    r"(?ix)\b("
    r"FMVSS|Euro\s*NCAP|NHTSA|IIHS|CNS|GB[/ ]?\d|AIS[- ]?\d+"
    r"|Federal\s+Motor\s+Vehicle\s+Safety"
    r")\b"
)


def asked_regulation_ids(
    question: str | None,
    regulation_scope: str | None = None,
) -> list[str]:
    """UNECE regulation ids named in the question and/or gold ``regulation_scope``."""
    found: set[str] = set()
    scope = _normalize_reg_id(regulation_scope)
    if scope:
        found.add(scope)
    for m in _REG_MENTION_RE.finditer(question or ""):
        n = m.group(1) or m.group(2)
        if n:
            found.add(f"UN-ECE-R{int(n)}")
    return sorted(found)


def retrieved_regulation_ids(sources: Sequence[Any] | None) -> list[str]:
    """Unique normalized regulation ids from retrieved/cited chunk metadata."""
    out: list[str] = []
    seen: set[str] = set()
    for item in sources or []:
        rid = _citation_regulation_id(item)
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
    return out


def regulation_match_check(
    *,
    question: str | None,
    regulation_scope: str | None = None,
    retrieved_sources: Sequence[Any] | None = None,
    answer_declined: bool = False,
) -> dict[str, Any]:
    """Shared guard: does retrieved regulation match the regulation asked?

    Used by both ``hallucination_probe`` and ``out_of_scope`` (not category forks).

    Fail-closed on uncertainty:
    - Foreign / non-UNECE asks (FMVSS, Euro NCAP, …) never match indexed retrieves.
    - Asked UNECE with no overlapping retrieved id → mismatch.
    - Ambiguous ask (no named UNECE id and not clearly foreign) → uncertain;
      treated as requiring a decline unless already declined.
    When match is not confirmed, the case passes this gate only if the answer
    declined (honest not-found / abstain).
    """
    asked = asked_regulation_ids(question, regulation_scope)
    retrieved = retrieved_regulation_ids(retrieved_sources)
    foreign_ask = bool(_FOREIGN_CORPUS_RE.search(question or ""))
    asked_indexed = [r for r in asked if r in INDEXED_REGULATION_IDS]
    asked_outside = [r for r in asked if r not in INDEXED_REGULATION_IDS]

    if foreign_ask or asked_outside:
        match_status = "mismatch"
        reason = "asked_outside_indexed_corpus"
        matches = False
    elif not asked:
        match_status = "uncertain"
        reason = "asked_regulation_uncertain"
        matches = False  # fail-closed
    elif not retrieved:
        match_status = "mismatch"
        reason = "no_retrieved_regulation"
        matches = False
    elif set(asked_indexed) & set(retrieved):
        match_status = "match"
        reason = "retrieved_overlaps_asked"
        matches = True
    else:
        match_status = "mismatch"
        reason = "retrieved_does_not_match_asked"
        matches = False

    need_decline = not matches
    gate_pass = (not need_decline) or bool(answer_declined)
    return {
        "check": "regulation_match",
        "pass": gate_pass,
        "matches": matches,
        "match_status": match_status,
        "reason": reason,
        "asked_regulation_ids": asked,
        "retrieved_regulation_ids": retrieved,
        "foreign_ask": foreign_ask,
        "need_decline": need_decline,
        "answer_declined": bool(answer_declined),
    }


def run_custom_hard_gates(
    case: dict[str, Any],
    *,
    answer: str,
    retrieved_chunk_ids: Sequence[str] | None = None,
    cited_chunk_ids: Sequence[str] | None = None,
    cited_sources: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run the applicable hard gates for a golden case; overall pass/fail.

    For ``numeric_safety`` / ``compliance_check`` / ``cross_regulation`` all three
    checks run. Other categories return ``applicable=False`` without failing.
    """
    category = str(case.get("category") or "").strip().lower()
    question = str(case.get("question") or "")
    scope = case.get("regulation_scope") or case.get("regulation_id")
    must_not = case.get("must_not_contain") or case.get("banned_answer_substrings") or []

    if category not in CUSTOM_HARD_GATE_CATEGORIES:
        return {
            "applicable": False,
            "category": category or None,
            "pass": True,
            "checks": {},
        }

    citations_for_grounding = list(cited_chunk_ids or [])
    if not citations_for_grounding and cited_sources:
        citations_for_grounding = [
            _citation_chunk_id(s) for s in cited_sources if _citation_chunk_id(s)
        ]

    citations_for_cross: Sequence[Any]
    if cited_sources:
        citations_for_cross = list(cited_sources)
    else:
        citations_for_cross = citations_for_grounding

    checks = {
        "numeric_verbatim": numeric_verbatim_check(
            question, answer, must_not_contain=must_not
        ),
        "citation_grounding": citation_grounding_check(
            citations_for_grounding, retrieved_chunk_ids
        ),
        "cross_regulation": cross_regulation_check(
            citations_for_cross, scope, question=question
        ),
    }
    # For numeric_safety, numeric_verbatim is mandatory.
    # For cross_regulation, cross_regulation check is mandatory.
    # citation_grounding always applies when citations exist.
    overall = all(c.get("pass") for c in checks.values())
    return {
        "applicable": True,
        "category": category,
        "pass": overall,
        "checks": checks,
    }
