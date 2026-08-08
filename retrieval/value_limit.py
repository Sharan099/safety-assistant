"""Value-vs-limit (measured number + pass/fail) query helpers.

Biases retrieval toward injury-criteria / performance-limit clauses and away
from instrumentation / calibration / ISO 6487 channel-filter text.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Measured value + regulation cue + pass/fail intent.
_NUMERIC_RE = re.compile(
    r"(?ix)"
    r"("
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|kN|kn|m/?s|m/sec|ms|g\s*/\s*min|g/min|g)\b"
    r"|\b(?:of|was|is|measured|produced|recorded)\s+\d+(?:[.,]\d+)?\b"
    r"|\b\d+(?:[.,]\d+)?\s*(?:millimet(?:er|re)s?)\b"
    r"|\bleakage\s+rate\s+is\s+\d+(?:[.,]\d+)?\b"
    r")"
)
_REG_RE = re.compile(
    r"(?ix)\b(?:UN[- ]?(?:ECE[- ]?)?R\s*\d+|Regulation\s+No\.?\s*\d+|R\s*\d{2,3})\b"
)
_PASS_FAIL_RE = re.compile(
    r"(?ix)\b("
    r"pass(?:es|ed)?|fail(?:s|ed)?|comply|complies|compliance|"
    r"satisf(?:y|ies|ied)|conform(?:s|ity)?|within\s+(?:the\s+)?limit|"
    r"exceed(?:s|ed)?|acceptable|acceptable\s+under"
    r")\b"
)

# Prefer these in retrieved text for limit questions.
_LIMIT_POSITIVE = re.compile(
    r"(?ix)("
    r"performance\s+criteria|"
    r"injury\s+criter(?:ion|ia)|"
    r"shall\s+not\s+exceed|"
    r"less\s+than\s+or\s+equal|"
    r"\bRDC\b|\bPSPF\b|\bHPC(?:36)?\b|\bHIC(?:15|36)?\b|\bThCC\b|\bTHCC\b|"
    r"\bTTI\b|\bTCFC\b|\bFFC\b|"
    r"Rib\s+Deflection\s+Criterion|"
    r"Viscous\s+Criterion|"
    r"Soft\s+Tissue\s+Criterion|"
    r"Pubic\s+Symphysis\s+Peak\s+Force|"
    r"Head\s+Performance\s+Criterion|"
    r"Thorax\s+Compression\s+Criterion|"
    r"abdomen\s+performance\s+criterion|"
    r"pelvis\s+performance\s+criterion|"
    r"thorax\s+performance\s+criteria|"
    r"fuel[- ]?feed|"
    r"leakage\s+shall\s+not\s+exceed|"
    r"30\s*g\s*/\s*min|"
    r"electrolyte\s+leakage|"
    r"no\s+(?:liquid\s+)?electrolyte\s+leakage|"
    r"passenger\s+compartment"
    r")"
)

# Strongly demote sensor / calibration / channel-filter annex text.
_LIMIT_NEGATIVE = re.compile(
    r"(?ix)("
    r"ISO\s*6487|"
    r"\bCFC\b|"
    r"\bCAC\b|"
    r"channel\s+frequency|"
    r"channel\s+filter|"
    r"certification|"
    r"calibrat(?:e|ion|ed)|"
    r"instrumentation|"
    r"dummy\s+components|"
    r"measurements?\s+to\s+be\s+made|"
    r"filter(?:ing|ed)\s+at|"
    r"full\s+rib\s+module\s+certification|"
    r"drop\s+height"
    r")"
)

# Prefer clauses that state a numeric/regulatory limit, not "HPC calculated" procedure.
_LIMIT_LANGUAGE_RE = re.compile(
    r"(?ix)("
    r"shall\s+not\s+exceed|"
    r"less\s+than\s+or\s+equal|"
    r"shall\s+be\s+(?:less|≤)|"
    r"must\s+not\s+exceed|"
    r"performance\s+criteria\s+shall"
    r")"
)

# Post-crash electrical / REESS text — wrong neighbor for HPC/RDC/etc. limit Qs.
_ELECTRICAL_SAFETY_RE = re.compile(
    r"(?ix)("
    r"\bREESS\b|"
    r"electrical?\s+shock|"
    r"high\s+voltage|"
    r"electric\s+power\s+train|"
    r"protection\s+against\s+electric|"
    r"physical\s+protection|"
    r"coupling\s+system\s+for\s+charging"
    r")"
)

# Electrolyte containment requirement (not isolation-resistance measurement).
_ELECTROLYTE_CONTAINMENT_RE = re.compile(
    r"(?ix)("
    r"electrolyte\s+leakage.{0,100}passenger\s+compart|"
    r"no\s+(?:liquid\s+)?electrolyte\s+leakage|"
    r"into\s+the\s+passenger\s+compartment|"
    r"aqueous\s+electrolyte\s+REESS|"
    r"non-aqueous\s+electrolyte\s+REESS"
    r")"
)

# Isolation-resistance *procedure* — competes with electrolyte containment in EV text.
_ISOLATION_MEASUREMENT_RE = re.compile(
    r"(?ix)("
    r"isolation\s+resistance\s+(?:measurement|test\s+instrument|measur)|"
    r"measuring\s+electric\s+resistance|"
    r"electrical\s+isolation\s+value\s+Ri|"
    r"test\s+method\s+for\s+measuring\s+electric\s+resistance|"
    r"Fifth\s+step\s+The\s+electrical\s+isolation"
    r")"
)

_ELECTROLYTE_TOPIC_RE = re.compile(
    r"(?ix)\belectrolyte\b|\bspillage\b|\belectrolyte\s+leak"
)

# Named injury metrics mentioned in the user question → retrieval extras.
_CRITERION_MENTIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)\brib\s+deflection\b|\bRDC\b"),
        "Rib Deflection Criterion (RDC) injury criteria performance criteria "
        "less than or equal to 42 mm",
    ),
    (
        re.compile(r"(?i)\bviscous\s+criterion\b|\bVC\b|\bV\s*\*\s*C\b"),
        "Viscous Criterion (VC) Soft Tissue Criterion injury criteria "
        "performance criteria 1.0 m/s",
    ),
    (
        re.compile(r"(?i)\bpubic\s+symphysis\b|\bPSPF\b"),
        "Pubic Symphysis Peak Force (PSPF) pelvis performance criterion "
        "injury criteria 6 kN",
    ),
    (
        re.compile(
            r"(?i)\bHPC(?:36)?\b|\bHIC(?:15|36)?\b|\bhead\s+performance\b|"
            r"\bhead\s+injury\b"
        ),
        "Head Performance Criterion (HPC) injury criteria shall not exceed 1000 "
        "less than or equal to 1,000",
    ),
    (
        re.compile(r"(?i)\bThCC\b|\bTHCC\b|\bthorax\s+compression\b"),
        "Thorax Compression Criterion (ThCC) injury criteria shall not exceed 42 mm",
    ),
    (
        re.compile(r"(?i)\bTTI\b|\bthoracic\s+trauma\b"),
        "Thoracic Trauma Index (TTI) injury criteria performance criteria",
    ),
    (
        re.compile(r"(?i)\bfuel\s*[- ]?\s*leak|\bleakage\s+rate\b|\bg\s*/\s*min\b|\bg/min\b"),
        "fuel-feed installation leakage rate shall not exceed 30 g/min continuous leakage",
    ),
    (
        re.compile(
            r"(?i)\belectrolyte\b|\bspillage\b|"
            r"\belectrolyte\s+leak(?:age)?\b"
        ),
        "electrolyte leakage REESS passenger compartment shall be no liquid "
        "electrolyte leakage into the passenger compartment spillage",
    ),
]

_GENERIC_LIMIT_EXTRAS = (
    "injury criteria performance criteria limit shall not exceed "
    "less than or equal to"
)


def is_value_vs_limit_query(question: str) -> bool:
    """True when the user supplies a measurement and asks pass/fail vs a regulation."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(
        _NUMERIC_RE.search(q) and _REG_RE.search(q) and _PASS_FAIL_RE.search(q)
    )


_DEFINITION_SEEKING_RE = re.compile(
    r"(?ix)\b(?:"
    r"what\s+does\b|"
    r"what\s+is\s+(?:a|an|the\s+)?(?:meaning|definition)\b|"
    r"define\b|definition\s+of|meaning\s+of|"
    r"stands?\s+for\b"
    r")\b"
)
_LIMIT_SEEKING_RE = re.compile(
    r"(?ix)\b(?:"
    r"limit|maximum|threshold|shall\s+not\s+exceed|"
    r"satisfy|comply|compliance|pass|fail"
    r")\b"
)


def is_definition_seeking_query(question: str) -> bool:
    """True when the user wants a definition/expansion, not a pass/fail verdict."""
    q = (question or "").strip()
    return bool(q and _DEFINITION_SEEKING_RE.search(q))


def is_compliance_prefer_limit_query(question: str) -> bool:
    """Pass/fail / comply asks that should prefer limit/requirement clauses.

    Covers classic value-vs-limit (measured number + pass/fail) plus
    electrolyte/REESS compliance probes that lack a numeric measurement but
    still need the containment requirement, not the REESS definition.

    Does **not** fire on pure definition asks ("What does REESS mean?",
    "Define isolation resistance") even when a named criterion acronym appears —
    those must keep definition clauses ranked highly.
    """
    q = (question or "").strip()
    if not q:
        return False
    if is_definition_seeking_query(q):
        return False
    if is_value_vs_limit_query(q):
        return True
    if _ELECTROLYTE_TOPIC_RE.search(q) and _PASS_FAIL_RE.search(q):
        return True
    if _PASS_FAIL_RE.search(q) and _REG_RE.search(q):
        return True
    # Named injury criterion + explicit limit/verdict language (not bare mention).
    if is_named_criterion_query(q) and _LIMIT_SEEKING_RE.search(q):
        return True
    return False


def expand_value_vs_limit_query(question: str) -> str:
    """Append injury-criterion keywords so BM25 prefers limit clauses over sensors."""
    q = (question or "").strip()
    if not q:
        return q
    extras: list[str] = [_GENERIC_LIMIT_EXTRAS]
    for pat, extra in _CRITERION_MENTIONS:
        if pat.search(q):
            extras.append(extra)
    # Deduplicate tokens while preserving order.
    seen: set[str] = set()
    bits: list[str] = [q]
    for block in extras:
        for tok in block.split():
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                bits.append(tok)
    expanded = " ".join(bits)
    logger.info("value_vs_limit query expand: %r → %r", q[:120], expanded[:200])
    return expanded


def criteria_focused_subquery(question: str) -> str | None:
    """Second hybrid subquery that targets the named criterion limit only."""
    q = (question or "").strip()
    if not q:
        return None
    for pat, extra in _CRITERION_MENTIONS:
        if pat.search(q):
            return extra
    if is_value_vs_limit_query(q):
        return _GENERIC_LIMIT_EXTRAS
    return None


def is_named_criterion_query(question: str) -> bool:
    """True when the question names an injury criterion (HPC, RDC, VC, …)."""
    q = (question or "").strip()
    if not q:
        return False
    return any(pat.search(q) for pat, _ in _CRITERION_MENTIONS)


def exact_regulatory_phrases(question: str) -> list[str]:
    """Canonical full-name phrases to favor in BM25 / post-fusion boost."""
    q = (question or "").strip()
    if not q:
        return []
    phrases: list[str] = []
    for pat, extra in _CRITERION_MENTIONS:
        if pat.search(q):
            # First three tokens of the criteria cue are usually the formal name.
            name = " ".join(extra.split()[:4]).strip()
            if name and name.lower() not in {p.lower() for p in phrases}:
                phrases.append(name)
    # Also keep short acronyms present in the question.
    for acr in (
        "HPC",
        "HPC36",
        "HIC",
        "RDC",
        "PSPF",
        "ThCC",
        "THCC",
        "TTI",
        "TCFC",
        "FFC",
        "VC",
    ):
        if re.search(rf"\b{acr}\b", q, re.I) and acr not in phrases:
            phrases.append(acr)
    return phrases


def _chunk_text(chunk: object) -> str:
    return " ".join(
        str(getattr(chunk, attr, "") or "")
        for attr in ("text", "enriched_text", "section_title", "section_number")
    )


def injury_criteria_boost(chunk: object, *, question: str = "") -> float:
    """Additive score bias: + for limit clauses, − for calibration/ISO 6487."""
    text = _chunk_text(chunk)
    if not text.strip():
        return 0.0
    pos = len(_LIMIT_POSITIVE.findall(text))
    neg = len(_LIMIT_NEGATIVE.findall(text))
    # Cap so one long Annex does not swamp everything.
    # Soften CFC/channel demotion when the chunk also states a real limit.
    has_limit_lang = bool(_LIMIT_LANGUAGE_RE.search(text))
    neg_weight = 0.25 if has_limit_lang else 0.55
    boost = min(pos, 6) * 0.35 - min(neg, 6) * neg_weight

    phrases = exact_regulatory_phrases(question)
    if phrases:
        low = text.lower()
        words = max(len(text.split()), 1)
        full_names = [p for p in phrases if len(p) > 5]
        acronyms = [p for p in phrases if len(p) <= 5]
        full_hits = sum(low.count(p.lower()) for p in full_names)
        acr_hits = sum(low.count(p.lower()) for p in acronyms)

        # Strong signal: formal name and/or criterion + limit language.
        if full_hits:
            boost += 0.9 + min(full_hits, 3) * 0.25
        if (full_hits or acr_hits) and has_limit_lang:
            boost += 1.35  # the actual "shall not exceed / ≤" limit clause
        elif acr_hits and not full_hits and not has_limit_lang:
            # Annex 8-style "HPC calculated" spam without a limit statement.
            boost -= min(acr_hits, 5) * 0.35
        elif acr_hits and full_hits == 0:
            boost += 0.25  # weak acronym-only presence

        # Density of *full names* only (ignore bare acronym spam).
        if full_hits:
            boost += min(full_hits / words * 80.0, 1.0)

        if is_named_criterion_query(question) and _ELECTRICAL_SAFETY_RE.search(text):
            # Do not demote REESS/electrical when the question IS about electrolyte /
            # electrical containment — only when an injury metric was named instead.
            if not _ELECTROLYTE_TOPIC_RE.search(question):
                if not (full_hits or (acr_hits and has_limit_lang)):
                    boost -= 1.1
                elif words > 800:
                    # Mega §5 with HPC limit + REESS: slight penalty vs focused leaf.
                    boost -= 0.25

    # Electrolyte / spillage: prefer containment requirement clauses; demote
    # isolation-resistance measurement procedures that share EV embedding space.
    if question and _ELECTROLYTE_TOPIC_RE.search(question):
        if _ELECTROLYTE_CONTAINMENT_RE.search(text):
            boost += 2.2
        elif re.search(r"(?i)\belectrolyte\s+leakage\b", text):
            boost += 1.2
        if _ISOLATION_MEASUREMENT_RE.search(text) and not _ELECTROLYTE_CONTAINMENT_RE.search(
            text
        ):
            boost -= 1.8

    # Compliance / pass-fail: demote definitions + pure test-procedure annex text
    # that share topic words with the actual limit clause.
    if question and is_compliance_prefer_limit_query(question):
        sec = str(getattr(chunk, "section_number", "") or "")
        if re.match(r'(?ix)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"[^"]+"\s+means\b', text):
            boost -= 1.6
        elif re.match(r"(?i)^2\.", sec) and "means" in text.lower()[:160]:
            boost -= 1.4
        if re.search(
            r"(?ix)procedure\s+for\s+calculat|peak\s+viscous\s+response|"
            r"is\s+calculated\s+as\s+the\s+instantaneous",
            text,
        ) and not has_limit_lang:
            boost -= 1.5
        if re.match(r"(?i)^annex\s*4", sec) and not has_limit_lang:
            boost -= 0.9
    return boost


def is_exact_term_boost_query(question: str) -> bool:
    """True when BM25 should be up-weighted for literal regulatory terms."""
    q = (question or "").strip()
    if not q:
        return False
    if is_named_criterion_query(q):
        return True
    return bool(_ELECTROLYTE_TOPIC_RE.search(q))


def bias_chunks_for_value_vs_limit(
    chunks: Sequence[T],
    *,
    question: str = "",
) -> list[T]:
    """Re-order hybrid/rerank candidates toward injury-criteria limit clauses."""
    if not chunks:
        return []
    # Never demote definitions when the user is asking what a term means.
    if question and is_definition_seeking_query(question):
        return list(chunks)
    # Apply for classic value-vs-limit AND broader compliance prefer-limit asks.
    if question and not (
        is_value_vs_limit_query(question)
        or is_named_criterion_query(question)
        or is_compliance_prefer_limit_query(question)
    ):
        return list(chunks)
    named_reg: str | None = None
    if question:
        try:
            from retrieval.enumerative import detect_named_regulation

            named_reg = detect_named_regulation(question)
        except Exception:  # noqa: BLE001
            named_reg = None
    scored: list[tuple[float, int, T]] = []
    for i, chunk in enumerate(chunks):
        base = float(getattr(chunk, "score", 0.0) or 0.0)
        boost = injury_criteria_boost(chunk, question=question)
        # Prefer the regulation named in the question (R95 HPC ≠ R94 electrical).
        if named_reg and str(getattr(chunk, "regulation_id", "") or "") == named_reg:
            boost += 0.75
        # Prefer main Specifications (clause 5 / 5.2.*) over Annex procedure text.
        sec = str(getattr(chunk, "section_number", "") or "")
        if re.match(r"^5(\.|$)", sec) or re.match(r"^5\.\d", sec):
            boost += 0.4
        if re.match(r"(?i)^Annex", sec):
            # Annex 4 performance-data for HPC is useful; pure procedure less so.
            text = _chunk_text(chunk)
            if _LIMIT_POSITIVE.search(text):
                boost += 0.15
            else:
                boost -= 0.25
        scored.append((base + boost, i, chunk))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out: list[T] = []
    for fused, _, chunk in scored:
        if hasattr(chunk, "model_copy"):
            try:
                out.append(chunk.model_copy(update={"score": fused}))  # type: ignore[attr-defined]
                continue
            except Exception:  # noqa: BLE001
                pass
        try:
            setattr(chunk, "score", fused)
        except Exception:  # noqa: BLE001
            pass
        out.append(chunk)
    if question:
        logger.info(
            "criteria bias reordered top=%s",
            [
                (
                    getattr(c, "section_number", None),
                    (getattr(c, "chunk_id", None) or "")[:12],
                    round(float(getattr(c, "score", 0) or 0), 3),
                )
                for c in out[:5]
            ],
        )
    return out


VALUE_VS_LIMIT_USER_INSTRUCTION = """\
VALUE-VS-LIMIT QUESTION — required answer structure:
The user provided a measured numeric value and asks whether the vehicle
passes / complies / satisfies the cited regulation. You MUST:
(a) Identify the specific injury-criterion LIMIT clause in the context
    (look for "performance criteria", "shall not exceed", or
    "less than or equal to" — NOT instrumentation or ISO 6487 filtering).
(b) State the regulatory limit value and unit from that clause.
(c) Explicitly compare the user's measured value against that limit.
(d) Conclude PASS or FAIL clearly (for upper-bound criteria: PASS if
    measured ≤ limit, otherwise FAIL).
If the injury-criterion limit clause is absent from the context, return
{"answer_segments": []} — do not substitute a calibration, CFC, CAC,
channel-filter, or certification passage as if it were the limit.
"""
