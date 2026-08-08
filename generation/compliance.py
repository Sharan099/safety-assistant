"""Deterministic compliance-check path (value-vs-limit) using structured limits.

Replaces prompt-only Fix 14 for pass/fail questions: measurements are parsed
from the query in code, limits are looked up from ``data/limits/*.json``, and
comparisons run in Python. The LLM may only phrase connecting prose around
already-computed verdicts and numbers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from generation.numeric_guard import normalize_number_token, normalize_unit
from ingestion.extract_limits import LimitRow, find_limit, load_all_limits, seed_known_limits

logger = logging.getLogger(__name__)

_PASS_FAIL_RE = re.compile(
    r"(?ix)\b("
    r"pass(?:es|ed)?|fail(?:s|ed)?|comply|complies|compliance|"
    r"satisf(?:y|ies|ied)|conform(?:s|ity)?|within\s+(?:the\s+)?limit|"
    r"exceed(?:s|ed)?|acceptable|does\s+the\s+vehicle\s+pass"
    r")\b"
)

# Explicit "does … comply / pass" asks — always require a structured verdict.
_EXPLICIT_COMPLY_RE = re.compile(
    r"(?ix)\b("
    r"does\s+(?:the\s+)?(?:vehicle|it)\s+(?:pass|comply|satisfy|conform)|"
    r"do(?:es)?\s+(?:this|the\s+result)\s+comply|"
    r"is\s+(?:the\s+)?(?:vehicle|result)\s+(?:compliant|acceptable)|"
    r"compliant\?"
    r")"
)

CANNOT_DETERMINE_MESSAGE = (
    "Cannot determine compliance from indexed content: the question asks whether "
    "the vehicle complies, but no measurable criterion or qualitative pass/fail "
    "scenario could be evaluated against a known requirement."
)

# Qualitative scenarios: observed fact in the question → fixed PASS/FAIL.
# (No measured number — e.g. electrolyte entered the passenger compartment.)
_QUALITATIVE_SCENARIOS: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (
        re.compile(
            r"(?ix)"
            r"electrolyte\s+(?:entered|enters|has\s+entered|leak(?:ed|s)?\s+into)"
            r".{0,60}passenger\s+compart"
            r"|passenger\s+compart.{0,60}electrolyte"
            r"|electrolyte\s+(?:in|inside)\s+the\s+passenger\s+compart"
        ),
        {
            "criterion": "Electrolyte leakage into passenger compartment",
            "verdict": "FAIL",
            "section_number": "5.2.8.2",
            "source_chunk_id": "665eccc7295ab7a1",
            "regulation_id": "UN-ECE-R94",
            "note": (
                "UN R94 requires no electrolyte leakage from the REESS into the "
                "passenger compartment; the stated entry is a FAIL."
            ),
        },
    ),
    (
        re.compile(
            r"(?ix)"
            r"\bREESS\s+remained\s+(?:mounted|attached|retained)\b|"
            r"\bREESS\s+(?:is|was)\s+(?:still\s+)?(?:mounted|attached|retained)\b"
        ),
        {
            "criterion": "REESS retention",
            "verdict": "PASS",
            "section_number": "5.2.8.3",
            "source_chunk_id": "",
            "regulation_id": "UN-ECE-R94",
            "note": (
                "REESS remaining mounted/attached satisfies the retention "
                "requirement (shall remain attached by at least one load path)."
            ),
        },
    ),
]

# Criterion cue → canonical lookup aliases (order: longer phrases first).
_CRITERION_CUES: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"(?i)\bpubic\s+symphysis(?:\s+peak\s+force)?\b|\bPSPF\b"),
        ["PSPF", "Pubic Symphysis Peak Force"],
    ),
    (
        re.compile(r"(?i)\brib\s+deflection(?:\s+criterion)?\b|\bRDC\b"),
        ["RDC", "Rib Deflection Criterion", "rib deflection"],
    ),
    (
        re.compile(
            r"(?i)\bthorax\s+compression(?:\s+criterion)?\b|\bThCC\b|\bTHCC\b|"
            r"\bchest\s+compression\b"
        ),
        ["ThCC", "THCC", "Thorax Compression Criterion", "chest compression"],
    ),
    (
        re.compile(
            r"(?i)\bhead\s+performance(?:\s+criterion)?\b|\bHPC(?:36)?\b|"
            r"\bHIC(?:15|36)?\b|\bhead\s+injury\b"
        ),
        ["HPC", "Head Performance Criterion", "HIC"],
    ),
    (
        re.compile(
            r"(?i)\bviscous\s+criterion\b|\bsoft\s+tissue\s+criterion\b|"
            r"\bVC\b|\bV\s*\*\s*C\b"
        ),
        ["VC", "Viscous Criterion", "Soft Tissue Criterion"],
    ),
    (
        re.compile(
            r"(?i)\bfuel\s*[- ]?\s*leak(?:age)?(?:\s+rate)?\b|"
            r"\bleakage\s+rate\b|\bfuel[- ]?feed\b"
        ),
        ["fuel leakage", "leakage rate", "fuel-feed leakage"],
    ),
]

# "HPC of 920", "chest compression of 32 mm", "fuel leakage of 40 g/min"
_NAMED_MEASURE_RE = re.compile(
    r"(?ix)"
    r"(?P<name>"
    r"HPC(?:36)?|HIC(?:15|36)?|"
    r"ThCC|THCC|RDC|PSPF|TTI|VC|"
    r"Head\s+Performance(?:\s+Criterion)?|"
    r"Thorax\s+Compression(?:\s+Criterion)?|"
    r"Rib\s+Deflection(?:\s+Criterion)?|"
    r"Pubic\s+Symphysis(?:\s+Peak\s+Force)?|"
    r"Viscous\s+Criterion|Soft\s+Tissue\s+Criterion|"
    r"chest\s+compression|chest\s+deflection|"
    r"fuel\s*[- ]?\s*leak(?:age)?(?:\s+rate)?|leakage\s+rate"
    r")"
    r"\s*(?:of|was|is|=|:)?\s*"
    r"(?P<num>\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<unit>g\s*/\s*min|g/min|mm|kN|kn|m\s*/\s*s|m/s|ms|%))?"
)

_UNIT_MEASURE_RE = re.compile(
    r"(?ix)"
    r"(?P<num>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>g\s*/\s*min|g/min|mm|kN|kn|m\s*/\s*s|m/s|ms|%)"
)

PHRASE_SYSTEM = """\
You write a short compliance explanation around a FIXED deterministic template.

TEMPLATE CONSTRAINT (Fix 22 — do not violate):
1. Overall verdict is already computed in Python (PASS / FAIL / INCOMPLETE /
   CANNOT_DETERMINE). Copy it EXACTLY — never invent or soften it.
2. Every criterion already has measured value, operator, limit, unit, and
   verdict. Copy those EXACT numbers and verdicts — never round, rescale,
   rename criteria, or change operators.
3. Your only job is connecting prose: "summary_prose" (2–4 sentences) and
   optional per-criterion "note" strings that phrase around the fixed fields.
4. Do NOT invent additional criteria, limits, or measurements.
5. Prefer structure: lead with the overall verdict, then briefly explain each
   criterion using the provided measured/limit pair.

Return JSON matching the schema. Copy overall_verdict and each criterion's
numeric fields EXACTLY from the user message.
"""

PHRASE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "compliance_phrase",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "overall_verdict": {"type": "string"},
                "summary_prose": {"type": "string"},
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "measured": {"type": "number"},
                            "limit": {"type": ["number", "null"]},
                            "unit": {"type": "string"},
                            "operator": {"type": "string"},
                            "verdict": {"type": "string"},
                            "note": {"type": "string"},
                        },
                        "required": [
                            "criterion",
                            "measured",
                            "verdict",
                        ],
                    },
                },
            },
            "required": ["overall_verdict", "summary_prose", "criteria"],
        },
    },
}


class MeasuredCriterion(BaseModel):
    """One user-stated measurement tied to a criterion cue."""

    criterion_query: str
    lookup_aliases: list[str] = Field(default_factory=list)
    measured_value: float
    measured_raw: str
    unit: str = ""
    span: tuple[int, int] = (0, 0)


class CriterionVerdict(BaseModel):
    criterion: str
    aliases_matched: list[str] = Field(default_factory=list)
    measured: float | None = None
    measured_display: str = ""
    limit: float | None = None
    limit_display: str | None = None
    unit: str = ""
    operator: str = ""
    verdict: str  # PASS | FAIL | LIMIT_NOT_FOUND | CANNOT_DETERMINE
    source_chunk_id: str = ""
    section_number: str = ""
    regulation_id: str = ""
    note: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """Numbers as separate fields for the frontend (not free-text)."""
        return {
            "criterion": self.criterion,
            "measured": self.measured,
            "measured_display": self.measured_display,
            "limit": self.limit,
            "limit_display": self.limit_display,
            "unit": self.unit,
            "operator": self.operator,
            "verdict": self.verdict,
            "source_chunk_id": self.source_chunk_id,
            "section_number": self.section_number,
            "regulation_id": self.regulation_id,
            "note": self.note,
        }


class ComplianceResult(BaseModel):
    is_compliance_check: bool = False
    overall_verdict: str = ""  # PASS | FAIL | INCOMPLETE
    criteria: list[CriterionVerdict] = Field(default_factory=list)
    summary_prose: str = ""
    answer_text: str = ""
    regulation_id: str | None = None
    # When LLM phrasing ran: actual served answerer (Portkey telemetry).
    phrasing_model: str = ""
    phrasing_provider: str = ""
    phrasing_target_index: int | None = None
    phrasing_was_fallback: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "overall_verdict": self.overall_verdict,
            "summary_prose": self.summary_prose,
            "criteria": [c.to_public_dict() for c in self.criteria],
            "regulation_id": self.regulation_id,
        }


def has_compliance_intent(question: str) -> bool:
    """True when the user asks pass/fail / comply / satisfy."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(_PASS_FAIL_RE.search(q) or _EXPLICIT_COMPLY_RE.search(q))


def is_compliance_check_query(question: str) -> bool:
    """True when compliance intent is present and we can (or must) verdict.

    Covers: numeric value-vs-limit, qualitative REESS/electrolyte scenarios, and
    bare "does the vehicle comply?" asks that must return CANNOT_DETERMINE rather
    than a procedural description with no conclusion.
    """
    q = (question or "").strip()
    if not q or not has_compliance_intent(q):
        return False
    if extract_measured_criteria(q) or extract_qualitative_scenarios(q):
        return True
    # Explicit comply/pass ask with no evaluable fact → still a compliance query
    # so the backend can emit CANNOT_DETERMINE.
    return bool(_EXPLICIT_COMPLY_RE.search(q))


def extract_qualitative_scenarios(question: str) -> list[CriterionVerdict]:
    """Map stated qualitative facts onto fixed PASS/FAIL requirement verdicts."""
    q = question or ""
    out: list[CriterionVerdict] = []
    seen: set[str] = set()
    for pat, meta in _QUALITATIVE_SCENARIOS:
        if not pat.search(q):
            continue
        name = str(meta["criterion"])
        if name in seen:
            continue
        seen.add(name)
        out.append(
            CriterionVerdict(
                criterion=name,
                aliases_matched=[name],
                measured=None,
                measured_display="",
                limit=None,
                limit_display="",
                unit="",
                operator="",
                verdict=str(meta["verdict"]),
                note=str(meta.get("note") or ""),
                source_chunk_id=str(meta.get("source_chunk_id") or ""),
                section_number=str(meta.get("section_number") or ""),
                regulation_id=str(meta.get("regulation_id") or ""),
            )
        )
    return out


def _aliases_for_name(name: str) -> list[str]:
    for pat, aliases in _CRITERION_CUES:
        if pat.search(name):
            return list(aliases)
    return [name.strip()]


def extract_measured_criteria(question: str) -> list[MeasuredCriterion]:
    """Parse (criterion, value, unit) pairs from the query — never via LLM."""
    q = question or ""
    found: list[MeasuredCriterion] = []
    seen_spans: list[tuple[int, int]] = []

    def _overlaps(span: tuple[int, int]) -> bool:
        a, b = span
        for x, y in seen_spans:
            if a < y and b > x:
                return True
        return False

    for m in _NAMED_MEASURE_RE.finditer(q):
        span = m.span()
        if _overlaps(span):
            continue
        name = m.group("name").strip()
        raw = m.group("num")
        unit = normalize_unit(m.group("unit")) or ""
        try:
            value = float(normalize_number_token(raw))
        except ValueError:
            continue
        seen_spans.append(span)
        found.append(
            MeasuredCriterion(
                criterion_query=name,
                lookup_aliases=_aliases_for_name(name),
                measured_value=value,
                measured_raw=raw.strip(),
                unit=unit,
                span=span,
            )
        )

    # Unit-only mentions near a criterion cue (e.g. "Rib Deflection of 45 mm"
    # already caught; "leakage rate is 35 g/min" via named pattern).
    if not found:
        for m in _UNIT_MEASURE_RE.finditer(q):
            span = m.span()
            if _overlaps(span):
                continue
            # Look backward ~60 chars for a criterion cue.
            window = q[max(0, span[0] - 60) : span[0]]
            aliases: list[str] | None = None
            cue_name = ""
            for pat, als in _CRITERION_CUES:
                cm = pat.search(window)
                if cm:
                    aliases = list(als)
                    cue_name = cm.group(0)
                    break
            if not aliases:
                continue
            raw = m.group("num")
            unit = normalize_unit(m.group("unit")) or ""
            try:
                value = float(normalize_number_token(raw))
            except ValueError:
                continue
            seen_spans.append(span)
            found.append(
                MeasuredCriterion(
                    criterion_query=cue_name or aliases[0],
                    lookup_aliases=aliases,
                    measured_value=value,
                    measured_raw=raw.strip(),
                    unit=unit,
                    span=span,
                )
            )

    return found


def compare_values(measured: float, operator: str, limit: float) -> str:
    """Return PASS or FAIL for an upper/lower-bound operator."""
    op = (operator or "<=").strip()
    if op in {"<=", "≤", "=<"}:
        return "PASS" if measured <= limit else "FAIL"
    if op in {">=", "≥", "=>"}:
        return "PASS" if measured >= limit else "FAIL"
    if op == "<":
        return "PASS" if measured < limit else "FAIL"
    if op == ">":
        return "PASS" if measured > limit else "FAIL"
    if op in {"=", "=="}:
        return "PASS" if abs(measured - limit) < 1e-9 else "FAIL"
    # Default UNECE style: shall not exceed → <=
    return "PASS" if measured <= limit else "FAIL"


def _format_number(value: float, raw: str | None = None) -> str:
    if raw and normalize_number_token(raw):
        # Prefer the user's own digit string.
        return raw.strip().replace(",", ".")
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"


def _ensure_limits_seeded() -> None:
    if not load_all_limits():
        seed_known_limits()
        logger.info("compliance: seeded known limits tables (empty data/limits/)")


def evaluate_compliance(
    question: str,
    *,
    regulation_id: str | None = None,
) -> ComplianceResult | None:
    """Run deterministic comparisons; return None if not a compliance query."""
    if not is_compliance_check_query(question):
        return None

    _ensure_limits_seeded()
    measurements = extract_measured_criteria(question)
    qualitative = extract_qualitative_scenarios(question)

    # Infer regulation when the question names exactly one.
    rid = regulation_id
    if not rid:
        try:
            from retrieval.enumerative import detect_named_regulation

            rid = detect_named_regulation(question)
        except Exception:  # noqa: BLE001
            rid = None

    # Multi-criteria / fuel composites without a named reg: prefer R95 when
    # fuel leakage is present (limit lives in R95 gold chunk), else R94 for HPC/ThCC.
    aliases_flat = {a.lower() for m in measurements for a in m.lookup_aliases}
    if not rid:
        if any("fuel" in a or "leakage" in a for a in aliases_flat):
            rid = "UN-ECE-R95"
        elif any(a in {"hpc", "thcc"} for a in aliases_flat):
            rid = "UN-ECE-R94"
        elif qualitative:
            rid = next(
                (c.regulation_id for c in qualitative if c.regulation_id),
                "UN-ECE-R94",
            )

    criteria: list[CriterionVerdict] = []
    for m in measurements:
        row: LimitRow | None = None
        for alias in m.lookup_aliases:
            row = find_limit(alias=alias, unit=m.unit or None, regulation_id=rid)
            if row is not None:
                break
        display_name = m.criterion_query
        if row is None:
            criteria.append(
                CriterionVerdict(
                    criterion=display_name,
                    aliases_matched=list(m.lookup_aliases),
                    measured=m.measured_value,
                    measured_display=_format_number(m.measured_value, m.measured_raw),
                    unit=m.unit,
                    verdict="LIMIT_NOT_FOUND",
                    note=f"limit not found for {display_name}",
                    regulation_id=rid or "",
                )
            )
            continue

        unit = m.unit or row.unit or ""
        verdict = compare_values(m.measured_value, row.operator, float(row.limit_value))
        criteria.append(
            CriterionVerdict(
                criterion=row.criterion_name or display_name,
                aliases_matched=list(m.lookup_aliases),
                measured=m.measured_value,
                measured_display=_format_number(m.measured_value, m.measured_raw),
                limit=float(row.limit_value),
                limit_display=_format_number(float(row.limit_value)),
                unit=unit,
                operator=row.operator or "<=",
                verdict=verdict,
                source_chunk_id=row.source_chunk_id,
                section_number=row.section_number,
                regulation_id=row.regulation_id,
            )
        )

    for qv in qualitative:
        if rid and not qv.regulation_id:
            qv = qv.model_copy(update={"regulation_id": rid})
        criteria.append(qv)

    if not criteria:
        result = ComplianceResult(
            is_compliance_check=True,
            overall_verdict="CANNOT_DETERMINE",
            criteria=[],
            regulation_id=rid,
            summary_prose=CANNOT_DETERMINE_MESSAGE,
            answer_text=CANNOT_DETERMINE_MESSAGE,
        )
        return result

    if any(c.verdict == "FAIL" for c in criteria):
        overall = "FAIL"
    elif any(c.verdict == "LIMIT_NOT_FOUND" for c in criteria):
        overall = "INCOMPLETE"
    elif criteria and all(c.verdict == "PASS" for c in criteria):
        overall = "PASS"
    else:
        overall = "INCOMPLETE"

    result = ComplianceResult(
        is_compliance_check=True,
        overall_verdict=overall,
        criteria=criteria,
        regulation_id=rid,
    )
    result.answer_text = render_compliance_answer(result)
    return result


def render_compliance_answer(result: ComplianceResult) -> str:
    """Deterministic answer text from structured fields (numbers never LLM-authored)."""
    if result.overall_verdict == "CANNOT_DETERMINE":
        return (result.summary_prose or CANNOT_DETERMINE_MESSAGE).strip()

    lines: list[str] = []
    overall = result.overall_verdict
    lines.append(f"Overall verdict: {overall}")
    if result.summary_prose:
        lines.append(result.summary_prose)
    lines.append("")
    for c in result.criteria:
        unit = f" {c.unit}" if c.unit else ""
        if c.verdict == "LIMIT_NOT_FOUND":
            lines.append(
                f"- {c.criterion}: measured {c.measured_display}{unit} — "
                f"limit not found for {c.criterion}"
            )
            continue
        if c.measured is None and not c.limit_display:
            # Qualitative PASS/FAIL (no numeric comparison).
            lines.append(
                f"- {c.criterion}: → {c.verdict}"
                + (f" ({c.note})" if c.note else "")
            )
        else:
            op = c.operator or "<="
            op_disp = "<=" if op in {"<=", "≤", "=<"} else op
            lines.append(
                f"- {c.criterion}: measured {c.measured_display}{unit} "
                f"{op_disp} limit {c.limit_display}{unit} "
                f"→ {c.verdict}"
                + (f" ({c.note})" if c.note else "")
            )
        if c.section_number or c.source_chunk_id:
            cite_bits = []
            if c.regulation_id:
                cite_bits.append(c.regulation_id.replace("UN-ECE-", ""))
            if c.section_number:
                cite_bits.append(f"§{c.section_number}")
            lines.append(f"  source: {', '.join(cite_bits) or c.source_chunk_id}")
    return "\n".join(lines).strip()


def phrase_compliance_with_llm(
    result: ComplianceResult,
    *,
    llm: Any,
    question: str,
) -> ComplianceResult:
    """Ask the LLM for connecting prose only; overwrite all numbers from ``result``."""
    fixed = {
        "overall_verdict": result.overall_verdict,
        "criteria": [
            {
                "criterion": c.criterion,
                "measured": c.measured,
                "limit": c.limit,
                "unit": c.unit,
                "operator": c.operator,
                "verdict": c.verdict,
            }
            for c in result.criteria
        ],
    }
    user = (
        f"Question: {question}\n\n"
        f"FIXED VALUES (do not alter — Fix 22 template):\n"
        f"{json.dumps(fixed, ensure_ascii=False)}\n\n"
        "Write summary_prose (2-4 sentences) that leads with the overall verdict "
        "and explains each criterion using the EXACT measured/limit/operator/"
        "verdict fields above. Optionally add a short note per criterion. "
        "Copy all numbers and verdicts exactly — phrase around them only."
    )
    try:
        out = llm.complete(
            messages=[
                {"role": "system", "content": PHRASE_SYSTEM},
                {"role": "user", "content": user},
            ],
            role="answer",
            question=f"compliance_phrase:{question[:80]}",
            response_format=PHRASE_RESPONSE_FORMAT,
            skip_cache=True,
            temperature=0.0,
        )
        result.phrasing_model = str(getattr(out, "model", "") or "")
        result.phrasing_provider = str(
            getattr(out, "served_provider", None) or getattr(out, "provider", "") or ""
        )
        result.phrasing_target_index = getattr(out, "target_index", None)
        result.phrasing_was_fallback = bool(getattr(out, "was_fallback", False))
        data = json.loads(out.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("compliance phrasing LLM failed (%s); using deterministic text", exc)
        result.answer_text = render_compliance_answer(result)
        return result

    # Enforce fixed numbers — discard any LLM mutation.
    result.summary_prose = str(data.get("summary_prose") or "").strip()
    notes = data.get("criteria") if isinstance(data.get("criteria"), list) else []
    for i, c in enumerate(result.criteria):
        if i < len(notes) and isinstance(notes[i], dict):
            c.note = str(notes[i].get("note") or "").strip()
    # overall_verdict stays computed
    result.answer_text = render_compliance_answer(result)
    return result


def sources_for_compliance(
    result: ComplianceResult,
    chunks: Sequence[Any] | None = None,
) -> list[Any]:
    """Build SourceChunk list from limit metadata / retrieved chunks.

    When a limit row's ``section_number`` is more precise than the indexed
    mega-chunk (e.g. ``5.3.6`` vs payload ``5``), sync the criterion's
    ``section_number`` to the cited chunk so prose § marks match citations.
    """
    from generation.answer import SourceChunk, to_source
    from retrieval.retrieve import RetrievedChunk

    by_id = {
        getattr(c, "chunk_id", ""): c
        for c in (chunks or [])
        if getattr(c, "chunk_id", None)
    }
    sources: list[Any] = []
    seen: set[str] = set()
    for c in result.criteria:
        cid = c.source_chunk_id
        if cid and cid not in seen:
            seen.add(cid)
            hit = by_id.get(cid)
            if hit is not None and isinstance(hit, RetrievedChunk):
                # Keep prose § aligned with the cited chunk's payload.
                hsec = (hit.section_number or "").strip()
                if hsec and hsec != (c.section_number or "").strip():
                    c.section_number = hsec
                sources.append(to_source(hit))
                continue
            sources.append(
                SourceChunk(
                    chunk_id=cid,
                    regulation_id=c.regulation_id,
                    section_number=c.section_number,
                    text="",
                    citation=(
                        f"[{c.regulation_id.replace('UN-ECE-', '') or c.regulation_id} "
                        f"§{c.section_number or '?'}, p.?]"
                    ),
                )
            )
            continue
        # Fallback: match retrieved chunk by section prefix (qualitative rules).
        sec = (c.section_number or "").strip()
        if not sec or not chunks:
            continue
        for hit in chunks:
            if not isinstance(hit, RetrievedChunk):
                continue
            hsec = (hit.section_number or "").strip()
            if hsec == sec or hsec.startswith(sec + ".") or hsec.startswith(sec + "/"):
                hid = hit.chunk_id or ""
                if hid and hid not in seen:
                    seen.add(hid)
                    if hsec and hsec != sec:
                        c.section_number = hsec
                    sources.append(to_source(hit))
                break
    return sources


def align_compliance_sections_with_sources(
    result: ComplianceResult,
    sources: Sequence[Any],
) -> ComplianceResult:
    """Re-render after syncing criterion section_numbers to cited sources."""
    by_id = {
        getattr(s, "chunk_id", ""): getattr(s, "section_number", "") or ""
        for s in sources
        if getattr(s, "chunk_id", None)
    }
    changed = False
    for c in result.criteria:
        sec = by_id.get(c.source_chunk_id or "")
        if sec and sec != (c.section_number or "").strip():
            c.section_number = sec
            changed = True
    if changed or not (result.answer_text or "").strip():
        result.answer_text = render_compliance_answer(result)
    return result
