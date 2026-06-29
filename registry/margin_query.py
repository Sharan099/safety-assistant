"""Query-time margin / close-to-limit handling — deterministic arithmetic, harness SQL for follow-ups."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from database.models import Chunk, Document, Regulation, TestResult
from registry.harness_limits import (
    compute_headroom_mm,
    derive_pass_fail,
    extract_limit_details,
    headroom_pct_of_limit,
)

# Exact transcript strings from production testing (do not paraphrase in tests).
MARGIN_TRANSCRIPT_TURN_A = "38.4mm thorax vs 42mm UN R94 limit, what's the margin?"
MARGIN_TRANSCRIPT_TURN_B = (
    "38.4mm vs 42mm limit, is that margin tight, what else is close to its limit?"
)

_MM_PAIR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s+.*?\bvs\.?\b.*?(\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE | re.DOTALL,
)
_MM_TWO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_MARGIN_ASK_RE = re.compile(
    r"\b(?:margin|headroom|clearance|vs\.?|versus|against\s+(?:the\s+)?limit)\b",
    re.IGNORECASE,
)
_CLOSE_LIMIT_RE = re.compile(
    r"(?:close\s+to\s+(?:its\s+)?limit|margin\s+tight|what\s+else\s+is\s+close)",
    re.IGNORECASE,
)
_CRITERION_HINT_RE = re.compile(r"\b(thorax|thcc|chest|hic|femur|tibia)\b", re.IGNORECASE)
_REG_RE = re.compile(r"\b(?:UN[\s_-]?)?(R\d{2,3})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedMeasurement:
    measured_mm: float
    limit_mm: float
    criterion: str
    regulation_code: str | None
    direction: str = "ceiling"


def is_margin_query(query: str) -> bool:
    nums = _MM_TWO_RE.findall(query)
    if len(nums) < 2:
        return False
    return bool(_MARGIN_ASK_RE.search(query)) or bool(_CLOSE_LIMIT_RE.search(query))


def parse_measurement_pair(query: str) -> ParsedMeasurement | None:
    m = _MM_PAIR_RE.search(query)
    if m:
        measured, limit = float(m.group(1)), float(m.group(2))
    else:
        nums = [float(x) for x in _MM_TWO_RE.findall(query)]
        if len(nums) < 2:
            return None
        measured, limit = nums[0], nums[1]

    crit_m = _CRITERION_HINT_RE.search(query)
    criterion = "ThCC"
    if crit_m:
        token = crit_m.group(1).lower()
        criterion = "ThCC" if token in ("thorax", "thcc", "chest") else token.upper()

    reg_m = _REG_RE.search(query)
    regulation_code = f"UN_{reg_m.group(1)}" if reg_m else None

    return ParsedMeasurement(
        measured_mm=measured,
        limit_mm=limit,
        criterion=criterion,
        regulation_code=regulation_code,
    )


def _lookup_limit_from_clause(db: Session, linked_clause: str, criterion: str) -> tuple[float, str, str] | None:
    if "#" not in linked_clause:
        return None
    reg_code, section = linked_clause.split("#", 1)
    reg = (
        db.query(Regulation)
        .filter(Regulation.regulation_code == reg_code.replace("UN_", "UN_"))
        .order_by(Regulation.effective_date.desc().nullslast())
        .first()
    )
    if not reg:
        reg = db.query(Regulation).filter(Regulation.regulation_code == reg_code).first()
    if not reg:
        return None
    chunk = (
        db.query(Chunk)
        .join(Document)
        .filter(Document.regulation_id == reg.id, Chunk.section == section)
        .first()
    )
    if not chunk:
        return None
    return extract_limit_details(criterion, chunk.chunk_text)


def margin_requires_harness_auth(query: str) -> bool:
    """Turn-B style margin queries list harness SQL rows — require login."""
    return bool(_CLOSE_LIMIT_RE.search(query))


def query_close_to_limit_results(
    db: Session,
    *,
    limit_mm: float,
    direction: str = "ceiling",
    top_n: int = 5,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Structured harness query: rank ingested test_results by smallest headroom to limit."""
    from database.models import Test
    from registry.harness_security import user_can_access_test

    rows = db.query(TestResult).all()
    if not rows:
        return []

    ranked: list[dict[str, Any]] = []
    for tr in rows:
        test = db.query(Test).filter(Test.test_id == tr.test_id).first()
        if test and not user_can_access_test(test, user_id):
            continue
        crit = tr.injury_criterion or "ThCC"
        limit_details = _lookup_limit_from_clause(db, tr.linked_regulation_clause or "", crit)
        if limit_details:
            lim, dirn, unit = limit_details
            if unit != "mm":
                continue
            effective_limit = lim
            effective_dir = dirn
        else:
            effective_limit = limit_mm
            effective_dir = direction

        headroom = compute_headroom_mm(float(tr.value), effective_limit, effective_dir)
        ranked.append(
            {
                "test_id": tr.test_id,
                "criterion": crit,
                "measured_mm": float(tr.value),
                "limit_mm": effective_limit,
                "headroom_mm": headroom,
                "headroom_pct": headroom_pct_of_limit(headroom, effective_limit),
                "pass_fail": derive_pass_fail(float(tr.value), effective_limit, effective_dir),
                "linked_clause": tr.linked_regulation_clause,
            }
        )

    ranked.sort(key=lambda x: x["headroom_mm"])
    return ranked[:top_n]


def build_margin_answer(db: Session, query: str, parsed: ParsedMeasurement, *, user_id: str | None = None) -> str:
    headroom = compute_headroom_mm(parsed.measured_mm, parsed.limit_mm, parsed.direction)
    pct = headroom_pct_of_limit(headroom, parsed.limit_mm)
    pf = derive_pass_fail(parsed.measured_mm, parsed.limit_mm, parsed.direction)

    lines = [
        "**Computed margin (code-derived — do not re-derive):** "
        f"{parsed.limit_mm:.1f} mm limit − {parsed.measured_mm:.1f} mm measured "
        f"= **{headroom:.1f} mm** headroom ({pct:.1f}% of limit). "
        f"Status vs limit: **{pf}**.",
    ]

    if _CLOSE_LIMIT_RE.search(query):
        if headroom <= 0:
            tight = "at or beyond the limit (no positive headroom)."
        elif pct < 10:
            tight = f"**tight** — only {headroom:.1f} mm ({pct:.1f}% of limit) remains."
        elif pct < 20:
            tight = f"**moderately tight** — {headroom:.1f} mm ({pct:.1f}% of limit) headroom."
        else:
            tight = f"**not especially tight** — {headroom:.1f} mm ({pct:.1f}% of limit) headroom."
        lines.append(f"Margin assessment: {tight}")

        close_rows = query_close_to_limit_results(
            db, limit_mm=parsed.limit_mm, direction=parsed.direction, user_id=user_id
        )
        lines.append("")
        lines.append("**Other criteria close to limit (harness test_results, SQL-ranked by headroom):**")
        if not close_rows:
            lines.append(
                "No harness test_results are ingested — cannot rank other measured criteria. "
                "This is not answered from regulation clause retrieval."
            )
        else:
            for row in close_rows:
                lines.append(
                    f"- {row['criterion']} test {row['test_id']}: "
                    f"{row['measured_mm']:.1f} mm vs {row['limit_mm']:.1f} mm limit "
                    f"→ **{row['headroom_mm']:.1f} mm** headroom ({row['headroom_pct']:.1f}%), {row['pass_fail']}"
                )
    elif re.search(r"\bwhat(?:'s| is) the margin\b", query, re.I):
        pass  # Turn A — margin line is sufficient

    reg_note = f" ({parsed.regulation_code})" if parsed.regulation_code else ""
    lines.append(f"Criterion context: {parsed.criterion}{reg_note}.")
    return "\n".join(lines)


def margin_chat_result(query: str, answer: str) -> dict[str, Any]:
    routing = {
        "model_key": "computed",
        "model_id": "harness_margin",
        "provider": "registry",
        "evidence_only": False,
        "cache_hit": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "steps": [],
        "route": "margin_compute",
    }
    return {
        "answer": answer,
        "sources": [],
        "timing": {"total_ms": 0.0, "route": "margin_compute"},
        "metadata": {
            "filters_applied": {},
            "latency_ms": 0.0,
            "routing": routing,
            "timing": {"total_ms": 0.0, "route": "margin_compute"},
            "response_route": "margin_compute",
        },
    }


def try_margin_query_response(db: Session, query: str, *, user_id: str | None = None) -> dict[str, Any] | None:
    if not is_margin_query(query):
        return None
    parsed = parse_measurement_pair(query)
    if not parsed:
        return None
    answer = build_margin_answer(db, query, parsed, user_id=user_id)
    return margin_chat_result(query, answer)
