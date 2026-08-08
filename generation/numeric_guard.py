"""Guard against LLM numeric hallucination of user-provided measurements.

Immediate mitigation for cases like: user said ``35 g/min``, model answered
``measured value of 3 g/min`` and inverted FAIL → PASS.

Extract user numbers *before* generation; after generation, reject answers that
restate a user measurement with a number that was never in the query.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_HALLUCINATION_PATH = ROOT / "numeric_hallucination.jsonl"

NUMERIC_CONFIRM_MESSAGE = (
    "I want to make sure I use your exact figures correctly — "
    "could you confirm the measured value you provided?"
)

# Unit-bearing measurements (preferred signal).
_UNIT_NUMBER_RE = re.compile(
    r"(?ix)"
    r"(?<![\w./-])"
    r"(?P<num>\d+(?:[.,]\d+)?)"
    r"\s*"
    r"(?P<unit>"
    r"g\s*/\s*min|g/min|"
    r"mm|millimet(?:er|re)s?|"
    r"kN|kn|"
    r"m\s*/\s*s|m/s|m/sec|"
    r"ms|"
    r"%"
    r")"
    r"\b"
)

# Bare measured numbers: "of 35", "was 1250", "recorded an HPC of 1250".
_BARE_MEASURED_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:leakage\s+rate|rate)\s+is\s+"
    r"|(?:measured|produced|recorded|observed)\s+(?:an?\s+\w+\s+)?(?:of\s+)?"
    r"|(?:value|reading|figure|deflection|force|criterion)\s+(?:of|was|is)\s+"
    r"|(?:of|was|is)\s+"
    r")"
    r"(?P<num>\d+(?:[.,]\d+)?)"
    r"(?!\s*(?:series|amendment|page|clause|section)\b)"
)

# Answer phrases that claim to restate the user's measurement.
_RESTATE_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"measured\s+(?:value|reading|figure)?\s*(?:of\s+)?"
    r"|the\s+measured\s+(?:value|reading|figure)?\s*(?:of\s+)?"
    r"|your\s+(?:measured\s+)?(?:value|figure|reading|measurement)\s*(?:of\s+)?"
    r"|user[- ](?:provided|stated|given)\s+(?:value|figure|measurement)\s*(?:of\s+)?"
    r"|provided\s+(?:value|figure|measurement)\s*(?:of\s+)?"
    r"|given\s+(?:value|figure|measurement)\s*(?:of\s+)?"
    r"|recorded\s+(?:value\s+)?(?:of\s+)?"
    r"|produced\s+(?:a\s+)?(?:value\s+of\s+)?"
    r"|leakage\s+rate\s+(?:is|of|was)\s+"
    r"|(?:value|reading|figure)\s+of\s+"
    r")"
    r"(?P<num>\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<unit>g\s*/\s*min|g/min|mm|kN|kn|m\s*/\s*s|m/s|ms|%))?"
)

_VERDICT_RE = re.compile(
    r"(?ix)\b(?:"
    r"pass(?:es|ed)?|fail(?:s|ed)?|"
    r"compl(?:y|ies|iance)|satisf(?:y|ies|ied)|"
    r"within\s+(?:the\s+)?limit|exceed(?:s|ed)?"
    r")\b"
)

# Same-unit number near comparison language (broader net than restatement verbs).
_UNIT_NEAR_COMPARE_RE = re.compile(
    r"(?ix)"
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>g\s*/\s*min|g/min|mm|kN|kn|m\s*/\s*s|m/s|ms|%)"
    r".{0,40}?"
    r"(?:less\s+than|greater\s+than|exceed|within|below|above|compared|versus|vs\.?)"
    r"|"
    r"(?:less\s+than|greater\s+than|exceed|within|below|above|measured|value\s+of)"
    r".{0,40}?"
    r"(?P<num2>\d+(?:[.,]\d+)?)\s*(?P<unit2>g\s*/\s*min|g/min|mm|kN|kn|m\s*/\s*s|m/s|ms|%)"
)


def normalize_number_token(raw: str) -> str:
    """Canonical string form: strip spaces, comma→dot decimal."""
    s = (raw or "").strip().replace(" ", "").replace(",", ".")
    # Avoid trailing dot artifacts.
    if s.endswith(".") and s.count(".") > 1:
        s = s[:-1]
    return s


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = re.sub(r"\s+", "", unit.strip().lower())
    u = u.replace("millimetres", "mm").replace("millimeters", "mm")
    u = u.replace("millimetre", "mm").replace("millimeter", "mm")
    if u in {"m/sec", "ms-1"}:
        u = "m/s"
    if u == "kn":
        u = "kn"
    if u in {"g/min", "g/min."}:
        u = "g/min"
    return u


@dataclass(frozen=True)
class UserNumber:
    raw: str
    normalized: str
    value: float
    unit: str | None = None

    def matches_token(self, token: str) -> bool:
        other = normalize_number_token(token)
        if not other:
            return False
        if self.normalized == other:
            return True
        # Numeric equality (35 == 35.0) but NOT 35 ≈ 3.
        try:
            return abs(float(self.normalized) - float(other)) < 1e-9
        except ValueError:
            return False


@dataclass
class NumericGuardResult:
    ok: bool
    user_numbers: list[UserNumber] = field(default_factory=list)
    offending_token: str | None = None
    reason: str = ""

    @property
    def rejected(self) -> bool:
        return not self.ok


def extract_user_numbers(question: str) -> list[UserNumber]:
    """Extract every measurement-like number the user stated in the query."""
    q = question or ""
    found: list[UserNumber] = []
    seen: set[tuple[str, str | None]] = set()

    def _add(num_raw: str, unit_raw: str | None) -> None:
        norm = normalize_number_token(num_raw)
        unit = normalize_unit(unit_raw)
        if not norm:
            return
        # Skip lone regulation-ish tiny ints without units (R95 handled elsewhere).
        try:
            value = float(norm)
        except ValueError:
            return
        key = (norm, unit)
        if key in seen:
            return
        seen.add(key)
        found.append(
            UserNumber(raw=num_raw.strip(), normalized=norm, value=value, unit=unit)
        )

    for m in _UNIT_NUMBER_RE.finditer(q):
        _add(m.group("num"), m.group("unit"))

    for m in _BARE_MEASURED_RE.finditer(q):
        # Skip if this span was already captured with a unit.
        start, end = m.span("num")
        window = q[start: min(len(q), end + 12)]
        if _UNIT_NUMBER_RE.search(window):
            continue
        _add(m.group("num"), None)

    return found


def _token_matches_any(token: str, user_numbers: Sequence[UserNumber]) -> bool:
    return any(u.matches_token(token) for u in user_numbers)


def _unit_compatible(answer_unit: str | None, user_numbers: Sequence[UserNumber]) -> bool:
    au = normalize_unit(answer_unit)
    if not au:
        return True
    user_units = {u.unit for u in user_numbers if u.unit}
    if not user_units:
        return True
    return au in user_units


def check_numeric_fidelity(
    question: str,
    answer: str,
) -> NumericGuardResult:
    """Return ok=False when the answer restates a user measurement incorrectly."""
    user_numbers = extract_user_numbers(question)
    if not user_numbers:
        return NumericGuardResult(ok=True, user_numbers=[])

    answer = answer or ""
    if not answer.strip():
        return NumericGuardResult(ok=True, user_numbers=list(user_numbers))

    # 1) Explicit restatement of the measured / provided value.
    for m in _RESTATE_RE.finditer(answer):
        token = m.group("num")
        unit = m.groupdict().get("unit")
        if _token_matches_any(token, user_numbers):
            continue
        # Wrong figure in a restatement clause — always reject when units match
        # or the user only supplied one measurement.
        if unit and not _unit_compatible(unit, user_numbers):
            # e.g. restating a limit in a different unit family — ignore.
            continue
        return NumericGuardResult(
            ok=False,
            user_numbers=list(user_numbers),
            offending_token=token,
            reason=(
                f"answer restates measured value as {token!r} but user query "
                f"numbers are {[u.raw for u in user_numbers]!r}"
            ),
        )

    # 2) Same-unit number near comparison language that isn't a user figure.
    #    Allows regulatory limits (30 g/min) only when they also appear as a
    #    distinct second number alongside a correct user figure — we only flag
    #    when a same-unit number looks like the *measured* side and mismatches.
    for m in _RESTATE_RE.finditer(answer):
        pass  # already handled

    # 3) Verdict present → every unit-bearing user measurement must appear verbatim.
    if _VERDICT_RE.search(answer):
        for u in user_numbers:
            if not u.unit:
                continue
            # Require the normalized digits to appear (35, 7.5, 0.8, 1250).
            if not _number_appears_verbatim(answer, u):
                return NumericGuardResult(
                    ok=False,
                    user_numbers=list(user_numbers),
                    offending_token=None,
                    reason=(
                        f"verdict answer omitted user-measured {u.raw}"
                        + (f" {u.unit}" if u.unit else "")
                    ),
                )

    # 4) Truncation / digit-drop heuristic: same unit, answer has a number that
    #    is a strict prefix/suffix of a user number (35 → 3) near restatement cues.
    for u in user_numbers:
        if not u.unit:
            continue
        unit_pat = re.escape(u.unit).replace(r"\/", r"\s*/\s*")
        for m in re.finditer(
            rf"(?ix)(?P<num>\d+(?:[.,]\d+)?)\s*{unit_pat}",
            answer,
        ):
            token = m.group("num")
            if _token_matches_any(token, user_numbers):
                continue
            # Likely limit values (shorter or unrelated) — only reject if the
            # answer number is a corrupted form of the user number.
            if _is_digit_corruption(u.normalized, normalize_number_token(token)):
                # Only preceding context — avoid flagging a regulatory limit that
                # sits just before a faithful "measured 42.5 mm" restatement.
                start = max(0, m.start() - 64)
                ctx_before = answer[start: m.start()].lower()
                if re.search(
                    r"measured|value\s+of|provided|your|leakage\s+rate|reading|figure\s+of",
                    ctx_before,
                ):
                    return NumericGuardResult(
                        ok=False,
                        user_numbers=list(user_numbers),
                        offending_token=token,
                        reason=(
                            f"answer figure {token!r} looks like a corruption of "
                            f"user-measured {u.raw!r}"
                        ),
                    )

    return NumericGuardResult(ok=True, user_numbers=list(user_numbers))


def _number_appears_verbatim(answer: str, user_number: UserNumber) -> bool:
    """True if the user's digits appear as a number token in the answer."""
    # Prefer exact raw / normalized forms as whole number tokens.
    candidates = {user_number.raw.strip(), user_number.normalized}
    # Also allow European comma form of the normalized value.
    if "." in user_number.normalized:
        candidates.add(user_number.normalized.replace(".", ","))
    for cand in candidates:
        if not cand:
            continue
        if re.search(rf"(?<![\d.]){re.escape(cand)}(?![\d.])", answer):
            return True
    return False


def _is_digit_corruption(user_norm: str, answer_norm: str) -> bool:
    """True when answer digits look like a truncated/altered form of the user value."""
    if not user_norm or not answer_norm or user_norm == answer_norm:
        return False
    # Strip decimal dots for digit-sequence comparison.
    u_digits = user_norm.replace(".", "")
    a_digits = answer_norm.replace(".", "")
    if not a_digits or not u_digits:
        return False
    if a_digits == u_digits:
        return False
    # Prefix/suffix truncation: 35→3, 1250→125, 7.5→7
    if u_digits.startswith(a_digits) or u_digits.endswith(a_digits):
        return len(a_digits) < len(u_digits)
    return False


def log_numeric_hallucination(
    *,
    question: str,
    answer: str,
    result: NumericGuardResult,
    trace_id: str = "",
    model: str = "",
    provider: str = "",
    path: Path | None = None,
) -> None:
    """Append a critical-incident record to ``numeric_hallucination.jsonl``."""
    out = path or NUMERIC_HALLUCINATION_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "severity": "CRITICAL",
        "incident": "numeric_hallucination",
        "question": question,
        "answer": answer,
        "user_numbers": [
            {"raw": u.raw, "normalized": u.normalized, "unit": u.unit}
            for u in result.user_numbers
        ],
        "offending_token": result.offending_token,
        "reason": result.reason,
        "trace_id": trace_id,
        "model": model,
        "provider": provider,
    }
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.critical(
        "numeric_hallucination rejected offending=%r reason=%s trace=%s",
        result.offending_token,
        result.reason,
        trace_id,
    )
