"""Multi-turn condensation checks for follow-up resolution."""

from __future__ import annotations

import logging
from typing import Any

from api.conversations import Turn
from retrieval.rewrite import condense_followup, rewrite_query

logger = logging.getLogger(__name__)


def _is_multi_turn_case(case: dict[str, Any]) -> bool:
    tags = {str(t).lower() for t in (case.get("tags") or [])}
    return (
        "multi-turn" in tags
        or "condensation" in tags
        or bool(case.get("history"))
        or bool(case.get("expect_condensed_contains"))
    )


def _history_turns(case: dict[str, Any]) -> list[Turn]:
    out: list[Turn] = []
    for row in case.get("history") or []:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        out.append(Turn(question=q, answer=str(row.get("answer") or "").strip()))
    return out


def _contains_all(text: str, needles: list[str]) -> list[str]:
    low = text.lower()
    return [n for n in needles if n.lower() not in low]


def _contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    low = text.lower()
    return any(n.lower() in low for n in needles)


def check_condensation_case(case: dict[str, Any], *, use_llm: bool = False) -> dict[str, Any]:
    """Assert condensed follow-up keeps topic + prior regulation context."""
    question = str(case.get("question") or "").strip()
    history = _history_turns(case)
    must = list(case.get("expect_condensed_contains") or [])
    any_of = list(case.get("expect_condensed_any") or [])
    expect_applied = case.get("expect_condensation_applied")

    condensed, applied = condense_followup(question, history, use_llm=use_llm)
    # Also exercise rewrite path (acronym expand after condense).
    rewritten = rewrite_query(question, use_llm=use_llm, history=history)

    if expect_applied is not None and bool(expect_applied) != applied:
        raise AssertionError(
            f"{case['id']}: condensation_applied={applied}, expected {expect_applied}"
        )
    if history and not applied:
        raise AssertionError(f"{case['id']}: expected condensation with non-empty history")
    if not history and applied:
        raise AssertionError(f"{case['id']}: condensation should skip when history is empty")

    missing = _contains_all(condensed, must)
    if missing:
        raise AssertionError(
            f"{case['id']}: condensed={condensed!r} missing required tokens {missing}"
        )
    if not _contains_any(condensed, any_of):
        raise AssertionError(
            f"{case['id']}: condensed={condensed!r} missing any of {any_of}"
        )

    # rewritten.condensed should match condense_followup output.
    if (rewritten.condensed or "").strip() != condensed.strip():
        raise AssertionError(
            f"{case['id']}: rewrite_query condensed mismatch "
            f"{rewritten.condensed!r} vs {condensed!r}"
        )

    logger.info(
        "condensation ok id=%s original=%r condensed=%r",
        case["id"],
        question,
        condensed,
    )
    return {
        "id": case["id"],
        "question": question,
        "condensed": condensed,
        "condensation_applied": applied,
    }


def run_condensation_eval(
    gold: list[dict[str, Any]],
    *,
    use_llm: bool = False,
) -> list[dict[str, Any]]:
    cases = [c for c in gold if _is_multi_turn_case(c)]
    if not cases:
        return []
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in cases:
        try:
            rows.append(check_condensation_case(case, use_llm=use_llm))
        except AssertionError as exc:
            failures.append(str(exc))
    if failures:
        raise AssertionError(
            "Multi-turn condensation failed:\n  - " + "\n  - ".join(failures)
        )
    return rows
