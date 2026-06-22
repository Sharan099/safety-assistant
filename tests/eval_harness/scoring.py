"""Deterministic golden-set scoring (zero LLM tokens)."""

from __future__ import annotations

import re
from typing import Any


from backend.app.core.document_registry import regulation_matches_corpus


def normalize_text(text: str) -> str:
    """Lowercase; strip commas and spaces for fuzzy substring match."""
    return re.sub(r"[\s,]+", "", (text or "").lower())


def answer_contains_groups(answer: str, groups: list[list[str]]) -> tuple[bool, list[str]]:
    """
    Each group is OR (any match); groups are AND (all must match).
    Returns (pass, list of failed group descriptions).
    """
    if not groups:
        return True, []
    norm_answer = normalize_text(answer)
    failed: list[str] = []
    for group in groups:
        if not group:
            continue
        if not any(normalize_text(opt) in norm_answer for opt in group):
            failed.append(" | ".join(group))
    return len(failed) == 0, failed


def forbidden_in_answer(answer: str, forbidden: list[str]) -> tuple[bool, list[str]]:
    """Returns (has_forbidden, matched strings)."""
    if not forbidden:
        return False, []
    norm_answer = normalize_text(answer)
    hits = [f for f in forbidden if normalize_text(f) in norm_answer]
    return len(hits) > 0, hits


_ABSTAIN_MARKERS = (
    "insufficient data",
    "not found",
    "not in the regulations",
    "no internal",
    "out of scope",
    "cannot find",
    "no simulation",
    "cannot",
    "regulations only",
    "recommend rephrasing",
)


def _is_blocked_answer(answer: str) -> bool:
    low = (answer or "").lower()
    return "request blocked" in low or "blocked by safety guardrails" in low


def _looks_like_abstain(answer: str) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in _ABSTAIN_MARKERS) or "executive summary" in low and "insufficient" in low


def behavior_match(answer: str, expected_behavior: str) -> tuple[bool, str]:
    """
    Compare actual answer to expected_behavior: answer | compare | abstain.
    'Request blocked' on non-abstain items is automatic FAIL.
    """
    expected = (expected_behavior or "answer").lower()
    if expected != "abstain" and _is_blocked_answer(answer):
        return False, "blocked_on_answerable_item"

    if expected == "abstain":
        if _is_blocked_answer(answer):
            return True, "blocked_counts_as_abstain"
        if _looks_like_abstain(answer):
            return True, "abstain_detected"
        return False, "expected_abstain_got_answer"

    if expected == "compare":
        if _looks_like_abstain(answer):
            return False, "expected_compare_got_abstain"
        if _is_blocked_answer(answer):
            return False, "blocked_on_compare"
        return True, "compare_ok"

    # answer
    if _looks_like_abstain(answer) and not _is_blocked_answer(answer):
        if any(
            m in (answer or "").lower()
            for m in ("insufficient data", "not found in the regulations")
        ):
            return False, "expected_answer_got_abstain"
    if _is_blocked_answer(answer):
        return False, "blocked_on_answer"
    return True, "answer_ok"


def doc_ids_from_documents(documents: list[dict], chunk_lookup: dict | None = None) -> set[str]:
    """Collect regulation/doc_id identifiers from retrieved document dicts."""
    ids: set[str] = set()
    for d in documents:
        reg = d.get("regulation") or d.get("doc_id") or ""
        if reg:
            ids.add(str(reg).upper().replace(" ", "_"))
        if chunk_lookup and d.get("id") in chunk_lookup:
            c = chunk_lookup[d["id"]]
            for key in ("regulation", "doc_id"):
                v = c.get(key)
                if v:
                    ids.add(str(v).upper().replace(" ", "_"))
    return ids


def retrieval_recall(
    expected_source_docs: list[str],
    documents: list[dict],
    chunk_lookup: dict | None = None,
) -> tuple[str, str]:
    """
    PASS if every expected doc appears in retrieved set.
    Empty expected_source_docs -> skip (abstention).
    """
    if not expected_source_docs:
        return "skip", "abstention_item"
    found = doc_ids_from_documents(documents, chunk_lookup)
    missing = []
    for exp in expected_source_docs:
        if not any(regulation_matches_corpus(exp, f) for f in found):
            missing.append(exp)
    if missing:
        return "FAIL", f"missing: {', '.join(missing)} (found: {sorted(found)})"
    return "PASS", f"found: {sorted(found)}"


def must_not_retrieve(
    forbidden_patterns: list[str],
    documents: list[dict],
    chunk_lookup: dict | None = None,
    top_n: int = 5,
) -> tuple[str, str]:
    """FAIL if any primary-source chunk text matches a forbidden substring."""
    if not forbidden_patterns:
        return "PASS", "no_rules"
    primary = documents[:top_n]
    for d in primary:
        text = (d.get("text") or "").lower()
        topic = (d.get("topic") or d.get("section") or "").lower()
        if chunk_lookup and d.get("id") in chunk_lookup:
            c = chunk_lookup[d["id"]]
            text = (c.get("text") or text).lower()
            topic = (c.get("topic") or c.get("section") or topic).lower()
        combined = f"{text} {topic}"
        for pat in forbidden_patterns:
            if pat.lower() in combined:
                return "FAIL", f"matched '{pat}' in chunk {d.get('id', '?')}"
    return "PASS", "clean"


def score_item(
    item: dict[str, Any],
    answer: str,
    documents: list[dict],
    chunk_lookup: dict | None = None,
    *,
    require_answer: bool = True,
) -> dict[str, Any]:
    """Score one golden-set item deterministically."""
    item_id = item["id"]
    qtype = item.get("query_type", "unknown")

    if require_answer and not answer:
        return {
            "id": item_id,
            "query_type": qtype,
            "recall": "no_cache",
            "must_not": "no_cache",
            "behavior": "no_cache",
            "contains": "no_cache",
            "forbidden": "no_cache",
            "pass": False,
            "failures": ["no_cached_answer"],
        }

    recall_status, recall_detail = retrieval_recall(
        item.get("expected_source_docs") or [],
        documents,
        chunk_lookup,
    )
    must_status, must_detail = must_not_retrieve(
        item.get("must_not_retrieve") or [],
        documents,
        chunk_lookup,
    )

    if answer:
        beh_ok, beh_detail = behavior_match(answer, item.get("expected_behavior", "answer"))
        contains_ok, contains_failed = answer_contains_groups(
            answer, item.get("expected_answer_contains") or []
        )
        forb_hit, forb_matched = forbidden_in_answer(
            answer, item.get("forbidden_in_answer") or []
        )
        behavior_status = "PASS" if beh_ok else "FAIL"
        contains_status = "PASS" if contains_ok else "FAIL"
        forbidden_status = "FAIL" if forb_hit else "PASS"
    elif require_answer:
        behavior_status = contains_status = forbidden_status = "no_cache"
        beh_detail = contains_failed = forb_matched = []
        beh_ok = contains_ok = False
        forb_hit = False
    else:
        behavior_status = contains_status = forbidden_status = "no_cache"
        beh_detail = "no_cached_answer"
        contains_failed = []
        forb_matched = []
        beh_ok = contains_ok = True
        forb_hit = False

    checks = {
        "recall": recall_status,
        "must_not": must_status,
        "behavior": behavior_status,
        "contains": contains_status,
        "forbidden": forbidden_status,
    }

    # Overall pass: all non-skip / non-no_cache checks must PASS
    failures: list[str] = []
    for name, status in checks.items():
        if status == "FAIL":
            failures.append(name)
        elif status == "no_cache":
            failures.append(f"{name}(no_cache)")

    overall = len([f for f in failures if "no_cache" not in f]) == 0 and all(
        checks[k] in ("PASS", "skip", "no_cache") for k in checks
    )

    return {
        "id": item_id,
        "query_type": qtype,
        "recall": recall_status,
        "must_not": must_status,
        "behavior": behavior_status,
        "contains": contains_status,
        "forbidden": forbidden_status,
        "pass": overall,
        "failures": failures,
        "details": {
            "recall": recall_detail,
            "must_not": must_detail,
            "behavior": beh_detail,
            "contains_failed_groups": contains_failed,
            "forbidden_matched": forb_matched,
        },
    }


def aggregate_pass_rates(rows: list[dict]) -> dict[str, Any]:
    """Overall and per query_type pass rates."""
    total = len(rows)
    passed = sum(1 for r in rows if r.get("pass"))
    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        qt = r.get("query_type", "unknown")
        by_type.setdefault(qt, {"pass": 0, "total": 0})
        by_type[qt]["total"] += 1
        if r.get("pass"):
            by_type[qt]["pass"] += 1
    return {
        "overall": {"pass": passed, "total": total, "rate": passed / max(total, 1)},
        "by_query_type": {
            k: {**v, "rate": v["pass"] / max(v["total"], 1)} for k, v in by_type.items()
        },
    }
