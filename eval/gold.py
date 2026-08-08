"""Load eval/golden_set.jsonl and normalize gold source keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from retrieval.retrieve import RetrievedChunk

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT / "golden_set.jsonl"
# Backward-compatible alias used by older callers.
DEFAULT_GOLD = DEFAULT_GOLDEN


def load_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Load JSONL gold cases (one JSON object per line).

    The canonical set is ``eval/golden_set.jsonl`` (**30** cases). The prior
    152-case set is archived at ``eval/golden_set_152_archive.jsonl``. There is
    intentionally no fallback to the deprecated 6-case ``eval/archive/gold.json``.
    """
    gold_path = path or DEFAULT_GOLDEN
    if not gold_path.is_file():
        raise FileNotFoundError(
            "golden_set.jsonl not found — refusing to "
            "silently fall back to the 6-case legacy set"
            + (f" (looked for {gold_path})" if gold_path != DEFAULT_GOLDEN else "")
        )

    cases: list[dict[str, Any]] = []
    text = gold_path.read_text(encoding="utf-8-sig")
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(_normalize_case(json.loads(line), i))
    if not cases:
        raise ValueError(f"Golden set empty: {gold_path}")
    return cases


def load_gold(path: Path | None = None) -> list[dict[str, Any]]:
    """Alias for load_golden_set."""
    return load_golden_set(path)


def _normalize_case(raw: dict[str, Any], index: int) -> dict[str, Any]:
    question = (raw.get("question") or raw.get("query") or "").strip()
    # New schema (Step 2): expected_behavior is the free-text pass criteria.
    answer = (
        raw.get("answer")
        or raw.get("ground_truth")
        or raw.get("expected_behavior")
        or ""
    ).strip()
    case_id = str(raw.get("id") or f"case_{index:03d}")
    if not question:
        raise ValueError(f"Gold case {case_id} missing question")

    sections = list(raw.get("expected_sections") or raw.get("gold_section_numbers") or [])
    if raw.get("expected_section_number"):
        sec = str(raw["expected_section_number"]).strip()
        if sec and sec not in [str(s).strip() for s in sections]:
            sections.insert(0, sec)
    section_ids = list(raw.get("expected_section_ids") or raw.get("gold_section_ids") or [])
    chunk_ids = list(raw.get("expected_chunk_ids") or raw.get("gold_chunk_ids") or [])
    if raw.get("expected_chunk_id"):
        cid = str(raw["expected_chunk_id"]).strip()
        if cid and cid not in chunk_ids:
            chunk_ids.append(cid)

    # New schema uses regulation_scope; legacy uses regulation_id.
    reg = (
        raw.get("regulation_id")
        or raw.get("regulation_scope")
        or ""
    )
    if reg is not None:
        reg = str(reg).strip()
    else:
        reg = ""
    if reg.lower() in {"null", "none"}:
        reg = ""

    category = str(raw.get("category") or "").strip().lower() or None
    severity_raw = str(raw.get("severity") or "").strip()
    severity = severity_raw.upper() if severity_raw else None

    expected_answer_contains = [
        str(s).strip()
        for s in (raw.get("expected_answer_contains") or [])
        if str(s).strip()
    ]
    must_not_contain = [
        str(s).strip()
        for s in (raw.get("must_not_contain") or raw.get("banned_answer_substrings") or [])
        if str(s).strip()
    ]
    expected_behavior = str(raw.get("expected_behavior") or "").strip() or None

    # Derive legacy expect_verdict from contains list when present.
    expect_verdict = str(raw.get("expect_verdict") or "").strip().upper() or None
    if not expect_verdict:
        upper_contains = {s.upper() for s in expected_answer_contains}
        if "FAIL" in upper_contains:
            expect_verdict = "FAIL"
        elif "PASS" in upper_contains:
            expect_verdict = "PASS"

    # Auto-build section_ids from regulation + section numbers when missing.
    if reg and sections and not section_ids:
        section_ids = [f"{reg}::{str(s).rstrip('.')}" for s in sections]

    tags = list(raw.get("tags") or [])
    if category and category not in {str(t).lower() for t in tags}:
        tags.append(category)
    if severity == "CRITICAL" and "ci-gate" not in {str(t).lower() for t in tags}:
        tags.append("ci-gate")

    abstention = bool(raw.get("abstention", False))
    expect_not_found = bool(raw.get("expect_not_found", False))
    expect_not_indexed_if_missing = bool(raw.get("expect_not_indexed_if_missing", False))
    if category in {"out_of_scope", "hallucination_probe", "prompt_injection"}:
        abstention = True
    if category == "hallucination_probe" and "not addressed" in (expected_behavior or "").lower():
        expect_not_found = True
    if category == "out_of_scope":
        expect_not_found = True
    beh_l = (expected_behavior or "").lower()
    if "not-indexed" in beh_l or "not indexed" in beh_l:
        expect_not_indexed_if_missing = True

    return {
        "id": case_id,
        "question": question,
        "query": question,
        "answer": answer,
        "ground_truth": answer,
        "regulation_id": reg or None,
        "regulation_scope": reg or None,
        "category": category,
        "expected_behavior": expected_behavior,
        "expected_answer_contains": expected_answer_contains,
        "must_not_contain": must_not_contain,
        "expected_sections": [str(s).strip() for s in sections],
        "expected_section_number": (
            str(raw.get("expected_section_number") or (sections[0] if sections else "")).strip()
            or None
        ),
        "expected_section_ids": [str(s).strip() for s in section_ids],
        "expected_chunk_ids": [str(s).strip() for s in chunk_ids],
        "gold_section_numbers": [str(s).strip() for s in sections],
        "gold_section_ids": [str(s).strip() for s in section_ids],
        "gold_chunk_ids": [str(s).strip() for s in chunk_ids],
        "tags": tags,
        "abstention": abstention,
        "expect_not_indexed": bool(raw.get("expect_not_indexed", False)),
        "expect_not_indexed_if_missing": expect_not_indexed_if_missing,
        "history": list(raw.get("history") or []),
        "expect_condensed_contains": [
            str(s).strip() for s in (raw.get("expect_condensed_contains") or []) if str(s).strip()
        ],
        "expect_condensed_any": [
            str(s).strip() for s in (raw.get("expect_condensed_any") or []) if str(s).strip()
        ],
        "expect_condensation_applied": raw.get("expect_condensation_applied"),
        "severity": severity,
        "expect_verdict": expect_verdict,
        "expect_limit_contains": [
            str(s).strip() for s in (raw.get("expect_limit_contains") or []) if str(s).strip()
        ],
        "expect_measured_contains": [
            str(s).strip()
            for s in (
                raw.get("expect_measured_contains")
                or [
                    x
                    for x in expected_answer_contains
                    if any(ch.isdigit() for ch in x) and x.upper() not in {"FAIL", "PASS"}
                ]
            )
            if str(s).strip()
        ],
        "expect_min_chunks": (
            int(raw["expect_min_chunks"])
            if raw.get("expect_min_chunks") is not None
            else None
        ),
        "expect_topic_keyword": str(raw.get("expect_topic_keyword") or "").strip() or None,
        "expect_context_keywords": [
            str(s).strip().lower()
            for s in (raw.get("expect_context_keywords") or [])
            if str(s).strip()
        ],
        "banned_answer_substrings": [s.lower() for s in must_not_contain],
        "expect_regulation_ids_only": [
            str(s).strip()
            for s in (raw.get("expect_regulation_ids_only") or ([reg] if (
                category == "cross_regulation" and reg and "never" in (expected_behavior or "").lower()
                and "r16" in (expected_behavior or "").lower()
            ) else []))
            if str(s).strip()
        ],
        "banned_regulation_ids": [
            str(s).strip()
            for s in (raw.get("banned_regulation_ids") or [])
            if str(s).strip()
        ],
        "expect_retrieved_regulations": [
            str(s).strip()
            for s in (raw.get("expect_retrieved_regulations") or [])
            if str(s).strip()
        ],
        "expect_answer_mentions_regulations": [
            str(s).strip()
            for s in (raw.get("expect_answer_mentions_regulations") or [])
            if str(s).strip()
        ],
        "expect_design_component": str(raw.get("expect_design_component") or "").strip()
        or None,
        "expect_min_covered_categories": (
            int(raw["expect_min_covered_categories"])
            if raw.get("expect_min_covered_categories") is not None
            else None
        ),
        "expect_applies": [
            str(s).strip()
            for s in (raw.get("expect_applies") or [])
            if str(s).strip()
        ],
        "repeat_retrieval": (
            int(raw["repeat_retrieval"])
            if raw.get("repeat_retrieval") is not None
            else None
        ),
        "expect_not_found": expect_not_found,
    }


def gold_keys(case: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in (
        "expected_chunk_ids",
        "expected_section_ids",
        "expected_sections",
        "gold_chunk_ids",
        "gold_section_ids",
        "gold_section_numbers",
    ):
        for item in case.get(field) or []:
            s = str(item).strip()
            if s:
                keys.add(s)
                keys.add(s.rstrip("."))
    return keys


def chunk_match_keys(chunk: RetrievedChunk) -> set[str]:
    keys: set[str] = set()
    for raw in (
        chunk.chunk_id,
        chunk.section_id,
        chunk.parent_section_id,
        chunk.section_number,
    ):
        if not raw:
            continue
        s = str(raw).strip()
        keys.add(s)
        keys.add(s.rstrip("."))
        if s.startswith("expanded::"):
            bare = s[len("expanded::") :]
            keys.add(bare)
            keys.add(bare.rstrip("."))
            if "::" in bare:
                keys.add(bare.split("::", 1)[1])
        if "::" in s:
            keys.add(s.split("::", 1)[1])
    return {k for k in keys if k}


def retrieved_ranked_keys(chunks: Iterable[RetrievedChunk]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        # Include chunk_id first so post-rebuild gold remaps (expected_chunk_ids)
        # score correctly; also keep section keys for section-number gold.
        primary = (chunk.chunk_id or chunk.section_id or chunk.section_number or "").strip()
        if primary.startswith("expanded::"):
            primary = primary[len("expanded::") :]
        candidates = [
            primary,
            (chunk.chunk_id or "").strip(),
            (chunk.section_id or "").strip(),
            (chunk.section_number or "").strip(),
        ]
        for c in candidates:
            c = c.rstrip(".")
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def is_hit(chunk: RetrievedChunk, gold: set[str]) -> bool:
    return bool(chunk_match_keys(chunk) & gold)
