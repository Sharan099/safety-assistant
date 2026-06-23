"""
UN R14 / UN R16 anchorage clause applicability enrichment.

Prepends structured APPLICABILITY headers to chunk text and emits metadata for
retrieval soft-boost / hard-filter (applies_to_category, anchorage_test_type).
"""

from __future__ import annotations

import re
from typing import Any

LOAD_RE = re.compile(
    r"\b\d{1,4}(?:[.,]\d+)?\s*(?:±\s*\d+)?\s*daN\b",
    re.I,
)
DURATION_RE = re.compile(
    r"(?:not less than|for at least|sustained for|withstand).{0,40}0\.2\s*second",
    re.I,
)
M1_N1_RE = re.compile(r"\bcategories\s+M1\s+and\s+N1\b", re.I)
NOT_M1_N1_RE = re.compile(r"\bcategories\s+other\s+than\s+M1\s+and\s+N1\b", re.I)
M3_N3_RE = re.compile(r"\bfor\s+M3\s+and\s+N3\b", re.I)
M2_N2_RE = re.compile(r"\bcategories\s+M2\s+and\s+N2\b", re.I)
TEST_LOAD_RE = re.compile(r"\btest\s+load\b|\btractive\s+force\b", re.I)

R14_DURATION_SNIPPET = (
    "Duration requirement (§6.3.3): belt anchorages must withstand the specified "
    "load for not less than 0.2 second."
)

ANCHORAGE_TEST_LABELS: dict[str, str] = {
    "three_point_retractor_pulley_upper": "Three-point belt, retractor with upper pulley/strap guide (§6.4.1)",
    "upper_torso_strap_anchorage": "Upper torso strap anchorage traction test (§6.4.1.2)",
    "lower_anchorage_traction": "Lower anchorage simultaneous traction test (§6.4.1.3)",
    "three_point_no_retractor_upper": "Three-point belt without retractor at upper anchorage (§6.4.2)",
    "upper_torso_no_retractor": "Upper torso strap load — no retractor at upper (§6.4.2.1)",
    "lower_anchorage_no_retractor": "Lower anchorage traction — no retractor at upper (§6.4.2.2)",
    "lap_belt_lower": "Lap belt lower anchorage test (§6.4.3)",
    "seat_integrated_anchorage": "Belt anchorages within/dispersed in seat structure (§6.4.4)",
    "special_type_belt_upper": "Special-type belt upper torso strap test (§6.4.5.1)",
    "special_type_belt_lower": "Special-type belt lower anchorage test (§6.4.5.2)",
    "special_type_belt_reduced_load": "Special-type belt — categories other than M1/N1 (§6.4.5.3)",
    "rearward_facing": "Rearward-facing seat anchorage test (§6.4.6)",
    "side_facing": "Side-facing seat anchorage test (§6.4.7)",
    "general_anchorage_requirements": "General anchorage strength requirements (§5)",
    "general_test_requirements": "General anchorage test requirements (§6.3)",
    "duration_hold_time": "Load application duration / hold time (§6.3.3)",
}


def _clause_prefix(clause: str | None) -> str:
    if not clause:
        return ""
    return clause.strip().rstrip(".")


def is_anchorage_clause_family(clause_number: str | None, regulation: str) -> bool:
    if regulation not in ("UN_R14", "UN_R16"):
        return False
    c = _clause_prefix(clause_number)
    if not c:
        return False
    if c.startswith("5."):
        return True
    if c.startswith("6.3.") or c.startswith("6.4"):
        return True
    return False


def resolve_anchorage_test_type(clause_number: str | None, section_title: str) -> str | None:
    c = _clause_prefix(clause_number)
    title = (section_title or "").lower()
    spaced = re.search(r"(\d+(?:\.\d+)+)\s+(\d+)", section_title or "")
    if spaced and not c.endswith(f".{spaced.group(2)}"):
        c = f"{spaced.group(1).rstrip('.')}.{spaced.group(2)}"
    if not c:
        return None
    mapping = {
        "6.4.1.1": "three_point_retractor_pulley_upper",
        "6.4.1.2": "upper_torso_strap_anchorage",
        "6.4.1.3": "lower_anchorage_traction",
        "6.4.2.1": "upper_torso_no_retractor",
        "6.4.2.2": "lower_anchorage_no_retractor",
        "6.4.3": "lap_belt_lower",
        "6.4.4": "seat_integrated_anchorage",
        "6.4.5.1": "special_type_belt_upper",
        "6.4.5.2": "special_type_belt_lower",
        "6.4.5.3": "special_type_belt_reduced_load",
        "6.4.6": "rearward_facing",
        "6.4.7": "side_facing",
        "6.3.3": "duration_hold_time",
    }
    if c in mapping:
        return mapping[c]
    if c.startswith("6.4.1"):
        return "three_point_retractor_pulley_upper"
    if c.startswith("6.4.2"):
        return "three_point_no_retractor_upper"
    if c.startswith("6.4.5"):
        return "special_type_belt_upper"
    if c.startswith("6.4."):
        return "general_test_requirements"
    if c.startswith("6.3"):
        return "general_test_requirements"
    if c.startswith("5."):
        return "general_anchorage_requirements"
    if "lap belt" in title:
        return "lap_belt_lower"
    if "upper torso" in title or "upper belt" in title:
        return "upper_torso_strap_anchorage"
    if "lower belt" in title or "lower anchorage" in title:
        return "lower_anchorage_traction"
    return None


def _category_tokens_from_text(text: str) -> list[str]:
    """Structured category tokens for retrieval boosting."""
    tokens: list[str] = []
    if M1_N1_RE.search(text) and not NOT_M1_N1_RE.search(text):
        tokens.append("M1_N1")
    if NOT_M1_N1_RE.search(text):
        tokens.append("NOT_M1_N1")
    if M3_N3_RE.search(text):
        tokens.append("M3_N3")
    if M2_N2_RE.search(text):
        tokens.append("M2_N2")
    if not tokens and re.search(r"\bM1\b", text):
        tokens.append("M1_N1")
    return list(dict.fromkeys(tokens))


def _category_display_from_text(text: str, clause_number: str | None) -> str:
    """Human-readable applicability line for the APPLICABILITY header."""
    parts: list[str] = []
    if M1_N1_RE.search(text) and not NOT_M1_N1_RE.search(text):
        parts.append("M1 and N1 vehicles")
    if NOT_M1_N1_RE.search(text):
        parts.append("all categories except M1 and N1")
    if M3_N3_RE.search(text):
        parts.append("M3 and N3 vehicles (sub-case of non-M1/N1 loads)")
    if M2_N2_RE.search(text):
        parts.append("M2 and N2 vehicles")

    loads: list[str] = []
    if re.search(r"1[,.]?350\s*(?:±\s*20)?\s*daN", text, re.I):
        loads.append("M1/N1 default load 1,350±20 daN")
    if re.search(r"675\s*(?:±\s*20)?\s*daN", text, re.I):
        loads.append("non-M1/N1 default 675±20 daN")
    if re.search(r"450\s*(?:±\s*20)?\s*daN", text, re.I):
        loads.append("M3/N3 load 450±20 daN")
    if re.search(r"740\s*(?:±\s*20)?\s*daN", text, re.I):
        loads.append("M3/N3 lap-belt load 740±20 daN")
    if re.search(r"2[,.]?225\s*(?:±\s*20)?\s*daN", text, re.I):
        loads.append("M1/N1 lap-belt load 2,225±20 daN")
    if re.search(r"1[,.]?110\s*(?:±\s*20)?\s*daN", text, re.I):
        loads.append("non-M1/N1 lap-belt load 1,110±20 daN")

    if parts and loads:
        return (
            f"{'; '.join(parts)} — "
            + "; ".join(loads)
            + f" (clause {clause_number or 'n/a'})"
        )
    if parts:
        return "; ".join(parts) + (f" (clause {clause_number})" if clause_number else "")
    if loads:
        return "; ".join(loads) + (f" (clause {clause_number})" if clause_number else "")
    if clause_number and clause_number.startswith("6.3.3"):
        return "All seat-belt anchorage strength tests in §6.4 (duration/hold-time rule)"
    return "All vehicle categories (see clause text for conditions)"


def extract_r14_duration_snippet(sections: list[tuple]) -> str | None:
    """Pull §6.3.3 hold-time sentence from parsed sections."""
    for _path, _title, _level, body, clause_number, _src in sections:
        c = _clause_prefix(clause_number)
        if c == "6.3.3" or c.startswith("6.3.3"):
            if DURATION_RE.search(body) or "0.2 second" in body.lower():
                return R14_DURATION_SNIPPET
    return R14_DURATION_SNIPPET


def bond_load_duration_sentences(sentences: list[str]) -> list[str]:
    """Keep load sentences paired with adjacent duration/hold-time sentences."""
    if not sentences:
        return sentences
    out: list[str] = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        if i + 1 < len(sentences):
            nxt = sentences[i + 1]
            load_here = bool(TEST_LOAD_RE.search(sent) or LOAD_RE.search(sent))
            duration_next = bool(DURATION_RE.search(nxt) or "0.2 second" in nxt.lower())
            if load_here and duration_next:
                out.append(f"{sent} {nxt}")
                i += 2
                continue
        out.append(sent)
        i += 1
    return out


def build_applicability_meta(
    *,
    regulation: str,
    clause_number: str | None,
    section_title: str,
    body: str,
    duration_snippet: str | None,
) -> dict[str, Any]:
    if not is_anchorage_clause_family(clause_number, regulation):
        return {}

    test_type = resolve_anchorage_test_type(clause_number, section_title)
    categories = _category_tokens_from_text(body)
    display = _category_display_from_text(body, clause_number)

    related: list[str] = []
    c = _clause_prefix(clause_number)
    if c.startswith("6.4") and regulation == "UN_R14":
        related.append("6.3.3")
    if c == "6.3.3":
        related.append("6.4")

    meta: dict[str, Any] = {
        "applies_to_category": categories or None,
        "anchorage_test_type": test_type,
        "anchorage_test_label": ANCHORAGE_TEST_LABELS.get(test_type or "", test_type),
        "applicability_display": display,
    }
    if related:
        meta["related_clause_ids"] = related
    if duration_snippet and LOAD_RE.search(body) and not DURATION_RE.search(body):
        meta["has_duration_link"] = True
    if DURATION_RE.search(body) or "0.2 second" in body.lower():
        meta["has_duration_requirement"] = True
    return meta


def format_applicability_header(meta: dict[str, Any], *, duration_snippet: str | None) -> str:
    lines = ["APPLICABILITY:"]
    if meta.get("applicability_display"):
        lines.append(f"  Applies to: {meta['applicability_display']}")
    label = meta.get("anchorage_test_label") or meta.get("anchorage_test_type")
    if label:
        lines.append(f"  Test type: {label}")
    if duration_snippet and meta.get("has_duration_link"):
        lines.append(f"  {duration_snippet}")
    elif meta.get("has_duration_requirement"):
        lines.append("  Duration: withstand specified load for not less than 0.2 second (§6.3.3)")
    return "\n".join(lines)


def enrich_section_body(
    *,
    regulation: str,
    clause_number: str | None,
    section_title: str,
    body: str,
    duration_snippet: str | None,
) -> tuple[str, dict[str, Any]]:
    """Return (enriched_body, applicability_metadata)."""
    meta = build_applicability_meta(
        regulation=regulation,
        clause_number=clause_number,
        section_title=section_title,
        body=body,
        duration_snippet=duration_snippet,
    )
    if not meta:
        return body, {}

    enriched = body
    if (
        duration_snippet
        and LOAD_RE.search(body)
        and not DURATION_RE.search(body)
        and _clause_prefix(clause_number).startswith("6.4")
    ):
        enriched = enriched.rstrip() + "\n\n" + duration_snippet

    header = format_applicability_header(meta, duration_snippet=duration_snippet)
    return f"{header}\n---\n{enriched}", meta


def prepend_applicability_to_chunk_text(text: str, meta: dict[str, Any], duration_snippet: str | None) -> str:
    """Prepend header to leaf chunk if not already present."""
    if not meta or "APPLICABILITY:" in text[:200]:
        return text
    header = format_applicability_header(meta, duration_snippet=duration_snippet)
    return f"{header}\n---\n{text}"
