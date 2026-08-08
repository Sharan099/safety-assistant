"""Detect false premises about regulation scope before retrieval/answering."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.sources import SOURCES

_REG_CODE_RE = re.compile(r"\b(?:UN[\s_-]?)?(R\d{2,3})\b", re.I)

_REG_SCOPE: dict[str, dict[str, list[str]]] = {
    "UN_R94": {
        "signals": ["frontal", "front impact", "head-on", "hic", "thcc"],
        "conflicts": ["side impact", "side collision", "lateral impact", "child restraint", "i-size", "isofix"],
    },
    "UN_R95": {
        "signals": ["side impact", "side collision", "lateral", "barrier intrusion", "chest"],
        "conflicts": ["frontal", "front impact", "head-on", "child restraint", "i-size", "seat belt anchorage"],
    },
    "UN_R16": {
        "signals": ["seat belt", "safety belt", "belt", "retractor", "anchorage", "webbing"],
        "conflicts": ["frontal impact test", "side barrier", "child restraint system", "i-size"],
    },
    "UN_R129": {
        "signals": ["child restraint", "i-size", "isofix", "height class", "q series"],
        "conflicts": ["frontal impact regulation", "side barrier regulation", "seat belt anchorage strength"],
    },
}

_TITLE_BY_CODE = {s["regulation_code"]: s["title"] for s in SOURCES}
_TOPIC_BY_CODE = {s["regulation_code"]: s["topic"] for s in SOURCES}


@dataclass
class PremiseIssue:
    regulation_code: str
    assumed_topic: str
    actual_topic: str
    message: str


def _normalize_reg(match: re.Match[str]) -> str:
    return f"UN_{match.group(1).upper()}"


def _mentioned_regs(query: str) -> list[str]:
    regs = {_normalize_reg(m) for m in _REG_CODE_RE.finditer(query)}
    low = query.lower()
    for num in ("94", "95", "16", "129"):
        if re.search(rf"\bregulation\s+(?:no\.?\s*)?{num}\b", low):
            regs.add(f"UN_R{num}")
    return sorted(regs)


def _query_mentions_any(query: str, phrases: list[str]) -> list[str]:
    low = query.lower()
    return [p for p in phrases if p in low]


def _r95_misframed_as_frontal(query: str) -> bool:
    """True when the question attributes frontal collision limits to UN R95.

    Multi-regulation questions that pair R94 with frontal and R95 with lateral
    (e.g. eval_10) must not trigger — bare 'frontal' in 'frontal protection'
    near R94 is not an R95 scope error.
    """
    low = query.lower()
    frontal_collision_phrases = (
        "frontal collision",
        "front impact",
        "head-on",
        "frontal hic",
        "frontal injury",
    )
    if not any(p in low for p in frontal_collision_phrases):
        return False
    # Correctly scoped multi-reg: R95 explicitly paired with lateral/side.
    if re.search(
        r"(?:regulation\s*(?:no\.?\s*)?95|r\s*95)[^,.]{0,80}"
        r"(?:lateral|side\s+impact|side\s+protection|side\s+collision)",
        low,
    ):
        return False
    if re.search(
        r"(?:lateral|side\s+impact|side\s+protection|side\s+collision)[^,.]{0,80}"
        r"(?:regulation\s*(?:no\.?\s*)?95|r\s*95)",
        low,
    ):
        return False
    return True


def check_premises(query: str) -> list[PremiseIssue]:
    """Flag contradictions between the question framing and known regulation scope."""
    issues: list[PremiseIssue] = []
    for reg in _mentioned_regs(query):
        scope = _REG_SCOPE.get(reg)
        if not scope:
            continue
        conflict_hits = _query_mentions_any(query, scope["conflicts"])
        signal_hits = _query_mentions_any(query, scope["signals"])
        if conflict_hits and not signal_hits:
            issues.append(
                PremiseIssue(
                    regulation_code=reg,
                    assumed_topic=conflict_hits[0],
                    actual_topic=_TOPIC_BY_CODE.get(reg, reg),
                    message=(
                        f"The question frames {reg} ({_TITLE_BY_CODE.get(reg, reg)}) "
                        f"around '{conflict_hits[0]}', but this corpus indexes it as: "
                        f"{_TOPIC_BY_CODE.get(reg, reg)}. "
                        f"Correct that assumption in your answer and cite only retrieved excerpts."
                    ),
                )
            )
            continue
        if reg == "UN_R95" and _r95_misframed_as_frontal(query):
            issues.append(
                PremiseIssue(
                    regulation_code=reg,
                    assumed_topic="frontal collision",
                    actual_topic=_TOPIC_BY_CODE.get(reg, reg),
                    message=(
                        f"The question asks for frontal collision injury limits under {reg}, "
                        f"but {reg} governs lateral/side impact protection. "
                        "Correct that scope error before discussing side-impact requirements."
                    ),
                )
            )
    return issues


def format_premise_notes(issues: list[PremiseIssue]) -> str:
    if not issues:
        return ""
    lines = ["Premise check (correct any false assumptions in the question):"]
    for issue in issues:
        lines.append(f"- {issue.message}")
    return "\n".join(lines)
