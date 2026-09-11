"""Evidence-sufficiency gate and bounded corrective retrieval (CLAUDE.md §8, §12).

Deterministic decisions, made before any LLM call:

    requested regulation absent from evidence  → ABSTAIN
    as-of date with no valid version           → ABSTAIN
    ambiguous query (no scope, no content words)→ ABSTAIN
    no evidence                                → ABSTAIN
    weak evidence + retry budget               → one deterministic rewrite, retrieve again
    conflicting versions in evidence           → proceed, with an explicit warning
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.domain.temporal import QueryScope
from safety_assistant.persistence.models import Regulation, RegulationVersion
from safety_assistant.retrieval.context import Evidence
from safety_assistant.retrieval.filters import significant_tokens

MIN_STRONG_EVIDENCE = 1
WEAK_QUERY_TERMS = 1

# Domain acronyms the lexical leg cannot bridge on its own. Expansion is a
# deterministic rewrite used only for the single corrective retry.
ACRONYMS: dict[str, str] = {
    "hpc": "head performance criterion",
    "hic": "head injury criterion",
    "thcc": "thorax compression criterion",
    "rdc": "rib deflection criterion",
    "apf": "abdominal peak force",
    "pspf": "pubic symphysis peak force",
    "tcfc": "tibia compression force criterion",
    "ffc": "femur force criterion",
    "ti": "tibia index",
    "vc": "viscous criterion",
    "v*c": "viscous criterion",
    "crs": "child restraint system",
    "ecrs": "enhanced child restraint system",
    "mdb": "mobile deformable barrier",
    "odb": "offset deformable barrier",
    "reess": "rechargeable electrical energy storage system",
    "sbr": "safety-belt reminder",
}


@dataclass
class GateDecision:
    proceed: bool
    abstain_reason: str | None = None
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    rewrite: str | None = None  # populated when a corrective retry is recommended


def rewrite_query(query: str) -> str | None:
    """Expand known acronyms; returns None when nothing changes."""
    out = query
    for short, long in ACRONYMS.items():
        pattern = re.compile(rf"(?<![\w*]){re.escape(short)}(?![\w*])", re.IGNORECASE)
        if pattern.search(out) and long not in out.lower():
            out = pattern.sub(f"{short.upper()} ({long})", out)
    return out if out != query else None


def _versions_valid_on(session: Session, regulation_keys: tuple[str, ...], day: datetime.date) -> int:
    stmt = (
        select(RegulationVersion.id)
        .join(Regulation, Regulation.id == RegulationVersion.regulation_id)
        .where(RegulationVersion.status.in_(["ACTIVE", "SUPERSEDED"]))
    )
    if regulation_keys:
        stmt = stmt.where(Regulation.regulation_key.in_(regulation_keys))
    rows = session.execute(stmt.add_columns(RegulationVersion.valid_from, RegulationVersion.valid_to)).all()
    return sum(1 for _, vf, vt in rows if (vf is None or vf <= day) and (vt is None or vt > day))


def evaluate_gate(
    session: Session,
    query: str,
    qs: QueryScope,
    evidence: list[Evidence],
    *,
    as_of: datetime.date | None,
    retries_left: int,
) -> GateDecision:
    terms = significant_tokens(query)
    if not qs.regulation_keys and not qs.has_exact_identifier and len(terms) <= WEAK_QUERY_TERMS:
        return GateDecision(
            False,
            "ambiguous_query",
            "The question does not name a regulation, criterion or topic. Specify the limit or clause you mean.",
        )

    if as_of is not None and qs.regulation_keys:
        if _versions_valid_on(session, tuple(qs.regulation_keys), as_of) == 0:
            return GateDecision(
                False,
                "no_version_valid_on_date",
                f"No ingested version of {', '.join(qs.regulation_keys)} was in force on {as_of.isoformat()}; "
                "the corpus only holds later consolidated texts, so the historical requirement cannot be stated.",
            )

    if qs.regulation_keys:
        known = set(session.scalars(select(Regulation.regulation_key)).all())
        unknown = [k for k in qs.regulation_keys if k not in known]
        if unknown and len(unknown) == len(qs.regulation_keys):
            return GateDecision(
                False,
                "requested_regulation_not_in_evidence",
                f"{', '.join(unknown)} is not an ingested source in this corpus; no answer can be given for it.",
            )
        present = {e.regulation_key for e in evidence}
        missing = [k for k in qs.regulation_keys if k not in present]
        if missing and len(missing) == len(qs.regulation_keys):
            if retries_left > 0 and (rw := rewrite_query(query)):
                return GateDecision(True, rewrite=rw, warnings=["corrective retry: acronym expansion"])
            return GateDecision(
                False,
                "requested_regulation_not_in_evidence",
                f"No evidence was retrieved from {', '.join(missing)}. It is either not in the corpus or "
                "does not cover this topic.",
            )

    if not evidence:
        if retries_left > 0 and (rw := rewrite_query(query)):
            return GateDecision(True, rewrite=rw, warnings=["corrective retry: acronym expansion"])
        return GateDecision(False, "no_evidence", "No relevant evidence was found in the active corpus.")

    warnings: list[str] = []
    strong = [
        e
        for e in evidence
        if (e.ranks.sparse and e.ranks.sparse <= 5) or (e.ranks.dense and e.ranks.dense <= 5) or e.ranks.exact
    ]
    if len(strong) < MIN_STRONG_EVIDENCE:
        if retries_left > 0 and (rw := rewrite_query(query)):
            return GateDecision(True, rewrite=rw, warnings=["corrective retry: weak evidence, acronym expansion"])
        warnings.append("weak evidence: no candidate ranked in the top 5 of any retrieval leg")

    by_reg: dict[str, set[str]] = {}
    for e in evidence:
        by_reg.setdefault(e.regulation_key, set()).add(e.version_label)
    for reg, versions in by_reg.items():
        if len(versions) > 1:
            warnings.append(
                f"evidence spans several versions of {reg}: {', '.join(sorted(versions))} — check validity dates"
            )
    return GateDecision(True, warnings=warnings)
