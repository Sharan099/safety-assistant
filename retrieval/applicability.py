"""APPLICABILITY pipeline — reason over EACH indexed regulation's Scope clause.

Fixes \"Which regulations apply to <vehicle>?\" stopping at a single hybrid hit
(e.g. BMW X3 EV → only R95 definitions):

1. Retrieve the Scope/applicability clause of EVERY indexed regulation.
2. Parse the vehicle description (category, mass, powertrain, seating).
3. Reason clause-by-clause → APPLIES / DOES_NOT_APPLY / CANNOT_DETERMINE per
   regulation, citing that regulation's scope chunk — never a bare single-reg answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

VERDICT_APPLIES = "APPLIES"
VERDICT_DOES_NOT = "DOES_NOT_APPLY"
VERDICT_UNKNOWN = "CANNOT_DETERMINE"
_VALID_VERDICTS = frozenset({VERDICT_APPLIES, VERDICT_DOES_NOT, VERDICT_UNKNOWN})

APPLICABILITY_SYSTEM_PROMPT = """\
You are a UNECE passive-safety regulation assistant deciding which INDEXED
regulations apply to a described vehicle/use-case.

Reply with a single JSON object:
{
  "answer_segments": [
    {
      "text": "<one short WHY clause referring to the scope text; no §/page chips>",
      "citation_chunk_id": "<exact chunk_id of THAT regulation's scope passage>",
      "category_id": "<regulation_id e.g. UN-ECE-R94>",
      "claim_kind": "APPLIES" | "DOES_NOT_APPLY" | "CANNOT_DETERMINE"
    }
  ]
}

STRICT RULES:
1. Emit EXACTLY ONE segment per indexed regulation listed in the user message.
2. claim_kind MUST be one of APPLIES / DOES_NOT_APPLY / CANNOT_DETERMINE.
3. Cite ONLY that regulation's provided Scope passage (citation_chunk_id).
4. Reason from the Scope text vs the vehicle description (category, mass,
   powertrain, seating). Do not invent mass/category limits absent from the
   passage — use CANNOT_DETERMINE when the indexed Scope is silent or incomplete.
5. Occupant-protection for an M1 / passenger EV typically implicates BOTH frontal
   (R94) AND side (R95) impact regs when their Scopes cover that vehicle class —
   do not stop at the first matching regulation.
6. Do not write section/page numbers or citation chips in "text".
"""

APPLICABILITY_USER_INSTRUCTION = """\
APPLICABILITY — one verdict per indexed regulation:
- APPLIES: Scope text covers this vehicle class / use-case.
- DOES_NOT_APPLY: Scope text clearly excludes this vehicle.
- CANNOT_DETERMINE: indexed Scope is missing, silent, or insufficient —
  say the engineer should check the full regulation.
Never answer with only a single regulation when multiple are indexed.
"""


@dataclass
class VehicleProfile:
    """Parsed vehicle cues from the user question (best-effort)."""

    raw: str
    categories: list[str] = field(default_factory=list)  # M1, N1, …
    mass_kg: float | None = None
    powertrain: str | None = None  # ev | hybrid | ice
    seating_hint: str | None = None
    model_hints: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "categories": list(self.categories),
            "mass_kg": self.mass_kg,
            "powertrain": self.powertrain,
            "seating_hint": self.seating_hint,
            "model_hints": list(self.model_hints),
        }


@dataclass
class ApplicabilityExpansion:
    question: str
    vehicle: VehicleProfile
    indexed_regulation_ids: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "vehicle": self.vehicle.to_public_dict(),
            "indexed_regulation_ids": list(self.indexed_regulation_ids),
        }


@dataclass
class RegApplicability:
    regulation_id: str
    verdict: str
    reason: str
    scope_chunk: Any | None = None
    heuristic: bool = True


@dataclass
class ApplicabilityRetrievalResult:
    chunks: list[Any]
    expansion: ApplicabilityExpansion
    per_regulation: dict[str, Any] = field(default_factory=dict)  # rid → scope chunk
    heuristics: list[RegApplicability] = field(default_factory=list)
    missing_scope: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "vehicle": self.expansion.vehicle.to_public_dict(),
            "indexed": list(self.expansion.indexed_regulation_ids),
            "scope_regs": sorted(self.per_regulation.keys()),
            "missing_scope": list(self.missing_scope),
            "heuristics": [
                {
                    "regulation_id": h.regulation_id,
                    "verdict": h.verdict,
                    "reason": h.reason,
                    "chunk_id": getattr(h.scope_chunk, "chunk_id", None),
                }
                for h in self.heuristics
            ],
            "n_chunks": len(self.chunks),
        }


_MASS_RE = re.compile(
    r"(?ix)(?:gvm|gvwr|mass|weight|kg)\s*(?:of\s+|not\s+exceeding\s+|≤\s*|<=\s*|under\s+)?"
    r"(\d[\d,\.]*)\s*(?:kg)?"
)
_CATEGORY_RE = re.compile(r"(?i)\b(M1|M2|M3|N1|N2|N3)\b")
_EV_RE = re.compile(
    r"(?ix)\b(ev|bev|electric(?:\s+vehicle)?|battery[- ]electric|reess|electrified)\b"
)
_HYBRID_RE = re.compile(r"(?ix)\b(hybrid|phev|hev)\b")
_ICE_RE = re.compile(r"(?ix)\b(ice|combustion|petrol|gasoline|diesel)\b")
_PASSENGER_RE = re.compile(
    r"(?ix)\b(passenger\s+car|suv|saloon|sedan|x[0-9]|3[ -]?series|occupant\s+protection)\b"
)
_CRS_RE = re.compile(r"(?ix)\b(child\s+restraint|crs|isofix|booster|infant)\b")


def parse_vehicle_profile(question: str) -> VehicleProfile:
    q = (question or "").strip()
    cats = [m.group(1).upper() for m in _CATEGORY_RE.finditer(q)]
    # Passenger-car / SUV / X3 → treat as M1 unless another category is named.
    if not cats and _PASSENGER_RE.search(q):
        cats = ["M1"]
    mass = None
    m = _MASS_RE.search(q)
    if m:
        try:
            mass = float(m.group(1).replace(",", ""))
        except ValueError:
            mass = None
    powertrain = None
    if _EV_RE.search(q):
        powertrain = "ev"
    elif _HYBRID_RE.search(q):
        powertrain = "hybrid"
    elif _ICE_RE.search(q):
        powertrain = "ice"
    seating = "crs" if _CRS_RE.search(q) else None
    models = []
    for pat in (r"(?i)\bBMW\s+X\d+\b", r"(?i)\bX\d+\s+EV\b", r"(?i)\bBMW\b"):
        mm = re.search(pat, q)
        if mm:
            models.append(mm.group(0))
    return VehicleProfile(
        raw=q,
        categories=list(dict.fromkeys(cats)),
        mass_kg=mass,
        powertrain=powertrain,
        seating_hint=seating,
        model_hints=models,
    )


def expand_applicability_query(question: str) -> ApplicabilityExpansion:
    from retrieval.retrieve import get_indexed_regulations

    vehicle = parse_vehicle_profile(question)
    indexed = sorted(
        {
            r.regulation_id
            for r in get_indexed_regulations()
            if (r.regulation_id or "").strip()
        }
    )
    return ApplicabilityExpansion(
        question=(question or "").strip(),
        vehicle=vehicle,
        indexed_regulation_ids=indexed,
    )


def _scope_mentions_category(scope_text: str, categories: Sequence[str]) -> bool | None:
    """True if scope clearly includes a category; False if excludes; None if silent."""
    blob = (scope_text or "").lower()
    if not blob:
        return None
    if not categories:
        # Generic passenger / vehicles language.
        if re.search(r"\b(vehicles? of category|category m1|m1 vehicles?)\b", blob):
            return True
        if re.search(r"\bapplies to vehicles\b", blob):
            return True
        return None
    hit = False
    for cat in categories:
        c = cat.lower()
        if re.search(rf"\bcategory\s+{re.escape(c)}\b", blob) or re.search(
            rf"\b{re.escape(c)}\b", blob
        ):
            hit = True
    if hit:
        return True
    # Explicit other categories only.
    if re.search(r"\bcategory\s+[mn]\d\b", blob) and not hit:
        return False
    return None


def _heuristic_for_reg(
    regulation_id: str,
    scope_text: str,
    vehicle: VehicleProfile,
    *,
    has_scope: bool,
) -> RegApplicability:
    rid = regulation_id
    if not has_scope:
        return RegApplicability(
            regulation_id=rid,
            verdict=VERDICT_UNKNOWN,
            reason=(
                "No Scope/applicability clause was retrieved for this indexed "
                "regulation — check the full regulation text."
            ),
            scope_chunk=None,
        )

    blob = scope_text or ""
    cats = vehicle.categories or ["M1"]  # default passenger assumption for SUV/X3
    cat_hit = _scope_mentions_category(blob, cats)

    # Child-restraint regulation: only applies when CRS is in scope of the ask.
    if rid.endswith("-R129") or "R129" in rid:
        if vehicle.seating_hint == "crs" or _CRS_RE.search(vehicle.raw or ""):
            return RegApplicability(
                regulation_id=rid,
                verdict=VERDICT_APPLIES if cat_hit is not False else VERDICT_UNKNOWN,
                reason=(
                    "Child restraint systems are in scope of the question; "
                    "confirm vehicle-category fit against the R129 Scope clause."
                ),
            )
        return RegApplicability(
            regulation_id=rid,
            verdict=VERDICT_DOES_NOT,
            reason=(
                "UN R129 Scope addresses child restraint systems (CRS); the "
                "question concerns general vehicle occupant protection without "
                "a CRS ask."
            ),
        )

    # Frontal / side impact / belts — passenger EV / M1.
    impact_or_belt = any(
        rid.endswith(suf) for suf in ("-R94", "-R95", "-R16")
    ) or any(x in rid for x in ("R94", "R95", "R16"))

    if impact_or_belt:
        # Mass gate when Scope states a kg limit and user gave mass.
        mass_ok = True
        if vehicle.mass_kg is not None:
            lim = re.search(
                r"(?i)(?:not exceeding|≤|<=|maximum mass.*?)\s*([0-9][0-9,\.]*)\s*kg",
                blob,
            )
            if lim:
                try:
                    limit = float(lim.group(1).replace(",", ""))
                    mass_ok = vehicle.mass_kg <= limit + 1e-6
                except ValueError:
                    mass_ok = True

        if cat_hit is False:
            return RegApplicability(
                regulation_id=rid,
                verdict=VERDICT_DOES_NOT,
                reason=(
                    "Indexed Scope categories do not include the described "
                    "vehicle category."
                ),
            )
        if cat_hit is True and mass_ok:
            why = (
                f"Indexed Scope covers category {', '.join(cats)}; "
                "vehicle description is consistent with that class"
            )
            if vehicle.powertrain == "ev":
                why += (
                    " (electric powertrain does not remove frontal/side/"
                    "belt applicability under this Scope)"
                )
            why += "."
            return RegApplicability(
                regulation_id=rid,
                verdict=VERDICT_APPLIES,
                reason=why,
            )
        if cat_hit is True and not mass_ok:
            return RegApplicability(
                regulation_id=rid,
                verdict=VERDICT_DOES_NOT,
                reason=(
                    "Described mass exceeds the mass band stated in the indexed "
                    "Scope clause."
                ),
            )
        # Silent category — for passenger-car / X3 / occupant-protection asks,
        # still mark APPLIES for R94/R95 when Scope discusses vehicles generally.
        if _PASSENGER_RE.search(vehicle.raw or "") or vehicle.categories:
            if re.search(
                r"(?i)vehicle|occupant|impact|collision|safety-belt|restraint",
                blob,
            ):
                return RegApplicability(
                    regulation_id=rid,
                    verdict=VERDICT_APPLIES,
                    reason=(
                        "Vehicle described as a passenger / occupant-protection "
                        "use-case; indexed Scope addresses vehicles in that domain. "
                        "Confirm category/mass details in the full regulation if needed."
                    ),
                )
        return RegApplicability(
            regulation_id=rid,
            verdict=VERDICT_UNKNOWN,
            reason=(
                "Indexed Scope does not clearly confirm or exclude this vehicle "
                "description — check the full regulation."
            ),
        )

    return RegApplicability(
        regulation_id=rid,
        verdict=VERDICT_UNKNOWN,
        reason=(
            "Indexed Scope retrieved, but applicability to this vehicle is not "
            "clear from the clause alone — check the full regulation."
        ),
    )


def retrieve_applicability(
    query: str,
    *,
    client: object | None = None,
    collection: str | None = None,
    expansion: ApplicabilityExpansion | None = None,
) -> ApplicabilityRetrievalResult:
    """Fetch Scope clause for each indexed regulation (deterministic, not hybrid)."""
    from retrieval.retrieve import fetch_scope_chunks, get_indexed_regulations

    expansion = expansion or expand_applicability_query(query)
    if not expansion.indexed_regulation_ids:
        # Refresh from live index.
        expansion.indexed_regulation_ids = sorted(
            {
                r.regulation_id
                for r in get_indexed_regulations(
                    client=client,  # type: ignore[arg-type]
                    collection=collection,
                )
                if (r.regulation_id or "").strip()
            }
        )

    per_reg: dict[str, Any] = {}
    missing: list[str] = []
    chunks: list[Any] = []
    for rid in expansion.indexed_regulation_ids:
        scope_hits = fetch_scope_chunks(
            regulation_id=rid,
            client=client,  # type: ignore[arg-type]
            collection=collection,
        )
        # Keep only this regulation (belt-and-suspenders).
        scope_hits = [
            c
            for c in scope_hits
            if (getattr(c, "regulation_id", None) or "").strip() == rid
        ]
        if not scope_hits:
            missing.append(rid)
            logger.warning("applicability: no scope chunk for %s", rid)
            continue
        # Prefer true Scope article (section 1) over accidental defs.
        preferred = [
            c
            for c in scope_hits
            if (getattr(c, "section_number", None) or "").strip() in {"1", "1."}
        ]
        chosen = (preferred or scope_hits)[0]
        per_reg[rid] = chosen
        chunks.append(chosen)

    heuristics: list[RegApplicability] = []
    for rid in expansion.indexed_regulation_ids:
        chunk = per_reg.get(rid)
        text = getattr(chunk, "text", "") if chunk is not None else ""
        h = _heuristic_for_reg(
            rid, text, expansion.vehicle, has_scope=chunk is not None
        )
        h.scope_chunk = chunk
        heuristics.append(h)

    result = ApplicabilityRetrievalResult(
        chunks=chunks,
        expansion=expansion,
        per_regulation=per_reg,
        heuristics=heuristics,
        missing_scope=missing,
    )
    logger.info(
        "applicability retrieved scopes=%s missing=%s vehicle=%s heuristics=%s",
        sorted(per_reg.keys()),
        missing,
        expansion.vehicle.to_public_dict(),
        [(h.regulation_id, h.verdict) for h in heuristics],
    )
    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.optimizations["applicability"] = True
            tr.optimizations["applicability_retrieval"] = result.to_public_dict()
    except Exception:  # noqa: BLE001
        pass
    return result


def format_applicability_context(result: ApplicabilityRetrievalResult) -> str:
    vehicle = result.expansion.vehicle
    parts = [
        f"Vehicle description (parsed): {json.dumps(vehicle.to_public_dict())}",
        f"Indexed regulations (MUST each receive a verdict): "
        + ", ".join(result.expansion.indexed_regulation_ids),
        "Heuristic prior (override only with clear Scope evidence):",
    ]
    for h in result.heuristics:
        parts.append(f"- {h.regulation_id}: {h.verdict} — {h.reason}")
    parts.append("")
    idx = 0
    for rid in result.expansion.indexed_regulation_ids:
        chunk = result.per_regulation.get(rid)
        parts.append(f"## Scope of `{rid}`")
        if chunk is None:
            parts.append("(No Scope clause retrieved for this regulation.)")
            continue
        idx += 1
        if hasattr(chunk, "context_block"):
            parts.append(
                f"category_id={rid}\n{chunk.context_block(index=idx)}"
            )
        else:
            parts.append(
                f"category_id={rid}\n[passage {idx}] chunk_id={chunk.chunk_id}\n"
                f"{getattr(chunk, 'text', '')}"
            )
    return "\n\n".join(parts)


def applicability_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "applicability_answer",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "answer_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "citation_chunk_id": {"type": "string"},
                                "category_id": {"type": "string"},
                                "claim_kind": {
                                    "type": "string",
                                    "enum": [
                                        VERDICT_APPLIES,
                                        VERDICT_DOES_NOT,
                                        VERDICT_UNKNOWN,
                                    ],
                                },
                            },
                            "required": [
                                "text",
                                "citation_chunk_id",
                                "category_id",
                                "claim_kind",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["answer_segments"],
                "additionalProperties": False,
            },
        },
    }


def normalize_verdict(raw: str | None) -> str:
    k = (raw or "").strip().upper().replace(" ", "_")
    if k in {"APPLY", "APPLICABLE", "YES", "COVERED"}:
        return VERDICT_APPLIES
    if k in {"DOES_NOT", "NOT_APPLICABLE", "NO", "EXCLUDED", "DOESNOTAPPLY"}:
        return VERDICT_DOES_NOT
    if k in _VALID_VERDICTS:
        return k
    if "NOT" in k and "APPLY" in k:
        return VERDICT_DOES_NOT
    if "CANNOT" in k or "UNKNOWN" in k or "INSUFFICIENT" in k:
        return VERDICT_UNKNOWN
    if "APPLY" in k:
        return VERDICT_APPLIES
    return VERDICT_UNKNOWN


def merge_llm_with_heuristics(
    segments: Sequence[Any],
    result: ApplicabilityRetrievalResult,
    allowed_ids: set[str],
) -> list[RegApplicability]:
    """One verdict per indexed reg: prefer grounded LLM, else heuristic."""
    by_reg: dict[str, RegApplicability] = {
        h.regulation_id: RegApplicability(
            regulation_id=h.regulation_id,
            verdict=h.verdict,
            reason=h.reason,
            scope_chunk=h.scope_chunk,
            heuristic=True,
        )
        for h in result.heuristics
    }
    for seg in segments:
        rid = (getattr(seg, "category_id", None) or "").strip()
        cid = (getattr(seg, "citation_chunk_id", None) or "").strip()
        if rid not in by_reg:
            continue
        chunk = result.per_regulation.get(rid)
        # Require cite to that reg's scope chunk when available.
        if chunk is not None:
            expect = (getattr(chunk, "chunk_id", None) or "").strip()
            if cid and cid != expect and cid not in allowed_ids:
                continue
            if cid and expect and cid != expect:
                # Wrong chunk — keep heuristic.
                continue
        verdict = normalize_verdict(
            getattr(seg, "claim_kind", None) or getattr(seg, "text", None)
        )
        why = (getattr(seg, "text", None) or "").strip() or by_reg[rid].reason
        by_reg[rid] = RegApplicability(
            regulation_id=rid,
            verdict=verdict,
            reason=why,
            scope_chunk=chunk if chunk is not None else by_reg[rid].scope_chunk,
            heuristic=False,
        )
    # Ensure every indexed reg is present.
    out: list[RegApplicability] = []
    for rid in result.expansion.indexed_regulation_ids:
        if rid in by_reg:
            out.append(by_reg[rid])
        else:
            out.append(
                RegApplicability(
                    regulation_id=rid,
                    verdict=VERDICT_UNKNOWN,
                    reason=(
                        "No applicability decision produced for this indexed "
                        "regulation — check the full regulation."
                    ),
                    scope_chunk=result.per_regulation.get(rid),
                )
            )
    return out


def render_applicability_answer(
    decisions: Sequence[RegApplicability],
    *,
    vehicle: VehicleProfile,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Structured multi-reg applicability board — never a single-reg dump."""
    from retrieval.multi_regulation import short_regulation_label

    lines = [
        "Applicability assessment",
        "",
        f"Vehicle: {vehicle.raw}",
    ]
    if vehicle.categories or vehicle.powertrain or vehicle.mass_kg is not None:
        bits = []
        if vehicle.categories:
            bits.append("categories=" + ",".join(vehicle.categories))
        if vehicle.powertrain:
            bits.append(f"powertrain={vehicle.powertrain}")
        if vehicle.mass_kg is not None:
            bits.append(f"mass_kg={vehicle.mass_kg:g}")
        lines.append("Parsed cues: " + "; ".join(bits))
    lines.append("")

    sources: list[Any] = []
    seen: set[str] = set()
    order = (VERDICT_APPLIES, VERDICT_DOES_NOT, VERDICT_UNKNOWN)
    by_v: dict[str, list[RegApplicability]] = {v: [] for v in order}
    for d in decisions:
        by_v.setdefault(normalize_verdict(d.verdict), []).append(d)

    headings = {
        VERDICT_APPLIES: "Applies",
        VERDICT_DOES_NOT: "Does not apply",
        VERDICT_UNKNOWN: (
            "Cannot determine from indexed scope — check the full regulation"
        ),
    }
    for verdict in order:
        rows = by_v.get(verdict) or []
        lines.append(f"## {headings[verdict]}")
        if not rows:
            lines.append("_None._")
            lines.append("")
            continue
        for d in rows:
            label = short_regulation_label(d.regulation_id)
            chip = ""
            chunk = d.scope_chunk
            if chunk is not None:
                chip = " " + chunk.citation_tag()
                cid = getattr(chunk, "chunk_id", "") or ""
                if cid and cid not in seen:
                    seen.add(cid)
                    sources.append(to_source(chunk))
            lines.append(f"- **{label}** — {d.reason.strip()}{chip}")
        lines.append("")

    # Hard guarantee: every decision row listed somewhere.
    if len(decisions) < 2:
        lines.append(
            "_Note: fewer than two indexed regulations were assessed; "
            "applicability answers must survey every indexed corpus._"
        )

    return "\n".join(lines).strip(), sources


def extractive_applicability_fallback(
    result: ApplicabilityRetrievalResult,
    *,
    to_source: Any,
) -> tuple[str, list[Any]]:
    """Deterministic multi-reg board from heuristics alone."""
    return render_applicability_answer(
        result.heuristics,
        vehicle=result.expansion.vehicle,
        to_source=to_source,
    )


def rebuild_applicability_result(
    question: str,
    chunks: Sequence[Any],
    *,
    meta: dict[str, Any] | None = None,
) -> ApplicabilityRetrievalResult:
    expansion = expand_applicability_query(question)
    meta = meta or {}
    per_reg: dict[str, Any] = {}
    for c in chunks:
        rid = (getattr(c, "regulation_id", None) or "").strip()
        if rid and rid not in per_reg:
            per_reg[rid] = c
    # Prefer meta heuristics when present.
    heuristics: list[RegApplicability] = []
    for row in meta.get("heuristics") or []:
        rid = str(row.get("regulation_id") or "").strip()
        if not rid:
            continue
        heuristics.append(
            RegApplicability(
                regulation_id=rid,
                verdict=normalize_verdict(str(row.get("verdict") or "")),
                reason=str(row.get("reason") or ""),
                scope_chunk=per_reg.get(rid),
            )
        )
    if not heuristics:
        for rid in expansion.indexed_regulation_ids:
            chunk = per_reg.get(rid)
            heuristics.append(
                _heuristic_for_reg(
                    rid,
                    getattr(chunk, "text", "") if chunk else "",
                    expansion.vehicle,
                    has_scope=chunk is not None,
                )
            )
            heuristics[-1].scope_chunk = chunk
    return ApplicabilityRetrievalResult(
        chunks=list(chunks),
        expansion=expansion,
        per_regulation=per_reg,
        heuristics=heuristics,
        missing_scope=[
            str(x) for x in (meta.get("missing_scope") or []) if str(x).strip()
        ],
    )
