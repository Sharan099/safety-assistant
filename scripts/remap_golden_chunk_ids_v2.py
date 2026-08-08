"""Content-grounded remap of golden expected_chunk_ids (NOT retrieval-circular).

The prior remapper (``remap_golden_chunk_ids.py``) used ``hybrid_search`` and kept
up to 5 positive-overlap hits, which polluted gold with irrelevant annex/definition
chunks and artificially depressed recall@5 / MRR.

This script:
1. Starts from ``golden_set.jsonl.preremap`` (or ``--gold``).
2. Keeps any pre-existing chunk_id that still exists and matches GT content.
3. Finds candidates by scrolling Qdrant on ``section_number`` + ``regulation_id``.
4. Scores candidates by ground-truth reference / expected-behavior / contains overlap.
5. Keeps only high-confidence matches (no hybrid fallback).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "eval" / "golden_set.jsonl"
DEFAULT_PRE = ROOT / "eval" / "golden_set.jsonl.preremap"
COLL = "regulations"
_CLAUSE_RE = re.compile(r"\b(\d+(?:\.\d+){1,})\b")
_SKIP_CONTAINS = {"pass", "fail", "r94", "r95", "r16", "r129"}


def _client() -> QdrantClient:
    return QdrantClient(path=str(ROOT / "data" / "qdrant"))


def _payload_text(p: dict[str, Any]) -> str:
    return str(p.get("text") or p.get("page_content") or "")


def fetch_chunk(client: QdrantClient, chunk_id: str) -> dict[str, Any] | None:
    pts, _ = client.scroll(
        collection_name=COLL,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not pts:
        return None
    p = pts[0].payload or {}
    return {
        "chunk_id": chunk_id,
        "section_number": str(p.get("section_number") or p.get("section") or ""),
        "regulation_id": str(p.get("regulation_id") or ""),
        "element_type": str(p.get("element_type") or p.get("content_type") or ""),
        "text": _payload_text(p),
    }


def scroll_by_section(
    client: QdrantClient,
    *,
    regulation_id: str | None,
    section_number: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    must: list[Any] = [
        qm.FieldCondition(key="section_number", match=qm.MatchValue(value=section_number))
    ]
    if regulation_id:
        must.append(
            qm.FieldCondition(
                key="regulation_id", match=qm.MatchValue(value=regulation_id)
            )
        )
    pts, _ = client.scroll(
        collection_name=COLL,
        scroll_filter=qm.Filter(must=must),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    out: list[dict[str, Any]] = []
    for pt in pts:
        p = pt.payload or {}
        cid = str(p.get("chunk_id") or "")
        if not cid:
            continue
        out.append(
            {
                "chunk_id": cid,
                "section_number": str(p.get("section_number") or p.get("section") or ""),
                "regulation_id": str(p.get("regulation_id") or ""),
                "element_type": str(p.get("element_type") or p.get("content_type") or ""),
                "text": _payload_text(p),
            }
        )
    return out


def _needles(case: dict[str, Any]) -> list[str]:
    refs = [str(x) for x in (case.get("ground_truth_reference_chunks") or [])]
    contains = [
        str(x)
        for x in (case.get("expected_answer_contains") or [])
        if str(x).strip().lower() not in _SKIP_CONTAINS
    ]
    beh = str(case.get("expected_behavior") or "")
    return [n for n in (refs + contains + [beh]) if n and len(n.strip()) >= 4]


def _looks_like_clause(num: str) -> bool:
    """Reject measurement-like decimals (0.8, 42.5, 1.0) mistaken for clause numbers."""
    parts = num.split(".")
    if len(parts) < 2:
        return num in {"1"}  # Scope
    try:
        head = int(parts[0])
        second = int(parts[1])
    except ValueError:
        return False
    if head == 0 or head > 20:
        return False
    # Reject X.0 measurement style (1.0 m/s) — real clauses are 5.2.1… not 1.0.
    if len(parts) == 2 and second == 0:
        return False
    # Prefer multi-level UNECE trees (5.2.1.2) over ambiguous two-part values.
    if len(parts) == 2 and head >= 10:
        return False
    return True


def _sections(case: dict[str, Any]) -> list[str]:
    sections: list[str] = [str(s) for s in (case.get("expected_sections") or []) if s]
    # Never mine expected_answer_contains — those are measured values / answer tokens.
    for blob in list(case.get("ground_truth_reference_chunks") or []) + [
        case.get("expected_behavior") or "",
        case.get("question") or "",
    ]:
        for m in _CLAUSE_RE.finditer(str(blob)):
            num = m.group(1)
            if num not in sections and _looks_like_clause(num):
                sections.append(num)

    # Keyword → canonical injury-criteria clauses (when GT omits explicit §).
    # IMPORTANT: use word-boundary matches — bare ``"hic" in blob`` false-positives
    # on ``which`` / ``vehicle``.
    blob = " ".join(
        [
            str(case.get("question") or ""),
            str(case.get("expected_behavior") or ""),
            " ".join(str(x) for x in (case.get("expected_answer_contains") or [])),
        ]
    ).lower()
    reg = str(case.get("regulation_scope") or case.get("regulation_id") or "").upper()

    def _has(*terms: str) -> bool:
        return any(re.search(rf"\b{re.escape(t)}\b", blob) for t in terms)

    if "R95" in reg or _has("r95"):
        if _has("rdc", "vc") or any(
            p in blob for p in ("rib deflection", "soft tissue")
        ):
            if "5.2.1.2" not in sections:
                sections.append("5.2.1.2")
        if "door" in blob:
            for s in ("5.3.2", "5.3.3.1", "Annex 4/5.2"):
                if s not in sections:
                    sections.append(s)
    if "R94" in reg or _has("r94"):
        if _has("hpc", "hic") or "head performance" in blob:
            if "5.2.1.1" not in sections:
                sections.append("5.2.1.1")
        if _has("thcc") or "thorax compression" in blob:
            if "5.2.1.4" not in sections:
                sections.append("5.2.1.4")
        if _has("tcfc") or "tibia" in blob:
            if "5.2.1.7" not in sections:
                sections.append("5.2.1.7")
        if "electrolyte" in blob or _has("reess"):
            if "5.2.8.2" not in sections:
                sections.append("5.2.8.2")
        if "scope" in blob or "vehicles are covered" in blob or "category m1" in blob:
            if "1" not in sections:
                sections.append("1")
    return sections


def _norm_nums(s: str) -> str:
    # "1,000" / "1.000" → "1000" for limit matching.
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", (s or "").lower())


def content_score(text: str, needles: list[str], sections: list[str]) -> float:
    t = _norm_nums(text or "")
    if not t:
        return 0.0
    score = 0.0
    for sec in sections:
        if sec and sec.lower() in t:
            score += 1.5
    for n in needles:
        n_l = _norm_nums(n.strip())
        if len(n_l) < 3:
            continue
        # Prefer longer distinctive windows from GT refs.
        if len(n_l) >= 40:
            windows = [n_l[i : i + 40] for i in range(0, min(160, len(n_l)), 40)]
            hits = sum(1 for w in windows if len(w) >= 20 and w in t)
            score += hits * 2.0
        elif n_l in t:
            score += 1.0 if len(n_l) >= 4 else 0.5
    # Strong phrase bonuses for injury criteria gold.
    for phrase, bonus in (
        ("rib deflection criterion", 2.0),
        ("soft tissue criterion", 2.0),
        ("head performance criterion", 2.0),
        ("electrolyte leakage", 2.0),
        ("femur force criterion", 2.0),
        ("category m1", 2.0),
        ("doors shall be closed, but not locked", 2.0),
        ("no door shall open", 2.0),
    ):
        if phrase in t:
            score += bonus
    return score


def remap_case(client: QdrantClient, case: dict[str, Any]) -> dict[str, Any]:
    out = dict(case)
    cat = str(case.get("category") or "").strip().lower()
    if cat in {
        "prompt_injection",
        "guardrail",
        "out_of_scope",
        "hallucination_probe",
    } and not case.get("expected_chunk_ids") and not case.get("ground_truth_reference_chunks"):
        out["expected_chunk_ids"] = []
        return out

    needles = _needles(case)
    sections = _sections(case)
    out["expected_sections"] = sections
    reg = case.get("regulation_scope") or case.get("regulation_id") or None
    if reg is not None:
        reg = str(reg).strip() or None

    old_ids = [str(x) for x in (case.get("expected_chunk_ids") or []) if str(x).strip()]
    candidates: dict[str, dict[str, Any]] = {}

    # 1) Keep old IDs if they still exist and score well.
    for cid in old_ids:
        hit = fetch_chunk(client, cid)
        if not hit:
            continue
        sc = content_score(hit["text"], needles, sections)
        hit["score"] = sc
        if sc >= 1.0 or (sections and hit["section_number"] in sections):
            candidates[cid] = hit

    # 2) Section scroll (primary, non-circular).
    for sec in sections:
        for hit in scroll_by_section(client, regulation_id=reg, section_number=sec):
            sc = content_score(hit["text"], needles, sections)
            hit["score"] = sc
            prev = candidates.get(hit["chunk_id"])
            if prev is None or sc > float(prev.get("score") or 0):
                if sc >= 1.0 or hit["section_number"] in sections:
                    candidates[hit["chunk_id"]] = hit

    # 3) Hybrid candidate pass DISABLED by default — it opens a second Qdrant
    #    client (lock conflict on local path) and re-introduces retrieval circularity.
    #    Section scroll + old-id retention is the source of truth.

    # 4) For figure cases, also accept figure element_type under parent section.
    if any("figure" in (n or "").lower() for n in needles + [case.get("expected_behavior") or ""]):
        for sec in sections:
            for hit in scroll_by_section(client, regulation_id=reg, section_number=sec, limit=40):
                et = (hit.get("element_type") or "").lower()
                if "figure" not in et and "figure" not in (hit.get("text") or "").lower()[:120]:
                    continue
                sc = content_score(hit["text"], needles, sections) + 1.0
                hit["score"] = sc
                prev = candidates.get(hit["chunk_id"])
                if prev is None or sc > float(prev.get("score") or 0):
                    candidates[hit["chunk_id"]] = hit

    ranked = sorted(candidates.values(), key=lambda h: float(h.get("score") or 0), reverse=True)

    def _section_compatible(hit: dict[str, Any]) -> bool:
        if not sections:
            return True
        sn = str(hit.get("section_number") or "")
        if sn in sections:
            return True
        # Child clauses of an expected parent (5.2.8.2 → 5.2.8.2.1).
        for s in sections:
            if s and (sn.startswith(s + ".") or s.startswith(sn + ".")):
                return True
        return False

    ranked = [h for h in ranked if _section_compatible(h)]
    # Strict: only keep score >= 2.0, max 3 ids (precision over recall for gold).
    chosen = [h["chunk_id"] for h in ranked if float(h.get("score") or 0) >= 2.0][:3]
    # If nothing strict but we have an exact section match with any score, keep best one.
    if not chosen and ranked:
        exact = [h for h in ranked if h.get("section_number") in sections]
        pool = exact or ranked
        if float(pool[0].get("score") or 0) >= 1.0:
            chosen = [pool[0]["chunk_id"]]

    out["expected_chunk_ids"] = chosen
    out["remapped_from"] = old_ids
    out["remap_method"] = "section_content_v2"
    out["remap_scores"] = {
        h["chunk_id"]: round(float(h.get("score") or 0), 3) for h in ranked[:8]
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=DEFAULT_PRE)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "golden_set.remapped_v2.jsonl")
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()

    text = args.gold.read_text(encoding="utf-8-sig")
    cases = [json.loads(l) for l in text.splitlines() if l.strip()]
    client = _client()
    remapped = [remap_case(client, c) for c in cases]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in remapped) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(remapped)} cases -> {args.out}")
    n_with = sum(1 for c in remapped if c.get("expected_chunk_ids"))
    n_empty_had = sum(
        1
        for c, o in zip(cases, remapped)
        if (c.get("expected_chunk_ids") or c.get("ground_truth_reference_chunks"))
        and not o.get("expected_chunk_ids")
    )
    print(f"cases_with_expected={n_with} emptied_despite_gold_signal={n_empty_had}")
    if args.inplace:
        target = DEFAULT_GOLD
        backup = target.with_suffix(".jsonl.preremap_v2bak")
        shutil.copy2(target, backup)
        shutil.copy2(args.out, target)
        print(f"Backed up {target} -> {backup}; overwritten in place")


if __name__ == "__main__":
    main()
