"""Remap golden_set expected_chunk_ids onto the post-rebuild Qdrant index.

.. deprecated::
    This hybrid-search remapper is **retrieval-circular** and pollutes gold with
    weak-overlap annex/definition hits. Use ``scripts/remap_golden_chunk_ids_v2.py``
    (section scroll + content scoring) instead.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from eval.gold import load_golden_set
from retrieval.retrieve import hybrid_search

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "eval" / "golden_set.jsonl"
_CLAUSE_RE = re.compile(r"\b(\d+(?:\.\d+){1,})\b")


def _overlap_score(chunk_text: str, needles: list[str]) -> float:
    t = (chunk_text or "").lower()
    if not t or not needles:
        return 0.0
    hits = 0
    for n in needles:
        n = (n or "").strip().lower()
        if len(n) < 4:
            continue
        if n in t:
            hits += 1
    return hits / max(1, len([n for n in needles if len((n or "").strip()) >= 4]))


def remap_case(case: dict[str, Any], *, top_k: int = 20) -> dict[str, Any]:
    out = dict(case)
    q = case["question"]
    reg = case.get("regulation_id") or case.get("regulation_scope")
    refs = list(case.get("ground_truth_reference_chunks") or [])
    contains = list(case.get("expected_answer_contains") or [])
    needles = refs + contains + [case.get("expected_behavior") or ""]

    # Section numbers implied by references / behavior.
    sections: list[str] = list(case.get("expected_sections") or [])
    for blob in refs + [case.get("expected_behavior") or ""]:
        for m in _CLAUSE_RE.finditer(blob):
            num = m.group(1)
            if num not in sections and len(num) >= 3:
                sections.append(num)
    if sections:
        out["expected_sections"] = sections

    # Cases with no retrieval gold (injection/guardrail/oos) stay empty.
    if not case.get("expected_chunk_ids") and not refs and not sections:
        if case.get("category") in {
            "prompt_injection",
            "guardrail",
            "out_of_scope",
            "hallucination_probe",
        }:
            out["expected_chunk_ids"] = []
            return out

    hits = hybrid_search(q, regulation_id=reg if reg else None, top_k=top_k)
    scored: list[tuple[float, str, str]] = []
    for h in hits:
        score = _overlap_score(h.text, needles)
        if sections and (h.section_number or "") in sections:
            score += 0.5
        if h.content_type == "figure" and any("figure" in (n or "").lower() for n in needles):
            score += 0.3
        scored.append((score, h.chunk_id, h.section_number or ""))
    scored.sort(reverse=True)
    # Keep positive-score hits; fall back to top-3 retrieved if nothing overlaps.
    chosen = [cid for sc, cid, _ in scored if sc > 0.0][:5]
    if not chosen and hits:
        chosen = [h.chunk_id for h in hits[:3]]
    out["expected_chunk_ids"] = chosen
    out["remapped_from"] = list(case.get("expected_chunk_ids") or [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "golden_set.remapped.jsonl")
    ap.add_argument("--inplace", action="store_true", help="Backup + overwrite --gold")
    args = ap.parse_args()

    cases = load_golden_set(args.gold)
    remapped = [remap_case(c) for c in cases]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in remapped) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(remapped)} cases -> {args.out}")
    if args.inplace:
        backup = args.gold.with_suffix(".jsonl.preremap")
        shutil.copy2(args.gold, backup)
        shutil.copy2(args.out, args.gold)
        print(f"Backed up {args.gold} -> {backup}; overwritten in place")


if __name__ == "__main__":
    main()
