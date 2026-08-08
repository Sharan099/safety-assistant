"""One-off audit: verify remapped expected_chunk_ids against real corpus text."""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parents[1]
COLL = "regulations"


def fetch_by_chunk_ids(client: QdrantClient, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cid in ids:
        pts, _ = client.scroll(
            collection_name=COLL,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=cid))]
            ),
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        if pts:
            p = pts[0].payload or {}
            out[cid] = {
                "section": p.get("section_number") or p.get("section"),
                "reg": p.get("regulation_id"),
                "element_type": p.get("element_type") or p.get("content_type"),
                "text": (p.get("text") or p.get("page_content") or "")[:600],
                "exists": True,
            }
        else:
            out[cid] = {"exists": False, "text": "", "section": None, "reg": None}
    return out


def ranked_chunk_ids(retrieved_keys: list[str]) -> list[str]:
    out: list[str] = []
    for k in retrieved_keys or []:
        if "::" in k:
            continue
        if k.startswith("Annex"):
            continue
        if k.replace(".", "").replace("/", "").isdigit():
            continue
        if len(k) >= 12:
            out.append(k)
    return out


def content_ok(text: str, *, contains: list[str], behavior: str, refs: list[str], sections: list[str]) -> dict:
    """Heuristic: does chunk text look like the intended gold content?"""
    t = (text or "").lower()
    reasons: list[str] = []
    score = 0
    # section hint from behavior e.g. §5.2.8.2
    for sec in sections:
        if sec and sec.lower() in t:
            score += 2
            reasons.append(f"section:{sec}")
    for needle in contains:
        n = (needle or "").strip().lower()
        if len(n) < 2:
            continue
        # skip PASS/FAIL — those are answer labels, not chunk content
        if n in {"pass", "fail"}:
            continue
        if n in t:
            score += 1
            reasons.append(f"contains:{needle}")
    for ref in refs[:2]:
        # use a distinctive 40-char window from the GT reference
        ref_l = (ref or "").lower()
        for window in (ref_l[i : i + 40] for i in range(0, min(120, len(ref_l)), 40)):
            window = window.strip()
            if len(window) >= 20 and window in t:
                score += 3
                reasons.append("gt_ref_overlap")
                break
    beh = (behavior or "").lower()
    for token in ("electrolyte", "rib deflection", "42 mm", "1.0 m/s", "hpc", "scope", "door", "figure 1", "neck"):
        if token in beh and token in t:
            score += 1
            reasons.append(f"beh:{token}")
    return {"score": score, "reasons": reasons, "ok": score >= 2}


def main() -> None:
    client = QdrantClient(path=str(ROOT / "data" / "qdrant"))
    eval_d = json.loads(
        (ROOT / "eval/results/stage5_rebuild-retrieval-only-20260807T232116Z.json").read_text(
            encoding="utf-8"
        )
    )
    cur = {
        json.loads(l)["id"]: json.loads(l)
        for l in (ROOT / "eval/golden_set.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    pre = {
        json.loads(l)["id"]: json.loads(l)
        for l in (ROOT / "eval/golden_set.jsonl.preremap").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    rows = eval_d["retrieval"]["cases"]
    ranked_rows = []
    for r in rows:
        g = cur.get(r["id"]) or {}
        exp = list(g.get("expected_chunk_ids") or [])
        if not exp:
            continue
        top5 = ranked_chunk_ids(r.get("retrieved_keys") or [])[:5]
        overlap = set(exp) & set(top5)
        ranked_rows.append(
            (
                0 if overlap else 1,
                -(r.get("mrr") or 0.0),
                -(r.get("recall@5") or 0.0),
                r["id"],
                r,
                exp,
                top5,
                overlap,
            )
        )
    ranked_rows.sort(reverse=True)

    sample = []
    seen: set[str] = set()
    for _a, _b, _c, _id, r, exp, top5, overlap in ranked_rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        sample.append((r, exp, top5, overlap))
        if len(sample) >= 10:
            break

    report: list[dict] = []
    print(f"SAMPLE SIZE {len(sample)}")
    for r, exp, top5, overlap in sample:
        cid = r["id"]
        g = cur[cid]
        p = pre.get(cid) or {}
        contains = list(g.get("expected_answer_contains") or [])
        behavior = g.get("expected_behavior") or ""
        refs = list(
            g.get("ground_truth_reference_chunks")
            or p.get("ground_truth_reference_chunks")
            or []
        )
        sections = list(g.get("expected_sections") or [])
        # also parse section-like from behavior
        import re

        for m in re.finditer(r"\b(\d+(?:\.\d+){1,})\b", behavior):
            if m.group(1) not in sections:
                sections.append(m.group(1))

        all_ids = list(dict.fromkeys(exp + top5))
        texts = fetch_by_chunk_ids(client, all_ids)

        exp_judgments = []
        for eid in exp:
            info = texts.get(eid) or {}
            j = content_ok(
                info.get("text") or "",
                contains=contains,
                behavior=behavior,
                refs=refs,
                sections=sections,
            )
            exp_judgments.append(
                {
                    "chunk_id": eid,
                    "exists": info.get("exists"),
                    "section": info.get("section"),
                    "reg": info.get("reg"),
                    "text_preview": (info.get("text") or "")[:220],
                    **j,
                }
            )

        ret_judgments = []
        for rid in top5:
            info = texts.get(rid) or {}
            j = content_ok(
                info.get("text") or "",
                contains=contains,
                behavior=behavior,
                refs=refs,
                sections=sections,
            )
            ret_judgments.append(
                {
                    "chunk_id": rid,
                    "exists": info.get("exists"),
                    "section": info.get("section"),
                    "reg": info.get("reg"),
                    "text_preview": (info.get("text") or "")[:220],
                    **j,
                }
            )

        any_exp_ok = any(j["ok"] and j["exists"] for j in exp_judgments)
        any_exp_missing = any(not j["exists"] for j in exp_judgments)
        best_exp = max((j["score"] for j in exp_judgments), default=0)
        best_ret = max((j["score"] for j in ret_judgments), default=0)

        verdict = "remap_looks_correct"
        if any_exp_missing and not any_exp_ok:
            verdict = "remap_broken_missing_ids"
        elif not any_exp_ok:
            verdict = "remap_wrong_content"
        elif best_ret > best_exp and not overlap:
            verdict = "remap_ok_but_better_chunk_retrieved"
        elif not overlap:
            verdict = "remap_ok_retrieval_miss"

        row = {
            "id": cid,
            "mrr": r.get("mrr"),
            "recall@5": r.get("recall@5"),
            "question": r["question"],
            "behavior": behavior[:220],
            "contains": contains,
            "pre_expected": p.get("expected_chunk_ids"),
            "now_expected": exp,
            "remapped_from": g.get("remapped_from"),
            "top5_retrieved": top5,
            "top5_overlap": sorted(overlap),
            "expected_judgments": exp_judgments,
            "retrieved_judgments": ret_judgments,
            "verdict": verdict,
        }
        report.append(row)

        print("\n" + "=" * 80)
        print(
            f"CASE {cid} mrr={r.get('mrr')} recall@5={r.get('recall@5')} "
            f"overlap={sorted(overlap)} VERDICT={verdict}"
        )
        print("Q:", r["question"][:160])
        print("behavior:", behavior[:200])
        print("contains:", contains)
        if refs:
            print("GT ref[0]:", str(refs[0])[:240])
        print("pre:", p.get("expected_chunk_ids"))
        print("now:", exp)
        print("top5:", top5)
        print("--- EXPECTED ---")
        for j in exp_judgments:
            print(
                f"  {j['chunk_id']} exists={j['exists']} sec={j['section']} "
                f"ok={j['ok']} score={j['score']} {j['reasons']}"
            )
            print(
                f"    {j['text_preview'].encode('ascii', 'replace').decode('ascii')!r}"
            )
        print("--- RETRIEVED TOP5 ---")
        for j in ret_judgments:
            print(
                f"  {j['chunk_id']} exists={j['exists']} sec={j['section']} "
                f"ok={j['ok']} score={j['score']} {j['reasons']}"
            )
            print(
                f"    {j['text_preview'].encode('ascii', 'replace').decode('ascii')!r}"
            )

    out_path = ROOT / "eval" / "results" / "remap_gold_audit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    from collections import Counter

    print("verdict_counts", Counter(r["verdict"] for r in report))


if __name__ == "__main__":
    main()
