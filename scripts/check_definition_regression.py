"""Definition-query regression check vs compliance prefer-limit bias (Task 2).

Compares retrieval with bias enabled vs disabled for definition-seeking probes,
and asserts the router / prefer-limit gate does not treat them as compliance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(override=False)

from retrieval.router import QueryIntent, classify_query
from retrieval.value_limit import (
    is_compliance_prefer_limit_query,
    is_definition_seeking_query,
)


def _top_roles(chunks, n: int = 5) -> list[str]:
    out = []
    for c in chunks[:n]:
        text = (c.text or c.enriched_text or "")[:200]
        role = "def" if ("means" in text.lower()[:120] or 'means "' in text.lower()) else "other"
        if "shall not exceed" in text.lower() or "≤" in text or "<=" in text:
            role = "limit"
        out.append(
            f"{c.chunk_id[:12]}@{c.section_number or '?'}[{role}]"
        )
    return out


def main() -> int:
    from retrieval.retrieve import hybrid_search, retrieve
    from retrieval.value_limit import bias_chunks_for_value_vs_limit

    probes = [
        json.loads(line)
        for line in (ROOT / "eval" / "definition_probe.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    # Also include a compliance control that SHOULD prefer limits.
    controls = [
        {
            "id": "ctrl_cmp_004",
            "question": (
                "REESS remained mounted, but electrolyte entered the passenger "
                "compartment. Does the vehicle comply?"
            ),
            "expect_prefer_limit": True,
            "expect_intent": QueryIntent.COMPLIANCE_CHECK,
        }
    ]

    print("=" * 72)
    print("DEFINITION vs COMPLIANCE prefer-limit gate")
    print("=" * 72)
    failures = 0

    for p in probes:
        q = p["question"]
        routed = classify_query(q, llm=None)
        prefer = is_compliance_prefer_limit_query(q)
        defining = is_definition_seeking_query(q)
        print(f"\n[{p['id']}] {q}")
        print(
            f"  definition_seeking={defining} prefer_limit={prefer} "
            f"intent={routed.intent.value} reason={routed.reason}"
        )
        if not defining:
            print("  FAIL: expected definition_seeking=True")
            failures += 1
        if prefer:
            print("  FAIL: prefer_limit must be False for definition asks")
            failures += 1
        if routed.intent == QueryIntent.COMPLIANCE_CHECK:
            print("  FAIL: router classified definition ask as COMPLIANCE_CHECK")
            failures += 1

        raw = hybrid_search(q, top_k=20, regulation_id=p.get("regulation_scope"))
        biased = bias_chunks_for_value_vs_limit(raw, question=q)
        # Bias must be a no-op for definition asks (same order).
        raw_ids = [c.chunk_id for c in raw]
        biased_ids = [c.chunk_id for c in biased]
        noop = raw_ids == biased_ids
        print(f"  bias_noop={noop} top5={_top_roles(raw)}")
        if not noop:
            print("  FAIL: bias reordered definition-seeking results")
            failures += 1

        full = retrieve(
            q,
            top_k=8,
            regulation_id=p.get("regulation_scope"),
            rewrite=False,
            do_rerank=True,
            small_to_big=True,
        )
        print(f"  pipeline_top5={_top_roles(full)}")

    for c in controls:
        q = c["question"]
        routed = classify_query(q, llm=None)
        prefer = is_compliance_prefer_limit_query(q)
        print(f"\n[{c['id']}] {q[:80]}…")
        print(
            f"  prefer_limit={prefer} intent={routed.intent.value} "
            f"(expect prefer={c['expect_prefer_limit']})"
        )
        if prefer != c["expect_prefer_limit"]:
            print("  FAIL: control prefer_limit mismatch")
            failures += 1
        if routed.intent != c["expect_intent"]:
            print(
                f"  WARN: intent={routed.intent.value} "
                f"expected {c['expect_intent'].value}"
            )

    print("\n" + "=" * 72)
    print(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
