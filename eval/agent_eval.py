"""Evaluate agent traces (per-step citation coverage), not just final answers."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.citations import citation_coverage, check_claims
from agent.loop import run_agent
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT / "agent_golden.jsonl"
RESULTS = ROOT / "results"


def load_agent_golden(path: Path | None = None) -> list[dict[str, Any]]:
    gold_path = path or DEFAULT_GOLDEN
    text = gold_path.read_text(encoding="utf-8-sig")
    cases = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        cases.append(
            {
                "id": raw.get("id") or f"agent_{i:03d}",
                "task": (raw.get("task") or raw.get("question") or "").strip(),
                "mode": raw.get("mode"),
                "min_coverage": float(raw.get("min_coverage") or 0.9),
                "expect_tools": list(raw.get("expect_tools") or []),
                "expect_regs": list(raw.get("expect_regs") or []),
            }
        )
    return cases


def score_trace(result, *, min_coverage: float = 0.9) -> dict[str, Any]:
    tool_steps = [s for s in result.steps if s.tool not in {"plan"}]
    coverages = [s.citation_coverage for s in tool_steps if s.tool != "plan"]
    final_checks = check_claims(
        result.answer or result.report_markdown or result.table_markdown,
        allowed=[s.citation for s in result.sources if s.citation],
    )
    final_cov = citation_coverage(final_checks)
    tools_used = [s.tool for s in result.steps]
    return {
        "trace_id": result.trace_id,
        "mode": result.mode,
        "n_steps": len(result.steps),
        "tools_used": tools_used,
        "step_citation_coverage_mean": round(statistics.mean(coverages), 4) if coverages else 1.0,
        "step_citation_coverage_min": round(min(coverages), 4) if coverages else 1.0,
        "final_citation_coverage": round(final_cov, 4),
        "ungrounded_claim_count": result.ungrounded_claim_count,
        "overall_citation_coverage": result.overall_citation_coverage,
        "n_sources": len(result.sources),
        "faithfulness_passed": bool(
            result.overall_citation_coverage >= min_coverage
            and result.ungrounded_claim_count == 0
            and final_cov >= min_coverage
        ),
        "not_found": result.not_found,
    }


def run_agent_eval(
    *,
    gold_path: Path | None = None,
    min_coverage: float = 0.9,
) -> dict[str, Any]:
    cases = load_agent_golden(gold_path)
    llm = LLMClient()
    rows = []
    for case in cases:
        result = run_agent(case["task"], llm=llm)
        scored = score_trace(result, min_coverage=case.get("min_coverage") or min_coverage)
        expect_tools = case.get("expect_tools") or []
        tools_ok = all(t in scored["tools_used"] for t in expect_tools) if expect_tools else True
        expect_regs = [r.upper() for r in (case.get("expect_regs") or [])]
        regs_found = {
            (s.regulation_id or "").upper()
            for s in result.sources
        }
        regs_ok = all(
            any(er in rr or rr in er for rr in regs_found) for er in expect_regs
        ) if expect_regs else True
        # For not-indexed regs (FMVSS), allow compare tool without sources from that side
        if case.get("mode") and case["mode"] != result.mode:
            mode_ok = False
        else:
            mode_ok = True
        passed = bool(scored["faithfulness_passed"] and tools_ok and mode_ok)
        rows.append(
            {
                "id": case["id"],
                "task": case["task"],
                "passed": passed,
                "tools_ok": tools_ok,
                "regs_ok": regs_ok,
                "mode_ok": mode_ok,
                **scored,
            }
        )

    n = len(rows) or 1
    report = {
        "n": len(rows),
        "pass_rate": round(sum(1 for r in rows if r["passed"]) / n, 4),
        "avg_final_citation_coverage": round(
            sum(r["final_citation_coverage"] for r in rows) / n, 4
        ),
        "avg_step_citation_coverage_min": round(
            sum(r["step_citation_coverage_min"] for r in rows) / n, 4
        ),
        "cases": rows,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Evaluate agent traces for citation faithfulness.")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--min-coverage", type=float, default=0.9)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    report = run_agent_eval(gold_path=args.gold, min_coverage=args.min_coverage)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS / "agent_latest.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2))
    print(f"Wrote {out}")
    failed = [c for c in report["cases"] if not c["passed"]]
    if failed:
        print(f"FAIL: {len(failed)}/{report['n']} agent cases", file=sys.stderr)
        for c in failed:
            print(f"  - {c['id']}: cov={c['final_citation_coverage']}", file=sys.stderr)
        return 1
    print("PASS: agent trace eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
