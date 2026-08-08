"""Plan multi-step agent work: decompose → tool steps."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.regs import normalize_regulation
from agent.state import PlannedStep
from generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

PLAN_SYSTEM = """\
You plan tool calls for a UNECE passive-safety regulation agent.
Return ONLY JSON:
{
  "mode": "qa" | "compare" | "report",
  "steps": [
    {"tool": "retrieve"|"compare_regulations"|"lookup_table"|"draft_report",
     "args": {...}, "rationale": "..."}
  ]
}
Rules:
- Prefer retrieve / lookup_table before draft_report.
- For comparisons use compare_regulations once (it retrieves both sides).
- For gap-analysis / memo requests use retrieve steps then draft_report.
- Max 6 steps. Never invent regulation text in the plan.
- Do NOT plan design_implication / applicability / checklist_gen / retest_scope here —
  those Layer 3–5 modes are selected by the hybrid intent router, not free-form planning.
"""


_INTENT_TOOL: dict[str, str] = {
    "DESIGN_IMPLICATION": "design_implication",
    "APPLICABILITY": "applicability",
    "CHECKLIST_GEN": "checklist_gen",
    "RETEST_SCOPE": "retest_scope",
}


def plan_for_query_intent(intent: str, task: str) -> tuple[str, list[PlannedStep]]:
    """Fixed bounded plan for Layer 3–5 intents (one specialized tool step)."""
    tool = _INTENT_TOOL.get((intent or "").strip().upper())
    if not tool:
        return "qa", []
    mode = tool
    rationale = {
        "design_implication": "Expand component→concepts; retrieve across likely regs",
        "applicability": "Survey Scope of every indexed regulation for this vehicle",
        "checklist_gen": "Build per-category homologation / test-prep checklist",
        "retest_scope": "Cite modification clauses; informational only — verify with authority",
    }.get(tool, "Layer 3–5 multi-step pipeline")
    return mode, [
        PlannedStep(
            tool=tool,  # type: ignore[arg-type]
            args={"query": task},
            rationale=rationale,
        )
    ]


def detect_mode(task: str) -> str:
    t = (task or "").lower()
    if any(k in t for k in ("gap analysis", "memo", "draft report", "report:", "write a report")):
        return "report"
    if any(k in t for k in ("compare", " vs ", "versus", "difference between", "compliance comparison")):
        return "compare"
    return "qa"


def _extract_regs(task: str) -> list[str]:
    found: list[str] = []
    for m in re.finditer(
        r"\b(R\s*\d{2,3}|UN[-\s]?ECE[-\s]?R\s*\d{2,3}|FMVSS\s*-?\s*\d+)\b",
        task,
        re.I,
    ):
        rid = normalize_regulation(m.group(0))
        if rid and rid not in found:
            found.append(rid)
    return found


def _heuristic_plan(task: str) -> tuple[str, list[PlannedStep]]:
    mode = detect_mode(task)
    regs = _extract_regs(task)
    steps: list[PlannedStep] = []

    if mode == "compare":
        topic = re.sub(
            r"\b(compare|versus|vs\.?|difference between|compliance comparison[:\s]*)\b",
            " ",
            task,
            flags=re.I,
        )
        topic = re.sub(
            r"\b(R\s*\d{2,3}|UN[-\s]?ECE[-\s]?R\s*\d{2,3}|FMVSS\s*-?\s*\d+)\b",
            " ",
            topic,
            flags=re.I,
        )
        topic = re.sub(r"\s+", " ", topic).strip(" :?-") or task
        a = regs[0] if regs else "UN-ECE-R94"
        b = regs[1] if len(regs) > 1 else ("FMVSS-208" if "fmvss" in task.lower() else "UN-ECE-R95")
        steps.append(
            PlannedStep(
                tool="compare_regulations",
                args={"reg_a": a, "reg_b": b, "topic": topic},
                rationale="Side-by-side cited comparison",
            )
        )
        return mode, steps

    if mode == "report":
        # Decompose into retrieve foci then draft
        foci = []
        if "r95" in task.lower() or "UN-ECE-R95" in (regs or []):
            foci.append(("UN-ECE-R95", "side impact injury criteria requirements"))
        if "test setup" in task.lower() or "gap" in task.lower():
            foci.append((regs[0] if regs else "UN-ECE-R95", "test procedure requirements"))
        if not foci:
            foci = [(regs[0] if regs else None, task)]
        for reg, q in foci[:3]:
            steps.append(
                PlannedStep(
                    tool="retrieve",
                    args={"regulation": reg, "query": q},
                    rationale="Gather cited evidence",
                )
            )
        steps.append(
            PlannedStep(
                tool="draft_report",
                args={
                    "title": "Gap analysis memo",
                    "sections": [
                        {"heading": "Scope", "focus": task},
                        {"heading": "Regulatory requirements", "focus": "requirements limits criteria"},
                        {"heading": "Gaps / open items", "focus": "test procedure requirements"},
                        {"heading": "Recommendations", "focus": "requirements"},
                    ],
                },
                rationale="Synthesize fully-cited memo from evidence",
            )
        )
        return mode, steps

    # qa — prefer table lookup for criterion-ish questions
    if re.search(r"\b(hic|thcc|tti|vc|deflection|limit|criterion)\b", task, re.I):
        reg = regs[0] if regs else "UN-ECE-R94"
        crit = re.search(r"\b(HIC15|HIC36|HIC|ThCC|TTI|VC|chest deflection)\b", task, re.I)
        steps.append(
            PlannedStep(
                tool="lookup_table",
                args={
                    "regulation": reg,
                    "criterion": crit.group(0) if crit else task,
                },
                rationale="Criterion / table lookup",
            )
        )
    else:
        steps.append(
            PlannedStep(
                tool="retrieve",
                args={"regulation": regs[0] if regs else None, "query": task},
                rationale="Retrieve grounded passages",
            )
        )
    return mode, steps


def plan_task(task: str, *, llm: LLMClient | None = None) -> tuple[str, list[PlannedStep]]:
    """Return (mode, steps). Uses heuristics for mock; optional LLM JSON plan for groq."""
    task = (task or "").strip()
    if not task:
        return "qa", []

    client = llm or LLMClient()
    if client.provider == "mock":
        return _heuristic_plan(task)

    try:
        result = client.complete(
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": task},
            ],
            role="rewrite",
            question=task,
            chunk_ids=[],
            max_tokens=512,
        )
        text = result.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        mode = str(data.get("mode") or detect_mode(task))
        steps: list[PlannedStep] = []
        for raw in data.get("steps") or []:
            tool = str(raw.get("tool") or "")
            if tool not in {"retrieve", "compare_regulations", "lookup_table", "draft_report"}:
                continue
            steps.append(
                PlannedStep(
                    tool=tool,  # type: ignore[arg-type]
                    args=dict(raw.get("args") or {}),
                    rationale=str(raw.get("rationale") or ""),
                )
            )
        if steps:
            return mode, steps[:6]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM plan failed (%s); using heuristics", exc)

    return _heuristic_plan(task)
