"""LangGraph orchestrator — linear six-agent crash-development crew."""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from backend.app.agents.registry import (
    build_countermeasure_agent,
    build_knowledge_agent,
    build_program_manager_agent,
    build_regulation_agent,
    build_root_cause_agent,
    build_simulation_agent,
)
from backend.app.agents.schemas import CrewReport, CrewState


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in citations:
        key = c.get("marker", "") + (c.get("label") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _assemble_report(state: CrewState) -> dict[str, Any]:
    outputs = state.get("agent_outputs") or {}
    sim = outputs.get("simulation") or {}
    rca = outputs.get("root_cause") or {}
    know = outputs.get("knowledge") or {}
    cm = outputs.get("countermeasure") or {}
    pm = outputs.get("program_manager") or {}

    failing = [
        m.get("name", "")
        for m in sim.get("metrics", [])
        if m.get("status") == "fail"
    ]
    root_lines = [
        f"{r.get('metric')}: {r.get('hypothesis')}"
        for r in rca.get("root_causes", [])
    ]
    case_lines = [
        f"{c.get('program')}: {c.get('failure_mode')} → {c.get('outcome')}"
        for c in know.get("similar_cases", [])
    ]
    cm_lines = [
        f"#{c.get('rank')} {c.get('action')} ({c.get('targets_metric')})"
        for c in sorted(cm.get("countermeasures", []), key=lambda x: x.get("rank", 99))
    ]
    tickets = pm.get("jira_tickets") or []
    action_items = [t.get("title", "") for t in tickets if t.get("title")]

    summary = pm.get("report_markdown", "")[:400]
    if not summary:
        summary = (
            f"Crash crew analysis: {len(failing)} failing metric(s), "
            f"{len(root_lines)} root-cause hypothesis(es), "
            f"{len(cm_lines)} countermeasure(s)."
        )

    return CrewReport(
        summary=summary,
        failing_metrics=failing,
        root_cause=root_lines,
        similar_cases=case_lines,
        countermeasures=cm_lines,
        action_items=action_items,
    ).model_dump()


def _node_finalize(state: CrewState) -> CrewState:
    timing = state.get("timing") or {}
    per_agent = timing.get("per_agent_ms") or {}
    total = sum(per_agent.values()) if per_agent else 0
    citations = _dedupe_citations(state.get("citations") or [])
    return {
        **state,
        "citations": citations,
        "report": _assemble_report(state),
        "timing": {**timing, "total_ms": round(total, 2)},
    }


def _build_graph():
    g = StateGraph(CrewState)
    g.add_node("simulation", build_simulation_agent())
    g.add_node("regulation", build_regulation_agent())
    g.add_node("root_cause", build_root_cause_agent())
    g.add_node("knowledge", build_knowledge_agent())
    g.add_node("countermeasure", build_countermeasure_agent())
    g.add_node("program_manager", build_program_manager_agent())
    g.add_node("finalize", _node_finalize)

    g.set_entry_point("simulation")
    g.add_edge("simulation", "regulation")
    g.add_edge("regulation", "root_cause")
    g.add_edge("root_cause", "knowledge")
    g.add_edge("knowledge", "countermeasure")
    g.add_edge("countermeasure", "program_manager")
    g.add_edge("program_manager", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


crew = _build_graph()


def run_crew(
    crash_result: str,
    *,
    vehicle: str = "",
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the crew from API layer."""
    t0 = time.perf_counter()
    initial: CrewState = {
        "crash_input": crash_result,
        "crash_summary": vehicle or "crash development analysis",
        "vehicle": vehicle,
        "user_id": user_id,
        "session_id": session_id,
        "agent_queries": {},
        "agent_outputs": {},
        "citations": [],
        "timing": {},
    }
    result = crew.invoke(initial)
    timing = result.get("timing") or {}
    if not timing.get("total_ms"):
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["timing"] = timing
    return result
