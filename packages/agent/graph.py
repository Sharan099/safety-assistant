"""The single stateful LangGraph investigation agent — TRD.md Section 3/17-18.

```
load_runs -> quality_gate -> [blocked | continue]
continue: -> global_response -> configuration_diff -> comparability
        -> signal_plan -> signal_analysis -> knowledge_retrieval
        -> historical_retrieval -> hypothesis_and_evidence -> END
blocked: -> blocked_finalize -> END
```

One graph, no swarm (TRD.md Section 3). Every node calls a deterministic
`packages.agent.tools` function or `packages.agent.evidence`; the graph
itself performs no engineering calculation and the LLM (if configured) only
drafts hypothesis *text* — see packages/agent/evidence.py.

Session/LLM access is via closures (`build_graph(session, llm=...)`)
rather than LangGraph's own dependency injection — the state dict stays
plain, JSON-serializable data (TRD.md Section 23/24), not live ORM/HTTP
objects.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from packages.agent import tools
from packages.agent.evidence import evidence_and_hypothesis
from packages.agent.llm import LLMProvider
from packages.agent.persistence import (
    compute_global_response as persistence_compute_global_response,
)
from packages.agent.persistence import (
    persist_comparability,
    persist_configuration_diff,
    persist_quality_results,
    persist_signal_analysis,
)
from packages.agent.state import InvestigationState
from packages.domain.core import SimulationRun
from packages.domain.investigation import Investigation, InvestigationRun


def build_graph(session: Session, *, llm: LLMProvider | None = None) -> CompiledStateGraph:  # type: ignore[type-arg]
    def _investigation(state: InvestigationState) -> Investigation:
        investigation = session.get(Investigation, state["investigation_id"])
        assert investigation is not None
        return investigation

    def load_runs(state: InvestigationState) -> dict[str, Any]:
        tools.load_run(session, state["run_a_id"])
        tools.load_run(session, state["run_b_id"])
        return {}

    def quality_gate(state: InvestigationState) -> dict[str, Any]:
        run_a = tools.load_run(session, state["run_a_id"])
        run_b = tools.load_run(session, state["run_b_id"])
        # Persisted (not just kept in graph state) so anything reading the
        # database afterward — the Copilot context builder, a report, a
        # human — sees the same result the agent based its reasoning on.
        # Found missing here originally via tests/agent/test_copilot_context.py.
        summary_a, summary_b = persist_quality_results(session, _investigation(state), run_a, run_b)
        session.commit()
        return {"quality_a": summary_a.model_dump(), "quality_b": summary_b.model_dump()}

    def route_after_quality(state: InvestigationState) -> str:
        qa, qb = state["quality_a"], state["quality_b"]
        if qa["overall_status"] == "FAIL" or qb["overall_status"] == "FAIL":
            return "blocked"
        return "continue"

    def blocked(state: InvestigationState) -> dict[str, Any]:
        return {"blocked_reason": "quality_gate_failed", "review_required": True}

    def blocked_finalize(state: InvestigationState) -> dict[str, Any]:
        _investigation(state).state = "BLOCKED"
        session.commit()
        return {}

    def global_response(state: InvestigationState) -> dict[str, Any]:
        run_a = tools.load_run(session, state["run_a_id"])
        run_b = tools.load_run(session, state["run_b_id"])
        gr = persistence_compute_global_response(session, run_a, run_b)
        return {"global_response": gr.model_dump()} if gr else {}

    def configuration_diff(state: InvestigationState) -> dict[str, Any]:
        run_a = tools.load_run(session, state["run_a_id"])
        run_b = tools.load_run(session, state["run_b_id"])
        diff = persist_configuration_diff(session, _investigation(state), run_a, run_b)
        session.commit()
        return {"configuration_diff": [d.model_dump() for d in diff]} if diff else {}

    def comparability(state: InvestigationState) -> dict[str, Any]:
        run_a = tools.load_run(session, state["run_a_id"])
        run_b = tools.load_run(session, state["run_b_id"])
        investigation = _investigation(state)

        # Recompute rather than reconstruct from the (already-dumped-to-dict)
        # state, matching the API endpoints' "each step is self-sufficient"
        # design (apps/api/routers/investigations.py) — persist_* functions
        # are cheap, deterministic, and idempotent (upsert-by-delete).
        quality_a, quality_b = persist_quality_results(session, investigation, run_a, run_b)
        gr = persistence_compute_global_response(session, run_a, run_b)
        diffs = persist_configuration_diff(session, investigation, run_a, run_b)

        summary = persist_comparability(
            session,
            investigation,
            run_a,
            run_b,
            quality_a,
            quality_b,
            global_response=gr,
            configuration_diffs=diffs,
        )
        session.commit()
        return {"comparability": summary.model_dump()}

    def signal_plan(state: InvestigationState) -> dict[str, Any]:
        return {"signal_plan": tools.select_signal_plan(state["primary_metric"])}

    def signal_analysis(state: InvestigationState) -> dict[str, Any]:
        run_a = tools.load_run(session, state["run_a_id"])
        run_b = tools.load_run(session, state["run_b_id"])
        investigation = _investigation(state)
        results = {}
        for signal_name in state.get("signal_plan", []):
            result = persist_signal_analysis(session, investigation, run_a, run_b, signal_name)
            if result is not None:
                results[signal_name] = result.model_dump()
        session.commit()
        return {"signal_results": results}

    def knowledge_retrieval(state: InvestigationState) -> dict[str, Any]:
        query = f"{state['question']} {state['primary_metric']}"
        return {"knowledge_evidence": tools.tool_retrieve_knowledge(session, query, limit=5)}

    def historical_retrieval(state: InvestigationState) -> dict[str, Any]:
        cases = tools.tool_retrieve_historical_cases(session, state["primary_metric"], state["investigation_id"])
        return {"historical_cases": cases}

    def hypothesis_and_evidence(state: InvestigationState) -> dict[str, Any]:
        investigation = session.get(Investigation, state["investigation_id"])
        assert investigation is not None
        evidence, hypothesis = evidence_and_hypothesis(session, investigation, state, llm=llm)
        investigation.state = "ENGINEER_REVIEW"
        session.commit()
        return {
            "hypotheses": [{"id": str(hypothesis.id), "title": hypothesis.title, "status": hypothesis.status}],
            "review_required": True,
        }

    graph: StateGraph[InvestigationState] = StateGraph(InvestigationState)
    graph.add_node("load_runs", load_runs)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("blocked", blocked)
    graph.add_node("blocked_finalize", blocked_finalize)
    graph.add_node("global_response", global_response)
    graph.add_node("configuration_diff", configuration_diff)
    graph.add_node("comparability", comparability)
    graph.add_node("signal_plan", signal_plan)
    graph.add_node("signal_analysis", signal_analysis)
    graph.add_node("knowledge_retrieval", knowledge_retrieval)
    graph.add_node("historical_retrieval", historical_retrieval)
    graph.add_node("hypothesis_and_evidence", hypothesis_and_evidence)

    graph.set_entry_point("load_runs")
    graph.add_edge("load_runs", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate", route_after_quality, {"blocked": "blocked", "continue": "global_response"}
    )
    graph.add_edge("blocked", "blocked_finalize")
    graph.add_edge("blocked_finalize", END)
    graph.add_edge("global_response", "configuration_diff")
    graph.add_edge("configuration_diff", "comparability")
    graph.add_edge("comparability", "signal_plan")
    graph.add_edge("signal_plan", "signal_analysis")
    graph.add_edge("signal_analysis", "knowledge_retrieval")
    graph.add_edge("knowledge_retrieval", "historical_retrieval")
    graph.add_edge("historical_retrieval", "hypothesis_and_evidence")
    graph.add_edge("hypothesis_and_evidence", END)

    return graph.compile()


def run_investigation(
    session: Session, investigation_id: uuid.UUID, *, llm: LLMProvider | None = None
) -> InvestigationState:
    investigation = session.get(Investigation, investigation_id)
    if investigation is None:
        raise ValueError(f"investigation not found: {investigation_id}")

    run_rows = session.query(InvestigationRun).filter_by(investigation_id=investigation.id).all()
    by_role = {r.role: r.simulation_run_id for r in run_rows}
    run_a = session.get(SimulationRun, by_role["BASELINE"])
    run_b = session.get(SimulationRun, by_role["COMPARISON"])
    assert run_a is not None and run_b is not None

    from packages.domain.core import SignalDefinition

    primary_metric = "chest_deflection"  # safe default: the only fully-modeled example plan (PR-007)
    if investigation.primary_metric_id is not None:
        signal_def = session.get(SignalDefinition, investigation.primary_metric_id)
        if signal_def is not None:
            primary_metric = signal_def.canonical_name

    initial_state: InvestigationState = {
        "investigation_id": investigation.id,
        "question": investigation.question,
        "run_a_id": run_a.run_id,
        "run_b_id": run_b.run_id,
        "primary_metric": primary_metric,
    }

    graph = build_graph(session, llm=llm)
    final_state: InvestigationState = graph.invoke(initial_state)  # type: ignore[assignment]
    return final_state
