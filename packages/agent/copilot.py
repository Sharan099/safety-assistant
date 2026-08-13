"""The Copilot LangGraph — CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 5/7/9.

```
load_context -> classify_intent -> dispatch_tools -> ground_response -> END
```

Reuses `packages.agent.tools` (the same deterministic tool layer
`packages/agent/graph.py` uses) rather than performing any calculation
itself — Phase 6: "The LLM must not calculate engineering values when
deterministic tools exist." The LLM (if configured) only rephrases the
already-grounded deterministic response text in `ground_response`; its
absence never blocks an answer (TRD.md Section 30).

Each node's output is one "step" — the API layer (apps/api/routers/copilot.py)
streams these via `.stream()` as they happen, which is what gives the UI
real-time workflow visibility without a second mechanism.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from packages.agent import tools
from packages.agent.copilot_context import InvestigationContext, build_investigation_context
from packages.agent.copilot_intents import Intent, classify_intent, extract_signal
from packages.agent.llm import LLMMessage, LLMProvider, LLMUnavailableError
from packages.domain.copilot import CopilotConversation, CopilotMessage, CopilotToolCall
from packages.domain.core import SimulationRun
from packages.domain.investigation import (
    ControlledComparisonRequest,
    Evidence,
    Hypothesis,
    HypothesisEvidenceLink,
    Investigation,
)

STANDARD_SUGGESTED_ACTIONS = [
    "Why is this hypothesis leading?",
    "Find contradictions",
    "Analyze crash pulse",
    "Analyze relevant signals",
    "Show supporting evidence",
    "Find documentation",
    "Find similar cases",
    "What evidence is missing?",
    "Suggest next analysis",
]


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    status: str  # SUCCESS | FAILURE
    result_summary: str
    started_at: datetime.datetime
    completed_at: datetime.datetime


@dataclass
class HandlerResult:
    response_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    new_hypothesis_id: uuid.UUID | None = None
    suggested_actions: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)


class CopilotState(TypedDict, total=False):
    investigation_id: uuid.UUID
    message: str
    context: dict[str, Any]  # InvestigationContext.model_dump()
    intent: str
    extracted_signal: str | None
    tool_calls: list[dict[str, Any]]
    evidence_refs: list[str]
    new_hypothesis_id: str | None
    response: str
    suggested_actions: list[str]
    unknowns: list[str]
    llm_degraded: bool


def _record(
    tool_name: str, arguments: dict[str, Any], status: str, summary: str, started: datetime.datetime
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=tool_name,
        arguments=arguments,
        status=status,
        result_summary=summary,
        started_at=started,
        completed_at=datetime.datetime.now(datetime.UTC),
    )


def _evidence_by_id(context: InvestigationContext) -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in context.current_evidence}


# --- intent handlers -------------------------------------------------------
# Every handler has the same shape: (session, investigation, context, run_a,
# run_b, message, extracted_signal) -> HandlerResult. Dispatch table at the
# bottom of the module.


def _handle_explain_evidence(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    if not context.current_hypotheses:
        return HandlerResult(
            response_text="No hypothesis has been proposed for this investigation yet — run the agent "
            "or the individual analysis steps first.",
            tool_calls=[_record("get_hypotheses", {}, "SUCCESS", "0 hypotheses found", started)],
            unknowns=["no hypothesis proposed yet"],
        )

    leading = context.current_hypotheses[0]
    evidence_lookup = _evidence_by_id(context)

    wants_contradictions = "contradict" in message.lower()
    ids = leading["contradicting_evidence_ids"] if wants_contradictions else leading["supporting_evidence_ids"]
    cited = [evidence_lookup[eid] for eid in ids if eid in evidence_lookup]

    if wants_contradictions:
        if not cited:
            # PRD_COPILOT_UPDATE.md Section 6: distinguish "no direct
            # contradiction found" from "hypothesis proven" — never imply the
            # latter just because nothing contradicts it yet.
            text = (
                f'No direct contradiction found for hypothesis "{leading["title"]}". This does not mean the '
                "hypothesis is proven — only that no contradicting evidence has been gathered."
            )
        else:
            bullets = "\n".join(f"- {e['content']}" for e in cited)
            text = f'Evidence contradicting "{leading["title"]}":\n{bullets}'
    else:
        if not cited:
            text = f'"{leading["title"]}" (status: {leading["status"]}) has no supporting evidence recorded.'
        else:
            bullets = "\n".join(f"- {e['content']}" for e in cited)
            text = f'"{leading["title"]}" (status: {leading["status"]}) is supported by:\n{bullets}'

    return HandlerResult(
        response_text=text,
        tool_calls=[
            _record(
                "get_evidence", {"hypothesis_id": leading["id"]}, "SUCCESS", f"{len(cited)} evidence item(s)", started
            )
        ],
        evidence_refs=[e["id"] for e in cited],
    )


def _handle_compare_runs(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    result = tools.tool_compare_global_response(session, run_a, run_b)
    if result is None:
        return HandlerResult(
            response_text="Global crash pulse (vehicle_pulse) is not available for one or both runs.",
            tool_calls=[_record("compare_global_response", {}, "FAILURE", "vehicle_pulse unavailable", started)],
            unknowns=["vehicle_pulse signal unavailable"],
        )
    materially_different = result["material_difference_detected"]
    metrics_text = "; ".join(
        f"{m['name']}: Run A={m['run_a_value']:.2f}, Run B={m['run_b_value']:.2f} "
        f"({'materially different' if m['materially_different'] else 'comparable'})"
        for m in result["metrics"]
    )
    text = f"Global crash pulse comparison: {metrics_text}. " + (
        "The global pulse differs materially — per PR-005, resolve this before attributing any "
        "downstream occupant-response difference to a local component."
        if materially_different
        else "The global pulse is comparable between runs."
    )
    return HandlerResult(
        response_text=text,
        tool_calls=[_record("compare_global_response", {}, "SUCCESS", metrics_text, started)],
    )


def _handle_analyze_signal(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    signal_name = extracted_signal or context.primary_metric
    if signal_name is None:
        return HandlerResult(
            response_text="No signal was recognized in that question, and this investigation has no primary "
            "metric set to fall back on.",
            unknowns=["no signal recognized"],
        )

    result = tools.tool_analyze_signal(session, run_a, run_b, signal_name)
    if result is None:
        return HandlerResult(
            response_text=f"Signal '{signal_name}' is not available for one or both runs.",
            tool_calls=[_record("analyze_signal", {"signal": signal_name}, "FAILURE", "signal unavailable", started)],
            unknowns=[f"signal {signal_name} unavailable"],
        )

    divergence = result.get("divergence")
    if divergence is not None:
        text = (
            f"{signal_name}: Run A peak={result['run_a_features']['peak']:.2f}, "
            f"Run B peak={result['run_b_features']['peak']:.2f}, correlation={result['correlation']:.3f}. "
            f"First divergence at {divergence['time_ms']:.1f} ms."
        )
    else:
        text = (
            f"{signal_name}: Run A peak={result['run_a_features']['peak']:.2f}, "
            f"Run B peak={result['run_b_features']['peak']:.2f}, correlation={result['correlation']:.3f}. "
            "No divergence event detected above the configured threshold."
        )
    return HandlerResult(
        response_text=text,
        tool_calls=[_record("analyze_signal", {"signal": signal_name}, "SUCCESS", text, started)],
    )


def _handle_retrieve_knowledge(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    results = tools.tool_retrieve_knowledge(session, message, limit=5)
    if not results:
        # PRD_COPILOT_UPDATE.md Section 8: "If the corpus cannot support the
        # answer, say so" — never fabricate a citation.
        return HandlerResult(
            response_text="No relevant documentation was found in the ingested corpus for that question.",
            tool_calls=[_record("retrieve_knowledge", {"query": message}, "SUCCESS", "0 relevant results", started)],
            unknowns=["no supporting documentation found"],
        )
    lines = [
        f"- {r['document_title']} ({r['authority_level']}), p.{r.get('page_start', '?')}: {r['content'][:200]}"
        for r in results
    ]
    text = "Relevant documentation:\n" + "\n".join(lines)
    return HandlerResult(
        response_text=text,
        tool_calls=[_record("retrieve_knowledge", {"query": message}, "SUCCESS", f"{len(results)} result(s)", started)],
    )


def _handle_retrieve_history(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    metric = context.primary_metric or ""
    cases = tools.tool_retrieve_historical_cases(session, metric, investigation.id)
    if not cases:
        # PRD.md PR-010 / this file's own docstring: no historical-case table
        # is populated yet in V1 — say so, never invent a precedent.
        return HandlerResult(
            response_text="No historical cases are available yet — this system only draws on closed, "
            "prior investigations, and none exist for this primary metric.",
            tool_calls=[_record("retrieve_historical_cases", {}, "SUCCESS", "0 cases", started)],
            unknowns=["no historical cases available"],
        )
    lines = [f"- {c['title']}: {c['finding']}" for c in cases]
    text = "Similar historical cases:\n" + "\n".join(lines)
    return HandlerResult(
        response_text=text,
        tool_calls=[_record("retrieve_historical_cases", {}, "SUCCESS", f"{len(cases)} case(s)", started)],
    )


def _handle_challenge_hypothesis(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    """PRD_COPILOT_UPDATE.md Section 6 / CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md
    Phase 9: engineer input -> alternative hypothesis -> relevant signal
    selection -> deterministic analysis -> evidence -> comparison with
    current hypothesis -> response. Never auto-finalizes anything."""
    started = datetime.datetime.now(datetime.UTC)
    if extracted_signal is None:
        return HandlerResult(
            response_text="An alternative explanation was requested, but no specific signal was recognized "
            "in the message — try naming one (e.g. 'torso rotation').",
            unknowns=["no signal named in challenge"],
        )

    result = tools.tool_analyze_signal(session, run_a, run_b, extracted_signal)
    tool_calls = [
        _record(
            "analyze_signal",
            {"signal": extracted_signal},
            "SUCCESS" if result is not None else "FAILURE",
            "analyzed" if result is not None else "signal unavailable",
            started,
        )
    ]
    if result is None:
        return HandlerResult(
            response_text=f"Could not analyze '{extracted_signal}' — signal unavailable for one or both runs.",
            tool_calls=tool_calls,
            unknowns=[f"signal {extracted_signal} unavailable"],
        )

    divergence = result.get("divergence")
    evidence = Evidence(
        investigation_id=investigation.id,
        evidence_type="CALCULATED",
        source_type="detect_first_divergence",
        content=(
            f"{extracted_signal} diverged at {divergence['time_ms']:.1f} ms (engineer-requested alternative check)"
            if divergence is not None
            else f"{extracted_signal} shows no detected divergence (engineer-requested alternative check)"
        ),
        value=divergence or {"correlation": result["correlation"]},
        calculation_version=result["provenance"]["algorithm_version"],
    )
    session.add(evidence)
    session.flush()

    hypothesis = Hypothesis(
        investigation_id=investigation.id,
        title=f"Alternative: {extracted_signal.replace('_', ' ')} as a contributor",
        description=f"Engineer-proposed alternative explanation, raised in response to: {message}",
        status="PROPOSED",
        confidence_basis="engineer-challenge",
        affected_components={"signal": extracted_signal},
        created_by="engineer_challenge",
    )
    session.add(hypothesis)
    session.flush()
    session.add(HypothesisEvidenceLink(hypothesis_id=hypothesis.id, evidence_id=evidence.id, relationship="SUPPORTS"))
    session.flush()

    comparison = ""
    if context.current_hypotheses:
        leading = context.current_hypotheses[0]
        comparison = (
            f' The current leading hypothesis is "{leading["title"]}" (status: {leading["status"]}); this '
            "alternative has not been compared against it beyond both being recorded — that comparison "
            "requires engineer judgement, not an automatic decision."
        )

    text = (
        f"Recorded an alternative hypothesis: {extracted_signal.replace('_', ' ')} as a contributor. "
        + (
            f"Divergence detected at {divergence['time_ms']:.1f} ms."
            if divergence is not None
            else "No divergence detected."
        )
        + comparison
    )
    return HandlerResult(
        response_text=text,
        tool_calls=tool_calls,
        evidence_refs=[str(evidence.id)],
        new_hypothesis_id=hypothesis.id,
    )


def _handle_request_next_analysis(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    plan = tools.select_signal_plan(context.primary_metric) if context.primary_metric else []
    already_analyzed = set(context.selected_signals)
    remaining = [s for s in plan if s not in already_analyzed]

    unknowns = [
        f"missing evidence: {h['title']}" for h in context.current_hypotheses if not h["supporting_evidence_ids"]
    ]

    if remaining:
        text = f"Signals not yet analyzed for this investigation's plan: {', '.join(remaining)}."
    elif not context.current_hypotheses:
        text = "All planned signals have been analyzed, but no hypothesis has been generated yet — run the agent."
    else:
        text = (
            "All planned signals have been analyzed. Consider requesting a controlled comparison to "
            "isolate the leading hypothesis, or reviewing knowledge/historical evidence."
        )

    return HandlerResult(
        response_text=text,
        tool_calls=[
            _record("select_signal_plan", {"primary_metric": context.primary_metric}, "SUCCESS", text, started)
        ],
        suggested_actions=[f"Analyze {s}" for s in remaining] or STANDARD_SUGGESTED_ACTIONS,
        unknowns=unknowns,
    )


def _handle_show_source(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    documentary = [e for e in context.current_evidence if e["evidence_type"] == "DOCUMENTARY"]
    if not documentary:
        return HandlerResult(
            response_text="No documentary (LS-DYNA/regulatory) evidence has been cited in this investigation yet.",
            unknowns=["no documentary evidence cited"],
        )
    latest = documentary[-1]
    return HandlerResult(
        response_text=f"Source: {latest['content']}",
        tool_calls=[_record("get_evidence", {"evidence_type": "DOCUMENTARY"}, "SUCCESS", "found source", started)],
        evidence_refs=[latest["id"]],
    )


def _handle_controlled_comparison(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    started = datetime.datetime.now(datetime.UTC)
    hypothesis_id = uuid.UUID(context.current_hypotheses[0]["id"]) if context.current_hypotheses else None
    if hypothesis_id is None:
        return HandlerResult(
            response_text="A controlled comparison needs a hypothesis to isolate — none exists for this "
            "investigation yet.",
            unknowns=["no hypothesis to isolate"],
        )
    request = ControlledComparisonRequest(
        investigation_id=investigation.id,
        hypothesis_id=hypothesis_id,
        requested_change={"note": message},
        controlled_variables={},
        reason=message,
        status="RECORDED",
    )
    session.add(request)
    session.flush()
    # BACKEND_SCHEMA.md Section 44: V1 records requests; it does not
    # autonomously launch a commercial solver job.
    text = (
        "Recorded a controlled-comparison request against the current leading hypothesis. This system "
        "does not launch solver jobs automatically — an engineer must execute and submit results."
    )
    return HandlerResult(
        response_text=text,
        tool_calls=[
            _record(
                "create_controlled_comparison_request",
                {"hypothesis_id": str(hypothesis_id)},
                "SUCCESS",
                "recorded",
                started,
            )
        ],
    )


def _handle_general(
    session: Session,
    investigation: Investigation,
    context: InvestigationContext,
    run_a: SimulationRun,
    run_b: SimulationRun,
    message: str,
    extracted_signal: str | None,
) -> HandlerResult:
    leading = context.current_hypotheses[0]["title"] if context.current_hypotheses else "none proposed yet"
    text = (
        f"Investigation state: {context.investigation_state}. Primary metric: {context.primary_metric or 'not set'}. "
        f"Leading hypothesis: {leading}."
    )
    return HandlerResult(response_text=text, suggested_actions=STANDARD_SUGGESTED_ACTIONS)


_HANDLERS = {
    "EXPLAIN_EVIDENCE": _handle_explain_evidence,
    "COMPARE_RUNS": _handle_compare_runs,
    "ANALYZE_SIGNAL": _handle_analyze_signal,
    "ANALYZE_DIVERGENCE": _handle_analyze_signal,  # same tool; response text already covers divergence
    "RETRIEVE_KNOWLEDGE": _handle_retrieve_knowledge,
    "RETRIEVE_HISTORY": _handle_retrieve_history,
    "CHALLENGE_HYPOTHESIS": _handle_challenge_hypothesis,
    "REQUEST_NEXT_ANALYSIS": _handle_request_next_analysis,
    "SHOW_SOURCE": _handle_show_source,
    "CONTROLLED_COMPARISON": _handle_controlled_comparison,
    "GENERAL_INVESTIGATION_QUESTION": _handle_general,
}


def _draft_natural_language(message: str, deterministic_text: str, llm: LLMProvider | None) -> tuple[str, bool]:
    """Returns (text, llm_degraded). The deterministic text is always
    correct/grounded; the LLM's only job is to rephrase it more naturally —
    never to add a fact. TRD.md Section 30: its absence never blocks an
    answer."""
    if llm is None:
        return deterministic_text, False
    try:
        response = llm.complete(
            [
                LLMMessage(
                    role="system",
                    content="Rephrase the following grounded engineering-investigation answer in natural, "
                    "concise prose. Do not add any fact, number, or source not already present in it.",
                ),
                LLMMessage(role="user", content=f"Question: {message}\n\nGrounded answer:\n{deterministic_text}"),
            ],
            max_tokens=300,
        )
        return response.content, False
    except LLMUnavailableError:
        return deterministic_text, True


def build_copilot_graph(session: Session, *, llm: LLMProvider | None = None) -> CompiledStateGraph:  # type: ignore[type-arg]
    def load_context(state: CopilotState) -> dict[str, Any]:
        context = build_investigation_context(session, state["investigation_id"])
        return {"context": context.model_dump(mode="json")}

    def classify(state: CopilotState) -> dict[str, Any]:
        intent: Intent = classify_intent(state["message"])
        return {"intent": intent, "extracted_signal": extract_signal(state["message"])}

    def dispatch(state: CopilotState) -> dict[str, Any]:
        investigation = session.get(Investigation, state["investigation_id"])
        assert investigation is not None
        context = InvestigationContext.model_validate(state["context"])
        run_a = tools.load_run(session, context.run_a["run_id"])
        run_b = tools.load_run(session, context.run_b["run_id"])

        handler = _HANDLERS[state["intent"]]
        result = handler(session, investigation, context, run_a, run_b, state["message"], state.get("extracted_signal"))
        session.commit()

        return {
            "response": result.response_text,
            "tool_calls": [
                {
                    "tool_name": t.tool_name,
                    "arguments": t.arguments,
                    "status": t.status,
                    "result_summary": t.result_summary,
                    "started_at": t.started_at.isoformat(),
                    "completed_at": t.completed_at.isoformat(),
                }
                for t in result.tool_calls
            ],
            "evidence_refs": result.evidence_refs,
            "new_hypothesis_id": str(result.new_hypothesis_id) if result.new_hypothesis_id else None,
            "suggested_actions": result.suggested_actions,
            "unknowns": result.unknowns,
        }

    def ground(state: CopilotState) -> dict[str, Any]:
        text, degraded = _draft_natural_language(state["message"], state["response"], llm)
        return {"response": text, "llm_degraded": degraded}

    graph: StateGraph[CopilotState] = StateGraph(CopilotState)
    graph.add_node("load_context", load_context)
    graph.add_node("classify_intent", classify)
    graph.add_node("dispatch_tools", dispatch)
    graph.add_node("ground_response", ground)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "classify_intent")
    graph.add_edge("classify_intent", "dispatch_tools")
    graph.add_edge("dispatch_tools", "ground_response")
    graph.add_edge("ground_response", END)

    return graph.compile()


def _get_or_create_conversation(session: Session, investigation_id: uuid.UUID) -> CopilotConversation:
    conversation = session.query(CopilotConversation).filter_by(investigation_id=investigation_id).one_or_none()
    if conversation is None:
        conversation = CopilotConversation(investigation_id=investigation_id)
        session.add(conversation)
        session.flush()
    return conversation


def run_copilot_turn(
    session: Session, investigation_id: uuid.UUID, message: str, *, llm: LLMProvider | None = None
) -> CopilotState:
    """One question -> one persisted user message + one persisted assistant
    message (with its CopilotToolCall rows) -> final CopilotState.

    Validates the investigation exists before anything else — an unknown
    investigation_id must fail clearly, not silently create an orphaned
    conversation (CLAUDE_CODE_COPILOT_CHANGE_REQUEST.md Phase 12: "invalid
    investigation" is an explicit required test case).
    """
    if session.get(Investigation, investigation_id) is None:
        raise ValueError(f"investigation not found: {investigation_id}")
    if not message.strip():
        raise ValueError("message must not be empty")

    conversation = _get_or_create_conversation(session, investigation_id)
    session.add(CopilotMessage(conversation_id=conversation.id, role="user", content=message))
    session.commit()

    graph = build_copilot_graph(session, llm=llm)
    final_state: CopilotState = graph.invoke({"investigation_id": investigation_id, "message": message})  # type: ignore[assignment]

    assistant_message = CopilotMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=final_state.get("response", ""),
        evidence_refs=final_state.get("evidence_refs") or [],
        suggested_actions=final_state.get("suggested_actions") or [],
        unknowns=final_state.get("unknowns") or [],
        llm_degraded=final_state.get("llm_degraded", False),
    )
    session.add(assistant_message)
    session.flush()

    for call in final_state.get("tool_calls", []):
        session.add(
            CopilotToolCall(
                message_id=assistant_message.id,
                tool_name=call["tool_name"],
                arguments=call["arguments"],
                status=call["status"],
                result_summary=call["result_summary"],
                started_at=datetime.datetime.fromisoformat(call["started_at"]),
                completed_at=datetime.datetime.fromisoformat(call["completed_at"]),
            )
        )
    session.commit()

    return final_state
