"""Bounded LangGraph orchestration (CLAUDE.md §9).

    START → parse_query → route_intent
        ├─ standard        (exact_lookup / technical_qa / definition / historical)
        ├─ comparison      one scoped retrieval per named regulation, merged evidence
        └─ change_analysis section diff between the two latest versions (+ retrieval)
      → evidence_gate ── weak & budget left → rewrite → retrieve once more
                      ── insufficient → abstain
                      ── sufficient → generate → citation_validate → END

Deterministic code decides routing, scope, dates, budgets and validation; the
LLM only synthesises from evidence under schema. Budgets are enforced by the
tool layer (`agents/tools`) and by `Budget.check_deadline`.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from safety_assistant.agents.state import AgentState, Budget, BudgetExceeded
from safety_assistant.agents.tools import Tools
from safety_assistant.domain.temporal import parse_query_scope
from safety_assistant.generation.citations import validate_draft
from safety_assistant.generation.grounding import evaluate_gate
from safety_assistant.generation.prompts.grounded_v1 import PROMPT_VERSION, SYSTEM, build_user_message
from safety_assistant.generation.schemas import GroundedDraft
from safety_assistant.observability import metrics, span
from safety_assistant.providers.llm import LLMError, LLMMessage, LLMProvider
from safety_assistant.retrieval import RetrievalService, ScopeFilter
from safety_assistant.retrieval.context import Evidence
from safety_assistant.security import injection_signals

log = logging.getLogger(__name__)
MAX_COMPARED_REGULATIONS = 2


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


class RegulatoryAgent:
    def __init__(
        self,
        session: Session,
        retrieval: RetrievalService,
        llm: LLMProvider | None,
        budget: Budget | None = None,
        llm_data_classes: tuple[str, ...] = ("PUBLIC",),
    ):
        self.tools = Tools(session, retrieval)
        self.session = session
        self.llm = llm
        self.budget = budget or Budget()
        self.llm_data_classes = frozenset(llm_data_classes)
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes

    def parse_query(self, state: AgentState) -> AgentState:
        qs = parse_query_scope(state["query"], today=state.get("today"))
        base = state["base_scope"]
        scope = dataclasses.replace(
            base,
            as_of=base.as_of or qs.as_of,
            regulation_keys=base.regulation_keys or tuple(qs.regulation_keys),
            include_superseded=base.include_superseded or qs.historical_hint or qs.change_hint,
        )
        signals = injection_signals(state["query"])
        warnings = list(state.get("warnings", []))
        if signals:
            warnings.append("prompt-injection pattern detected in question; instructions in it were ignored")
        return {
            **state,
            "intent": qs.intent,
            "scope": scope,
            "regulation_keys": list(scope.regulation_keys),
            "warnings": warnings,
        }

    def route_intent(self, state: AgentState) -> AgentState:
        intent, keys = state["intent"], state["regulation_keys"]
        if intent == "comparison" and len(keys) >= 2:
            route = "comparison"
        elif intent == "change_analysis" and len(keys) == 1 and self.tools.regulation_exists(keys[0]):
            route = "change_analysis"
        else:
            route = "standard"
        return {**state, "route": route}

    def retrieve_standard(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        query = state["rewrites"][-1] if state.get("rewrites") else state["query"]
        result = self.tools.search_regulations(state, query, state["scope"])
        timings = {**state.get("timings", {}), f"retrieval_{state['retrieval_attempts']}": _ms(t0)}
        return {**state, "retrieval": result, "evidence": result.bundle.evidence, "timings": timings}

    def retrieve_comparison(self, state: AgentState) -> AgentState:
        """One scoped retrieval per regulation so neither text starves the other."""
        t0 = time.perf_counter()
        subs = []
        merged: list[Evidence] = []
        per_reg_k = max(3, (state.get("k") or 8) // 2)
        for key in state["regulation_keys"][:MAX_COMPARED_REGULATIONS]:
            scope = dataclasses.replace(state["scope"], regulation_keys=(key,))
            r = self.tools.search_regulations(state, state["query"], scope, k=per_reg_k)
            subs.append(r)
            merged.extend(r.bundle.evidence)
        for i, e in enumerate(merged, start=1):  # stable ids across the merged bundle
            e.evidence_id = f"E{i}"
        warnings = list(state["warnings"])
        if len(state["regulation_keys"]) > MAX_COMPARED_REGULATIONS:
            warnings.append(f"comparison limited to the first {MAX_COMPARED_REGULATIONS} regulations named")
        primary = subs[0] if subs else None
        return {
            **state,
            "retrieval": primary,
            "sub_results": subs,
            "evidence": merged,
            "timings": {**state.get("timings", {}), "retrieval_comparison": _ms(t0)},
            "warnings": warnings,
        }

    def retrieve_change_analysis(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        key = state["regulation_keys"][0]
        d = self.tools.compare_versions(state, key)
        warnings = list(state["warnings"])
        extra = None
        scope = dataclasses.replace(state["scope"], include_superseded=True)
        if d is None:
            warnings.append(
                f"only one verified version of {key} is in the corpus; a text-level change analysis is not possible — "
                "the amendment chain from the cover page is reported instead"
            )
        else:
            lines = [f'<change_analysis regulation="{d.regulation_key}" from="{d.from_version}" to="{d.to_version}">']
            lines.append(f"added: {', '.join(c.path for c in d.added) or 'none'}")
            lines.append(f"removed: {', '.join(c.path for c in d.removed) or 'none'}")
            lines.append(f"changed: {', '.join(c.path for c in d.changed) or 'none'}; unchanged: {d.unchanged}")
            for c in (d.changed + d.added)[:12]:
                lines.append(f'<section path="{c.path}" kind="{c.kind}" normative="{c.normative_after}">')
                lines.append(c.diff or "(new text)")
                lines.append("</section>")
            lines.append("</change_analysis>")
            extra = "\n".join(lines)
            # focus retrieval on the changed clauses of the newer version
            scope = dataclasses.replace(scope, version_ids=(str(d.to_version_id),))
        result = self.tools.search_regulations(state, state["query"], scope)
        return {
            **state,
            "retrieval": result,
            "evidence": result.bundle.evidence,
            "extra_context": extra,
            "warnings": warnings,
            "timings": {**state.get("timings", {}), "retrieval_change": _ms(t0)},
        }

    def evidence_gate(self, state: AgentState) -> AgentState:
        result = state["retrieval"]
        assert result is not None
        retries_left = self.budget.max_retrieval_attempts - state["retrieval_attempts"]
        if state["route"] != "standard":
            retries_left = 0  # comparison/change routes already spent their budget deliberately
        query = state["rewrites"][-1] if state.get("rewrites") else state["query"]
        decision = evaluate_gate(
            self.session,
            query,
            result.query_scope,
            state["evidence"],
            as_of=state["scope"].as_of,
            retries_left=retries_left,
        )
        warnings = state["warnings"] + decision.warnings
        rewrites = list(state.get("rewrites", []))
        if decision.proceed and decision.rewrite:
            rewrites.append(decision.rewrite)
        return {**state, "decision": decision, "warnings": warnings, "rewrites": rewrites}

    def generate(self, state: AgentState) -> AgentState:
        if self.llm is None:
            return {
                **state,
                "mode": "EVIDENCE_ONLY",
                "warnings": state["warnings"] + ["no LLM configured: evidence-only mode"],
            }
        confidential = [e for e in state["evidence"] if e.data_class not in self.llm_data_classes]
        if confidential:
            return {
                **state,
                "mode": "EVIDENCE_ONLY",
                "warnings": state["warnings"]
                + [
                    "evidence includes data classes the configured LLM provider is not cleared for "
                    f"({sorted({e.data_class for e in confidential})}): evidence-only mode"
                ],
            }
        if state.get("llm_calls", 0) >= self.budget.max_llm_calls:
            return {
                **state,
                "mode": "EVIDENCE_ONLY",
                "warnings": state["warnings"] + ["LLM call budget exhausted: evidence-only mode"],
            }
        self.budget.check_deadline(state["started"])
        scope = state["scope"]
        scope_note = (
            f"intent={state['intent']}; route={state['route']}; "
            f"regulations={','.join(state['regulation_keys']) or 'any'}; as_of={scope.as_of or 'current'}"
        )
        user = build_user_message(state["query"], state["evidence"], scope_note)
        if state.get("extra_context"):
            user = user.replace("<question>", f"{state['extra_context']}\n\n<question>", 1)
        if state.get("project_context"):
            pc = f"<project_context>\n{state['project_context']}\n</project_context>"
            user = user.replace("<question>", f"{pc}\n\n<question>", 1)
        if state.get("conversation_context"):
            ctx = f"<conversation_context>\n{state['conversation_context']}\n</conversation_context>"
            user = user.replace("<question>", f"{ctx}\n\n<question>", 1)
        t0 = time.perf_counter()
        try:
            with span("llm.generate", provider=self.llm.name, model=self.llm.model):
                resp = self.llm.generate(
                    [LLMMessage(role="system", content=SYSTEM), LLMMessage(role="user", content=user)],
                    schema=GroundedDraft,
                    temperature=0.0,
                    max_tokens=2000,  # reasoning routes count their thinking against this budget
                )
        except LLMError as exc:
            metrics.LLM_CALLS.labels(provider=self.llm.name, outcome=type(exc).__name__).inc()
            metrics.STAGE_LATENCY.labels(stage="llm").observe(time.perf_counter() - t0)
            log.warning("llm failure trace=%s err=%s", state.get("trace_id"), exc)
            return {
                **state,
                "llm_calls": state.get("llm_calls", 0) + 1,
                "mode": "EVIDENCE_ONLY",
                "warnings": state["warnings"] + [f"generation unavailable ({type(exc).__name__}): evidence-only mode"],
                "timings": {**state["timings"], "llm": _ms(t0)},
            }
        draft: GroundedDraft = resp.parsed  # type: ignore[assignment]
        metrics.LLM_CALLS.labels(provider=resp.provider, outcome="ok").inc()
        metrics.STAGE_LATENCY.labels(stage="llm").observe(time.perf_counter() - t0)
        for kind in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if resp.usage and kind in resp.usage:
                metrics.LLM_TOKENS.labels(provider=resp.provider, kind=kind).inc(resp.usage[kind])
        versions = {**state.get("versions", {}), "llm_model": resp.model, "llm_provider": resp.provider}
        return {
            **state,
            "draft": draft,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "versions": versions,
            "tokens": resp.usage,
            "warnings": state["warnings"] + list(draft.warnings),
            "timings": {**state["timings"], "llm": _ms(t0)},
        }

    def citation_validate(self, state: AgentState) -> AgentState:
        draft = state.get("draft")
        if draft is None:
            return state
        if not draft.claims:
            return {**state, "mode": "ABSTAINED", "abstain_reason": "weak_evidence", "message": draft.answer}
        kept, report = validate_draft(draft, state["evidence"])
        warnings = list(state["warnings"])
        if draft.insufficient_evidence:
            # The model flagged partial coverage but still made cited claims: the validator decides what
            # survives; the flag becomes a warning instead of discarding a grounded answer.
            warnings.append("the model reports that the evidence covers the question only partly")
        if any(c.kind == "CALCULATION" for c in kept):
            warnings.append("contains a value derived from the cited evidence (calculation shown): verify before use")
        if report.dropped_claims:
            metrics.CITATION_FAILURES.inc(report.dropped_claims)
            warnings.append(f"{report.dropped_claims} claim(s) removed: failed citation/numeric validation")
        if not kept:
            return {
                **state,
                "validation": report,
                "warnings": warnings,
                "mode": "ABSTAINED",
                "abstain_reason": "validation_failed",
                "message": "The generated answer could not be verified against the evidence and was withheld.",
            }
        return {
            **state,
            "validation": report,
            "warnings": warnings,
            "draft": GroundedDraft(answer=draft.answer, claims=kept, warnings=draft.warnings),
            "mode": "GENERATED",
        }

    def abstain(self, state: AgentState) -> AgentState:
        d = state["decision"]
        assert d is not None
        return {**state, "mode": "ABSTAINED", "abstain_reason": d.abstain_reason, "message": d.message}

    # ------------------------------------------------------------------ edges

    @staticmethod
    def _after_route(state: AgentState) -> str:
        return state["route"]

    @staticmethod
    def _after_gate(state: AgentState) -> str:
        d = state["decision"]
        assert d is not None
        if not d.proceed:
            return "abstain"
        if d.rewrite:
            return "retry"
        return "generate"

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        for name in (
            "parse_query",
            "route_intent",
            "retrieve_standard",
            "retrieve_comparison",
            "retrieve_change_analysis",
            "evidence_gate",
            "generate",
            "citation_validate",
            "abstain",
        ):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("parse_query")
        g.add_edge("parse_query", "route_intent")
        g.add_conditional_edges(
            "route_intent",
            self._after_route,
            {
                "standard": "retrieve_standard",
                "comparison": "retrieve_comparison",
                "change_analysis": "retrieve_change_analysis",
            },
        )
        for n in ("retrieve_standard", "retrieve_comparison", "retrieve_change_analysis"):
            g.add_edge(n, "evidence_gate")
        g.add_conditional_edges(
            "evidence_gate",
            self._after_gate,
            {"abstain": "abstain", "retry": "retrieve_standard", "generate": "generate"},
        )
        g.add_edge("generate", "citation_validate")
        g.add_edge("citation_validate", END)
        g.add_edge("abstain", END)
        return g.compile()

    # ------------------------------------------------------------------ run

    def run(
        self,
        query: str,
        *,
        scope: ScopeFilter | None = None,
        k: int | None = None,
        today: Any = None,
        trace_id: str | None = None,
        conversation_context: str | None = None,
        project_context: str | None = None,
    ) -> AgentState:
        initial: AgentState = {
            "query": query,
            "conversation_context": conversation_context,
            "project_context": project_context,
            "base_scope": scope or ScopeFilter(),
            "k": k,
            "today": today,
            "trace_id": trace_id or uuid.uuid4().hex,
            "started": time.perf_counter(),
            "budget": self.budget,
            "rewrites": [],
            "sub_results": [],
            "evidence": [],
            "retrieval_attempts": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "timings": {},
            "warnings": [],
            "versions": {"prompt_version": PROMPT_VERSION},
            "extra_context": None,
            "decision": None,
            "draft": None,
            "validation": None,
            "retrieval": None,
            "tokens": None,
            "mode": "EVIDENCE_ONLY",
            "abstain_reason": None,
            "message": None,
        }
        try:
            with span("agent.run", trace_id=initial["trace_id"]):
                final: AgentState = self.graph.invoke(initial, config={"recursion_limit": 25})
        except BudgetExceeded as exc:
            log.warning("agent budget exceeded trace=%s: %s", initial["trace_id"], exc)
            final = {
                **initial,
                "mode": "ABSTAINED",
                "abstain_reason": "generation_unavailable",
                "message": f"Request exceeded its processing budget ({exc}).",
                "warnings": [str(exc)],
            }
        final["timings"] = {**final.get("timings", {}), "total": _ms(initial["started"])}
        metrics.ANSWERS.labels(
            mode=final.get("mode", "?"),
            abstain_reason=final.get("abstain_reason") or "",
            route=final.get("route") or "",
        ).inc()
        metrics.STAGE_LATENCY.labels(stage="answer_total").observe(final["timings"]["total"] / 1000)
        return final
