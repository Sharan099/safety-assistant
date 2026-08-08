"""Citation-strict agent loop: plan → tools → reason → synthesize.

Hybrid entry (Layer 1–2 vs 3–5):
- Explicit compare / report tasks keep the free-form tool planner.
- DESIGN_IMPLICATION / APPLICABILITY / CHECKLIST_GEN / RETEST_SCOPE use a
  fixed one-step plan that runs the specialized multi-step pipelines.
- FACTUAL_LOOKUP / COMPLIANCE_CHECK short-circuit to the fast ``answer_question``
  path — never a multi-tool agent exploration.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from agent.citations import citation_coverage, enforce_grounded_text, extract_citations
from agent.planner import detect_mode, plan_for_query_intent, plan_task
from agent.state import AgentResult, AgentStepTrace, PlannedStep
from agent.tools import (
    tool_applicability,
    tool_checklist_gen,
    tool_compare_regulations,
    tool_design_implication,
    tool_draft_report,
    tool_lookup_table,
    tool_retest_scope,
    tool_retrieve,
)
from generation.answer import SourceChunk, answer_question
from generation.llm_client import LLMClient
from observability.context import reset_current_trace, set_current_trace
from observability.trace import new_trace
from retrieval.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

StepCallback = Callable[[AgentStepTrace], None]

_INTENT_TOOLS = frozenset(
    {"design_implication", "applicability", "checklist_gen", "retest_scope"}
)


def _record_step(
    *,
    tool: str,
    args: dict[str, Any],
    result: dict[str, Any],
    t0: float,
) -> AgentStepTrace:
    citations = list(result.get("citations") or extract_citations(str(result.get("text") or "")))
    text = str(result.get("text") or result.get("report_markdown") or result.get("table_markdown") or "")
    cleaned, checks = enforce_grounded_text(text, allowed=citations)
    if cleaned != text:
        result["text"] = cleaned
        if "report_markdown" in result:
            result["report_markdown"] = cleaned
        if "table_markdown" in result and result.get("table_markdown") == text:
            result["table_markdown"] = cleaned
    return AgentStepTrace(
        step_id=str(uuid.uuid4()),
        tool=tool,  # type: ignore[arg-type]
        args=args,
        output_preview=cleaned[:1200],
        citations=citations,
        chunk_ids=list(result.get("chunk_ids") or []),
        claim_checks=checks,
        citation_coverage=citation_coverage(checks),
        latency_ms=round((time.time() - t0) * 1000.0, 2),
        error=None if result.get("ok", True) or result.get("text") else "tool_failed",
    )


def _execute_tool(
    step: PlannedStep,
    *,
    llm: LLMClient,
    evidence_chunks: list[RetrievedChunk],
    evidence_sources: list[SourceChunk],
) -> dict[str, Any]:
    tool = step.tool
    args = dict(step.args or {})
    if tool == "retrieve":
        return tool_retrieve(args.get("regulation"), str(args.get("query") or ""), llm=llm)
    if tool == "lookup_table":
        return tool_lookup_table(
            str(args.get("regulation") or ""),
            str(args.get("criterion") or args.get("query") or ""),
            llm=llm,
        )
    if tool == "compare_regulations":
        return tool_compare_regulations(
            str(args.get("reg_a") or ""),
            str(args.get("reg_b") or ""),
            str(args.get("topic") or ""),
            llm=llm,
        )
    if tool == "draft_report":
        return tool_draft_report(
            args.get("sections") or [],
            title=str(args.get("title") or "Engineering memo"),
            llm=llm,
            evidence_sources=evidence_sources,
            evidence_chunks=evidence_chunks,
        )
    if tool == "design_implication":
        return tool_design_implication(str(args.get("query") or ""), llm=llm)
    if tool == "applicability":
        return tool_applicability(str(args.get("query") or ""), llm=llm)
    if tool == "checklist_gen":
        return tool_checklist_gen(str(args.get("query") or ""), llm=llm)
    if tool == "retest_scope":
        return tool_retest_scope(str(args.get("query") or ""), llm=llm)
    return {"ok": False, "text": f"Unknown tool: {tool}", "citations": [], "chunk_ids": []}


def _merge_sources(
    bag: list[SourceChunk],
    result: dict[str, Any],
    chunks_bag: list[RetrievedChunk],
) -> None:
    for raw in result.get("sources") or []:
        try:
            src = SourceChunk.model_validate(raw) if isinstance(raw, dict) else raw
        except Exception:  # noqa: BLE001
            continue
        if any(s.chunk_id == src.chunk_id for s in bag):
            continue
        bag.append(src)
    for ch in result.get("chunks") or []:
        if isinstance(ch, RetrievedChunk):
            if any(c.chunk_id == ch.chunk_id for c in chunks_bag):
                continue
            chunks_bag.append(ch)


def _result_from_answer(
    *,
    task: str,
    ans: Any,
    mode: str,
    plan: list[PlannedStep],
    steps_out: list[AgentStepTrace],
    trace: Any,
    client: LLMClient,
) -> AgentResult:
    metrics = trace.finalize()
    return AgentResult(
        task=task,
        mode=mode,
        answer=ans.answer or "",
        sources=list(ans.sources or []),
        plan=plan,
        steps=steps_out,
        trace_id=ans.trace_id or trace.trace_id,
        provider=ans.provider or client.provider,
        model=ans.model or client.large_model,
        not_found=bool(ans.not_found),
        overall_citation_coverage=1.0,
        ungrounded_claim_count=0,
        metrics=metrics if isinstance(metrics, dict) else (ans.metrics or {}),
        query_intent=ans.query_intent,
        execution_layer=ans.execution_layer,
        multi_step=bool(ans.multi_step),
        mode_disclaimer=ans.mode_disclaimer,
        mode_disclaimer_title=ans.mode_disclaimer_title,
    )


def run_agent(
    task: str,
    *,
    llm: LLMClient | None = None,
    on_step: StepCallback | None = None,
    max_steps: int = 6,
) -> AgentResult:
    """Plan → execute tools → synthesize, enforcing citations on every step."""
    task = (task or "").strip()
    client = llm or LLMClient()
    trace = new_trace(task)
    token = set_current_trace(trace)
    steps_out: list[AgentStepTrace] = []
    sources: list[SourceChunk] = []
    chunks: list[RetrievedChunk] = []
    table_md = ""
    report_md = ""
    answer = ""
    query_intent: str | None = None
    execution_layer: str | None = None
    multi_step = False
    mode_disclaimer: str | None = None
    mode_disclaimer_title: str | None = None

    try:
        from retrieval.hybrid_layers import (
            ExecutionLayer,
            execution_layer_for,
            layer_public_dict,
        )
        from retrieval.router import classify_query

        heuristic_mode = detect_mode(task)
        routed = classify_query(task, llm=client, use_llm=False, log=False)
        layer = execution_layer_for(routed)
        layer_info = layer_public_dict(routed)
        query_intent = routed.intent.value
        execution_layer = layer_info.get("execution_layer")
        multi_step = bool(layer_info.get("multi_step"))
        trace.optimizations["query_intent"] = routed.to_public_dict()
        trace.optimizations["execution_layer"] = layer_info

        # Explicit compare/report language keeps the free-form agent planner.
        if heuristic_mode in {"compare", "report"}:
            mode, plan = plan_task(task, llm=client)
        elif layer is ExecutionLayer.MULTI_STEP:
            # Bounded Layer 3–5: fixed one-tool plan (design/applicability/checklist/retest).
            mode, plan = plan_for_query_intent(routed.intent.value, task)
            logger.info(
                "hybrid agent multi_step intent=%s tool_plan=%s",
                routed.intent.value,
                [p.tool for p in plan],
            )
        else:
            # Layer 1–2 fast path: never open a multi-tool exploration for factual/compliance.
            mode = "fast_path"
            plan = [
                PlannedStep(
                    tool="retrieve",
                    args={"query": task, "regulation": routed.regulation_id},
                    rationale="Fast-path delegate to answer_question (Layer 1–2)",
                )
            ]
            plan_trace = AgentStepTrace(
                step_id=str(uuid.uuid4()),
                tool="plan",
                args={"mode": mode, "intent": routed.intent.value, "layer": "fast"},
                output_preview="fast_path→answer_question",
                citations=[],
                citation_coverage=1.0,
            )
            steps_out.append(plan_trace)
            if on_step:
                on_step(plan_trace)
            t0 = time.time()
            ans = answer_question(task, llm=client, skip_answer_cache=True)
            step_trace = _record_step(
                tool="retrieve",
                args={"query": task, "fast_path": True},
                result={
                    "ok": not ans.not_found,
                    "text": ans.answer or "",
                    "citations": [s.citation for s in (ans.sources or []) if s.citation],
                    "chunk_ids": [s.chunk_id for s in (ans.sources or []) if s.chunk_id],
                    "sources": [s.model_dump() for s in (ans.sources or [])],
                },
                t0=t0,
            )
            steps_out.append(step_trace)
            if on_step:
                on_step(step_trace)
            return _result_from_answer(
                task=task,
                ans=ans,
                mode=mode,
                plan=plan,
                steps_out=steps_out,
                trace=trace,
                client=client,
            )

        plan_trace = AgentStepTrace(
            step_id=str(uuid.uuid4()),
            tool="plan",
            args={
                "mode": mode,
                "intent": query_intent,
                "layer": execution_layer,
            },
            output_preview=str([p.model_dump() for p in plan])[:1200],
            citations=[],
            citation_coverage=1.0,
        )
        steps_out.append(plan_trace)
        if on_step:
            on_step(plan_trace)

        if not plan:
            result = AgentResult(
                task=task,
                mode=mode,
                answer="No plan produced for empty or unsupported task.",
                not_found=True,
                provider=client.provider,
                trace_id=trace.trace_id,
                query_intent=query_intent,
                execution_layer=execution_layer,
                multi_step=multi_step,
            )
            trace.not_found = True
            result.metrics = trace.finalize()
            return result

        for planned in plan[:max_steps]:
            t0 = time.time()
            tool_result = _execute_tool(
                planned,
                llm=client,
                evidence_chunks=chunks,
                evidence_sources=sources,
            )
            _merge_sources(sources, tool_result, chunks)
            if tool_result.get("mode_disclaimer"):
                mode_disclaimer = str(tool_result.get("mode_disclaimer"))
                mode_disclaimer_title = str(
                    tool_result.get("mode_disclaimer_title") or ""
                ) or None
            if tool_result.get("execution_layer"):
                execution_layer = str(tool_result.get("execution_layer"))
                multi_step = bool(tool_result.get("multi_step"))
            if tool_result.get("query_intent"):
                query_intent = str(tool_result.get("query_intent"))
            step_trace = _record_step(
                tool=planned.tool,
                args=planned.args,
                result=tool_result,
                t0=t0,
            )
            steps_out.append(step_trace)
            if on_step:
                on_step(step_trace)

            if planned.tool == "compare_regulations":
                table_md = str(tool_result.get("table_markdown") or "")
                answer = str(tool_result.get("text") or "")
            elif planned.tool == "draft_report":
                report_md = str(tool_result.get("report_markdown") or tool_result.get("text") or "")
                answer = report_md
            elif planned.tool in {"retrieve", "lookup_table"} | _INTENT_TOOLS:
                answer = str(tool_result.get("text") or answer)

        # Final synthesize only for free-form qa (not Layer 3–5 intent tools —
        # those already returned a finished grounded answer).
        if mode == "qa" and answer and not any(p.tool in _INTENT_TOOLS for p in plan):
            allowed = [s.citation for s in sources if s.citation]
            answer, checks = enforce_grounded_text(answer, allowed=allowed)
            syn = AgentStepTrace(
                step_id=str(uuid.uuid4()),
                tool="synthesize",
                args={},
                output_preview=answer[:1200],
                citations=allowed,
                chunk_ids=[s.chunk_id for s in sources],
                claim_checks=checks,
                citation_coverage=citation_coverage(checks),
            )
            steps_out.append(syn)
            if on_step:
                on_step(syn)

        all_checks = [c for st in steps_out for c in st.claim_checks]
        ungrounded = sum(1 for c in all_checks if not c.grounded)
        overall = citation_coverage(all_checks) if all_checks else 1.0
        not_found = not sources and not answer

        trace.chunk_ids = [s.chunk_id for s in sources]
        trace.citations = [
            {
                "chunk_id": s.chunk_id,
                "citation": s.citation,
                "regulation_id": s.regulation_id,
                "section_number": s.section_number,
                "page_number": s.page_number,
            }
            for s in sources
        ]
        trace.not_found = not_found
        trace.faithfulness_passed = overall >= 0.95 and ungrounded == 0
        metrics = trace.finalize()

        return AgentResult(
            task=task,
            mode=mode,
            answer=answer,
            table_markdown=table_md,
            report_markdown=report_md,
            sources=sources,
            plan=plan,
            steps=steps_out,
            trace_id=trace.trace_id,
            provider=client.provider,
            model=client.large_model,
            not_found=not_found,
            overall_citation_coverage=overall,
            ungrounded_claim_count=ungrounded,
            metrics=metrics,
            query_intent=query_intent,
            execution_layer=execution_layer,
            multi_step=multi_step,
            mode_disclaimer=mode_disclaimer,
            mode_disclaimer_title=mode_disclaimer_title,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failed")
        trace.error = str(exc)
        metrics = trace.finalize()
        return AgentResult(
            task=task,
            answer=f"Agent error: {exc}",
            steps=steps_out,
            trace_id=trace.trace_id,
            provider=client.provider,
            not_found=True,
            metrics=metrics,
            query_intent=query_intent,
            execution_layer=execution_layer,
            multi_step=multi_step,
        )
    finally:
        reset_current_trace(token)
