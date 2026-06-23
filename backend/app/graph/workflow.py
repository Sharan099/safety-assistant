"""
LangGraph RAG workflow:
  validate_input → retrieve → rerank → build_prompt → generate → validate_output
"""

import os
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from backend.app.core.observability import trace_run, trace_span
from backend.app.core.settings import (
    ENABLE_GROUNDING_GATE,
    GROUNDING_MIN_RERANK_PROB,
    GROUNDING_MIN_SEMANTIC,
    INJECTION_BLOCKED_MESSAGE,
    LOW_GROUNDING_ABSTAIN_MESSAGE,
)
from backend.app.guardrails.validator import SafetyGuardrails
from backend.app.llm.groq_client import GroqLLM
from backend.app.retrieval.citations import (
    assess_grounding,
    build_citations,
    derive_answer_flags,
    detect_category_value_misattribution,
    detect_knowledge_boundary_flags,
    detect_scope_overclaim_flags,
    enrich_doc_provenance,
    validate_category_citation_authority,
)
from backend.app.gateway.errors import (
    GENERATION_UNAVAILABLE_MESSAGE,
    is_raw_provider_error,
    sanitize_user_answer,
)
from backend.app.graph.prompt_budget import (
    estimate_tokens,
    passage_char_budget,
    strip_chunk_boilerplate,
)
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker

# Low-grounding abstain — short, distinct from injection block.
LOW_GROUNDING_ABSTAIN_REPLY = LOW_GROUNDING_ABSTAIN_MESSAGE

# Legacy alias kept for imports/tests that reference ABSTAIN_REPLY.
ABSTAIN_REPLY = LOW_GROUNDING_ABSTAIN_REPLY


class RAGState(TypedDict, total=False):
    query: str
    documents: list[dict]
    citations: list[dict]
    flags: list[dict]
    grounding: dict
    context: str
    prompt: str
    answer: str
    guardrails_input: dict
    guardrails_output: dict
    timing: dict
    metadata: dict
    error: str


def _generation_failed_state(state: RAGState, answer: str, error_code: str) -> RAGState:
    """Strip misleading grounding/citations when generation did not succeed."""
    meta = {**(state.get("metadata") or {}), "response_state": "generation_failed"}
    grounding = dict(state.get("grounding") or {})
    grounding.pop("confidence_band", None)
    grounding["generation_failed"] = True
    return {
        **state,
        "answer": sanitize_user_answer(answer, generation_failed=True),
        "error": error_code,
        "citations": [],
        "flags": [],
        "grounding": grounding,
        "metadata": meta,
    }


def _build_grounded_context(documents: list[dict], citations: list[dict]) -> str:
    """
    Build context where every passage is prefixed with a citation marker [S#]
    and its provenance, grouped so the LLM never blurs legal regulations with
    rating protocols.
    """
    from config import MAX_CONTEXT_TOKENS

    legal: list[str] = []
    rating: list[str] = []
    reference: list[str] = []
    char_cap = passage_char_budget(MAX_CONTEXT_TOKENS, len(documents), overhead_tokens=120)

    for d, c in zip(documents, citations):
        raw = strip_chunk_boilerplate(d.get("text", "") or "")
        text = raw[:char_cap]
        parent = strip_chunk_boilerplate((d.get("parent_context", "") or "").strip())
        block = (
            f"[{c['marker']}] {c['label']} ({c['doc_type_label']})\n"
            f"{text}"
        )
        if parent and parent[:60] not in text:
            block += f"\n[section context] {parent[: min(300, char_cap // 2)]}"

        if c["doc_type"] == "legal_regulation":
            legal.append(block)
        elif c["doc_type"] == "rating_protocol":
            rating.append(block)
        else:
            reference.append(block)

    parts: list[str] = []
    if legal:
        parts.append("=== LEGAL REGULATIONS (binding) ===\n" + "\n\n".join(legal))
    if rating:
        parts.append(
            "=== RATING PROTOCOLS (consumer assessment, NOT legally binding) ===\n"
            + "\n\n".join(rating)
        )
    if reference:
        parts.append(
            "=== ENGINEERING REFERENCES (non-binding) ===\n" + "\n\n".join(reference)
        )
    return "\n\n".join(parts)


# User-turn instructions: context delivery (persona + structure live in config.SYSTEM_PROMPT).
_ANSWER_RULES = (
    "Answer using ONLY the retrieved [S#] passages below. Follow CONDITIONAL FORMAT "
    "and APPLICABILITY VERIFICATION rules from your system instructions.\n"
    "Before quoting any load/duration number: verify vehicle category and anchorage "
    "test type match the question; if ambiguous, ask or break down by category — "
    "never pick one clause arbitrarily.\n"
    "For definition/compare queries: quote each term's clause separately; verify text "
    "differs before claiming definitions are identical; do not convert units unless "
    "the source states the converted value.\n"
    "When 3+ passages have DIFFERENT applicability headers: build a category→value "
    "mapping table FIRST (category: value [S#]) using ONLY passages whose applicability "
    "header matches that category — then write prose. Exclude any passage whose header "
    "explicitly excludes the category, even if the category name appears in contrast text.\n"
    "Label ANY claim not traceable to [S#] as 'General knowledge (outside retrieved corpus):' "
    "— including vehicle-classification facts used to interpret the question.\n"
    "State the narrowest applicable scope (vehicle category, seat type, test direction) "
    "in the opening sentence when evidence is subset-specific.\n"
    "For a single-value lookup: one line + [S#]. For analysis: use structured sections "
    "only when content exists — omit empty sections.\n"
    "Never blur Legal (UN/ECE, FMVSS) with Rating (Euro NCAP) or Reference handbooks."
)


def _build_prompt(query: str, context: str, mode_name: str | None = None) -> str:
    from backend.app.graph.prompt_templates import template_for

    mode_rules = template_for(mode_name)
    return f"""{_ANSWER_RULES}

MODE INSTRUCTIONS: {mode_rules}

RETRIEVED CONTEXT (each passage tagged [S#])
{context}

QUESTION: {query}
"""


class RAGWorkflow:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self._llm = None
        self._llm_is_gateway = False
        self.guardrails = SafetyGuardrails()
        self.graph = self._build_graph()

    def _get_llm(self):
        """Return the LLM client behind the generate node.

        When ENABLE_GATEWAY is set, this is the Intelligent Multi-LLM Gateway
        (Groq -> Claude Haiku -> Claude Sonnet, with cache + failover). Otherwise
        it is the original GroqLLM — preserving v2.2 behaviour exactly.
        """
        if self._llm is None:
            from backend.app.core.settings import ENABLE_GATEWAY

            if ENABLE_GATEWAY:
                try:
                    from backend.app.core.services import get_gateway

                    gateway = get_gateway()
                    if gateway is not None:
                        logger.info("Workflow LLM: Intelligent Multi-LLM Gateway")
                        self._llm = gateway
                        self._llm_is_gateway = True
                        return self._llm
                except Exception as exc:
                    logger.warning(
                        f"Gateway init failed, falling back to GroqLLM: {exc}"
                    )

            if not os.getenv("GROQ_API_KEY"):
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to .env in the project root."
                )
            logger.info("Connecting to Groq LLM...")
            self._llm = GroqLLM()
            self._llm_is_gateway = False
        return self._llm

    def _build_routing_context(self, state: RAGState):
        """Assemble the gateway routing inputs from existing pipeline state.

        Reuses the already-computed grounding assessment and retrieval signals,
        and reads conversation depth / feedback history from the store. All
        reads are best-effort and never break a chat request.
        """
        from backend.app.core import store
        from backend.app.gateway.types import RoutingContext

        meta = state.get("metadata") or {}
        query = state.get("query", "")
        grounding = state.get("grounding") or {}
        docs = state.get("documents") or []
        user_id = meta.get("user_id")
        session_id = meta.get("session_id")

        try:
            depth = store.session_depth(session_id)
        except Exception:
            depth = 0
        try:
            downvote = store.recent_downvote_rate(user_id)
        except Exception:
            downvote = 0.0
        try:
            perf = store.model_performance()
        except Exception:
            perf = {}
        try:
            scope = self.retriever.detect_regs(query)
        except Exception:
            scope = []

        return RoutingContext(
            prompt=state.get("prompt", ""),
            query=query,
            grounding=grounding,
            retrieval={
                "doc_count": len(docs),
                "best_semantic": grounding.get("best_semantic"),
                "reranker_used": meta.get("reranker_used"),
                "citations": state.get("citations"),
            },
            conversation_depth=depth,
            user_id=user_id,
            session_id=session_id,
            feedback_downvote_rate=downvote,
            model_performance=perf,
            scope=scope,
            mode=meta.get("mode"),
            llm_tier_floor=int(meta.get("llm_tier_floor") or 1),
        )

    def _node_validate_input(self, state: RAGState) -> RAGState:
        with trace_span("validate_input", {"query": state.get("query", "")}) as span:
            gr = self.guardrails.validate_input(state["query"])
            span["outputs"] = self.guardrails.to_dict(gr)
            timing = state.get("timing", {})
            response_state = gr.input_state  # injection_blocked | answerable
            return {
                **state,
                "guardrails_input": self.guardrails.to_dict(gr),
                "timing": timing,
                "metadata": {
                    **(state.get("metadata") or {}),
                    "input_blocked": gr.blocked,
                    "response_state": response_state,
                },
            }

    def _node_retrieve(self, state: RAGState) -> RAGState:
        if state.get("metadata", {}).get("input_blocked"):
            return state
        meta = state.get("metadata") or {}
        logger.info(f"Retrieving for: {state['query'][:80]}...")
        with trace_span("retrieve", {"query": state["query"]}) as span:
            result = self.retriever.retrieve(
                state["query"],
                mode=meta.get("mode"),
            )
            span["outputs"] = {
                "doc_count": len(result["documents"]),
                "semantic": result["semantic_count"],
                "bm25": result["bm25_count"],
                "queries": result.get("queries", []),
                "intent_flags": result.get("intent_flags", []),
                "query_breadth": result.get("query_breadth", {}),
            }
            timing = {**state.get("timing", {}), "retrieval_ms": result["latency_ms"]}
            meta = {
                **(state.get("metadata") or {}),
                "query_breadth": result.get("query_breadth", {}),
                "retrieved_chunk_count": len(result["documents"]),
            }
            return {**state, "documents": result["documents"], "timing": timing, "metadata": meta}

    def _node_rerank(self, state: RAGState) -> RAGState:
        if state.get("metadata", {}).get("input_blocked"):
            return state
        logger.info(f"Reranking {len(state.get('documents', []))} candidates...")
        with trace_span("rerank", {"candidates": len(state.get("documents", []))}) as span:
            meta = state.get("metadata") or {}
            mode_name = meta.get("mode")
            from backend.app.core.modes import get_mode

            mode_cfg = get_mode(mode_name)
            from backend.app.core.settings import TOP_K_AFTER_RERANK
            from backend.app.retrieval.query_breadth import assess_query_breadth, effective_rerank_k

            breadth = assess_query_breadth(state["query"])
            rerank_k = effective_rerank_k(breadth, default_k=TOP_K_AFTER_RERANK)
            result = self.reranker.rerank(
                state["query"],
                state.get("documents", []),
                force_strong=mode_cfg.force_strong_reranker,
                top_k=rerank_k,
            )
            span["outputs"] = {"top_k": len(result["documents"])}
            timing = {**state.get("timing", {}), "rerank_ms": result["latency_ms"]}
            meta = {
                **(state.get("metadata") or {}),
                "reranker_used": result.get("reranker_used", False),
                "rerank_k": rerank_k,
                "post_rerank_chunk_count": len(result["documents"]),
            }
            return {
                **state,
                "documents": result["documents"],
                "timing": timing,
                "metadata": meta,
            }

    def _node_build_prompt(self, state: RAGState) -> RAGState:
        if state.get("metadata", {}).get("input_blocked"):
            return state
        docs = state.get("documents", [])
        meta = state.get("metadata") or {}
        mode_name = meta.get("mode")
        from backend.app.core.modes import get_mode

        mode_cfg = get_mode(mode_name)
        chunk_lookup = getattr(self.retriever, "_chunk_by_id", {}) or {}
        docs = [enrich_doc_provenance(d, chunk_lookup) for d in docs]

        # Build structured citations for every retrieved passage.
        citations = build_citations(docs, chunk_lookup) if docs else []

        # Grounding gate: abstain if retrieval confidence is too low.
        grounding = assess_grounding(
            docs,
            reranker_used=bool(meta.get("reranker_used")),
            min_semantic=mode_cfg.grounding_min_semantic,
            min_rerank_prob=mode_cfg.grounding_min_rerank_prob,
        )
        abstain = ENABLE_GROUNDING_GATE and grounding.get("should_abstain", False)

        context = _build_grounded_context(docs, citations) if docs else ""
        if abstain:
            logger.warning(
                f"Grounding gate abstaining: reason={grounding.get('reason')} "
                f"confidence={grounding.get('confidence')}"
            )
            # On abstention there is no grounded answer: suppress sources/flags.
            citations = []

        prompt = (
            _build_prompt(state["query"], context, mode_name=mode_name)
            if context and not abstain
            else ""
        )
        if not context:
            logger.warning("No documents after retrieval/rerank")
        # Answer-level flags are derived in run() once the final answer is known,
        # so they can be scoped to the markers the answer actually cited.
        return {
            **state,
            "context": context,
            "prompt": prompt,
            "citations": citations,
            "flags": [],
            "grounding": grounding,
            "metadata": {
                **meta,
                "abstain": abstain,
                "response_state": "low_grounding_abstain" if abstain else "answerable",
            },
        }

    def _node_generate(self, state: RAGState) -> RAGState:
        meta = state.get("metadata") or {}
        if meta.get("input_blocked"):
            return {
                **state,
                "answer": INJECTION_BLOCKED_MESSAGE,
                "metadata": {**meta, "response_state": "injection_blocked"},
            }
        if meta.get("abstain"):
            return {
                **state,
                "answer": LOW_GROUNDING_ABSTAIN_REPLY,
                "metadata": {**meta, "response_state": "low_grounding_abstain"},
            }
        if not state.get("context"):
            return {
                **state,
                "answer": LOW_GROUNDING_ABSTAIN_REPLY,
                "metadata": {**meta, "response_state": "low_grounding_abstain"},
            }
        logger.info("Calling LLM...")
        with trace_span("generate", {"prompt_len": len(state.get("prompt", ""))}) as span:
            try:
                llm = self._get_llm()
                if self._llm_is_gateway:
                    ctx = self._build_routing_context(state)
                    out = llm.generate(state["prompt"], routing_context=ctx)
                else:
                    out = llm.generate(state["prompt"])
            except Exception as exc:
                logger.exception("LLM generation failed")
                return _generation_failed_state(state, str(exc), "llm_exception")

            span["outputs"] = {
                "answer_len": len(out["answer"]),
                "model": out.get("model"),
                "provider": out.get("provider"),
                "tier": out.get("tier"),
                "cache_hit": out.get("cache_hit"),
            }
            if out.get("generation_failed") or out.get("error"):
                return _generation_failed_state(
                    state, out.get("answer", ""), out.get("error", "generation_failed")
                )
            if is_raw_provider_error(out.get("answer", "")):
                return _generation_failed_state(state, out["answer"], "raw_provider_error")

            timing = {**state.get("timing", {}), "llm_ms": out["latency_ms"]}
            # Additive routing metadata (only present when the gateway is active).
            gw_meta = {
                k: out[k]
                for k in (
                    "model",
                    "provider",
                    "tier",
                    "cache_hit",
                    "fallback_used",
                    "route_score",
                    "route_reasons",
                    "cost_usd",
                    "cost_saved_usd",
                    "prompt_tokens",
                    "completion_tokens",
                    "routing_diagnostic",
                )
                if out.get(k) is not None
            }
            logger.info(f"LLM answer ready in {out['latency_ms']}ms")
            from backend.app.guardrails.output_sanitizer import sanitize_model_output

            answer, sanitize_warnings = sanitize_model_output(out["answer"])
            if sanitize_warnings:
                meta = {**(state.get("metadata") or {}), "sanitize_warnings": sanitize_warnings}
            return {
                **state,
                "answer": answer,
                "timing": timing,
                "metadata": {**(state.get("metadata") or {}), **gw_meta, **(
                    {"sanitize_warnings": sanitize_warnings} if sanitize_warnings else {}
                )},
            }

    def _node_validate_output(self, state: RAGState) -> RAGState:
        gr = self.guardrails.validate_output(state.get("answer", ""))
        return {**state, "guardrails_output": self.guardrails.to_dict(gr)}

    def _route_after_input(self, state: RAGState) -> str:
        if state.get("metadata", {}).get("input_blocked"):
            return "generate"
        return "retrieve"

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("validate_input", self._node_validate_input)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("rerank", self._node_rerank)
        g.add_node("build_prompt", self._node_build_prompt)
        g.add_node("generate", self._node_generate)
        g.add_node("validate_output", self._node_validate_output)

        g.set_entry_point("validate_input")
        g.add_conditional_edges(
            "validate_input",
            self._route_after_input,
            {"retrieve": "retrieve", "generate": "generate"},
        )
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "build_prompt")
        g.add_edge("build_prompt", "generate")
        g.add_edge("generate", "validate_output")
        g.add_edge("validate_output", END)
        return g.compile()

    @trace_run("rag_workflow")
    def run(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        mode: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        from backend.app.core.modes import get_default_mode, get_mode

        if role == "manager" and not mode:
            mode = "management_view"
        mode = mode or get_default_mode()
        mode_cfg = get_mode(mode)
        t0 = time.perf_counter()
        logger.info(f"Chat workflow start [{mode}]: {query[:80]}...")
        try:
            final = self.graph.invoke(
                {
                    "query": query,
                    "timing": {},
                    "metadata": {
                        "user_id": user_id,
                        "session_id": session_id,
                        "mode": mode,
                        "role": role or "engineer",
                        "llm_temperature": mode_cfg.temperature,
                        "llm_tier_floor": mode_cfg.llm_tier_override,
                    },
                }
            )
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            timing = {**final.get("timing", {}), "total_ms": total_ms}
            logger.info(f"Chat workflow done in {total_ms}ms")

            answer = final.get("answer", "")
            grounding = final.get("grounding", {})
            meta = final.get("metadata", {}) or {}
            citations = final.get("citations", [])
            should_abstain = bool(grounding.get("should_abstain"))
            generation_failed = bool(
                final.get("error")
                or meta.get("response_state") == "generation_failed"
                or grounding.get("generation_failed")
            )

            if generation_failed:
                citations = []
                flags = []
                grounding = {**grounding, "confidence_band": None, "generation_failed": True}
                answer = sanitize_user_answer(answer, generation_failed=True)
            elif should_abstain or meta.get("abstain"):
                # Abstention: no grounded answer -> no sources, no flags.
                citations = []
                flags: list[dict[str, Any]] = []
            else:
                # Scope answer-level flags to the markers the answer actually
                # cited (deduplicated by regulation inside derive_answer_flags).
                flags = derive_answer_flags(citations, answer_text=answer)
                cat_flags = validate_category_citation_authority(
                    query,
                    citations,
                    final.get("documents", []),
                    getattr(self.retriever, "_chunk_by_id", {}),
                )
                flags.extend(cat_flags)
                flags.extend(
                    detect_category_value_misattribution(
                        query,
                        answer,
                        citations,
                        final.get("documents", []),
                        getattr(self.retriever, "_chunk_by_id", {}),
                    )
                )
                flags.extend(
                    detect_knowledge_boundary_flags(query, answer, should_abstain=False)
                )
                flags.extend(
                    detect_scope_overclaim_flags(
                        answer,
                        citations,
                        final.get("documents", []),
                        getattr(self.retriever, "_chunk_by_id", {}),
                    )
                )

            # Additive gateway routing summary (empty dict when gateway is off).
            gateway = {
                k: meta[k]
                for k in (
                    "model",
                    "provider",
                    "tier",
                    "cache_hit",
                    "fallback_used",
                    "route_score",
                    "route_reasons",
                    "cost_usd",
                    "cost_saved_usd",
                    "prompt_tokens",
                    "completion_tokens",
                    "routing_diagnostic",
                )
                if k in meta
            }

            return {
                "query": query,
                "answer": answer,
                "documents": final.get("documents", []),
                "citations": citations,
                "flags": flags,
                "grounding": grounding,
                "context": final.get("context", ""),
                "prompt": final.get("prompt", ""),
                "generation_failed": generation_failed,
                "error": final.get("error") if generation_failed else None,
                "guardrails": {
                    "input": final.get("guardrails_input", {}),
                    "output": final.get("guardrails_output", {}),
                },
                "gateway": gateway,
                "timing": timing,
            }
        except Exception as exc:
            logger.exception("Workflow failed")
            return {
                "query": query,
                "answer": GENERATION_UNAVAILABLE_MESSAGE,
                "documents": [],
                "citations": [],
                "flags": [],
                "grounding": {"generation_failed": True},
                "generation_failed": True,
                "error": str(exc),
                "timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 2)},
            }
