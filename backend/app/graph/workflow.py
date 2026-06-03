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
from backend.app.guardrails.validator import SafetyGuardrails
from backend.app.llm.groq_client import GroqLLM
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker


class RAGState(TypedDict, total=False):
    query: str
    documents: list[dict]
    context: str
    prompt: str
    answer: str
    guardrails_input: dict
    guardrails_output: dict
    timing: dict
    metadata: dict
    error: str


def _compress_context(documents: list[dict]) -> str:
    lines = []
    for d in documents:
        reg = d.get("regulation", "REG")
        title = d.get("title", "") or d.get("section_title", "")
        heading = d.get("heading_path", "")
        text = (d.get("text", "") or "")[:700]
        header = f"{heading}\n{title}".strip() if heading else title
        block = f"\n=== {reg} ===\n{header}\n\n{text}"
        # Parent-child: include parent section context when available.
        parent = (d.get("parent_context", "") or "").strip()
        if parent and parent[:60] not in text:
            block += f"\n\n[section context] {parent[:400]}"
        lines.append(block)
    return "\n".join(lines)


def _build_prompt(query: str, context: str) -> str:
    return f"""
You are PSA AI, an expert passive safety and homologation engineering assistant.

USER QUESTION
{query}

RETRIEVED REGULATION CONTEXT
{context}

Answer using ONLY the context above. Use clear markdown sections.
If information is missing, say: Information not found in regulations.
"""


class RAGWorkflow:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self._llm: GroqLLM | None = None
        self.guardrails = SafetyGuardrails()
        self.graph = self._build_graph()

    def _get_llm(self) -> GroqLLM:
        if self._llm is None:
            if not os.getenv("GROQ_API_KEY"):
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to .env in the project root."
                )
            logger.info("Connecting to Groq LLM...")
            self._llm = GroqLLM()
        return self._llm

    def _node_validate_input(self, state: RAGState) -> RAGState:
        with trace_span("validate_input", {"query": state.get("query", "")}) as span:
            gr = self.guardrails.validate_input(state["query"])
            span["outputs"] = self.guardrails.to_dict(gr)
            timing = state.get("timing", {})
            return {
                **state,
                "guardrails_input": self.guardrails.to_dict(gr),
                "timing": timing,
                "metadata": {**(state.get("metadata") or {}), "input_blocked": gr.blocked},
            }

    def _node_retrieve(self, state: RAGState) -> RAGState:
        if state.get("metadata", {}).get("input_blocked"):
            return state
        logger.info(f"Retrieving for: {state['query'][:80]}...")
        with trace_span("retrieve", {"query": state["query"]}) as span:
            result = self.retriever.retrieve(state["query"])
            span["outputs"] = {
                "doc_count": len(result["documents"]),
                "semantic": result["semantic_count"],
                "bm25": result["bm25_count"],
                "queries": result.get("queries", []),
                "intent_flags": result.get("intent_flags", []),
            }
            timing = {**state.get("timing", {}), "retrieval_ms": result["latency_ms"]}
            return {**state, "documents": result["documents"], "timing": timing}

    def _node_rerank(self, state: RAGState) -> RAGState:
        if state.get("metadata", {}).get("input_blocked"):
            return state
        logger.info(f"Reranking {len(state.get('documents', []))} candidates...")
        with trace_span("rerank", {"candidates": len(state.get("documents", []))}) as span:
            result = self.reranker.rerank(state["query"], state.get("documents", []))
            span["outputs"] = {"top_k": len(result["documents"])}
            timing = {**state.get("timing", {}), "rerank_ms": result["latency_ms"]}
            return {**state, "documents": result["documents"], "timing": timing}

    def _node_build_prompt(self, state: RAGState) -> RAGState:
        if state.get("metadata", {}).get("input_blocked"):
            return state
        docs = state.get("documents", [])
        context = _compress_context(docs) if docs else ""
        prompt = _build_prompt(state["query"], context) if context else ""
        if not context:
            logger.warning("No documents after retrieval/rerank")
        return {**state, "context": context, "prompt": prompt}

    def _node_generate(self, state: RAGState) -> RAGState:
        meta = state.get("metadata") or {}
        if meta.get("input_blocked"):
            reason = state.get("guardrails_input", {}).get("block_reason", "blocked")
            return {
                **state,
                "answer": (
                    "Your query was blocked by safety guardrails "
                    f"({reason}). Please rephrase your question."
                ),
            }
        if not state.get("context"):
            return {
                **state,
                "answer": "No relevant passive safety regulation information found.",
            }
        logger.info("Calling Groq LLM...")
        with trace_span("generate", {"prompt_len": len(state.get("prompt", ""))}) as span:
            try:
                out = self._get_llm().generate(state["prompt"])
            except Exception as exc:
                logger.exception("Groq generation failed")
                return {
                    **state,
                    "answer": f"LLM error: {exc}",
                    "error": str(exc),
                }
            span["outputs"] = {
                "answer_len": len(out["answer"]),
                "model": out.get("model"),
            }
            timing = {**state.get("timing", {}), "llm_ms": out["latency_ms"]}
            logger.info(f"Groq answer ready in {out['latency_ms']}ms")
            return {**state, "answer": out["answer"], "timing": timing}

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
    def run(self, query: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        logger.info(f"Chat workflow start: {query[:80]}...")
        try:
            final = self.graph.invoke({"query": query, "timing": {}, "metadata": {}})
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            timing = {**final.get("timing", {}), "total_ms": total_ms}
            logger.info(f"Chat workflow done in {total_ms}ms")
            return {
                "query": query,
                "answer": final.get("answer", ""),
                "documents": final.get("documents", []),
                "context": final.get("context", ""),
                "prompt": final.get("prompt", ""),
                "guardrails": {
                    "input": final.get("guardrails_input", {}),
                    "output": final.get("guardrails_output", {}),
                },
                "timing": timing,
            }
        except Exception as exc:
            logger.exception("Workflow failed")
            return {
                "query": query,
                "answer": f"Error: {exc}",
                "documents": [],
                "error": str(exc),
                "timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 2)},
            }
