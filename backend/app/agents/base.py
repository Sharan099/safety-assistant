"""Agent factory — retrieval, grounding gate, gateway, pydantic-validated JSON."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from backend.app.agents.schemas import AGENT_SCHEMAS, CrewState
from backend.app.core.authority_tier import chunk_authority_tier
from backend.app.core.modes import get_mode
from backend.app.core.settings import ENABLE_GROUNDING_GATE, TOP_K_AFTER_RERANK
from backend.app.retrieval.citations import (
    assess_grounding,
    build_citations,
    detect_authority_blur_flags,
    enrich_doc_provenance,
)
from backend.app.retrieval.query_breadth import assess_query_breadth, effective_rerank_k

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    m = _JSON_BLOCK_RE.search(text)
    raw = m.group(1).strip() if m else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def _filter_tiers(docs: list[dict], allowed: tuple[str, ...] | None) -> list[dict]:
    if not allowed:
        return docs
    return [d for d in docs if chunk_authority_tier(d) in allowed]


def _build_context(docs: list[dict], citations: list[dict]) -> str:
    parts: list[str] = []
    for doc, cit in zip(docs, citations):
        marker = cit.get("marker", "S?")
        tier = cit.get("authority_tier_badge", "")
        header = f"[{marker}] {cit.get('label', '')} ({tier})"
        body = re.sub(r"^[#\[].*?\]?\n+", "", doc.get("text", ""), count=1).strip()
        parts.append(f"{header}\n{body[:600]}")
    return "\n\n---\n\n".join(parts)


def _get_llm():
    from backend.app.core.services import get_gateway
    from backend.app.llm.groq_client import GroqLLM

    gateway = get_gateway()
    if gateway is not None:
        return gateway, True
    return GroqLLM(), False


def _generate(
    prompt: str,
    *,
    query: str,
    mode: str,
    tier: int,
    grounding: dict,
    retrieval_meta: dict,
) -> dict[str, Any]:
    llm, is_gateway = _get_llm()
    if is_gateway:
        from backend.app.gateway.types import RoutingContext

        ctx = RoutingContext(
            prompt=prompt,
            query=query,
            grounding=grounding,
            retrieval=retrieval_meta,
            mode=mode,
            llm_tier_floor=tier,
        )
        return llm.generate(prompt, routing_context=ctx)
    return llm.generate(prompt)


def make_agent(
    name: str,
    *,
    mode: str,
    allowed_tiers: tuple[str, ...] | None,
    tier: int,
    system_prompt: str,
    build_query: Callable[[CrewState], str],
    build_user: Callable[[CrewState, str, str], str] | None = None,
    skip_retrieval: bool = False,
) -> Callable[[CrewState], CrewState]:
    """Return a LangGraph node that runs one specialist agent."""

    schema_cls = AGENT_SCHEMAS[name]

    def _node(state: CrewState) -> CrewState:
        t0 = time.perf_counter()
        query = build_query(state)
        agent_queries = {**(state.get("agent_queries") or {}), name: query}

        if skip_retrieval:
            user_body = (build_user or (lambda s, q, c: q))(state, query, "")
            prompt = f"{system_prompt}\n\n---\n\n{user_body}\n\nRespond with JSON only."
            out = _generate(
                prompt,
                query=query,
                mode=mode,
                tier=tier,
                grounding={},
                retrieval_meta={},
            )
            try:
                parsed = schema_cls.model_validate(_extract_json(out.get("answer", "")))
                payload = parsed.model_dump()
            except Exception as exc:
                logger.warning(f"Agent {name} JSON parse failed: {exc}")
                payload = {"status": "insufficient_data"}
            timing = {
                **(state.get("timing") or {}),
                "per_agent_ms": {
                    **(state.get("timing", {}).get("per_agent_ms") or {}),
                    name: round((time.perf_counter() - t0) * 1000, 2),
                },
            }
            return {
                **state,
                "agent_queries": agent_queries,
                "agent_outputs": {**(state.get("agent_outputs") or {}), name: payload},
                "timing": timing,
            }

        from backend.app.core.services import get_retriever, get_reranker

        retriever = get_retriever()
        reranker = get_reranker()
        mode_cfg = get_mode(mode)

        result = retriever.retrieve(query, mode=mode)
        docs = _filter_tiers(result["documents"], allowed_tiers)
        breadth = assess_query_breadth(query)
        rerank_k = effective_rerank_k(breadth, default_k=TOP_K_AFTER_RERANK)
        rr = reranker.rerank(
            query,
            docs,
            force_strong=mode_cfg.force_strong_reranker,
            top_k=rerank_k,
        )
        docs = rr["documents"]
        docs = docs[: mode_cfg.retrieval_k]
        chunk_lookup = getattr(retriever, "_chunk_by_id", {}) or {}
        docs = [enrich_doc_provenance(d, chunk_lookup) for d in docs]
        citations = build_citations(docs, chunk_lookup) if docs else []

        grounding = assess_grounding(
            docs,
            reranker_used=bool(rr.get("reranker_used")),
            min_semantic=mode_cfg.grounding_min_semantic,
            min_rerank_prob=mode_cfg.grounding_min_rerank_prob,
        )
        abstain = ENABLE_GROUNDING_GATE and grounding.get("should_abstain", False)

        if abstain or not docs:
            payload = {"status": "insufficient_data"}
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return {
                **state,
                "agent_queries": agent_queries,
                "agent_outputs": {**(state.get("agent_outputs") or {}), name: payload},
                "citations": (state.get("citations") or []) + citations,
                "timing": {
                    **(state.get("timing") or {}),
                    "per_agent_ms": {
                        **(state.get("timing", {}).get("per_agent_ms") or {}),
                        name: elapsed,
                    },
                },
            }

        context = _build_context(docs, citations)
        user_body = (build_user or (lambda s, q, c: f"Query: {q}\n\nSources:\n{c}"))(
            state, query, context
        )
        prompt = f"{system_prompt}\n\n---\n\n{user_body}\n\nRespond with JSON only."

        out = _generate(
            prompt,
            query=query,
            mode=mode,
            tier=tier,
            grounding=grounding,
            retrieval_meta={"doc_count": len(docs), "best_semantic": grounding.get("best_semantic")},
        )
        try:
            parsed = schema_cls.model_validate(_extract_json(out.get("answer", "")))
            payload = parsed.model_dump()
            blur = detect_authority_blur_flags(json.dumps(payload), citations)
            if blur:
                logger.warning(f"Agent {name} authority blur flags: {blur}")
        except Exception as exc:
            logger.warning(f"Agent {name} failed validation: {exc}")
            payload = {"status": "insufficient_data"}

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            **state,
            "agent_queries": agent_queries,
            "agent_outputs": {**(state.get("agent_outputs") or {}), name: payload},
            "citations": (state.get("citations") or []) + citations,
            "timing": {
                **(state.get("timing") or {}),
                "per_agent_ms": {
                    **(state.get("timing", {}).get("per_agent_ms") or {}),
                    name: elapsed,
                },
            },
        }

    _node.__name__ = f"agent_{name}"
    return _node
