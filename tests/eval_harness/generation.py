"""Generate RAG answers for evaluation cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _documents_to_contexts(documents: list[dict], chunk_lookup: dict | None = None) -> list[str]:
    contexts: list[str] = []
    for d in documents:
        text = d.get("text") or ""
        if chunk_lookup and d.get("id") in chunk_lookup:
            text = chunk_lookup[d["id"]].get("text") or text
        if text.strip():
            contexts.append(text[:2000])
    return contexts


def generate_answer_for_item(item: dict, workflow=None, retriever=None, reranker=None) -> dict[str, Any]:
    """
    Run full RAG pipeline once for a golden-set item.
    Returns answer, contexts, documents, and token usage.
    """
    skip_llm = os.getenv("EVAL_SKIP_LLM", "").lower() in ("1", "true", "yes")

    if workflow is None:
        from backend.app.core.services import get_retriever, get_reranker, get_workflow
        retriever = retriever or get_retriever()
        reranker = reranker or get_reranker()
        workflow = get_workflow()

    chunk_lookup = getattr(retriever, "_chunk_by_id", None) if retriever else None

    if skip_llm:
        # Zero tokens: retrieval + rerank only
        result = retriever.retrieve(item["question"])
        docs = result["documents"]
        if reranker:
            rr = reranker.rerank(item["question"], docs)
            docs = rr["documents"]
        return {
            "answer": "",
            "contexts": _documents_to_contexts(docs, chunk_lookup),
            "documents": docs,
            "tokens": {"prompt": 0, "completion": 0},
        }

    out = workflow.run(item["question"])
    docs = out.get("documents") or []
    contexts = _documents_to_contexts(docs, chunk_lookup)
    if not contexts and out.get("context"):
        contexts = [out["context"][:4000]]

    gw = out.get("gateway") or {}
    tokens = {
        "prompt": int(gw.get("prompt_tokens") or 0),
        "completion": int(gw.get("completion_tokens") or 0),
    }

    return {
        "answer": out.get("answer", ""),
        "contexts": contexts,
        "documents": docs,
        "tokens": tokens,
        "grounding": out.get("grounding"),
        "guardrails": out.get("guardrails"),
    }
