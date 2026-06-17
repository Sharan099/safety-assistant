"""Shared singletons — load artifacts once, reuse across requests."""

import os
import threading

from loguru import logger

from backend.app.graph.workflow import RAGWorkflow
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker

_lock = threading.RLock()
_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_workflow: RAGWorkflow | None = None
_gateway = None  # type: ignore[var-annotated]
_warmed = False

# Readiness / self-test state (gates the frontend until the pipeline is proven).
_selftest_lock = threading.RLock()
_ready = False
_selftest: dict | None = None
SELF_TEST_QUERY = "What are the UN R14 seat belt anchorage strength requirements?"


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                logger.info("Initializing shared HybridRetriever...")
                _retriever = HybridRetriever()
    return _retriever


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker()
    return _reranker


def get_workflow() -> RAGWorkflow:
    global _workflow
    if _workflow is None:
        with _lock:
            if _workflow is None:
                logger.info("Initializing shared LangGraph workflow...")
                _workflow = RAGWorkflow(
                    retriever=get_retriever(),
                    reranker=get_reranker(),
                )
    return _workflow


def get_gateway():
    """Shared Intelligent Multi-LLM Gateway, or None when disabled.

    The gateway's semantic cache REUSES the retriever's BGE embedding model via
    `HybridRetriever.embed_text`, so no second embedding model is loaded.
    """
    global _gateway
    from backend.app.core.settings import ENABLE_GATEWAY

    if not ENABLE_GATEWAY:
        return None
    if _gateway is None:
        with _lock:
            if _gateway is None:
                from backend.app.gateway.gateway import LLMGateway

                logger.info("Initializing Intelligent Multi-LLM Gateway...")
                retriever = get_retriever()
                _gateway = LLMGateway(embed_fn=retriever.embed_text)
    return _gateway


def pipeline_status() -> dict:
    """Cheap status for GET /health — does not load models or JSON artifacts."""
    return {
        "pipeline_warmed": _warmed,
        "retriever_loaded": _retriever is not None,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "reranker_enabled": os.getenv("ENABLE_RERANKER", "false").lower() == "true",
    }


def warmup_pipeline() -> None:
    """
    Fast startup: load JSON + BM25 + vector matrix only.
    Heavy ML models load on first chat unless PRELOAD_ML_MODELS=true.
    """
    global _warmed
    if _warmed:
        return

    logger.info("Pipeline warmup (artifacts only)...")
    get_retriever()
    get_workflow()

    if os.getenv("PRELOAD_ML_MODELS", "false").lower() == "true":
        logger.info("PRELOAD_ML_MODELS=true — loading embedding + reranker...")
        try:
            get_retriever().warmup()
        except Exception as exc:
            logger.warning(f"Embedding warmup failed (BM25 still works): {exc}")
        if os.getenv("ENABLE_RERANKER", "false").lower() == "true":
            try:
                get_reranker().warmup()
            except Exception as exc:
                logger.warning(f"Reranker warmup failed: {exc}")
    else:
        logger.info(
            "ML models will load on first chat. "
            "Set PRELOAD_ML_MODELS=true to load at startup."
        )

    _warmed = True
    logger.info("Backend ready — POST /api/v1/chat")


def run_self_test() -> dict:
    """
    Run one end-to-end query so the frontend only lets users chat once the
    pipeline is proven working. Cached after the first successful run.

    Retrieval must succeed for readiness. The LLM call is best-effort: if Groq is
    unavailable or rate-limited we still report ready=True (retrieval works) but
    flag llm_ok=False so the UI can warn.
    """
    global _ready, _selftest
    if _ready and _selftest is not None:
        return _selftest

    with _selftest_lock:
        if _ready and _selftest is not None:
            return _selftest

        result: dict = {
            "ready": False,
            "retrieval_ok": False,
            "llm_ok": False,
            "llm_configured": bool(os.getenv("GROQ_API_KEY")),
            "detail": "",
        }
        try:
            retriever = get_retriever()
            retriever.warmup()  # load embedding model
            r = retriever.retrieve(SELF_TEST_QUERY)
            result["retrieval_ok"] = len(r.get("documents", [])) > 0
            result["doc_count"] = len(r.get("documents", []))
        except Exception as exc:
            result["detail"] = f"retrieval failed: {exc}"
            logger.error(f"Self-test retrieval failed: {exc}")
            _selftest = result
            return result

        skip_llm = os.getenv("READY_SKIP_LLM", "false").lower() == "true"
        if result["llm_configured"] and not skip_llm:
            try:
                out = get_workflow().run(SELF_TEST_QUERY)
                result["llm_ok"] = bool(out.get("answer")) and not out.get("error")
                if out.get("error"):
                    result["detail"] = f"llm error: {out['error']}"
            except Exception as exc:
                result["detail"] = f"llm self-test failed: {exc}"
                logger.warning(f"Self-test LLM call failed (non-fatal): {exc}")

        # Ready as long as retrieval works; LLM is best-effort.
        result["ready"] = result["retrieval_ok"]
        _ready = result["ready"]
        _selftest = result
        logger.info(f"Self-test complete: {result}")
        return result


def readiness() -> dict:
    """Non-blocking snapshot for GET /ready; triggers the self-test lazily."""
    if _ready and _selftest is not None:
        return _selftest
    return run_self_test()
