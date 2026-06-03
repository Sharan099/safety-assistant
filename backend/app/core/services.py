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
_warmed = False


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
