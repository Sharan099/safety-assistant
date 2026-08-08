"""Cross-encoder (or Cohere) reranking of hybrid candidates."""

from __future__ import annotations

import logging
import os
from typing import Sequence

from dotenv import load_dotenv

from retrieval.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_N = 5


def _env_min_score() -> float | None:
    """Optional absolute relevance floor (``RERANK_MIN_SCORE``). Empty / off → disabled."""
    raw = (os.getenv("RERANK_MIN_SCORE") or "").strip()
    if not raw or raw.lower() in {"none", "off", "disable", "-"}:
        return None
    return float(raw)


def _env_score_margin() -> float | None:
    """Optional relative floor: keep scores ≥ (best_in_window − margin)."""
    raw = (os.getenv("RERANK_SCORE_MARGIN") or "").strip()
    if not raw or raw.lower() in {"none", "off", "disable", "-"}:
        return None
    return float(raw)


class Reranker:
    """Local CrossEncoder by default; optional Cohere Rerank API."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self.provider = (provider or os.getenv("RERANK_PROVIDER") or "local").strip().lower()
        self.model_name = model_name or os.getenv("RERANK_MODEL") or DEFAULT_MODEL
        self.api_key = (api_key or os.getenv("COHERE_API_KEY") or "").strip()
        self._model = None

        if self.provider not in {"local", "cohere", "none"}:
            raise ValueError(f"Unsupported RERANK_PROVIDER={self.provider!r}")

    def _load_local(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from ingestion.hf_auth import ensure_hf_auth

            ensure_hf_auth()
            logger.info("Loading reranker %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        if self.provider == "none":
            # Preserve incoming order (identity scores).
            return [float(len(texts) - i) for i in range(len(texts))]
        if self.provider == "cohere":
            return self._score_cohere(query, texts)
        model = self._load_local()
        pairs = [(query, t) for t in texts]
        # Keep predict in-process on Windows: a process pool / DataLoader worker
        # can re-enter __main__ and deadlock on local Qdrant.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            scores = model.predict(
                pairs,
                batch_size=min(16, max(1, len(pairs))),
                show_progress_bar=False,
                convert_to_numpy=True,
                pool=None,
            )
        except TypeError:
            scores = model.predict(pairs)
        return [float(s) for s in scores]

    def _score_cohere(self, query: str, texts: Sequence[str]) -> list[float]:
        if not self.api_key:
            raise RuntimeError("RERANK_PROVIDER=cohere but COHERE_API_KEY is not set")
        import httpx

        model = self.model_name if self.model_name.startswith("rerank-") else "rerank-english-v3.0"
        with httpx.Client(timeout=60.0, trust_env=True) as client:
            resp = client.post(
                "https://api.cohere.com/v2/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "query": query,
                    "documents": list(texts),
                    "top_n": len(texts),
                },
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"Cohere rerank HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        scores = [0.0] * len(texts)
        for item in data.get("results") or []:
            idx = int(item["index"])
            scores[idx] = float(item.get("relevance_score") or 0.0)
        return scores


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_n: int | None = None,
    min_score: float | None = None,
    score_margin: float | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Reorder candidates with a cross-encoder; keep top_n above relevance floors.

    After scoring, take the top ``RERANK_TOP_K`` window, then optionally:

    - ``min_score`` / ``RERANK_MIN_SCORE`` — absolute score floor
    - ``score_margin`` / ``RERANK_SCORE_MARGIN`` — keep scores within ``margin``
      of the best score in the window (relative floor)

    When a floor would empty the window, the single best-scoring candidate is
    kept so callers never get an empty context from thresholding alone.
    """
    load_dotenv()
    if not chunks:
        return []
    top_n = top_n if top_n is not None else int(os.getenv("RERANK_TOP_K", str(DEFAULT_TOP_N)))
    if min_score is None:
        min_score = _env_min_score()
    if score_margin is None:
        score_margin = _env_score_margin()
    reranker = reranker or Reranker()

    prefer_limit = False
    try:
        from ingestion.enrich import role_prefixed_text
        from retrieval.value_limit import is_compliance_prefer_limit_query

        prefer_limit = is_compliance_prefer_limit_query(query)
    except Exception:  # noqa: BLE001
        role_prefixed_text = None  # type: ignore[assignment]

    def _doc_text(c: RetrievedChunk) -> str:
        raw = (c.text or c.enriched_text or "").strip()
        if prefer_limit and role_prefixed_text is not None:
            return role_prefixed_text(raw, section_number=c.section_number or "")
        return raw

    texts = [_doc_text(c) for c in chunks]
    scores = reranker.score(query, texts)
    ranked = sorted(
        zip(chunks, scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    scored: list[RetrievedChunk] = [
        chunk.model_copy(update={"score": float(score)}) for chunk, score in ranked
    ]

    window = scored[:top_n]
    out = list(window)
    if min_score is not None:
        out = [c for c in out if float(c.score or 0.0) >= float(min_score)]
    if score_margin is not None and window:
        best = float(window[0].score or 0.0)
        floor = best - float(score_margin)
        out = [c for c in out if float(c.score or 0.0) >= floor]
    if not out and scored:
        out = [scored[0]]
        logger.info(
            "rerank floors emptied top-%d (min_score=%s margin=%s); keeping best=%.4f",
            top_n,
            min_score,
            score_margin,
            float(scored[0].score or 0.0),
        )

    try:
        from observability.context import get_current_trace

        tr = get_current_trace()
        if tr is not None:
            tr.add_rerank(model=reranker.model_name, calls=1)
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        "rerank query=%r candidates=%d kept=%d top_n=%d min_score=%s margin=%s provider=%s",
        query[:80],
        len(chunks),
        len(out),
        top_n,
        min_score,
        score_margin,
        reranker.provider,
    )
    return out
