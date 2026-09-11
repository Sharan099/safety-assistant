"""Real cross-encoder reranker (fastembed TextCrossEncoder, ONNX). Full
re-score with the model's raw relevance logit — not additive."""

from __future__ import annotations

from safety_assistant.providers.rerankers.base import RerankCandidate

DEFAULT_CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    model_version = "v1"

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        if not candidates:
            return []
        return [float(s) for s in self._model.rerank(query, [c.content for c in candidates])]
