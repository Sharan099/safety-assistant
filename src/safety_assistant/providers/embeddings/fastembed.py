"""Real semantic embeddings via fastembed (ONNX Runtime, no torch).

Model chosen by measurement (evals/experiments/embedding_benchmark.py,
docs/ADR/0014): all-MiniLM-L6-v2, 384-d. Loading costs ~1-2 s once per
process; `embed_*` is CPU-bound — callers on an async event loop must
offload to a thread (see api/dependencies).
"""

from __future__ import annotations


class FastEmbedProvider:
    model_version = "v1"

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # local import: ORT import cost only when used

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dimensions = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        (vector,) = self._model.embed([text])
        return vector.tolist()  # type: ignore[no-any-return]
