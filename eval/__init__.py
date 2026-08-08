"""Evaluation package — golden set, retrieval metrics, RAGAS + DeepEval."""

__all__ = [
    "load_golden_set",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]


def __getattr__(name: str):
    if name == "load_golden_set":
        from eval.gold import load_golden_set

        return load_golden_set
    if name in {"mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"}:
        from eval import metrics as m

        return getattr(m, name)
    raise AttributeError(name)
