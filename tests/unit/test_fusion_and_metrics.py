import uuid

from safety_assistant.evaluation.metrics import hit_at_k, mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from safety_assistant.retrieval.fusion import reciprocal_rank_fusion

A, B, C, D = (uuid.uuid4() for _ in range(4))


def test_rrf_rewards_agreement_and_never_sums_raw_scores() -> None:
    fused = reciprocal_rank_fusion([[A, B, C], [B, A, D]])
    assert fused[A] == fused[B] > fused[C] == fused[D]
    assert abs(fused[A] - (1 / 61 + 1 / 62)) < 1e-12


def test_rrf_weights_and_empty_lists() -> None:
    fused = reciprocal_rank_fusion([[A], [B]], weights=[2.0, 1.0])
    assert fused[A] == 2 * fused[B]
    assert reciprocal_rank_fusion([]) == {}


def test_metrics_basic() -> None:
    ranked, rel = [A, B, C, D], {B, D}
    assert recall_at_k(ranked, rel, 2) == 0.5 and recall_at_k(ranked, rel, 4) == 1.0
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert hit_at_k(ranked, rel, 1) == 0.0 and hit_at_k(ranked, rel, 2) == 1.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert 0 < ndcg_at_k(ranked, rel, 4) < 1
    assert ndcg_at_k([B, D, A, C], rel, 4) == 1.0


def test_metrics_undefined_without_relevance_and_mean_skips_none() -> None:
    assert recall_at_k([A], set(), 5) is None and reciprocal_rank([A], set()) is None
    assert mean([None, 1.0, 0.0]) == 0.5 and mean([None]) is None
