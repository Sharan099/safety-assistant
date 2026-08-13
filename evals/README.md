# evals

Standing evaluation harnesses — distinct from `tests/`, which asserts
specific known values. These run the real pipeline against a golden dataset
and report aggregate metrics, per `IMPLEMENTATION_PLAN.md` Phase 12/19.

```powershell
uv run python evals/scenario_eval.py       # numerical: divergence/config-diff/quality accuracy, all 10 SCN scenarios
uv run python evals/retrieval_eval.py      # retrieval: Recall@5, Recall@10, MRR against evals/golden_retrieval_set.yaml
uv run python evals/level3_hybrid_eval.py  # Level 3: per-stage comparison — BM25-only / Dense-only / RRF / full pipeline+reranker, plus NDCG@10 (TRD_LEVEL3.md §27-31)
```

`level3_hybrid_eval.py` exists specifically to satisfy TRD_LEVEL3.md §30's
"do not claim hybrid retrieval improves performance until measured" — the
numbers it prints are the actual measurement, not tuned. On the current
golden set, BM25-only outperforms raw RRF (the interim
`HashingEmbeddingProvider`'s dense leg, `docs/ADR/0007`, dilutes the fused
ranking); the reranker (`docs/ADR/0012`) recovers RRF back to BM25-only's
quality and genuinely improves NDCG@10 over RRF-only. Structured CAE
retrieval correctness is covered by `tests/retrieval/test_structured.py`
(entity/relationship correctness against real persisted decks), not this
script — TRD_LEVEL3.md §14/`docs/ADR/0012`: structured search is
complementary to RAG, not a candidate fused into the same ranked list.

Run these after changing `packages/analysis/synthetic.py`, the divergence
detector's threshold, `packages/retrieval`, or the ingested knowledge
corpus — not just once. See `docs/ADR/0008` for what running them the first
time actually caught (two real bugs, not hypothetical ones).

## Not built yet

- **Agent eval** (tool selection, evidence grounding, hypothesis quality) —
  `tests/agent/test_graph.py` covers the two clearest cases (SCN-001
  produces a correct hypothesis, SCN-010 blocks correctly) but there's no
  standing harness across all 10 scenarios yet.
- **Citation accuracy / source authority correctness** — needs page-level
  ground truth per golden query, not just the document.
- **Product metrics** (investigation completion, reviewer corrections, time
  to finding) — requires real usage data this V1 doesn't have yet.
