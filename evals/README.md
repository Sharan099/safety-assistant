# evals

Standing evaluation harnesses — distinct from `tests/`, which asserts
specific known values. These run the real pipeline against a golden dataset
and report aggregate metrics, per `IMPLEMENTATION_PLAN.md` Phase 12/19.

```powershell
uv run python evals/scenario_eval.py       # numerical: divergence/config-diff/quality accuracy, all 10 SCN scenarios
uv run python evals/retrieval_eval.py      # retrieval: Recall@5, Recall@10, MRR against evals/golden_retrieval_set.yaml
uv run python evals/level3_hybrid_eval.py  # Level 3: per-stage comparison — BM25-only / Dense-only / RRF / full pipeline+reranker, plus NDCG@10 (TRD_LEVEL3.md §27-31)
uv run python evals/embedding_benchmark.py # Final-Fix P0: candidate embedding models vs the mock hashing tier (docs/ADR/0014)
uv run python evals/reranker_benchmark.py  # Final-Fix P1: candidate cross-encoder rerankers vs the lexical/authority heuristic (docs/ADR/0015)
```

`level3_hybrid_eval.py` exists specifically to satisfy TRD_LEVEL3.md §30's
"do not claim hybrid retrieval improves performance until measured" — the
numbers it prints are the actual measurement, not tuned. As of
`docs/ADR/0014` (real semantic embeddings replaced the hashing placeholder),
BM25-only and Dense-only are both individually strong, but naive RRF fusion
dilutes slightly below either one alone on the current 8-query golden set —
a real RRF characteristic, not a bug; the reranker recovers past both and
is the best of all four legs. Structured CAE retrieval correctness is
covered by `tests/retrieval/test_structured.py` (entity/relationship
correctness against real persisted decks), not this script —
TRD_LEVEL3.md §14/`docs/ADR/0012`: structured search is complementary to
RAG, not a candidate fused into the same ranked list.

`embedding_benchmark.py` / `reranker_benchmark.py`
(`PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md` §5/§6/§21) compare real candidate
models against each other and against the existing default, writing
results to `evals/results/*.json`. Both surfaced genuine, non-obvious
findings: the winning embedding model also indexed 6x faster than the
runner-up; the winning-on-quality reranker was rejected as the *default*
anyway because its latency (~3.5s/query) fails the "must remain practical"
requirement for an interactive tool — quality alone isn't the whole gate.

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
