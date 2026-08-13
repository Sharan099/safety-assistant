# evals

Standing evaluation harnesses — distinct from `tests/`, which asserts
specific known values. These run the real pipeline against a golden dataset
and report aggregate metrics, per `IMPLEMENTATION_PLAN.md` Phase 12/19.

```powershell
uv run python evals/scenario_eval.py     # numerical: divergence/config-diff/quality accuracy, all 10 SCN scenarios
uv run python evals/retrieval_eval.py    # retrieval: Recall@5, Recall@10, MRR against evals/golden_retrieval_set.yaml
```

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
