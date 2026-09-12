# Evaluation

## Dataset

`evals/datasets/regulatory_v1.yaml` — 47 cases, 16 slices, 39 with section-level truth. Each case: `case_id, query, query_type, difficulty, jurisdiction, as_of_date, expected_regulation_key(s), expected_version_label, expected_section_paths / prefixes, key_facts, acceptable_citations, answerability, review_status, notes`. Relevance is resolved at run time from section paths (stable across re-chunking). Growth target 200–500; `review_status: DRAFT` marks cases awaiting a second reviewer.

## Retrieval metrics

`uv run safety-assistant eval-retrieval [--legs dense sparse hybrid_rrf hybrid_rrf_rerank full]` computes Recall@5/10/20, Precision@k, Hit@k, MRR, nDCG@10, regulation-hit@5, p50/p95 latency, overall and per slice, and writes `evals/results/retrieval_<dataset>_<timestamp>.json` with git SHA, corpus fingerprint (chunk count, per-version parser/chunker/parsed_hash), embedding model, reranker and retrieval config.

Current results: see README "Measured results" and `evals/results/retrieval_regulatory_v1_latest.json`.

## Regression gate

`tests/retrieval_regression` (runs only against the real corpus): 23 stable cases must keep their regulation in the top 5 and their clause in the top 10; full-pipeline MRR floor 0.60 (measured 0.636–0.664).

## Generation and refusal

Verified by contract tests with a schema-compliant mock (`tests/integration/test_api_ask.py`, `tests/unit/test_citations_and_gate.py`): supported claims kept, unsupported numbers/ids dropped, abstention on no-version-on-date / unknown regulation / ambiguous query, evidence-only on LLM failure, injection flagged. **Not published**: LLM-judged correctness, groundedness, citation completeness, refusal precision/recall on the gold set — no LLM provider was available. To produce them: configure `LLM_PROVIDER`, run `/ask` over the dataset, score `key_facts` coverage and `acceptable_citations` against `answer.claims`, and record results with the same provenance fields.

## Experiments

E0 dense, E2 hybrid, E3 hybrid+rerank and the full system are the legs of the runner. Pre-rebuild embedding/reranker benchmarks are kept in `evals/baselines/`. New experiments go under `evals/experiments/` and must record dataset version, git SHA, parser/chunker/embedding versions, config and timestamp.

## Load

`scripts/eval/load_test.py` → `evals/results/load_*.json` (users, duration, rps, p50/p95/p99, errors, throttled).
