# Evaluation

Every number in this document was produced by code in this repository; result files under `evals/results/`
carry dataset version, git SHA, corpus fingerprint, model/config versions and timestamps. Nothing is estimated.

## Datasets

| Dataset | Cases | Provenance |
|---|---|---|
| `evals/datasets/regulatory_v1.yaml` | 47 (39 with section-level truth), 16 slices | human-written and reviewed against the ingested text |
| `evals/datasets/regulatory_v2.yaml` | **262** | v1 (47) + 200 `AUTO_GROUNDED` cases + 15 hand-written unanswerable / out-of-scope / ambiguous / adversarial cases |

`AUTO_GROUNDED` cases are produced by `scripts/eval/generate_cases.py`: for 240 sampled normative sections
(CLAUSE/DEFINITION of the four UN regulations, stratified definitions/numeric/other) the LLM proposes up to two
engineer-style questions with `key_facts`; a case is kept only if **every key fact is a verbatim span of the section**
(whitespace-normalised) and the section path/regulation key come from the database. Nothing is filtered by whether
retrieval finds it. `scripts/eval/build_dataset.py` then samples 200 of the 335 candidates round-robin over
regulation × query type, and scopes every second query ("In UN R16, …") so the set holds both realistic-scoped and
hard-unscoped questions (`notes` records which). Each case names the routing model that generated it. These cases are
labelled `AUTO_GROUNDED`, never `REVIEWED`; the human-written v1 set remains the reference for paraphrase-heavy behaviour
(see "RRF weights" below for why that matters).

Relevance rule (both sets): a retrieved chunk is relevant when its section path, or any path merged into it, matches
`expected_section_paths` / `expected_section_prefixes`; regulation-level hit = regulation key match.

## Retrieval

`uv run safety-assistant eval-retrieval --dataset <yaml> [--legs …]` measures each leg independently
(dense · sparse · hybrid RRF · + reranker · full = + exact-clause leg + parent/cross-ref expansion) and writes
`evals/results/retrieval_<dataset>_<timestamp>.json` (k_eval = 20; Recall/Precision/Hit@5/10/20, MRR, nDCG@10,
regulation-hit@5, p50/p95 latency; overall and per slice). `scripts/eval/grid.py` runs configuration grids on the same
metrics without an LLM.

### Production configuration (2026-09-13, corpus 21,910 chunks, git 66461d2)

`RERANKER=cross_encoder` (Xenova/ms-marco-MiniLM-L-6-v2) over the top-12 fused candidates, RRF weights dense 0.75 /
sparse 1.0 / exact 2.0.

| Dataset | leg | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 | RegHit@5 | p50 ms¹ |
|---|---|---|---|---|---|---|---|---|---|
| v1 (47) | dense | 0.546 | 0.639 | 0.677 | 0.667 | 0.470 | 0.480 | 0.864 | 112 |
| v1 | sparse | 0.686 | 0.821 | 0.870 | 0.795 | 0.546 | 0.582 | 0.955 | 110 |
| v1 | hybrid RRF | 0.650 | 0.851 | 0.875 | 0.769 | 0.583 | 0.615 | 0.955 | 201 |
| v1 | + cross-encoder | 0.844 | 0.877 | 0.901 | 0.949 | 0.753 | 0.768 | 0.977 | 1,740 |
| v1 | **full** | **0.844** | **0.877** | **0.926** | **0.949** | **0.756** | **0.768** | **0.977** | 1,707 |
| v2 (262) | dense | 0.608 | 0.660 | 0.703 | 0.638 | 0.480 | 0.512 | 0.871 | 92 |
| v2 | sparse | 0.896 | 0.930 | 0.961 | 0.926 | 0.720 | 0.761 | 0.988 | 115 |
| v2 | hybrid RRF | 0.818 | 0.916 | 0.949 | 0.848 | 0.652 | 0.704 | 0.984 | 204 |
| v2 | + cross-encoder | 0.911 | 0.931 | 0.953 | 0.938 | 0.807 | 0.829 | 0.992 | 1,249 |
| v2 | **full** | **0.911** | **0.931** | **0.957** | **0.938** | **0.808** | **0.829** | **0.992** | 1,444 |

¹ laptop CPU (i5-8250U) while other evaluation jobs ran; the cross-encoder dominates (~1.2 s for 12 pairs).

### What was tried (optimisation record)

| Change | v1 MRR | v2 MRR | Decision |
|---|---|---|---|
| baseline (heuristic reranker, equal RRF weights) — 2026-09-12 | 0.627 | 0.709 | — |
| sparse-heavy RRF (sparse 3.0, dense 0.5) | 0.617 | 0.736 | **rejected** — helps the LLM-generated set (lexically close to the clause text) but regresses the human-written set |
| dense 0.75, sparse 1.0 | 0.644 | 0.713 | **adopted** — improves both sets and R@10 |
| cross-encoder over all fused candidates | 0.754 | 0.810 (0.814 uncapped) | quality yes, ~9 s/query no |
| cross-encoder over top-12 (`RETRIEVAL_RERANK_TOP_N=12`) | 0.756 | 0.808 | **adopted** — keeps the gain at ~1.5 s |
| cross-encoder over top-24 | — | 0.814 | not worth 4 s |

Weakest v2 slices with the final configuration are `regulation_clause_identifier` and `cross_reference`
(n ≤ 2, not statistically meaningful); `exception_condition` is where fusion used to lose to BM25 and where the
cross-encoder helps most.

### Regression gate

`tests/retrieval_regression` runs against the real corpus with the test profile (heuristic reranker, so it is
download-free and fast): 23 stable v1 cases keep their regulation in the top 5 and clause in the top 10;
v1 full-pipeline MRR ≥ 0.60; **v2 full-pipeline MRR ≥ 0.68 and R@10 ≥ 0.90** (heuristic configuration measured
0.713 / 0.950). Every retrieval-affecting change re-runs it in CI.

## Generation (end to end, real LLM)

`scripts/eval/judged.py` runs the real pipeline per case — retrieval → gate → generation → validation — with the
configured LLM and scores each answer.

**Setup**: `LLM_PROVIDER=openai_compatible`, `LLM_MODEL=auto` through the local `freellmapi` gateway, which routes to
free-tier models (gpt-oss-120b, gpt-oss-safeguard-20b, gemini-3-flash, minimax-m2.7, … recorded per answer) with
fallback on rate limits. Answers are cached per pipeline fingerprint (`evals/cache/answers/`, gitignored).
Judged 2026-09-13, git 66461d2, pipeline fingerprint `2c81f18ff7d18dea`, 262 cases, p50 9.1 s per answer
(gateway routing/retries dominate).

### Deterministic metrics (all 262 cases)

| Metric | Value | Definition |
|---|---|---|
| refusal accuracy | **0.943** | unanswerable/ambiguous → ABSTAINED (11/11 not-in-corpus & out-of-scope; 3/4 ambiguous); answerable → answered (232/246) |
| citation hit | **0.919** | at least one citation points at the expected regulation + clause |
| citation precision | 0.714 | share of all citations that do |
| fact coverage | **0.823** | share of `key_facts` present in the answer (verbatim, all numbers, or ≥ 70 % of content words for non-numeric facts) |
| evidence coverage | **0.977** | share of `key_facts` present in the retrieved evidence (context-recall proxy) |
| grounding ok | **0.965** | the citation/numeric validator accepted the draft without dropping claims |
| injection resisted | **1.000** | 6/6 adversarial questions: planted values never stated in the answer, no prompt leakage, injection flagged in warnings |

Modes: GENERATED 222 · EVIDENCE_ONLY 11 · ABSTAINED 29 (of which 14 answerable cases abstained — the false-refusal rate is
5.7 %).

### LLM-judged metrics

| Framework | n | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|---|
| RAGAS 0.4 (judge = same gateway, `auto`) | 100 answerable, evenly sampled | 0.742 | 0.774 | 0.866 | 0.960 |
| DeepEval 4.2 (judge = same gateway) | 26 | 1.000 | 0.907 | 0.880 (contextual) | — |

Both judges ran sequentially through the free gateway with per-record caching and telemetry disabled; DeepEval's run
was stopped after 26 records when a gateway call hung — the sample is what completed, not a selection. The two
faithfulness scores diverge because RAGAS decomposes answers into statements and penalises any it cannot map to the
contexts (attribution sentences such as "according to UN R94 Rev.4, valid from 2021-06-09" count against it), while
DeepEval and the repository's own validator (grounding ok 0.965) judge the claims that carry the requirement. Treat the
RAGAS faithfulness figure as a lower bound and the deterministic validator as the contract.

### Findings that changed the system

- **False abstentions from the numeric validator.** Models attribute requirements with clause paths, revision labels and
  dates that exist only in the evidence *attributes* shown in the prompt; the validator compared claim numbers against chunk
  text only and dropped correct claims ("not in cited evidence: 2012, 26, 6, 7"). Fixed in `generation/citations.py`:
  numbers from the cited evidence's citation label, section path, version label, validity dates and pages count as
  supported; invented values are still rejected (unit test).
- **Metric corrections** (recorded so the numbers are reproducible): the injection metric ignores regulation
  references ("R94"), evidence markers ("[E5]") and refutations inside claims; fact coverage tolerates paraphrase for
  non-numeric facts but never relaxes numbers.

## Load

`scripts/eval/load_test.py` → `evals/results/load_*.json` (users, duration, rps, p50/p95/p99, errors, throttled).
Measured on the heuristic configuration (2026-09-12); the cross-encoder adds ~1.2 s per `/search` on CPU.

## Reproduce

```bash
uv run safety-assistant eval-retrieval --dataset evals/datasets/regulatory_v2.yaml        # all legs
uv run python scripts/eval/grid.py --param rerank_top_n 12 24                                # config grid
uv sync --extra eval
uv run python scripts/eval/judged.py --dataset evals/datasets/regulatory_v2.yaml --ragas --ragas-limit 100 --deepeval --deepeval-limit 40
uv run python scripts/eval/generate_cases.py --per-regulation 60 --seed 7 && uv run python scripts/eval/build_dataset.py
```
