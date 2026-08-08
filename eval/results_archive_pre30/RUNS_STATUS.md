# Eval run status

Human-readable ledger of timestamped folders under `eval/results/`.
Machine index: `results_index.json` (same statuses).

Statuses: `IN PROGRESS - N/TOTAL` | `COMPLETE` | `ABANDONED`

| run_id | status | artifacts | notes |
|--------|--------|-----------|-------|
| `20260805T152635Z` | **COMPLETE** | `partial_results.jsonl`, `results.json`, `dashboard.png`, `scoring_providers_by_batch.jsonl`, `resume_overflow_console.log` | Full 150/150 with `EVAL_JUDGE_OVERFLOW_CONFIG`. FreeLLMAPI carried most judge/scoring after Groq/Google caps. Gate: **NOT PRODUCTION READY** (exit 1 = thresholds, not a crash). ~$1.18, 4.24h. |

## Cleaned up (no folder left)

| former run | status | what happened |
|------------|--------|---------------|
| Pre-batching full run (~2026-08-05 16:30 local) | **ABANDONED** | Died at case 76/150 on HTTP 429. Predated `partial_results.jsonl` checkpointing - **zero scored cases on disk**. Console logs (`run_full_console.{err,log}`, `smoke_subset_console.log`) removed 2026-08-05 as clutter. |
| `livebatchtest1` | **ABANDONED** | Verification-only test run; artifacts already deleted. No dangling code/config references remain (only this ledger notes the cleanup). |

## Non-run folders

| folder | purpose |
|--------|---------|
| `figure_adjacent/` | Sample figure-adjacent chunk extracts (not an eval run). |
