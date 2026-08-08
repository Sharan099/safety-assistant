# Archived: `evaluation/` package

Moved here from the repo root as a **one-cycle cleanup buffer** (do not delete
yet). This was the orphan HTTP/API RAGAS stack (`python -m evaluation.run`),
parallel to and superseded by `eval.run_full` + `eval/golden_set.jsonl`.

- Not in the hatch wheel / not imported by live app or CI.
- **`scoring.py` removed** (2026-08-05 cleanup) — RAGAS for live eval is only
  `eval/scoring/ragas_scorer.py`; security is only `eval/scoring/security_scorer.py`.
- `run.py` is a stub that exits with instructions to use `eval.run_full`.
- Stale `results.json` is kept on disk for history but **must not** be git-tracked.
- Nested `archive/cases.json` is the old 10-case fixture.
