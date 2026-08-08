# CLEANUP_REPORT.md

**Date:** 2026-08-05  
**Scope:** Audit-only re-run of Passes 1–3 (inventory / dead code within files / architecture consistency), with emphasis on eval after real run history exists.  
**Action taken:** **None deleted.** This file is a report only.

**Context that changes the prior audit:** A live batched eval run now exists (`eval/results/20260805T152635Z`, 4/150 cases, resumable). An earlier pre-batching full run died at 76/150 with **zero** persisted case rows. That makes “two implementations of the same concern” and throwaway diagnosis scripts concrete risks, not theoretical ones.

---

## Executive summary

| Priority | Finding | Verdict (recommendation only) |
|----------|---------|-------------------------------|
| P0 | `eval/results/_run_full.bat` embeds live API keys + DB password | Scrub/delete + rotate keys (next cleanup pass) |
| P0 | Three full-eval stacks: `eval.run_full` (batched, live), `eval.run` (CI/README), `evaluation.run` (orphan API 10-case) | Pick one canonical path; demote/archive the others |
| P1 | Dual RAGAS / DeepEval implementations (`eval.ragas_eval`+`deepeval_eval` vs `eval.scoring.*`) | Collapse or label legacy vs current |
| P1 | CI `quality-gate.yml` still runs `eval.run --full`, not `eval.run_full` | Align CI with the batched path or document intentional split |
| P2 | 11/15 `scripts/*.py` undocumented; several with zero external refs | Quarantine / document / delete in a later pass |
| P2 | Empty scaffolds `eval/scoring/{deepteam_scores,retrieval}.py` | Fill or remove |
| P3 | README claims ~40 goldens + `latest.json` tag layout; disk has 150-case JSONL + timestamped `run_full` dirs | Refresh docs |

**Confirmed absences (good news):**
- There is **no** second on-disk copy of `run_full.py` (no `run_full_old.py` / non-batched twin file). The supersession risk is **module-level** (`eval.run` vs `eval.run_full`), not a leftover file duplicate.
- Preflight and smoke each have a **single** implementation (`eval/preflight_check.py`, `eval/smoke_subset.py`), wired into `run_full` and covered by tests.
- Orphaned pre-batching console logs were already removed (see `eval/results/RUNS_STATUS.md`).

---

## Pass 1 — Inventory / dependency graph

### 1.1 Live eval path (what today’s history actually uses)

```
preflight_check ──► smoke_subset ──► run_full (batched)
                         │                │
                         │                ├─► scoring.ragas_scorer
                         │                ├─► scoring.security_scorer
                         │                ├─► scoring.custom_checks
                         │                ├─► partial_results.jsonl + run_config.json
                         │                └─► render_dashboard (full or --partial)
                         └─ lazy-imports score_one_case from run_full
```

| Module | Role | Callers |
|--------|------|---------|
| `eval/run_full.py` | Canonical full orchestrator (batch_size, `--resume`, budget ceiling) | CLI, smoke, dashboard (aggregate helpers), `tests/test_run_full.py` |
| `eval/preflight_check.py` | Per-provider Portkey ping | `run_full`, CLI, `tests/test_preflight_check.py` |
| `eval/smoke_subset.py` | 5-case full-pipeline smoke | `run_full`, CLI, `tests/test_smoke_subset.py` |
| `eval/render_dashboard.py` | PNG dashboard (full + `--partial`) | `run_full` end, CLI, `tests/test_render_dashboard.py` |
| `eval/scoring/ragas_scorer.py` | RAGAS for `run_full` / smoke | `run_full`, `tests/test_ragas_scorer.py` |
| `eval/scoring/security_scorer.py` | Guardrail / injection / hallucination | `run_full`, smoke, tests |
| `eval/scoring/custom_checks.py` | Hard gates | `run_full`, ragas_scorer, tests |
| `eval/nntplib_shim.py` | Py3.13 DeepTeam shim | `run_full`, smoke, security_scorer — **KEEP** |

### 1.2 Parallel / legacy eval stacks (same concern, different modules)

| Stack | Entry | Batching / resume | Scorers | Who still points at it |
|-------|-------|-------------------|---------|------------------------|
| **A — current** | `python -m eval.run_full` | Yes | `eval.scoring.*` | Live ops, unit tests, partial dashboard |
| **B — mid/CI** | `python -m eval.run` / console script `passive-safety-eval` | No | `eval.ragas_eval` + `eval.deepeval_eval` | Root README, `pyproject.toml` entry, `.github/workflows/quality-gate.yml` + `main.yml` |
| **C — orphan** | `python -m evaluation.run` | No (10 fixed HTTP cases) | `evaluation.scoring` | Self + comment in `app/config.py` only; **not** in hatch `packages` |

**This is the “two implementations of the same concern” finding.** There is one `run_full.py` file; the older non-batched concern lives as `eval/run.py` (+ separate `evaluation/` package), not as a second `run_full`.

### 1.3 Scorer duplication graph

| Concern | Implementation A (live full eval) | Implementation B (legacy scorecard) | Implementation C |
|---------|-----------------------------------|-------------------------------------|------------------|
| RAGAS | `eval/scoring/ragas_scorer.py` | `eval/ragas_eval.py` (only `eval/run.py`) | `evaluation/scoring.py::run_ragas` |
| DeepEval / security | `eval/scoring/security_scorer.py` | `eval/deepeval_eval.py` (faithfulness/relevancy for `--full`) | — |
| Custom gates | `eval/scoring/custom_checks.py` | (partially overlapping regression checks in `regression_gate.py`) | — |
| Call-budget estimate | `run_full.estimate_run_llm_calls` | `generation_eval.estimate_groq_calls` | — |

Thin re-exports (optional, not independently used by `scoring/__init__.py`):  
`eval/scoring/{ragas_scores,deepeval_scores,custom}.py`.

Empty scaffolds (no logic):  
`eval/scoring/deepteam_scores.py`, `eval/scoring/retrieval.py`.

### 1.4 Gold / case assets

| Asset | Cases | Status |
|-------|------:|--------|
| `eval/golden_set.jsonl` | 150 | Canonical for `run_full` |
| `eval/categories/*.jsonl` | shards | Produced/maintained via `scripts/consolidate_golden_set.py` |
| `eval/gold.json` | 6 | Legacy fallback in `eval/gold.py` if JSONL missing |
| `evaluation/cases.json` | 10 | Only for orphan `evaluation.run` |
| `evaluation/results.json` | — | Stale artifact dated **2026-07-23**, still on disk / tracked |

### 1.5 `scripts/` inventory vs documentation

Documented in `scripts/README.md` (4):  
`prepare_hf_cache.py`, `manual_test_portkey_fallback.py`, `seed_domain_lexicon.py`, `benchmark_gemini_latency.py`.

**Undocumented (11):**

| Script | External name refs (approx.) | Notes |
|--------|------------------------------|-------|
| `audit_hpc_retrieval.py` | 0 outside itself | One-off HPC retrieval audit |
| `verify_section_fix.py` | 0 | One-off Qdrant scroll |
| `run_determinism_probe.py` | 0 | Thin CLI; logic covered via `determinism_eval` tests |
| `debug_docling_null_sections.py` | self | Debug dump |
| `inspect_ordering.py` | self | Docling order probe |
| `audit_r16_retractor.py` | self | Domain audit (2026-08-05) |
| `audit_figure_adjacent_chunks.py` | self → `eval/results/figure_adjacent/` | Audit artifacts |
| `audit_index.py` | self → mismatch/sample JSON | Index health |
| `backfill_ingested_at.py` | self | One-shot migration |
| `generate_deepteam_adversarial_goldens.py` | self → candidates JSON | Generator — keep if regenerating gold |
| `consolidate_golden_set.py` | categories README | Maintainer tool (~45KB) |

### 1.6 Today’s diagnosis leftovers (2026-08-05)

| Artifact | Status | Recommendation |
|----------|--------|----------------|
| Pre-batching console logs (`run_full_console.*`, `smoke_subset_console.log`) | Already deleted | Done |
| `eval/results/_run_full.bat` | Still present; **plaintext secrets** | Delete/scrub + rotate (P0) |
| `eval/results/20260805T152635Z/` | KEEP — resumable IN PROGRESS 4/150 | Documented in `RUNS_STATUS.md` / `results_index.json` |
| `eval/results/dashboard.png` (partial) | Generated via `--partial` | KEEP as example |
| `fix17_28_thinking_benchmark.json` | Bench output | KEEP or regenerate via documented script |
| `query_intent_audit.jsonl` | Runtime audit from router | Operational log; not an eval run |
| Root `audit_samples.json` (if present untracked) | Dup of audit dump | Candidate delete later |
| `requirements.txt` alongside `pyproject.toml` | Possible leftover pin file | Decide single source of truth |

### 1.7 Test coverage map (eval)

| Covered | Not covered / thin |
|---------|-------------------|
| `run_full`, smoke, preflight, dashboard, ragas_scorer, security_scorer, custom_checks, regression_gate, determinism, Fix 17/28 timeouts | `evaluation/*`, `ragas_eval.py`, `deepeval_eval.py`, `agent_eval.py`, most `scripts/*`, `quality_gate` (CI-only) |

---

## Pass 2 — Dead code within files

### 2.1 Empty / scaffold modules (clear dead weight)

| Path | Evidence | Suggested disposition |
|------|----------|------------------------|
| `eval/scoring/deepteam_scores.py` | Docstring: “Scaffold only — no scoring logic yet.” (3 lines) | DELETE or implement |
| `eval/scoring/retrieval.py` | Same scaffold pattern | DELETE or implement (retrieval metrics already live in `eval/metrics.py` + `retrieval_eval.py`) |
| `eval/scoring/{ragas_scores,deepeval_scores,custom}.py` | Thin re-exports; `__init__.py` imports concrete modules directly | MERGE into `__init__` or DELETE if unused |

### 2.2 Entire modules that look superseded (not unused symbols — unused *stacks*)

| Path | Why it looks dead / superseded | Caveat |
|------|--------------------------------|--------|
| `evaluation/run.py` + `evaluation/scoring.py` + `cases.json` + stale `results.json` | Outside hatch packages; only `app/config.py` comment references; pyc from July | Confirm no offline operator still uses HTTP API eval before delete |
| `eval/run_retrieval_only.py` | Thin wrapper; only self-docstring references | Overlaps default `eval.run` retrieval-only mode |
| `eval/ragas_eval.py` / `eval/deepeval_eval.py` | Only imported by `eval/run.py` | Still required while CI/README use `eval.run --full` |
| `eval/gold.json` | 6-case legacy; JSONL is canonical | Keep only as fallback or delete after `gold.py` stops preferring it |

### 2.3 Within-file notes (important files)

**`eval/run_full.py` (~57KB, 35 top-level symbols)**  
Batching, partial persistence, cost buckets, thresholds, and scoring dispatch all appear wired (also exercised by `tests/test_run_full.py`). No obvious orphaned public helper from manual review. Cost helpers (`_accumulate_case_cost`, `_overall_usage_from_per_category`, etc.) are used in the batch loop — not dead.

**`eval/render_dashboard.py`**  
Partial path (`build_results_from_partial`, banner, pending grey bars) is live and tested. No clear dead exports.

**`eval/preflight_check.py` / `eval/smoke_subset.py`**  
Not prototypes left beside a “real” version — they *are* the production helpers. Early diagnosis may have used console redirects (`_run_full.bat`), not alternate Python modules.

**`eval/regression_gate.py` (~60KB)**  
Large private check surface used by `eval.run` regression path + `tests/test_regression_gate.py`. Not dead; overlaps conceptually with `scoring/custom_checks` (architecture smell, not unused code).

**`eval/condensation_eval.py`**  
Only relevant if `eval.run` still invokes it; not on the `run_full` path. Treat as legacy-scorecard adjunct.

### 2.4 Scripts that were truly throwaway vs promote

| Script | Throwaway? | Promote into `eval/`? |
|--------|------------|------------------------|
| `benchmark_gemini_latency.py` | No | Already documented; keep under `scripts/` |
| `manual_test_portkey_fallback.py` | No | Keep as manual E2E |
| `generate_deepteam_adversarial_goldens.py` | No (generator) | Optional: `eval/tools/` later |
| `consolidate_golden_set.py` | No (maintainer) | Optional: `eval/tools/` later |
| `audit_*`, `debug_docling_*`, `inspect_ordering`, `verify_section_fix`, `run_determinism_probe`, `backfill_ingested_at` | Yes / one-shot | **Do not promote** — quarantine or delete in a deletion pass |

---

## Pass 3 — Architecture consistency

### 3.1 Entry-point drift

| Claimed / wired | Actual primary path today | Consistency |
|-----------------|---------------------------|-------------|
| README “Eval (RAGAS + DeepEval)” → `python -m eval.run` | Operators use `python -m eval.run_full` | **Drift** — README omits `run_full`, preflight, smoke, dashboard, `--resume` |
| `passive-safety-eval = eval.run:main` | Same as mid stack | Console script does not expose batched full eval |
| CI `quality-gate.yml` → `eval.run --full` then `eval.quality_gate` on `latest.json` | `run_full` writes timestamped dirs + `results_index.json`, not the old tag/`latest.json` layout | **CI does not exercise the live batched pipeline** |
| `app/config.py` mentions `python -m evaluation.run` | Package not in hatch wheel | Stale comment / orphan stack |

### 3.2 Coupling smells (not wrong, but sharp edges)

1. **`smoke_subset` ↔ `run_full` lazy cycle** — smoke lazy-imports `score_one_case` because `run_full` imports smoke at load time. Works, but extracting `score_one_case` to `eval/scoring/dispatch.py` (or similar) would clarify the graph.
2. **`render_dashboard` imports `run_full`** for `aggregate_per_category` / thresholds — UI renderer depends on orchestrator module. Prefer shared `eval/aggregate.py` if the graph keeps growing.
3. **Scripts use `sys.path.insert(0, ROOT)`** — bypass package install; inconsistent with `python -m eval.*`.

### 3.3 Product boundaries

| Surface | Purpose | Do not confuse with |
|---------|---------|---------------------|
| `eval/render_dashboard.py` → `dashboard.png` | Offline eval readiness / partial progress | Frontend `MetricsPanel` (live chat cost/latency) |
| `eval/results/RUNS_STATUS.md` + `results_index.json` | Human/machine run ledger | Scorecard `latest.json` from `eval.run` |
| `eval/quality_gate.py` | Post-scorecard CI thresholds | `eval/regression_gate.py` (per-case gates inside scorecard runs) |

### 3.4 Documentation / counts out of date

- README: golden set “~40” → disk has **150** in `golden_set.jsonl`.
- README results path `eval/results/<tag>-…json` + `latest.json` → `run_full` uses `eval/results/<UTC_run_id>/`.
- `scripts/README.md` lists 4 of 15 scripts.

---

## Recommended cleanup order (for a *future* deletion pass — not done here)

1. **Secrets:** scrub/delete `_run_full.bat`; rotate exposed keys.
2. **Canonical CLI decision:** document `eval.run_full` as the full-eval path; either point CI/`passive-safety-eval` at it, or explicitly name `eval.run` “legacy scorecard / CI retrieval+RAGAS”.
3. **Retire or archive `evaluation/`** (+ stale `results.json`) after confirming no users.
4. **Collapse scorer stacks** once CI no longer needs `ragas_eval`/`deepeval_eval`, or wrap them as thin adapters over `eval.scoring.*`.
5. **Remove empty scaffolds** `scoring/deepteam_scores.py`, `scoring/retrieval.py`; drop unused re-export shims if grep stays clean.
6. **Quarantine undocumented one-off scripts** (audit/debug/verify/backfill/determinism probe) into `scripts/_archive/` or delete.
7. **Refresh README + scripts README** (150 cases, `run_full` flow, preflight/smoke/dashboard/`--partial`).

---

## Explicit non-findings (avoid false deletions)

- **No duplicate `run_full.py` file** beside the batched version — do not hunt for a phantom `run_full_legacy.py`.
- **Do not delete** `eval/results/20260805T152635Z/` — legitimate IN PROGRESS run.
- **Do not delete** `preflight_check.py` / `smoke_subset.py` as “prototypes” — they are production.
- **Do not delete** `nntplib_shim.py` — required on Py3.13 for DeepTeam imports.
- **Do not delete** `benchmark_gemini_latency.py` / Fix 17/28 artifacts without replacing the documented latency workflow.

---

## Appendix — Quick reference tree (eval-focused)

```
eval/
  run_full.py          ← LIVE full eval (batched + resume)
  run.py               ← LEGACY/CI scorecard (non-batched)
  run_retrieval_only.py← thin / likely redundant with eval.run default
  preflight_check.py   ← LIVE
  smoke_subset.py      ← LIVE
  render_dashboard.py  ← LIVE (+ --partial)
  ragas_eval.py        ← only for eval.run
  deepeval_eval.py     ← only for eval.run
  scoring/             ← LIVE scorers for run_full
    ragas_scorer.py, security_scorer.py, custom_checks.py
    deepteam_scores.py, retrieval.py   ← EMPTY scaffolds
  golden_set.jsonl     ← 150 cases (canonical)
  gold.json            ← 6-case legacy fallback
  results/
    20260805T152635Z/  ← KEEP (partial)
    RUNS_STATUS.md     ← ledger
    _run_full.bat      ← SECRETS — delete next pass
evaluation/            ← ORPHAN parallel package (not in wheel)
scripts/               ← 4 documented + 11 undocumented one-offs
```
