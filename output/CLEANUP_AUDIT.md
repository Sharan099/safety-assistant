# Repository Cleanup Audit

**Generated:** 2026-06-29  
**Scope:** Full repo scan for Phase 2 (audit only — **no deletions performed**).  
**Live pipeline:** `app/main.py` → `api/routes.py` → `registry/search.py` → `backend/app/gateway/` + `vectorization/` + `database/` + `scheduler/`.

---

## Summary counts (approximate)

| Category | Count | Action |
|----------|-------|--------|
| ACTIVE | Core packages + live scripts | Keep |
| AUDIT-TRAIL | Logs, backups, quarantine, PRDs, manifests | **Never delete** without explicit exception |
| SUPERSEDED | Older RAGAS CSVs/logs, deprecated fetch scripts | Candidate for Phase 3 removal after your approval |
| UNCLEAR | `hf-space-push/`, `sessions/`, root `config` duplicates | Review before any delete |

---

## ACTIVE — required by live pipeline

### Core Python packages
| Path | Role |
|------|------|
| `app/` | FastAPI entry (`app/main.py`), settings |
| `api/` | REST routes (`/api/v1/chat`, `/health`, `/coverage`) |
| `registry/` | Search, coverage, ingest validation, harness, margin, chat intent |
| `backend/app/gateway/` | LLM gateway (imported by `registry/search.py`) |
| `database/` | SQLAlchemy models + session |
| `vectorization/` | Nomic embedder, reranker, structure chunker, indexer |
| `ingestion/` | Clause/table chunking (used by `ingest_storage`, structure chunker) |
| `parser/` | PDF parsing (`pdf_parser`, `structure_extract`) |
| `scheduler/` | Celery tasks + APScheduler |
| `crawler/` | UNECE/Euro NCAP/NHTSA crawlers (scheduler + allowlist tests) |
| `regulation_discovery/registry/version_parser.py` | Filename → reg metadata (ingest) |
| `frontend/src/` | Next.js chat UI |
| `tests/` | Pytest suite (75+ tests) |

### Live scripts (`scripts/`)
| Script | Role |
|--------|------|
| `ingest_offline_reg.py` | Manual PDF batch ingest (primary acquisition path) |
| `ocr_quarantine_ingest.py` | OCR recovery for scanned PDFs |
| `ingest_storage.py` | Storage → chunk → embed pipeline |
| `ingest_harness_test_records.py` | Harness test_results ingest |
| `run_ragas_eval.py` / `run_ragas_score.py` / `run_ragas_evaluation_final.py` | RAGAS eval harness |
| `ragas_plot_summary.py` | PNG chart from CSV |
| `run_e2e_demo.py` | Live demo capture |
| `print_query_timing.py` | Latency breakdown |
| `test_llm_apis.py` | Groq/Anthropic smoke test |
| `diagnose_retrieval_gaps.py` / `investigate_r16_retractor_retrieval.py` | Retrieval diagnostics |
| `live_security_regression.py` / `live_margin_regression.py` | Security/margin probes |
| `create_harness_tables.py` | DB schema for harness |
| `check_db_schema.py` | Schema verification |
| `run_structure_rechunk.py` | Structure re-chunk batch |
| `consolidate_downloads.py` | Download folder hygiene |

### Config & data (active)
| Path | Role |
|------|------|
| `coverage_expected.yaml` | FR-17 coverage spec |
| `docker-compose.yml` | Redis / optional full stack |
| `requirements.txt` | Python deps |
| `safety_registry.db` | Live corpus DB |
| `storage/UNECE/` | Validated PDF storage |
| `data/staging/` | Manual ingest drop zone |
| `tests/data/ragas_cases.json` | RAGAS question set |

---

## AUDIT-TRAIL — do not delete (project history & evidence)

Per governance rule — **excluded from cleanup even if “unused”:**

| Path | Reason |
|------|--------|
| `output/HUMAN_ACTION_RUNBOOK.md` | Human acquisition runbook |
| `output/CONSOLIDATION_FINAL.md` | Consolidation evidence |
| `output/OCR_QUARANTINE_LOG.md` | OCR recovery log |
| `output/HARNESS_AND_FETCH_LOG.md` | Harness/fetch audit |
| `output/AUTONOMOUS_RUN_LOG.md` | Autonomous run log |
| `output/FINAL_CONSOLIDATION_LOG.md` | Milestone log |
| `output/*_diagnosis*` / `output/unece_403_diagnosis.md` | Network/corpus diagnoses |
| `output/acquired_manifest.json` | Acquisition manifest |
| `data/backups/*.db` | DB snapshots (incl. 22/22 milestone) |
| `data/quarantine/` / `data/quarantine_harness/` | Quarantined PDFs/records |
| `Passive_Safety_Assistant_PRD.md` | Product PRD |
| `Regulation_KnowledgeBase_Registry_PRD.md` | Registry PRD |
| `coverage_expected.yaml` | Coverage contract |
| `archive/corpus_removed/` | Removed PDF archive (provenance) |

---

## SUPERSEDED — safe to remove **after your approval** (successor named)

### RAGAS reports (older runs → `output/ragas_evaluation_final.csv`)
| File | Superseded by | Why safe |
|------|---------------|----------|
| `output/ragas_report.csv` | `ragas_evaluation_final.csv` | Pre-retrieval-fix 5-case run |
| `output/ragas_report_before_retrieval_fix.csv` | same | Baseline before dense/sparse fix |
| `output/ragas_report_after_retrieval_fix.csv` | same | Intermediate 5-case |
| `output/ragas_report_clean.csv` | same | Groq-interim judge, 5 cases |
| `output/ragas_report_authoritative.csv` | same | 5-case Claude attempt; incomplete vs 10-case final |
| `output/ragas_report*.retrieval.json` (all except `ragas_evaluation_final.retrieval.json`) | `ragas_evaluation_final.retrieval.json` | Stale retrieval snapshots |
| `output/ragas_eval_run.log`, `ragas_eval_run2.log` | final eval logs | Console captures only |
| `output/ragas_score_run.log`, `ragas_score_run2.log`, `ragas_score_groq.log`, `ragas_score_groq2.log`, `ragas_score_final.log`, `ragas_score_after_fix.log` | final run | Stale judge logs |
| `output/ragas_retrieval_after_fix.log` | N/A | One-off log |

### Deprecated acquisition scripts (superseded by manual ingest runbook)
| File | Superseded by | Why safe |
|------|---------------|----------|
| `scripts/acquire_open_network.py` | `output/HUMAN_ACTION_RUNBOOK.md` + `ingest_offline_reg.py` | UNECE 403 blocks headless; runbook marks deprecated |
| `scripts/prep_missing_regs_fetch.py` | same | Fetch probe only; not production path |
| `scripts/run_live_crawler.py` | manual staging ingest | Live crawler blocked on this network |

### One-off / merged utility scripts
| File | Superseded by | Why safe |
|------|---------------|----------|
| `scripts/finish_ragas_clean.py` | `run_ragas_score.py` | Ad-hoc re-score; logic merged into score script |
| `scripts/patch_retrieval_cases.py` | `run_ragas_eval.py` | Dev patch utility |
| `scripts/list_authoritative_ragas_bundle.py` | `run_ragas_evaluation_final.py` | Bundle checklist superseded |

### Old eval artifacts
| File | Superseded by | Why safe |
|------|---------------|----------|
| `output/chunking_ab_baseline.json` | `output/structure_rechunk_report.json` | Earlier A/B |
| `output/embed_run.log`, `embed_run2.log`, `embed_run3.log`, `embed_run_v2.log` | current DB state | Embed batch logs |
| `output/r95_reingest_*.log` | `safety_registry.db` | Single-reg reingest logs |
| `output/golden_retrieval_eval_run.log`, `golden_retrieval_eval_run2.log` | `golden_retrieval_eval.json` | Duplicate logs |
| `output/overnight_run.log`, `phase0_*.log`, `paddle_ocr_all.log`, `batch_ingest.log` | milestone logs in AUDIT-TRAIL | Operational logs (optional delete — **not AUDIT-TRAIL by name**; mark optional) |

---

## UNCLEAR — do not delete without explicit decision

| Path | Notes |
|------|-------|
| `hf-space-push/` | Full HuggingFace Space deployment bundle (parallel app stack). **Not imported** by local `app/main.py`, but may still be the publish target for HF Space. Confirm before removal. |
| `sessions/` | Local session workspace samples (`ingest_sess01`, etc.) — appears copied from HF session-ingest dev; **not referenced** by live API. Likely dev fixtures. |
| `monitoring/` | Only `__init__.py` — empty stub, never imported. |
| `docs/` | May contain design notes — verify contents before classifying. |
| `archive/corpus_removed/` | **AUDIT-TRAIL** (listed above) — not superseded despite “archive” name. |
| `scripts/r14_synthesis_retest.py`, `r14_table_synthesis_70b.py`, `probe_groq_20b.py`, `probe_faithfulness_20b.py`, `resynth_case.py` | R14 investigation one-offs; no imports from pipeline. Keep until R14 table synthesis is stable, or archive to `scripts/archive/`. |
| `scripts/diagnose_r14_annex6.py`, `diagnose_retrieval_gap.py` | Diagnostic — still useful; not superseded. |
| `regulation_discovery/` (except `version_parser.py`) | Large config/crawler discovery module — only `version_parser` imported by live ingest; rest may be future/legacy. |
| Root `config.py` (if present) vs `app/config.py` | Verify no duplicate; live app uses `app/config.py`. |
| `frontend/node_modules/`, `frontend/.next/`, `__pycache__/` | Regenerable build artifacts — safe to gitignore/clean locally, **do not commit-delete**. |
| `output/e2e_demo_results.json` / `.md` | Demo snapshot — useful regression reference; not strictly AUDIT-TRAIL. |

---

## Phase 3 gate

1. Review SUPERSEDED table above and reply with approved paths.
2. Agent will: **git commit current state** → run full tests + `/api/v1/ready` → delete approved items only → re-test → **separate cleanup commit**.
3. AUDIT-TRAIL items will **not** be deleted unless you explicitly name an exception.

---

## Recommended first cleanup batch (smallest, lowest risk)

If you want a minimal approved set to start:
- `output/ragas_report.csv`
- `output/ragas_report_before_retrieval_fix.csv`
- `output/ragas_report_after_retrieval_fix.csv`
- `output/ragas_report_clean.csv`
- `output/ragas_report_authoritative.csv`
- `output/ragas_report.retrieval.json`
- `output/ragas_report_after_retrieval_fix.retrieval.json`
- `output/ragas_report_authoritative.retrieval.json`
- `output/ragas_report_interim_20bjudge.retrieval.json`
- `output/ragas_*_run.log` (all ragas score/eval console logs listed above)

**Keep:** `output/ragas_evaluation_final.csv`, `output/ragas_evaluation_final.png`, `output/ragas_evaluation_final.retrieval.json` (once eval completes).
