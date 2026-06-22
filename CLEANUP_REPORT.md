# CLEANUP_REPORT.md — Phase 0 Repo Audit & Cleanup Plan

**Status:** **EXECUTED** — Phase 0 complete (2026-06-22).  
**Scope:** Passive-safety RAG corpus, ingestion layout, stale artifacts, registry alignment.

---

## Executive summary

The indexed corpus has **50 PDFs** and **28,341 chunks**. Retrieval noise comes from:

1. **Wrong-domain docs** (ISO 26262 functional safety, FuSa training, ADS/cyber regs, GDPR CELEX, etc.)
2. **Duplicate / overlapping sources** (UN R16 as both `UN_R16.pdf` and `ECE-R-16-Regulation.pdf`; R137 as `UN_R137.pdf` and `R137e.pdf`; three Safety/CAE companion variants)
3. **Reference handbooks treated like regulations** (`SAFETY_COMPANION`, `CAE_COMPANION`) — mix frontal/side/rear values → contradictory chest-deflection answers
4. **Ingestion Python code living beside PDFs** under `data/` (should be code-only separation)
5. **Stale artifacts** (`output/markdown/`, `regulation_chunks.json`, `regulation_embeddings.json`, `ingest_manifest.json`) all reference removed/duplicate docs until re-ingest

**Recommended outcome:** shrink active corpus to **~18–22 primary sources**, archive the rest, re-tag companions as `reference`, remove ISO 26262 from index, then **full re-ingest → re-chunk → re-embed**.

---

## 1. Corpus classification (all 50 indexed PDFs)

### 1.1 KEEP — core passive-safety corpus (16 files)

| File | Role | Notes |
|------|------|-------|
| `data/UN_R94.pdf` | Legal — frontal (ODB) | Primary |
| `data/UN_R137.pdf` | Legal — full-width frontal | Primary |
| `data/UN_R95.pdf` | Legal — side impact | Primary |
| `data/UN_R135.pdf` | Legal — pole side | Primary |
| `data/UN_R14.pdf` | Legal — belt anchorages | Primary |
| `data/UN_R16.pdf` | Legal — belts / restraints | Primary |
| `data/UN_R17.pdf` | Legal — seats / head restraints | Primary |
| `data/FMVSS_208.pdf.pdf` | Legal — US frontal | Primary |
| `data/EURO_NCAP_FRONTAL.pdf.pdf` | Rating — frontal protocol | Primary |
| `data/EURO_NCAP_SIDE.pdf.pdf` | Rating — side protocol | Primary |
| `data/EURO_NCAP_REAR.pdf.pdf` | Rating — rear protocol | Primary |
| `data/EURO_NCAP_VRU.pdf.pdf` | Rating — VRU protocol | Primary |
| `data/regulations/euro-ncap-protocol-crash-protection-frontal-impact-v11.pdf` | Rating — frontal (alt) | **Dedup review** — keep one frontal NCAP source |
| `data/regulations/euro-ncap-protocol-crash-protection-side-impact-v11.pdf` | Rating — side (alt) | **Dedup review** |
| `data/regulations/euro-ncap-protocol-overall-assessment-v100.pdf` | Rating — overall | Optional keep |

### 1.2 RE-TAG — keep in archive corpus, `doc_type: reference`, down-rank (4 files)

| File | Action | Reason |
|------|--------|--------|
| `data/SAFETY_COMPANION.pdf.pdf` | **Re-tag** `reference` | Handbook; mixes test types — caused contradictory limits |
| `data/CAE_COMPANION.pdf.pdf` | **Re-tag** `reference` | Same |
| `data/regulations/CAE-Companion-2025-26.pdf` | **Dedup** → archive duplicate | Near-duplicate of CAE companion |
| `data/regulations/SafetyCompanion-2026.pdf` | **Dedup** → archive duplicate | Near-duplicate of Safety companion |

Phase 1 will add: exclude from `legal_limit` queries; `value_type` classifier per chunk.

### 1.3 REMOVE from index — archive PDF, delete markdown/chunks (30 files)

| File | Reason |
|------|--------|
| `data/ISO_26262.pdf.pdf` | **Functional safety (ASIL)** — not crashworthiness; drives wrong ASIL answers |
| `data/regulations/GlobalSpec-ASIL-Rating-Article.pdf` | FuSa / ASIL marketing article |
| `data/regulations/SGS TUEV Brochure Training FUSA Auto EN A4 13 1.pdf` | ISO 26262 training brochure |
| `data/regulations/13069a-ads2.0_090617_v9a_tag (1).pdf` | Autonomous driving / ADS — not passive safety corpus |
| `data/regulations/R155e (2) (1).pdf` | UN R155 cybersecurity — out of scope |
| `data/regulations/R156e (2).pdf` | UN R156 SW updates — out of scope |
| `data/regulations/R152am7e.pdf` | Active safety (pedestrian AEB) — not passive crashworthiness |
| `data/regulations/CELEX_32016R0679_EN_TXT.pdf` | GDPR — noise |
| `data/regulations/CELEX_32010L0040_EN_TXT.pdf` | EU type-approval framework — low signal / huge |
| `data/regulations/CELEX_32018R0858_EN_TXT.pdf` | EU regulation — verify; likely noise |
| `data/regulations/CELEX_32019R2144_EN_TXT.pdf` | EU regulation — verify; likely noise |
| `data/regulations/CELEX_32022R2236_EN_TXT.pdf` | EU regulation — verify; likely noise |
| `data/regulations/CELEX_42010X0528(03)_EN_TXT.pdf` | EU document — noise |
| `data/regulations/CELEX_42015X0710(01)_EN_TXT.pdf` | EU document — noise |
| `data/regulations/CELEX_42020X0486_EN_TXT.pdf` | EU document — noise |
| `data/regulations/CELEX_42020X0576_EN_TXT.pdf` | EU document — noise |
| `data/regulations/OJ_L_202502161_EN_TXT.pdf` | EU OJ — noise |
| `data/regulations/ECE-R-16-Regulation.pdf` | **Duplicate** of `UN_R16.pdf` |
| `data/regulations/R137e.pdf` | **Duplicate** of `UN_R137.pdf` |
| `data/regulations/R017r6e.pdf` | Likely overlaps `UN_R17.pdf` — archive after diff |
| `data/regulations/R029r2e.pdf` | Tyre regulation — out of scope |
| `data/regulations/R029r2am5e.pdf` | Tyre regulation — out of scope |
| `data/regulations/R127r3am3e.pdf` | Pedestrian — optional VRU; defer to Phase 1 |
| `data/regulations/engproc-85-00038.pdf` | Academic paper — noise |
| `data/regulations/d24106255.pdf` | Unknown internal doc — noise |
| `data/regulations/28-inf01.pdf` | Infotainment — noise |
| `data/regulations/tp-208-14_tag.pdf` | TPMS — noise |
| `data/regulations/whitepaper-virtual-validation-at-in-tech.pdf` | Vendor whitepaper — optional archive |
| `data/regulations/4.-Final_New_Submit_Introduction_to_Passive_Safety_Onoda_MMC_240528.pdf` | Slide deck — low structure |
| `data/regulations/UsedVehicles_Kigali_Passive_General_Safety.pdf` | Policy doc — noise |
| `data/regulations/[E_]ECE_TRANS_WP.29_GRSP_2010_20-EN.pdf` | Working party minutes — noise |

**Archive destination (proposed):** `archive/corpus_removed/` (PDFs preserved, not indexed).

---

## 2. Code & folder hygiene (proposed moves)

### 2.1 Current layout problem

```
data/
  *.py                    ← ingestion code (should not sit with corpus)
  *.pdf                   ← core corpus (good)
  regulations/
    *.pdf                   ← mixed corpus + duplicates
```

### 2.2 Target layout

```
data/
  corpus/                   ← PDFs only (+ manifest JSON)
    legal/
    rating/
    reference/              ← companions (indexed but down-ranked)
  manifest/
    corpus_manifest.json    ← authoritative list of indexed doc_ids

ingestion/                  ← moved from data/*.py
  __init__.py
  paddle_ocr_converter.py
  docling_converter.py
  hierarchical_chunker.py
  embed_chunks.py

archive/
  corpus_removed/           ← ISO_26262, CELEX, duplicates, etc.
```

### 2.3 Files to MOVE (not delete)

| From | To |
|------|-----|
| `data/paddle_ocr_converter.py` | `ingestion/paddle_ocr_converter.py` |
| `data/docling_converter.py` | `ingestion/docling_converter.py` |
| `data/hierarchical_chunker.py` | `ingestion/hierarchical_chunker.py` |
| `data/embed_chunks.py` | `ingestion/embed_chunks.py` |
| `data/__init__.py` | `ingestion/__init__.py` (or remove if empty) |

**Import updates required in:** `config.py`, `scripts/run_batch_ingestion.py`, `scripts/run_ingestion_pipeline.py`, `scripts/overnight_run.py`, `scripts/compare_ocr_pipeline.py`, `README.md`, tests.

### 2.4 PDF moves (after confirmation)

| From | To |
|------|-----|
| `data/UN_*.pdf`, `data/FMVSS_*.pdf`, `data/EURO_NCAP_*.pdf` | `data/corpus/legal/` or `data/corpus/rating/` |
| `data/SAFETY_COMPANION.pdf.pdf`, `data/CAE_COMPANION.pdf.pdf` | `data/corpus/reference/` |
| 30 removed PDFs | `archive/corpus_removed/` |

---

## 3. Registry & config changes (Phase 0 prep, apply after corpus cut)

| File | Change |
|------|--------|
| `backend/app/core/document_registry.py` | Remove or disable `ISO` entry; split `EURO_NCAP` per protocol; add `doc_type=reference` flags for companions |
| `config.py` `SUPPORTED_REGULATIONS` | Remove `ISO`; keep `CAE_REFERENCE`, `SAFETY_REFERENCE` as reference-only |
| `ingestion/hierarchical_chunker.py` `detect_regulation_type()` | Stop mapping `"iso"` → ISO; narrow `"safety"` keyword (too broad) |
| `frontend/src/app/page.tsx` | Remove "ISO 26262" from example questions / sidebar after corpus cut |

---

## 4. Stale artifacts — REGENERATE (do not hand-edit)

After corpus cleanup, **delete and rebuild**:

| Artifact | Action |
|----------|--------|
| `output/markdown/*.md` for removed/archived docs | Delete matching files |
| `output/ingest_manifest.json` | Regenerate via ingestion pipeline |
| `output/regulation_chunks.json` | Regenerate via `hierarchical_chunker` |
| `output/regulation_embeddings.json` | Regenerate via `embed_chunks` |
| `output/chunking_diagnostics.txt` | Regenerate |
| `hf-space-push/output/*` | Re-sync via `prepare_hf_space.ps1` after rebuild |

**Do not delete** until corpus files are archived and ingestion paths are updated.

---

## 5. Files proposed for DELETION (pending your OK)

### 5.1 Markdown outputs (30 files — mirrors removed PDFs)

All under `output/markdown/` matching Section 1.3 filenames, e.g.:

- `ISO_26262.pdf.md`
- `GlobalSpec-ASIL-Rating-Article.md`
- `SGS TUEV Brochure Training FUSA Auto EN A4 13 1.md`
- All `CELEX_*.md` (9 files)
- `ECE-R-16-Regulation.md`, `R137e.md`, `R017r6e.md`, … (full list in Section 1.3)

### 5.2 Duplicate markdown (keep newer canonical source)

- `CAE-Companion-2025-26.md` if `CAE_COMPANION.pdf.md` kept
- `SafetyCompanion-2026.md` if `SAFETY_COMPANION.pdf.md` kept

### 5.3 Evaluation test cases (update, not delete)

- `tests/test_cases_5.json` — remove `R_ISO` ASIL question (ISO removed)
- `tests/test_cases_20.json` — audit for ISO / companion-only cases

### 5.4 No source code deleted in Phase 0

Only moves + archive. Old `data/*.py` removed **after** imports point to `ingestion/`.

---

## 6. Phase 0 verification checklist (run after cleanup executes)

| # | Check | Command / test |
|---|--------|----------------|
| 1 | ISO not in manifest | `python scripts/audit_corpus.py --assert-no-iso` |
| 2 | Core 16 docs present | `audit_corpus.py --assert-core` |
| 3 | No ingestion `.py` under `data/` | `audit_corpus.py --assert-layout` |
| 4 | Chunk count dropped sensibly | expect ~40–60% fewer chunks vs 28,341 |
| 5 | Unique chunk IDs | chunker abort-on-duplicate (existing guard) |
| 6 | Embeddings match chunks | `embed_chunks` validation |
| 7 | `document_registry` has no ISO index entry | unit test |

---

## 7. Risk notes

- **HF / Railway deploy:** Re-embed ~200–400 MB JSON + LFS push after rebuild.
- **Backward compatibility:** Add `CORPUS_VERSION=2` flag in `config.py` until Phase 4 filtering ships.
- **Companion books:** Keeping them indexed but down-ranked (Phase 1) is safer than deleting — engineers still want CAE context for non-legal queries.
- **Euro NCAP dedup:** Recommend keeping `data/EURO_NCAP_*.pdf.pdf` quartet; archive the three `euro-ncap-protocol-*.pdf` duplicates unless diff shows unique content.

---

## 8. Execution results

| Item | Result |
|------|--------|
| Corpus PDFs | 17 (8 legal, 7 rating, 2 reference) |
| Archived | 33 PDFs → `archive/corpus_removed/` |
| Chunks | **14,554** (was 28,341) |
| Embeddings | **14,554** (1:1 match) |
| OCR errors | 0 |
| Phase 0 tests | `tests/test_phase0_audit.py` — 9/9 passed |

Re-ingest log: `output/phase0_reingest.log` (~7h total, embedding ~6.5h on CPU).

---

## 9. Confirmation (resolved)

Approved 2026-06-22 — executed as written.

---

## 10. Next phases (preview)

| Phase | Deliverable |
|-------|-------------|
| 1 | Metadata schema + chunk-level classifier |
| 2 | Chunk quality gate + `scripts/inspect_chunks.py` |
| 3 | `scripts/benchmark_models.py` + `output/model_selection.md` |
| 4 | Hard metadata pre-filter in `hybrid.py` |
| 5 | Conditional system prompt + output sanitizer |
| 6 | Frontend upload + `POST /documents` |
| 7 | `tests/golden_set.json` + CI regression |
| 8 | Full unit / integration / e2e suite |

---

## Appendix A — Ingestion code already outside `data/regulations/`

These are in `data/` root (not in `regulations/`) but still violate "data = PDFs only":

- `paddle_ocr_converter.py`
- `docling_converter.py`
- `hierarchical_chunker.py`
- `embed_chunks.py`
- `__init__.py`

## Appendix B — Current indexed count by category (approximate)

| Category | Count |
|----------|------:|
| Core legal UN/FMVSS | 8 |
| Euro NCAP | 7 |
| Reference companions | 4 |
| Remove (wrong domain / noise) | ~30 |
| **Total** | **50** |
