# Human Action Runbook

Manual steps that require a normal browser or paid API keys. **Automated UNECE acquisition is not supported** — the official register is behind Cloudflare bot protection that blocks headless browsers. Download PDFs yourself; this project ingests what you drop into `data/staging/`.

---

## 1. Download UNECE passive-safety regulations (manual browser)

### Why base + series both matter

Each regulation needs **two** PDFs for full coverage:

1. **Base regulation** — the original consolidated text (paragraphs like R16 §7.6.2 live here).
2. **Current amendment series** — the latest “NN series of amendments” consolidating changes.

Ingesting **only** the amendment series creates a **silent partial gap**: the regulation appears in the DB, but base-only paragraphs are missing (e.g. R16 retractor locking §7.6.2 was unreachable until `UN_R16_base.pdf` was ingested).

### Workflow

1. Open the addenda index for the regulation’s number range (table below).
2. Find the regulation row on that page.
3. Download the **base** PDF (often named like `R016r6e.pdf` or similar — revision without `am`).
4. Download the **latest amendment series** PDF (often named like `R016r6am8e.pdf` — contains `am` + series number).
5. Rename to the **Save as** filenames in the table (recommended) and copy both into `data/staging/`.
6. Run batch ingest (section 2).

Official register root: [UNECE WP.29 1958 Agreement standards](https://unece.org/transport/vehicle-regulations-wp29/standards)

### All 21 expected UNECE regulations

| Reg | Title | Addenda page | Base — save as | Series — save as (current as of audit) |
|-----|-------|--------------|----------------|----------------------------------------|
| **R11** | Door latches and hinges | [0–20](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-0-20) | `UN_R11_base.pdf` | `UN_R11_<NN>Series.pdf` *(check index for latest NN)* |
| **R12** | Steering mechanism — driver protection | [0–20](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-0-20) | `UN_R12_base.pdf` | `UN_R12_04Series.pdf` |
| **R14** | Safety-belt anchorages and ISOFIX | [0–20](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-0-20) | `UN_R14_base.pdf` | `UN_R14_07Series.pdf` |
| **R16** | Safety-belts and restraint systems | [0–20](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-0-20) | `UN_R16_base.pdf` | `UN_R16_08Series.pdf` |
| **R17** | Seats, anchorages and head restraints | [0–20](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-0-20) | `UN_R17_base.pdf` | `UN_R17_09Series.pdf` |
| **R21** | Interior fittings | [21–40](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-21-40) | `UN_R21_base.pdf` | `UN_R21_02Series.pdf` |
| **R25** | Head restraints (legacy) | [21–40](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-21-40) | `UN_R25_base.pdf` | `UN_R25_<NN>Series.pdf` *(check index)* |
| **R29** | Commercial vehicle cab protection | [21–40](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-21-40) | `UN_R29_base.pdf` | `UN_R29_02Series.pdf` |
| **R32** | Rear-end collision structural behaviour | [21–40](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-21-40) | `UN_R32_base.pdf` | `UN_R32_<NN>Series.pdf` *(check index)* |
| **R33** | Frontal collision structural behaviour (legacy) | [21–40](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-21-40) | `UN_R33_base.pdf` | `UN_R33_<NN>Series.pdf` *(check index)* |
| **R34** | Prevention of fire risks (fuel tanks) | [21–40](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-21-40) | `UN_R34_base.pdf` | `UN_R34_<NN>Series.pdf` *(check index)* |
| **R42** | Front and rear protective devices (bumpers) | [41–60](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-41-60) | `UN_R42_base.pdf` | `UN_R42_<NN>Series.pdf` *(check index)* |
| **R44** | Child restraint systems (legacy) | [41–60](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-41-60) | `UN_R44_base.pdf` | `UN_R44_04Series.pdf` |
| **R80** | Seats / anchorages (buses & coaches) | [61–80](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-61-80) | `UN_R80_base.pdf` | `UN_R80_03Series.pdf` |
| **R94** | Frontal collision occupant protection | [81–100](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-81-100) | `UN_R94_base.pdf` | `UN_R94_04Series.pdf` |
| **R95** | Lateral collision occupant protection | [81–100](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-81-100) | `UN_R95_base.pdf` | `UN_R95_05Series.pdf` |
| **R100** | Electric powertrain safety | [81–100](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-81-100) | `UN_R100_base.pdf` | `UN_R100_02Series.pdf` |
| **R127** | Pedestrian safety | [121–140](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-121-140) | `UN_R127_base.pdf` | `UN_R127_03Series.pdf` |
| **R129** | Enhanced Child Restraint Systems (i-Size) | [121–140](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-121-140) | `UN_R129_base.pdf` | `UN_R129_04Series.pdf` |
| **R135** | Pole side impact | [121–140](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-121-140) | `UN_R135_base.pdf` | `UN_R135_01Series.pdf` |
| **R137** | Frontal impact — restraint systems focus | [121–140](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-121-140) | `UN_R137_base.pdf` | `UN_R137_01Series.pdf` |
| **R153** | Fuel / electric integrity in rear impact | [141–160](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-141-160) | `UN_R153_base.pdf` | `UN_R153_01Series.pdf` |

**Notes**

- Series numbers marked *(check index)* change over time — always take the **highest** series number listed on the addenda page.
- Official browser downloads often keep UNECE names (`R016r6e.pdf`, `R016r6am8e.pdf`). The ingester can parse these, but renaming to the table above avoids ambiguity.
- **R94** is the reference “complete” regulation in the DB (base + series). Use it as a template.
- Optional supplementary amendments (e.g. R94 Amend 3) may be ingested as separate files with `amend` in the filename.

### Messy filenames — staging manifest

If you keep the browser’s original filename, create `data/staging/manifest.yaml`:

```yaml
files:
  - filename: R016r6e.pdf
    regulation_code: UN_R16
    amendment: Base
  - filename: R016r6am8e.pdf
    regulation_code: UN_R16
    amendment: "08 Series"
```

See `data/staging/manifest.yaml.example` for a starter template.

---

## 2. Batch ingest from staging

Drop all PDFs into `data/staging/`, then run:

```powershell
$env:PYTHONPATH="."
python scripts/ingest_offline_reg.py
```

**What the script does (every run)**

1. Snapshots `safety_registry.db` → `data/backups/safety_registry_batch_backup_<timestamp>.db`
2. Processes **every** `.pdf` in `data/staging/` (one file = one ingest attempt)
3. Maps each file via filename → `coverage_expected.yaml`, text-layer fallback, or `manifest.yaml`
4. Validates: `%PDF-` magic, opens, regulation id in text, minimum text layer
5. Promotes to `storage/`, version-resolves, structure-chunks, embeds, indexes
6. Quarantines failures to `data/quarantine/` with a reason
7. Skips duplicates (checksum / text-hash) — idempotent re-runs are safe
8. Writes run log to `output/HARNESS_AND_FETCH_LOG.md`
9. Prints a summary; exits non-zero if any file is UNMAPPED, REJECTED, or ERROR

**Single-file overrides**

```powershell
python scripts/ingest_offline_reg.py data/staging/R016r6e.pdf --regulation-code UN_R16 --amendment Base
```

**Verify coverage after ingest**

```powershell
python -c "from database.connection import SessionLocal; from registry.coverage import build_coverage_report; import json; print(json.dumps(build_coverage_report(SessionLocal())['summary'], indent=2))"
```

Target: `complete_count` = 21 (each reg has a base document, not amendment-only).

---

## 3. Authoritative Ragas evaluation & scoring

An authoritative Claude-based evaluation scores answers to the 5 baseline questions in `tests/data/ragas_cases.json`.

### (a) API keys

The `.env` file may contain `OPENROUTER_API_KEY`; if it returns `401 Unauthorized`, set a valid `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`.

For OpenRouter judging: `RAGAS_JUDGE_PROVIDER=openrouter`, `RAGAS_JUDGE_MODEL=anthropic/claude-sonnet-4`.

### (b) Commands

Generate retrieval snapshot locally (no LLM judges):

```powershell
$env:PYTHONPATH="."
python scripts/run_ragas_eval.py --skip-ragas
```

Output: `output/ragas_report.retrieval.json`

Score on an open network with a valid API key:

```powershell
python scripts/run_ragas_score.py output/ragas_report.retrieval.json output/ragas_report_authoritative.csv --judge-provider anthropic --require-anthropic
```

---

## 4. Key `.env` settings

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq (llama-3.3-70b) for pipeline answers |
| `ANTHROPIC_API_KEY` | Claude judge for Ragas scoring |
| `OPENROUTER_API_KEY` | Failover for Claude / Llama |
| `ENABLE_CROSS_REFERENCE_EXPANSION` | Expand retrieval via cross-referenced sections |

---

## 5. Do not use (deprecated)

- `scripts/acquire_open_network.py` — blocked by Cloudflare; will not succeed for UNECE.
- `scripts/prep_missing_regs_fetch.py` — raw HTTP; returns 403 from UNECE WAF.
- `scripts/run_live_crawler.py` — online crawl path; not viable for UNECE backfill.

Use manual browser download + `scripts/ingest_offline_reg.py` only.
