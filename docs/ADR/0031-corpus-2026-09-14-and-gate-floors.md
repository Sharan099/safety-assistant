# ADR-0031 — The 42-source corpus and re-baselined retrieval floors

Status: accepted · Date: 2026-09-14 · Related: 0030, README "Measured results"

## Decision
The verified corpus is the curated set built by `scripts/maintenance/build_registry.py` from the owner's delivery of 52 PDFs: 42 sources (27 UNECE texts, 49 CFR 571, four Euro NCAP protocols, an ISO 26262 overview, seven CAE manuals/sheets, two vendor handbooks). Duplicates, an older revision and an already-incorporated amendment sheet are recorded as dropped in the manifest; amendment sheets newer than their consolidated text are separate sources with their own key (`UN-R94-AMEND-05`) so they never supersede the full text; scanned texts are merged and OCR'd. `scripts/maintenance/prune_corpus.py` removes what left the registry and, on request, non-active versions, so the database holds exactly one version in force per source.

The regression-gate floors are re-baselined to the new corpus with the existing convention (measured value in the test profile minus a margin): v1 MRR 0.588 → 0.55, v2 MRR 0.648 → 0.62, v2 R@10 0.930 → 0.90 unchanged. The document-level gate becomes relative (SAC never worse than the baseline) because the absolute mismatch rate is a property of the corpus.

## Why
The larger corpus contains more near-identical clauses by design (R137 next to R94, R135 next to R95, R44 and R14 next to R129/R16, vendor handbooks that restate limits), so every absolute retrieval number is lower than on the 16-source corpus. The old floors would fail on any code change and on none; floors that guard regressions of the *code* have to be measured on the corpus the gate runs against. Two gold cases that targeted removed NHTSA reports were dropped; `r94-004` now accepts UN R137 as well because R137 §5.2.2.1 states the identical steering-wheel limits.

## Consequences
Measured results are reported per corpus in the README; the 16-source tables remain as history. Re-measure and re-record here when the registry changes again.
