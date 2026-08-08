# Archived diagnostic scripts

These are **historical** one-off diagnostics from this project's development.
They found real bugs (chunk metadata gaps, retrieval acronym failures, ELR
mis-chunking, non-deterministic retrieval). Preserve them as part of the
debugging story — they are not junk, but they are also not the live ops path.

Re-run from the repo root if you need to reproduce an old finding, e.g.
`python scripts/archive/verify_section_form.py`. Active maintenance scripts
stay in `scripts/` (see `scripts/README.md`).

| Script | What it found / verified |
|--------|--------------------------|
| `audit_hpc_retrieval.py` | Diagnosed HPC acronym expansion + dense/sparse/hybrid rank failures for “What is the HPC limit in UN R95?” |
| `verify_section_form.py` | Qdrant scroll that quantified missing `section_number` metadata (the R94 audit that reported ~24.8% nulls) |
| `run_determinism_probe.py` | Thin CLI over `eval.determinism_eval` — ran the checklist question 5× to confirm identical chunk sets |
| `debug_docling_null_sections.py` | Used during the R94 chunking metadata audit that found 24.8% null section numbers — inspected Docling annex/table heading shapes |
| `inspect_ordering.py` | Dumped Docling emit order for page/marker ranges while debugging clause order and heading association |
| `audit_r16_retractor.py` | Read-only sqlite payload audit that showed UN R16 ELR (§6.2.5.3) text buried under wrong/`null` section tags |
