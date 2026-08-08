# Eval notes — RAGAS NaN modes

## `ground_truth_reference_chunks`

- **Audit (pre-backfill):** `152/152` golden cases lacked `ground_truth_reference_chunks`
  (field absent). Scorer fell back to `expected_chunk_ids` fetch or weak
  `expected_answer_contains` / `expected_behavior` strings.
- **Backfill (fac + xrg):** `factual_lookup` **43/43** and `cross_regulation` **16/16**
  now carry verbatim regulation excerpts (PDF OCR / corpus chunks), spot-checked
  against `expected_answer_contains` (with decimal-comma / acronym aliases).
  Remaining categories still empty pending the same manual treatment.
- Scripts: `scripts/backfill_ground_truth_refs.py`,
  `scripts/backfill_ground_truth_refs_pass2.py` (PDF extraction helpers — not LLM).

## Faithfulness NaN pattern (`20260805T152635Z`)

| | faithfulness NaN (n=27) | scored (n=49) |
|--|--:|--:|
| median answer length | 668 | 186 |
| frac ≤120 chars | 0.0 | 0.10 |
| frac ≤200 chars | 0.11 | 0.57 |
| decline / not_found-ish | 0.04 | 0.33 |
| bullets / headings / checklists | higher | lower |

**Conclusion:** This is **not** an inherent RAGAS short-answer / "not found" limitation
in our suite. Short factual answers usually score; NaNs concentrate on longer
structured / multi-claim answers. Do not chase short-answer prompt workarounds for
this failure mode. See also README § "RAGAS metric caveats".
