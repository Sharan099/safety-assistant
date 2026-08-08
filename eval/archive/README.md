# Deprecated eval fixtures (archived)

These files are **not** used by the live eval path (`eval.gold.load_golden_set` →
`eval/golden_set.jsonl`, CI → `eval/ci_gate.jsonl`).

| File | Former role |
|------|-------------|
| `gold.json` | 6-case JSON array; was a silent fallback when `golden_set.jsonl` was missing |

Do not restore the silent fallback. Load this file only in explicit legacy tests.
