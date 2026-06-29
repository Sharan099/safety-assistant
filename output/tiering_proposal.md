# Step 5 tiering proposal (write-up only — NOT implemented)

**Status:** PROPOSAL — final decision waits on authoritative Claude baseline.  
**Date:** 2026-06-27

## Context

- **Fast tier** is now `openai/gpt-oss-20b` (Groq), not decommissioned Llama 3.1 8B.
- **Power tier** remains `llama-3.3-70b-versatile`.
- Gateway failover order (post fix): `70B → 20B → Anthropic → evidence-only`.
- Timing eval (`output/timing_eval_5q.json`, post gateway fail-fast): healthy LLM steps ~0.9–1.7s on 70B/20B when tiers available; worst ~4.3s when Anthropic connect timeout + evidence-only.

## Eval question mapping

| Case | Type | 70B needed? | Timing total | LLM ms | Notes |
|------|------|-------------|-------------:|-------:|-------|
| R94_chest_deflection | Single-clause numeric limit | **Fast candidate** | 3563 | 1679 | One clause (5.2.1.4 ThCC 42 mm); simple extraction |
| R14_isofix_annex6 | Named annex + table grid | **70B** | 1576 | 887 | Table in top-8 but synthesis still weak (ISOFIX framing); needs table reasoning |
| R16_retractor_types | Multi-type definitions + lock conditions | **70B** | 4256 | 3820* | Spans several §2.12.x definitions; structured list |
| R94_frontal_injury_criteria | Multi-criterion inventory | **70B** | 4262 | 3375* | Many injury criteria across 5.2.1.x; comparison-style breadth |
| R44_vs_R129_child_restraints | Cross-regulation comparison | **70B** | 346 | 1* | Multi-reg filter; needs 2048 output budget; evidence-only in timing run |

\*Timing run hit rate-limit / circuit-breaker cascade; LLM ms not representative of steady-state 70B.

## Proposed routing rules (if tiering is enabled later)

**Route to 20B fast tier:**
- Single regulation filter (`regulation_code` = one UN_Rxx).
- Question asks for one numeric limit, one definition, or one clause citation.
- Prompt fits under fast-tier effective token budget (~8k) without comparison expansion.
- Examples: R94 chest deflection, simple “what does §X say about Y” lookups.

**Keep on 70B:**
- Named annex / table extraction (R14 Annex 6 grid).
- Multi-clause injury-criteria or inventory questions (R94 frontal criteria).
- Cross-regulation comparisons (R44 vs R129) — already uses expanded query + `max_output_tokens=2048`.
- Section-spanning synthesis (R16 retractor types across §2.12.x).

**Never tier on retrieval alone** — routing should use query classifiers (comparison flag, annex/table mention, multi-reg filter) already partially present in `registry/search.py`.

## Is tiering still worth it?

**Marginal, not urgent.**

| Factor | 8B era | 20B era |
|--------|--------|---------|
| Quality gap vs 70B | Large | **Narrower** — 20B is a mid-tier model |
| Latency savings | ~30–50% when 70B succeeded | ~0.9–1.7s vs ~1.0–1.7s in timing eval — **small** |
| Risk | Obvious quality drops on tables | Subtle errors (R14 ISOFIX disclaimer) on fast tier |
| Ops complexity | Failover + rate limits | Same, plus tier mis-routing hurts trust |

**Recommendation:** Defer Step 5 implementation until authoritative Claude scores confirm which cases are quality-sensitive. If faithfulness on R94_chest_deflection is ≥0.85 on 70B and fast-tier shadow matches within 0.05, a **single-case fast route** for simple numeric lookups may be justified. Do **not** tier R14/R44/R94-frontal/R16 without explicit approval.

## What we need from authoritative eval

1. Per-question faithfulness delta if answers were re-generated on 20B (shadow mode — not run here).
2. Confirm R14 table question remains the hardest synthesis case.
3. Revisit cost model: 20B vs 70B Groq pricing may not justify routing complexity at n≈5 eval scale.

**FINAL tiering decision: yours, after `output/ragas_report_authoritative.csv` exists.**
