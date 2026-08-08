# Scripts

Active / reusable tooling. One-off diagnostics that already found their bugs live
under [`archive/`](archive/README.md) (preserved as project history, not deleted).

| Script | Purpose |
|--------|---------|
| `prepare_hf_cache.py` | Prefetch embedding/reranker weights into `docker/hf_cache/` for offline Docker builds |
| `manual_test_portkey_fallback.py` | **E2E proof:** invalid Groq key → Portkey falls back → cited `/chat` + dashboard provider |
| `seed_domain_lexicon.py` | Harvest acronym/concept **candidates** from Docling JSON for review (never auto-merges into production configs) |
| `benchmark_gemini_latency.py` | Fix 17/28: probe Gemini + NIM latency/tokens with thinking on vs off |
| `check_secret_patterns.py` | Grep-based scan for common API key / DB password shapes (used by pre-commit) |
| `install_git_hooks.py` | Copy `.githooks/pre-commit` → `.git/hooks/pre-commit` for this clone |
| `backfill_ingested_at.py` | Backfill `ingested_at` on existing Qdrant points (PDF mtime ISO UTC, or `--value unknown`) after schema adds the field |
| `consolidate_golden_set.py` | Rebuild `eval/golden_set.jsonl` + `eval/categories/*.jsonl` shards from the consolidator’s authored case tables — re-run after editing those tables |
| `generate_deepteam_adversarial_goldens.py` | Regenerate DeepTeam adversarial candidates and merge filtered cases into the golden set / category shards |
| `audit_index.py` | Read-only Qdrant `regulations` audit (null sections, pages, bboxes) — prefer this over archived one-off scrolls for new checks |
| `audit_figure_adjacent_chunks.py` | Inspect figure-adjacent chunk linkage; `--json-out` feeds `ingestion.vlm_figure_pass` |
| `confirm_stages_1_3.py` | Confirm Stages 1–3 across R94/R95/R16/R129 (+ optional additive figure upsert) |
| `ab_rerank_min_score.py` | Before/after retrieval-only A/B for `RERANK_MIN_SCORE` / `RERANK_SCORE_MARGIN` |
| `freellmapi_write_config.py` | Build `data/freellmapi/runtime.config.json` from `.env` keys for FreeLLMAPI declarative seed (gitignored) |

## Secret scanning (pre-commit)

Blocks commits whose staged files match Groq / Google / NVIDIA / OpenRouter / HF token
shapes, private-key blocks, DB URLs with embedded passwords, or non-placeholder
assignments to known secret env vars (the failure mode of launcher `.bat`/`.sh` scripts).

```bash
python scripts/install_git_hooks.py          # once per clone
python scripts/check_secret_patterns.py --self-test
```

## Portkey fallback (manual)

This is the single test that proves the "don't exhaust Groq limits" goal works end to
end (not just in config). It temporarily sets an invalid `GROQ_API_KEY`, posts a real
`/chat/sync` through the local Portkey gateway, asserts the answer was served by
NVIDIA NIM / Google / OpenRouter with citations, checks `/metrics/{trace_id}` +
`by_provider` aggregate, then restores `.env`.

**Prerequisites**

1. Local gateway up (no Portkey cloud account):

   ```bash
   docker compose up -d portkey
   # or: npx @portkey-ai/gateway
   # or: docker run --rm -p 8787:8787 portkeyai/gateway:latest
   ```

2. `.env` has a real `GROQ_API_KEY` (to restore) and at least one fallback key
   (`NVIDIA_API_KEY` preferred, else `GOOGLE_API_KEY` / `OPENROUTER_API_KEY`).

3. Qdrant already holds regulations (e.g. UN R94 ingested) so citations can be returned.

```bash
uv run python scripts/manual_test_portkey_fallback.py
```

Optional: `FALLBACK_TEST_PORT=8011` (default) if that port is busy.

## Golden-set maintenance

```bash
# After editing consolidator case tables:
python scripts/consolidate_golden_set.py

# Regenerate DeepTeam adversarial candidates / merge into goldens:
python scripts/generate_deepteam_adversarial_goldens.py
```

## Index metadata backfill

```bash
uv run python scripts/backfill_ingested_at.py
uv run python scripts/backfill_ingested_at.py --value unknown
```
