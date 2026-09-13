# Clean Rebuild / Repository Cleanup Plan

## Rule

A fresh wipe means a clean active working tree, not lost recovery.

Before deletion:

```bash
git status
git tag pre-v2-product-rebuild
git branch backup/pre-v2-product-rebuild
git rev-parse HEAD
```

Record baseline SHA and tests.

## File ledger

For every major tracked path:

| Path | Action | Reason | Replacement |
|---|---|---|---|
| frontend old screen | REWRITE | new product UX | new route/components |
| obsolete script | DELETE | superseded | Make target/new script |
| retrieval module | KEEP/MIGRATE | benchmarked behavior | target module |
| stale report | DELETE | Git retains history | current docs |

No deletion without ledger classification.

## Preserve/migrate proven capabilities

- ingestion lifecycle;
- parser safety/structure extraction;
- versioning;
- hybrid retrieval;
- exact-clause retrieval;
- RRF;
- reranking;
- authorization filters;
- evidence gate;
- citation validation;
- provider abstraction;
- production-fake prohibition;
- observability;
- eval datasets/baselines;
- required migrations/data fixtures;
- hardened container/IaC patterns.

## Rewrite candidates

- frontend structure and design system;
- API presentation layer coupled to old UX;
- conversation persistence if incomplete;
- document workspace API;
- user preferences/memory;
- product docs;
- README;
- Claude instructions and skills;
- tests tied only to deleted UI behavior.

## Delete candidates after replacement verification

- superseded reports;
- temporary logs/output;
- dead scripts;
- duplicate eval harnesses;
- old UI components no longer imported;
- obsolete fixtures;
- experiment code;
- unused dependencies;
- stale screenshots;
- local machine paths;
- deprecated env variables.

Use Git history instead of `archive/old-v1` source trees.

## Database caution

Do not wipe DB/migration history merely because the UI is rebuilt.

Reset/squash only if:
- no deployed data must be preserved;
- owner explicitly accepts clean DB initialization;
- a backup exists where needed;
- clean migration passes full integration/E2E.

Otherwise use forward migrations.

## Final cleanup proof

- clean `git status`;
- no references to deleted paths;
- one current architecture story;
- no obsolete product TODOs;
- fresh clone setup succeeds;
- full tests pass;
- frontend production build passes;
- Docker smoke passes.
