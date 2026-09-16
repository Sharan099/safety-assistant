# Safety Assistant v2 — Claude Code Instructions

## Product

Safety Assistant is a passive-safety regulatory intelligence workspace.

Primary user: passive-safety engineer.

Canonical documentation: `README.md` (product, architecture, security model, evaluation, operations) and `docs/ADR/` (decision records).

When code and docs disagree, investigate. Do not silently assume either is correct.

## Highest priorities

1. regulatory correctness;
2. authorization/data isolation;
3. provenance;
4. retrieval quality;
5. safe failure;
6. evidence-first UX;
7. maintainability;
8. cost/performance.

## Architecture invariants

- Apply authorization/source scope before retrieval ranking.
- Only READY/active authorized document versions may be retrieved.
- User uploads never become authoritative organization documents automatically.
- Chat history or memory is never regulatory evidence.
- Factual regulatory claims must map to retrieved evidence.
- Mock/fake embeddings/providers are forbidden in production.
- Private/workspace generated answers are not stored in a shared cross-user answer cache.
- Retrieval-affecting changes require retrieval regression tests.
- Schema changes require migrations unless a clean-baseline reset is explicitly approved/documented.
- Expensive ingestion is asynchronous.
- Failed ingestion artifacts do not enter serving retrieval.
- Do not add autonomous agents to deterministic retrieval without an ADR and demonstrated need.
- Do not add MCP to product architecture without an external interoperability requirement.

## Rebuild rule

Before deleting files:
1. verify/tag baseline;
2. list every path with its replacement in the commit message;
3. classify KEEP/MIGRATE/REWRITE/DELETE;
4. delete only ledger-approved paths after replacement/tests exist.

Git history is the archive. Do not keep a large `old/` tree.

## Frontend

Design principles are in README "Why the architecture looks this way" and the existing components under `frontend/components/`.

Build an evidence-first engineering workbench, not a generic ChatGPT clone.

Core routes:
- login;
- home;
- chat;
- documents;
- upload;
- document detail;
- ingestion;
- settings;
- role-gated admin.

Use the `ui-ux-implementation` skill for frontend work.

## Skills

Use relevant project Skills instead of loading procedures into the root context:
- `product-spec-guardian`
- `ui-ux-implementation`
- `rag-ingestion`
- `retrieval-evaluation`
- `security-review`
- `database-change-review`
- `pre-merge-gate`

## Workflow

For meaningful changes:

```text
inspect
→ state current behavior with file/symbol evidence
→ plan minimal coherent change
→ implement
→ focused tests
→ broader affected tests
→ independent review where risk is high
→ update docs/state
```

Do not perform unrelated refactors.

## Rebuild state

Record decisions as ADRs under `docs/ADR/` and releases in `CHANGELOG.md`.

## Security

Treat user input, uploads, retrieved text, external model output, filenames and URLs as untrusted.

Never expose secrets/auth tokens in logs/frontend.

Use `security-review` for auth, upload, data scope, cache, provider or admin changes.

## Database

Before schema changes:
- inspect current schema/migration head;
- use `database-change-review`;
- document migration/rollback;
- do not drop data by default.

## Evaluation

Preserve known retrieval baselines.

Never hard-code benchmark answers.
Never invent metrics.
Every reported number must be reproducible.

## Frontend quality

Required:
- loading/empty/error/permission states;
- keyboard accessibility;
- responsive behavior;
- Playwright for critical flows;
- browser visual verification at desktop and tablet widths.

## Token/context discipline

- search before reading large files;
- do not repeatedly reread canonical specs;
- load Skills only when relevant;
- at most three concurrent read-only subagents by default;
- subagent depth one by default;
- summarize progress in STATE.md;
- avoid verbose progress narration.

## Approval conditions

Do not proceed without an explicit recorded decision if a change would:
- weaken security;
- broaden data visibility;
- change authoritative-document policy;
- drop/squash DB history;
- reduce retrieval/eval thresholds;
- delete unclassified data/evals;
- add a major framework/service not in TRD.

## Definition of done

A task is not done until:
- behavior works;
- focused tests pass;
- relevant regressions pass;
- security implications reviewed;
- docs/contracts updated if changed;
- no unrelated diff;
- no secrets/temp artifacts;
- the implementation is explainable from code and tests.
