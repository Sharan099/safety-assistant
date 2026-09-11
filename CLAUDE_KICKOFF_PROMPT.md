# Claude Code Kickoff Prompt — Full Production Rebuild

Use the repository's root `CLAUDE.md` as the governing contract for this task.

I want you to perform a **complete production-oriented restructure of this Automotive Safety RAG repository**.

This is an implementation task, not a review-only task.

## Goal

Replace the current active repository structure with the clean final architecture defined in `CLAUDE.md`, while preserving and improving valuable existing behavior.

The old implementation should not remain as an ambiguous second active system. At the end there must be one obvious current application structure, one current README, one current architecture, and one current evaluation path.

## Required behavior

Start by inspecting the actual repository. Do not speculate about files you have not read.

Before deleting or rewriting anything:

1. inspect `git status`, branches/tags, recent log, repository tree, dependencies, tests, CI, deployment, evaluation artifacts, and Markdown;
2. establish a recoverable pre-rebuild checkpoint;
3. create/update `docs/rebuild-ledger.md`;
4. classify current files and components as `KEEP`, `MIGRATE`, `REWRITE`, `ARCHIVE`, or `DELETE`;
5. run the existing tests and evaluation/regression tools and record the baseline;
6. add characterization tests for important current behavior not covered by tests.

Then implement the rebuild milestone-by-milestone in the order defined by `CLAUDE.md`.

## Important architecture direction

Do not turn this into a collection of fashionable frameworks.

Preserve the strong ideas already present when they are sound:

- source validation;
- regulatory metadata/versioning;
- structure-aware parsing;
- dense + sparse hybrid retrieval;
- RRF;
- reranking;
- parent/context expansion;
- cross-reference handling;
- grounded citations;
- provider resilience;
- prompt/context budgeting;
- background ingestion;
- authentication/audit concepts;
- deterministic retrieval regression.

Strengthen the weak areas:

- representative labelled evaluation;
- explicit temporal/current-version semantics;
- incremental changed-section indexing;
- canonical data lineage;
- exact citation validation;
- production-mode dependency failure behavior;
- RBAC/security boundaries;
- confidential cache/provider policy;
- OpenTelemetry/operational metrics;
- load/fault testing;
- object storage;
- repeatable cloud/IaC deployment;
- CI security/evaluation gates;
- documentation consistency.

Do not change storage technology purely for novelty. Keep persistence/retrieval behind interfaces and justify database changes with an ADR and benchmark.

## Skills and subagents

Use project-local skills under `.claude/skills/` when relevant.

Good parallel subagent workstreams are:

- repository/dependency/documentation audit;
- security review;
- evaluation/retrieval-quality review;
- observability/SRE review;
- frontend review.

Keep tightly coupled refactors and single-file edits in the main context.

## Cleanup requirement

Once the replacement implementation passes the required gates:

- remove obsolete active source directories;
- remove duplicate implementations;
- remove stale top-level scripts;
- remove historical session/status Markdown;
- remove superseded architecture and TODO Markdown;
- remove machine-specific paths and contradictory documentation;
- archive only evidence that has durable benchmark/audit value;
- update imports, CI, Docker, README, and docs so nothing points to the old structure.

Do not delete files first and hope the replacement works later.

## Quality gates

Do not declare completion after code compiles.

Run the relevant final gates:

- formatting/lint;
- type checks;
- unit tests;
- integration tests;
- parser golden tests;
- security/adversarial tests;
- retrieval regression;
- evaluation quality gates;
- E2E tests;
- Docker/container build;
- dependency/secret/container scans where configured;
- frontend Playwright tests if the frontend is in scope;
- load/fault tests for production claims.

If a gate cannot be run because credentials/services are unavailable, do not fabricate success. Implement deterministic substitutes/fakes only in explicit test profiles and document exactly what remains unverified.

## Evaluation requirement

Create a versioned evaluation framework with explicit query slices and ground truth. Grow toward 200–500 curated questions. During this rebuild, establish a meaningful seed dataset and the infrastructure to scale it.

Report retrieval separately from generation.

Do not claim an improvement unless before/after results exist.

## Work continuity

This is a long task. Keep state in `docs/rebuild-ledger.md` and git. Do not create one progress Markdown per phase.

At every milestone:

- update the ledger;
- run the smallest meaningful tests/evals;
- fix regressions before proceeding;
- make a coherent checkpoint if permitted.

Continue until either:

A. the Definition of Done is met; or  
B. an external dependency/credential genuinely prevents a specific verification.

In case B, complete everything else, leave the repository in a runnable state, and report the exact remaining blocked verification—do not stop the entire rebuild.

## Final response

When finished, report only verified facts:

- final architecture;
- removed/migrated components;
- test counts by category;
- evaluation dataset size;
- measured metrics;
- security/observability/deployment work completed;
- known limitations;
- local run commands;
- production/staging commands if implemented;
- final git SHA if available.

Begin now by auditing the repository and establishing the rebuild ledger and baseline.
