# Testing, Evaluation and Release Plan — Safety Assistant v2

## Test categories

```text
unit
integration
api_contract
parser_golden
security
retrieval_regression
ingestion_e2e
frontend_unit
playwright_e2e
load
```

## Critical tests

### Authentication
- protected unauthenticated route → 401;
- insufficient role → 403;
- valid membership → allowed.

### Conversation
- create;
- restore;
- pagination;
- archive;
- user isolation;
- citation persistence.

### Upload
- valid PDF accepted;
- invalid magic rejected;
- oversized rejected;
- failed parse not READY;
- duplicate behavior;
- private scope;
- workspace scope;
- cross-user isolation.

### Ingestion
- lifecycle success;
- retry;
- poison document;
- idempotent rerun;
- version replacement;
- atomic READY activation.

### Retrieval
- authorization filter before exposure/ranking;
- stale versions excluded;
- expected clause ranking;
- exact clause;
- dense/sparse/RRF/rerank regression.

### Generation
- invented citation blocked;
- unsupported numeric/id claim rejected/removed;
- insufficient evidence abstains;
- provider outage → evidence only.

## Frontend E2E

Required:

```text
1. login → new chat → answer → open evidence
2. login → upload PDF → processing → READY → ask uploaded document
3. logout → login → restore conversation
4. user A private upload → user B denied
5. failed upload → actionable error UI
```

## Evaluation

Preserve the current retrieval baseline before v2 changes.

Expand progressively toward 200–500 labelled questions:
- clause;
- numeric;
- table;
- definition;
- version;
- multi-clause;
- cross-regulation;
- ambiguous;
- unanswerable;
- adversarial.

Do not gate a pure frontend restyle on LLM-judge metrics. Do gate retrieval-affecting changes on deterministic retrieval regression.

## CI gate

```text
lint/format
→ type check
→ unit
→ security
→ integration with Postgres/Redis
→ parser golden
→ retrieval regression
→ frontend checks
→ Playwright
→ dependency/secret scan
→ Docker build
→ container scan
→ SBOM
```

Use fast PR gates and heavier release/nightly gates if needed.

## Release

```text
PR
→ gates
→ staging
→ migrations
→ smoke
→ E2E
→ release approval
→ production
→ post-deploy smoke
```

Rollback must account for app/schema/index compatibility.

## Completion evidence

README metrics must include dataset/config/git SHA/environment context. Never replace missing measurements with aspirational numbers.
