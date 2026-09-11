# Production Rebuild Verification Checklist

Use this checklist near final cutover. Do not mark an item complete without evidence.

## Repository
- [ ] One active application architecture
- [ ] No duplicate backend implementations
- [ ] No obsolete imports/entrypoints
- [ ] No machine-specific paths
- [ ] No stale status/session Markdown in active docs
- [ ] README matches actual code/deployment
- [ ] Rebuild ledger records deletions/migrations

## Data and ingestion
- [ ] Immutable source artifact handling
- [ ] Regulation/document/version separation
- [ ] Publication and effective dates
- [ ] Supersession/amendment links
- [ ] Section/table/cross-reference structure
- [ ] Document/section/chunk fingerprints
- [ ] Incremental re-indexing
- [ ] Idempotency
- [ ] Quarantine/DLQ
- [ ] Atomic activation
- [ ] Freshness metrics

## Retrieval
- [ ] Dense baseline
- [ ] Sparse lexical baseline
- [ ] Hybrid fusion
- [ ] Reranking
- [ ] Metadata filters
- [ ] Temporal filters
- [ ] Parent/context expansion
- [ ] Cross-reference behavior bounded
- [ ] Retrieval regression in CI

## Generation
- [ ] Structured output
- [ ] Stable evidence IDs
- [ ] Citation validation
- [ ] Conflict handling
- [ ] Insufficient-evidence response
- [ ] Numeric/unit preservation tests
- [ ] No fake confidence

## Evaluation
- [ ] Versioned dataset
- [ ] Human-reviewed/labelled retrieval ground truth
- [ ] Query slices
- [ ] Recall@k
- [ ] MRR
- [ ] nDCG@k
- [ ] Citation correctness/completeness
- [ ] Groundedness/correctness
- [ ] Refusal metrics
- [ ] Numerical/table metrics
- [ ] Results tied to git SHA/config/model versions

## Security
- [ ] OIDC/RBAC or explicitly documented current auth level
- [ ] Privileged endpoints restricted
- [ ] Authorization before retrieval
- [ ] Upload/fetch validation
- [ ] SSRF controls
- [ ] Parser resource limits
- [ ] Prompt-injection suite
- [ ] Cross-user leakage tests
- [ ] Provider data-classification policy
- [ ] Confidential cache policy
- [ ] Secret/dependency/container scans

## Reliability and operations
- [ ] Liveness/readiness semantics
- [ ] Timeouts/retries/backoff
- [ ] Circuit breaker behavior
- [ ] Queue retry/DLQ
- [ ] Structured logs
- [ ] OpenTelemetry traces
- [ ] Operational metrics
- [ ] SLOs/alerts
- [ ] Load test
- [ ] Fault-injection test
- [ ] Backup/restore procedure
- [ ] Rollback procedure

## Delivery
- [ ] Reproducible Docker build
- [ ] Non-root/hardened runtime where practical
- [ ] Managed production data services or documented production target
- [ ] IaC
- [ ] Staging
- [ ] Controlled migrations
- [ ] Release workflow
- [ ] Current CHANGELOG
