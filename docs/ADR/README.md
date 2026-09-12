# Architecture Decision Records

One file per decision, sequential, never renumbered. ADRs 0001–0018 document the
pre-rebuild system (Passive Safety CAE Investigation Agent) and are kept as history;
where a rebuild ADR supersedes one it says so. The current architecture is ADR-0019
onward.

| ADR | Decision | Status |
|---|---|---|
| 0019 | PostgreSQL + pgvector single store; new schema/database cutover | accepted |
| 0020 | Structural chunking along the clause tree | accepted |
| 0021 | Hybrid dense + BM25 with RRF + exact-identifier leg | accepted |
| 0022 | Heuristic reranker default, cross-encoder opt-in | accepted |
| 0023 | Temporal regulation model (versions, validity, lifecycle) | accepted |
| 0024 | Citation contract, programmatic validation, abstention | accepted |
| 0025 | Bounded LangGraph agent | accepted |
| 0026 | Typed providers; fakes only in the test profile | accepted |
| 0027 | Security policy (allowlists, RBAC before retrieval, LLM data classes, no shared cache) | accepted |
| 0028 | Deployment on managed container infrastructure | accepted (Terraform unapplied) |
| 0001–0018 | pre-rebuild decisions | historical |
