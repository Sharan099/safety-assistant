# Security threat model

| Threat | Control | Test |
|---|---|---|
| Direct prompt injection in questions | signals detected and recorded; question is data inside `<question>`; answers only from validated claims | `tests/security/test_adversarial.py` |
| Indirect injection inside documents | evidence is data by contract; numbers/ids validated; source allowlist limits who can put text in the corpus | same; boundary documented in test |
| Malicious/oversized/wrong-type files | magic bytes, size cap, page cap, SHA-256 vs registry; quarantine with attempt budget | `test_wrong_magic_bytes_oversize_and_bad_hash_are_refused`, e2e quarantine |
| SSRF / redirects / internal networks | https only, host allowlist, public-IP DNS check per hop, manual redirect re-validation, size cap, content type | `tests/security/test_fetcher_ssrf.py` |
| Path abuse in storage | server-generated content-addressed keys; `..`/absolute URIs refused | `test_blob_store_keys_are_server_generated_and_traversal_is_refused` |
| Cross-user / data-class leakage | principal data classes narrow SQL scope before ranking; evidence lookup filtered the same way | `tests/security/test_data_isolation.py` |
| Cross-user / workspace / organization document leakage | one authorization predicate (`retrieval/authz.py`) evaluated in SQL before ranking and mirrored in the BM25 pre-filter; listings/detail/jobs/evidence use it; foreign ids look like missing ids (404) | `tests/unit/test_authz_equivalence.py`, `tests/integration/test_documents_upload.py`, `tests/security/test_user_isolation.py` |
| Session theft / CSRF on the browser session | HttpOnly SameSite=Lax cookie, `typ=session` claim, secret ≥ 32 chars in production, `X-Requested-With` required on cookie-authenticated mutations, rotation invalidates | `tests/unit/test_identity_sessions.py`, `tests/security/test_user_isolation.py` |
| Unsafe uploads (wrong type, oversized, poison PDF, path abuse) | bounded stream read, PDF magic, size cap, safe filename, server-generated content-addressed keys, full validation in the worker, quarantine is terminal, public error text only | `test_invalid_magic_and_oversized_are_rejected_at_the_boundary`, `test_poison_pdf_is_quarantined_*` |
| Silent authoritative promotion of uploads | scope changes only via `POST /documents/{id}/promote` with `document:promote`; READY required; audit event | `test_promotion_is_privileged_and_audited` |
| Conversation history used as regulatory evidence | prior turns rendered as `<conversation_context>` data with an explicit prompt rule; citations validated against current-request evidence only | `test_conversation_context_is_passed_as_data_not_evidence` |
| Judge/telemetry exfiltration during evaluation | RAGAS and DeepEval telemetry opted out; judges call the same gateway as the product; caches on disk are gitignored | `scripts/eval/judged.py` |
| Unauthorised privileged endpoints | scope checks on ingest/audit/admin routes; 401 without token, 403 without scope; audit actor recorded | `test_privileged_endpoints_require_auth_and_scope` |
| Stale / wrong-version answers | only ACTIVE (or ACTIVE|SUPERSEDED for as-of) versions retrievable; validity windows filtered in SQL | `test_non_active_versions_are_never_retrievable`, e2e temporal tests |
| Hallucinated citations / numbers | evidence-id existence + numeric fidelity validation; answer withheld if nothing survives | `test_hallucinated_citation_is_rejected`, unit tests |
| Provider policy violation (confidential data to an uncleared LLM) | `LLM_DATA_CLASSES` policy → evidence-only | `test_confidential_evidence_never_reaches_a_public_only_llm` |
| Resource exhaustion | request size limit (2,000 chars), rate limit per principal, page/size caps, bounded agent budgets, threadpool offload | rate limiter + budget tests, fault injection |
| Secrets exposure | no secrets in code/images; env/secret manager; production refuses dev password; gitleaks in CI | `test_production_refuses_fakes_and_dev_defaults` |
| Supply chain | uv.lock, pip-audit, bandit, semgrep, Trivy image scan, SBOM, non-root read-only container | `.github/workflows/security.yml`, `ci.yml` |

Residual risks: the injection detector is a signal, not a filter (by design); indirect injection that repeats genuine numbers passes numeric validation — human review and source allowlisting are the controls; the in-process rate limiter is per replica; OIDC has not been exercised against a live IdP in this repository; no malware scanner is wired into the upload path (boundary exists: `documents.service.create_upload`); the free LLM gateway used in development routes prompts to third-party free tiers — `LLM_DATA_CLASSES` keeps confidential evidence out of it unless the operator clears the provider.

Retention: `query_traces` store queries and principal ids (no tokens); define a retention job before storing personal data at scale.
