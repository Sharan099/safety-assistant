# Application Flows — Safety Assistant v2

## First login

```text
Login
→ identity verified
→ membership resolved
→ default workspace resolved
→ Home
→ recent conversations + document status
```

Failures:
- no membership → access-request/contact-admin state;
- IdP failure → retry;
- suspended user → denied.

## Ask verified regulations

```text
New investigation
→ source scope = Verified regulations
→ question
→ authentication
→ authorization policy
→ hybrid retrieval
→ evidence gate
→ generation
→ citation validation
→ persist messages + citations
→ render answer
→ engineer opens evidence
```

## Upload private document

```text
Documents
→ Upload
→ choose PDF
→ scope = Private
→ object storage
→ document/version created
→ job queued
→ VALIDATING
→ PARSING
→ CHUNKING
→ EMBEDDING
→ INDEXING
→ VERIFYING
→ READY
→ Ask this document
```

## Upload workspace document

Same pipeline, but access comes from workspace membership. Promotion to authoritative corpus is never automatic.

## Promote authoritative document

```text
Knowledge Admin
→ document detail
→ review provenance/metadata/validation
→ promote
→ privileged confirmation
→ authoritative lifecycle
→ active organization corpus
→ audit event
```

## Continue conversation

```text
Login
→ history
→ open thread
→ load messages + citations + source scope
→ continue question
→ use current authorized retrieval corpus
```

Old assistant text is context only. Current technical claims are grounded in current evidence.

## Ingestion failure

```text
Upload
→ processing
→ FAILED / QUARANTINED
→ UI shows safe reason + failed stage + diagnostic reference
→ retry/replace if allowed
```

Failed versions remain excluded from retrieval.

## Evidence-only fallback

```text
retrieval succeeds
→ generation unavailable/disallowed
→ answer_mode = EVIDENCE_ONLY
→ render evidence
→ persist mode
```

## Insufficient evidence

```text
question
→ retrieval/evidence below threshold
→ INSUFFICIENT_EVIDENCE
→ explain searched scope
→ optionally suggest broadening authorized scope or uploading material
```

Never invent a likely regulation.

## Cross-user security flow

For every conversation/document route:

```text
resource id
→ authenticated user
→ membership/policy filter
→ authorized resource query
```

Do not fetch globally first and rely on UI checks.
