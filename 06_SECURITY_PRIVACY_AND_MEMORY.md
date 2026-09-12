# Security, Privacy and User Memory — Safety Assistant v2

## Trust boundaries

Untrusted:
- user input;
- uploaded files;
- retrieved document text;
- external provider output;
- filenames;
- remote URLs;
- query parameters.

Trusted only after verification:
- authenticated identity;
- authorized membership;
- validated metadata;
- active document versions;
- server-side policy decisions.

## Memory model

Do not collapse different meanings of “memory.”

### A. Conversation history
Durable DB record of messages and citations. Used for resume/audit/continuity.

### B. Conversation summary
Optional compact context derived from older turns for token efficiency.

Rules:
- not a citation source;
- not regulatory truth;
- conversation-scoped;
- can be regenerated.

### C. User preferences
Explicit settings such as default workspace, language, answer density and theme.

### D. Regulatory knowledge
Never user memory. It lives in approved document versions, chunks, metadata and indexes.

## Multi-user isolation tests

```text
User A cannot list User B conversations.
User A cannot fetch User B conversation by guessed UUID.
User A cannot retrieve User B private chunks.
User A cannot see User B ingestion job.
Workspace A cannot retrieve Workspace B data.
Admin privilege cannot leak across organization boundaries.
```

## File upload security

Enforce:
- max bytes;
- max pages;
- magic/MIME;
- safe server-generated object key;
- no client-controlled filesystem path;
- timeouts;
- parser isolation where feasible;
- checksum;
- quarantine;
- optional malware scanner.

PDF text is untrusted content.

## Prompt injection

A document may contain instructions such as “ignore previous instructions.” Treat them as document content, not system instructions.

Defenses:
- instruction hierarchy;
- evidence isolation;
- no arbitrary tool execution from retrieved text;
- adversarial evals;
- citation/output validation.

## Provider data policy

Each provider config should declare:

```text
allowed_data_classes
region
retention policy
contract/DPA metadata where applicable
```

Do not route confidential content based on model-name substring logic.

## Audit

Audit privileged/security-relevant actions such as document promotion, role change, retry/reindex, archive/delete and admin corpus operations.

Do not log secrets or raw sensitive content unnecessarily.

## Delete semantics

Define separately:

```text
archive conversation
delete conversation
archive document
delete document
delete source object
delete chunks/embeddings
```

## Frontend security

- no provider keys in browser;
- CSP/security headers;
- safe Markdown rendering;
- sanitize/validate external links;
- avoid unsafe raw HTML rendering;
- do not trust frontend role flags;
- protect auth state;
- no sensitive content in client telemetry by default.
