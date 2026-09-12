# Safety Assistant v2 — Master Rebuild Plan

## Mission

Rebuild **Safety Assistant** as a production-oriented **Passive Safety Regulatory Intelligence Workspace**.

The primary user is a passive-safety engineer who needs to:

1. sign in securely;
2. ask technical questions across approved passive-safety regulations;
3. receive evidence-grounded answers with exact citations and provenance;
4. upload project/regulatory documents;
5. let the backend validate, parse, chunk, embed and index those documents asynchronously;
6. query uploaded documents together with the approved regulatory corpus, subject to authorization;
7. return to persistent conversation history;
8. maintain per-user preferences/history without allowing memory to become regulatory truth;
9. inspect document ingestion status, versions and failures;
10. trust that stale, unauthorized or incomplete evidence is never silently presented as authoritative.

## Existing strengths to preserve

The latest rebuild report already shows a strong backend: lifecycle ingestion, quarantine, atomic supersession, PostgreSQL + pgvector, hybrid retrieval with dense/BM25/exact-clause legs, RRF, reranking, parent/xref expansion, deterministic evidence gates, citation validation, RBAC/OIDC/API keys, observability, hardened containers, IaC and a working Next.js UI.

The v2 goal is therefore:

```text
preserve verified backend capabilities
        +
simplify architecture and product boundaries
        +
rebuild the frontend deliberately
        +
add first-class document workspace/upload UX
        +
add durable per-user conversations
        +
add safe user preferences/memory
        +
remove obsolete code/docs/scripts from the active tree
```

## Non-negotiable product rules

### 1. Evidence outranks memory

Chat history and preferences may influence continuity, wording and defaults, but never replace current retrieval evidence for a regulatory claim.

```text
current authorized corpus → retrieval → evidence → answer
```

not:

```text
old assistant answer → repeated as truth
```

### 2. Uploaded documents are scoped

Every document has an explicit scope:

```text
AUTHORITATIVE_ORG
WORKSPACE
PRIVATE_USER
```

User uploads default to `PRIVATE_USER` or an explicitly selected workspace. Promotion to `AUTHORITATIVE_ORG` requires a privileged approval flow.

### 3. Uploaded files use the full ingestion pipeline

Never:

```text
upload → embed immediately → query
```

Use:

```text
upload
→ object storage
→ validation
→ quarantine if unsafe/invalid
→ parsing
→ structure extraction
→ chunking
→ embedding
→ indexing
→ verification
→ READY
```

Only `READY` versions can participate in retrieval.

### 4. This is an engineering workbench, not a chatbot skin

The interface prioritizes:

- citations;
- source provenance;
- version;
- page/clause;
- retrieval scope;
- active workspace;
- document status;
- evidence state;
- traceability.

## Fresh-rebuild safety boundary

A fresh wipe means a clean active tree, not destruction of recoverability.

Before deletion:

```bash
git status
git tag pre-v2-product-rebuild
git branch backup/pre-v2-product-rebuild
git rev-parse HEAD
```

Claude must classify every tracked path into:

```text
KEEP
MIGRATE
REWRITE
DELETE
```

and write the decision to `docs/rebuild/FILE_LEDGER.md` before deletion.

Git history is the archive. Do not keep an `old/` implementation tree unless there is a specific runtime requirement.

## Target repository layout

```text
safety-assistant/
├── CLAUDE.md
├── AGENTS.md
├── README.md
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── Makefile
├── pyproject.toml
├── .env.example
│
├── src/safety_assistant/
│   ├── api/
│   ├── auth/
│   ├── domain/
│   ├── documents/
│   ├── ingestion/
│   ├── parsing/
│   ├── chunking/
│   ├── retrieval/
│   ├── generation/
│   ├── conversations/
│   ├── memory/
│   ├── policy/
│   ├── persistence/
│   ├── providers/
│   ├── observability/
│   └── workers/
│
├── frontend/
│   ├── app/
│   ├── components/{ui,shell,chat,evidence,documents,auth}/
│   ├── features/
│   ├── hooks/
│   ├── lib/
│   ├── styles/
│   └── tests/
│
├── migrations/
├── tests/{unit,integration,security,parser_golden,retrieval_regression,ingestion_e2e,api_contract}/
├── evals/{datasets,baselines,results}/
├── docs/{product,architecture,security,operations,adr,rebuild}/
├── infra/{docker,terraform,monitoring}/
├── scripts/{evaluation,maintenance}/
├── .claude/skills/
└── .github/workflows/
```

## Rebuild sequence

### R1 — design before frontend code

Approve:
1. problem and personas;
2. information architecture;
3. five primary user flows;
4. low-fidelity wireframes;
5. visual direction;
6. design tokens;
7. component inventory;
8. high-fidelity layouts;
9. accessibility rules;
10. responsive behavior.

### R2 — identity/workspace model

Implement or verify:

```text
User
Organization
Membership
Workspace
WorkspaceMembership
Conversation
Message
UserPreference
```

### R3 — document workspace

Implement:
- upload;
- document list/detail;
- source scope;
- version;
- ingestion status;
- failure states;
- retry where safe;
- archive/delete policy.

### R4 — ingestion lifecycle

Use explicit statuses:

```text
UPLOADED
VALIDATING
QUARANTINED
PARSING
CHUNKING
EMBEDDING
INDEXING
VERIFYING
READY
FAILED
ARCHIVED
```

### R5 — conversation product

Implement:
- new conversation;
- rename;
- list/search;
- archive;
- persistent messages;
- persistent citations;
- selected source scope;
- resume conversation.

### R6 — memory

MVP memory is intentionally small:

```text
preferred language
answer density
default workspace
recent regulation/source preferences
UI preferences
optional conversation summary for context compression
```

No hidden factual regulatory memory.

### R7 — evidence-first answer experience

Every answer can expose:

```text
answer
answer mode
citations
document title
regulation/version
clause/section
page
validity
source scope
source hash/id
evidence excerpt
```

## Definition of v2 MVP

The rebuild is complete only if:

- user can sign in;
- user sees an authorized workspace;
- user can create/resume conversations;
- messages/citations persist after logout/login;
- user can upload an allowed PDF;
- ingestion runs asynchronously;
- UI shows honest ingestion state;
- failed documents cannot be retrieved;
- READY private/workspace documents are searchable by authorized users;
- another user cannot retrieve private data;
- citations open evidence/source details;
- authoritative and private/workspace evidence are distinguishable;
- authorization is enforced in backend;
- cross-user leakage tests pass;
- retrieval regression remains above accepted baseline;
- E2E covers login → upload → READY → ask → evidence;
- fresh clone setup works;
- obsolete scripts/docs/components are gone from the active tree.

## Final completion report required from Claude

```text
1. starting SHA
2. final SHA
3. final tree summary
4. deleted-file ledger
5. preserved/migrated capability ledger
6. schema/migration summary
7. API contract summary
8. UI route map
9. tests by category
10. retrieval metrics vs baseline
11. security checks
12. Playwright result
13. Docker smoke result
14. known limitations
15. exact local run commands
```
