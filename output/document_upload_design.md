# User Document Upload — Design (crash / simulation data)

**Status:** Phase A0 **COMPLETE** — Phase A (structured upload) **unblocked**  
**Author:** Senior engineering review  
**Date:** 2026-06-29 (rev. 3)  
**Principle:** Reuse-first. Extends **harness domain** (`tests` / `test_results`) and **session-upload mode** (`sessions/` + structure chunking). Not a third ingestion subsystem.

---

## 0. Product decisions (locked)

| # | Decision | Detail |
|---|----------|--------|
| 1 | **Clause at upload: required** | Search-and-select against live corpus; not free-text clause refs. |
| 2 | **`user_id`: verified identity** | Session cookie → `get_current_user()` (`registry/auth.py`). |
| 3 | **Original PDF storage** | `storage/confidential/uploads/{user_id}/{upload_id}/original.pdf` |
| 4 | **Retention** | Until uploading user deletes; no MVP auto-expiry. |

---

## 1. Phase A0 — session auth (DONE)

| Component | Path |
|-----------|------|
| User + session tables | `database/models.py` (`User`, `AuthSession`, `Test.owner_user_id`) |
| Auth helpers | `registry/auth.py` — `get_current_user`, httpOnly cookie |
| Login API | `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`, `GET /api/v1/auth/me` |
| Seed users | `scripts/seed_auth_users.py` — `engineer_a`, `engineer_b`, `lead` |
| Harness isolation | `registry/harness_security.py` — owner check + audit with real `user_id` |
| Frontend | Login screen + guest mode (regulation-only); `credentials: include` |
| Tests | `tests/test_auth_sessions.py` — 401, audit, cross-user 403 |

**Runbook:** `python scripts/seed_auth_users.py` then login with `AUTH_SEED_PASSWORD` (default `changeme`).

---

Engineers need to upload crash-test / CAE artifacts and ask questions blending **regulation corpus** with **confidential data**.

| Need | Exists | Gap |
|------|--------|-----|
| Structured test numbers | `Test` / `TestResult` + harness ingest | JSON-only; no PDF upload |
| Narrative reports | HF `session_ingest.py` | Not wired to local API |
| Charts | — | Attach-only (post-MVP) |
| Confidentiality gates | `harness_security.py` | No verified `user_id` |
| UI upload | `/documents/upload` → **public** staging | Wrong path for confidential data |

---

## 3. Three upload types (one router)

```
Chat UI: Upload + type selector
        │
        ▼ POST /api/v1/user-uploads  (requires get_current_user)
Upload router (registry/user_upload.py)
  • confidential_tier=True default
  • store: storage/confidential/uploads/{user_id}/{upload_id}/
  • audit: TestAuditLog action=UPLOAD
        │
   ┌────┴────┬──────────────┐
   ▼         ▼              ▼
STRUCTURED  NARRATIVE    IMAGE (Phase C)
harness     session       attach only
```

### (a) Structured test report — MVP target

**Reuse:** Full harness chain after JSON is produced.

**Clause selection (search-and-select, not free text):**

```
User types: "chest deflection"
        ▼
GET /api/v1/clauses/search?q=chest+deflection&regulation_code=UN_R94 (optional)
        ▼
Returns resolvable candidates from DB:
  [{ regulation_code, section, document_name, snippet, linked_regulation_clause: "UN_R94#5.2.1.4" }]
        ▼
User picks one row → upload payload includes linked_regulation_clause (server-validated again on ingest)
```

Implementation reuses:

- `Chunk` + `Document` + `Regulation` join (same query as harness ingest clause check)
- Optional: thin wrapper over `RegulationSearchEngine` sparse/dense for “search” UX, but **selection** must resolve to an exact `section` row that exists in DB
- `extract_limit_details(criterion, chunk.chunk_text)` must succeed before upload is accepted (preview endpoint optional)

**Parser:** `registry/structured_test_pdf_parser.py` — deterministic regex on `PDFParser` output; **synthetic fixture PDFs only** until gate + audit + storage verified E2E.

**Development data:** Generate PDFs from `data/synthetic_test_report.json` (fitz). **No real confidential lab PDFs** in repo or CI.

### (b) Narrative — Phase B (not MVP)

Port HF `session_ingest` → `storage/confidential/sessions/{user_id}/{session_id}/` (not global `chunks`).

### (c) Image — Phase C (not MVP)

`TestAttachment` metadata only; no pixel OCR.

---

## 4. Confidential storage layout (separate from public corpus)

### Path convention

```
storage/
  UNECE/                          ← PUBLIC regulation corpus (existing)
  confidential/
    uploads/
      {user_id}/
        {upload_id}/
          original.pdf              ← immutable upload blob
          parse_manifest.json       ← parser output / errors
          quarantine/               ← if parse fails (optional subfolder)
    sessions/                       ← Phase B narrative (per-user)
      {user_id}/{session_id}/...
```

**Explicitly NOT used for confidential uploads:**

| Path | Purpose | Why excluded |
|------|---------|--------------|
| `data/staging/` | Manual **public** regulation acquisition | Batch `ingest_offline_reg.py` promotes to `storage/UNECE/` |
| `data/uploads/` (`settings.UPLOAD_DIR`) | Legacy `/documents/upload` staging | Same promotion pipeline |
| `storage/UNECE/` | Indexed public corpus | Never write user uploads here |

### Batch-ingest isolation (enforced in code)

1. **`ingest_offline_reg.py`** — only reads `data/staging/` (and explicit CLI paths). Add guard: reject paths under `storage/confidential/`.
2. **`/documents/upload`** — unchanged; docs comment: public corpus only.
3. **`consolidate_downloads.py` / crawlers** — no scan of `storage/confidential/`.
4. **`user_uploads.storage_path`** DB column stores absolute path; ingest scripts never glob `storage/confidential/**`.

Access control: same as harness data — `user_id` on row must match `get_current_user().id`; admin override out of MVP scope.

---

## 5. Retention policy (stated default)

| Policy | Value |
|--------|--------|
| **Default lifetime** | Until the **uploading user** deletes the upload |
| **MVP auto-expiry** | **None** |
| **Backup / snapshot** | Confidential tree excluded from routine `data/backups/` corpus DB snapshots unless user opts in (document separately) |
| **Delete API** | `DELETE /api/v1/user-uploads/{upload_id}` — removes `user_uploads` row, linked `Test`/`TestResult` if structured, files on disk, writes `TestAuditLog` `action=DELETE` |
| **Orphan handling** | Startup job (later): flag uploads with missing files; no silent purge in MVP |

---

## 6. Confidentiality model

**Default:** `confidential_tier=True` for all uploads.

**Reuse** `registry/harness_security.py`:

- `check_harness_access(db, model, user_id, test_ids)` on every chat touching upload-derived tests
- `audit_access` on UPLOAD, SELECT, DELETE
- Fail closed if unauthorized model and confidential sources in context

**Identity:** `user_id` in audit and row ownership = **only** `get_current_user().id` (post Phase A0).

---

## 7. API surface

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/api/v1/clauses/search` | Optional | Search-and-select candidates for upload |
| `POST` | `/api/v1/user-uploads` | **Required** | Multipart: `file`, `upload_type`, `linked_regulation_clause` (from picker) |
| `GET` | `/api/v1/user-uploads/{id}` | **Required** | Status; owner-only |
| `DELETE` | `/api/v1/user-uploads/{id}` | **Required** | Retention delete |
| `POST` | `/api/v1/chat` | **Required** for confidential context | Uses verified user for harness gate |

**Do not use** `POST /documents/upload` for confidential data.

### Citation extension

```json
{
  "source_kind": "regulation_corpus | harness_test | user_upload_narrative | user_upload_attachment",
  "confidential_tier": true,
  "test_id": "TEST-2026-94-001",
  "upload_id": "uuid"
}
```

---

## 8. UI design

1. Upload button + drag-drop on chat page
2. Type: Structured test report (MVP)
3. **Clause picker:** search box → results list from `/clauses/search` → user selects row (shows reg, section, snippet)
4. Progress → poll upload status
5. Answer banner + sidebar: **Regulation corpus** vs **Your uploaded report**

---

## 9. MVP slice (Phase A) — blocked on Phase A0 auth

### In scope (after auth)

1. Verified user uploads synthetic structured-test PDF
2. Clause from search-and-select → harness ingest → `tests` / `test_results`
3. Original PDF at `storage/confidential/uploads/{user_id}/{upload_id}/original.pdf`
4. `confidential_tier=True`; unauthorized model → 403 + audit
5. Chat cites real `test_id`; `source_kind=harness_test`
6. User can delete upload (retention policy)

### Out of scope

- Narrative session upload (B), images (C), LLM PDF parse, pixel OCR, spoofable identity

### Acceptance criteria

| # | Criterion |
|---|-----------|
| A0 | Unauthenticated request to `/user-uploads` → **401** |
| 1 | Synthetic PDF → correct `TestResult` + clause link passes harness validation |
| 2 | Parse fail → quarantine under confidential tree; no partial DB |
| 3 | Unauthorized model + confidential test in query → 403 + audit |
| 4 | Authorized model → no invented filenames in answer |
| 5 | Citation `source_kind=harness_test` in UI |
| 6 | File not visible under `data/staging/` or `storage/UNECE/` |
| 7 | `ingest_offline_reg.py` rejects `storage/confidential/` paths |

---

## 10. Effort estimate

| Phase | Days | Notes |
|-------|------|-------|
| **A0 Auth** | 2–4 | Product choice; session or OIDC; `get_current_user` + frontend login |
| A1 Storage + API + audit | 1.0 | After A0 |
| A2 Clause search API + picker UI | 1.0 | DB-backed resolvable clauses |
| A3 PDF parser + synthetic fixtures | 1.5 | |
| A4 Harness ingest wire + chat citations | 1.5 | |
| A5 Tests E2E | 1.0 | Synthetic PDF only |
| **Total after auth** | **~6–7 days** | Plus A0 |

---

## 11. Implementation sequence

```
Phase A0  ─ REAL AUTH (BLOCKER) ─ get_current_user, 401 on protected routes
Phase A1  ─ storage/confidential/ + user_uploads table + DELETE + batch-ingest guards
Phase A2  ─ /clauses/search + structured PDF parser + harness ingest
Phase A3  ─ chat source_kind + harness_retrieval helper + UI upload/picker/badges
Phase A4  ─ acceptance tests (synthetic PDF only)

Phase B  ─ narrative session upload
Phase C  ─ image attachments
```

---

## 12. Reuse map

| Asset | Path |
|-------|------|
| Harness ingest + clause validation | `scripts/ingest_harness_test_records.py` |
| Limit / pass-fail | `registry/harness_limits.py` |
| Model gate + audit | `registry/harness_security.py` |
| Models | `database/models.py` |
| Public upload (do not use) | `api/routes.py` `/documents/upload`, `app/config.py` `UPLOAD_DIR` |
| Public staging ingest | `scripts/ingest_offline_reg.py` |
| Synthetic harness JSON | `data/synthetic_test_report.json` |
| Frontend | `frontend/src/app/page.tsx` |

---

## 13. Status

- **Phase A0:** Complete — session auth verified (401, audit, cross-user isolation).
- **Phase A:** **Unblocked** — proceed with structured test-report PDF upload (synthetic data only).

---

*Rev. 3 — 2026-06-29. Phase A0 shipped; Phase A ready to build.*
