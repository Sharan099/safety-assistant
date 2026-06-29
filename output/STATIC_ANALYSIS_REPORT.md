# Static Analysis & Security Report

**Date:** 2026-06-29  
**Scope:** Post Phase A (structured confidential upload) + Phase A0 (session auth)  
**Test suite:** `125 passed` (`USE_MOCK_EMBEDDINGS=true`)  
**Readiness:** `/api/v1/ready` returns `ready: true` (live probe against `safety_registry.db`)

> **Note:** Snapshot/commit before harness/confidential/gateway changes is recommended if not already done. This pass applied only targeted HIGH fixes (MD5 fingerprinting, mypy on new upload code, TS styles).

---

## Executive summary

| Phase | Tool | Result |
|-------|------|--------|
| 1 | mypy (backend) | **Baseline blocked** on full tree (SQLAlchemy `Base`, async sessionmaker). **New upload/auth modules:** minor issues fixed in-parser; remainder LOW/incremental |
| 2 | ruff | **587** issues repo-wide (mostly style/unused imports). Phase A files: mostly FastAPI `Depends()` B008 (intentional pattern) |
| 3 | bandit | **0 HIGH** after MD5 fix; **5 MEDIUM**, **12 LOW** |
| 4 | pip-audit | **No known CVEs** in `requirements.txt` |
| 5 | tsc / eslint | **tsc clean** after styles fix; **eslint not configured** in frontend |

---

## Phase 1 — mypy (backend)

### Configuration

- Added `mypy.ini` with sane baseline: `check_untyped_defs`, `warn_return_any`, `ignore_missing_imports`, excludes `hf-space-push/`.

### Blockers (repo-wide — not fixed in this pass)

| Severity | Finding | Notes |
|----------|---------|-------|
| LOW | `database.models.Base` invalid as type | SQLAlchemy declarative pattern; needs `Mapped[]` migration or `mypy` plugin |
| LOW | `database.connection` async `sessionmaker` overload | Pre-existing |
| LOW | `parser.pdf_parser` table type `list[list[str \| None]]` | Pre-existing |

### HIGH / real bugs (new Phase A code — **fixed**)

| File | Issue | Fix |
|------|-------|-----|
| `registry/structured_test_pdf_parser.py` | `Match \| None` `.group()` after guard | Added `assert test_id_m is not None and date_m is not None` after error early-return |
| `registry/clause_search.py` | `Column[str]` passed to `extract_limit_details` | `str(chunk.chunk_text or "")` |

### Incremental annotation plan (LOW — not done)

1. Add `py.typed` + SQLAlchemy 2.0 `Mapped` types on `database/models.py` (largest ROI).
2. Type `registry/search.py` return dicts as `TypedDict` for chat/citation paths.
3. Enable `disallow_untyped_defs` on `registry/auth.py`, `registry/harness_security.py`, `api/user_upload_routes.py` first.

---

## Phase 2 — ruff (backend)

### Repo-wide

- **587** findings; **397** auto-fixable (mostly `F401` unused imports, `UP035` `typing.List` → `list`).
- Cross-reference `output/CLEANUP_AUDIT.md` before mass `--fix` to avoid duplicating planned cleanup.

### High-risk patterns in THIS codebase (reviewed)

| Location | Pattern | Risk | Action |
|----------|---------|------|--------|
| `api/routes.py` | `except Exception` on `/chat` (L585) | Could mask auth/gateway failures | **Intentional** — re-raises `HTTPException`; logs error. **Flag for review:** narrow to `SQLAlchemyError`, `ValueError` in future |
| `api/routes.py` | Multiple `except Exception` on ingest/health | Same | Pre-existing; not weakened in this pass |
| `registry/auth.py`, `harness_security.py`, `user_upload.py` | No bare `except` | — | **Clean** |
| `api/*` | `Depends()` in defaults (B008) | False positive | Standard FastAPI; ignore via ruff per-file or global ignore |

### Phase A files

- No mutable default args, no bare `except` in new upload path.
- `registry/clause_search.py` SIM110 simplified to `any(...)`.

---

## Phase 3 — bandit (security)

### SQL injection

| Finding | Verdict |
|---------|---------|
| `registry/sample_eval.py` B608 f-string SQL | **False positive (LOW confidence)** — placeholders are `?` with bound params; not user-controlled structure |

**Upload path:** No raw SQL in Phase A; SQLAlchemy ORM only. `linked_regulation_clause` validated via DB lookup before ingest.

### Path traversal — `storage/confidential/uploads/{user_id}/{upload_id}/`

| Control | Implementation |
|---------|----------------|
| `user_id` | From `get_current_user()` only (server session) |
| `upload_id` | Server-generated UUID |
| Path join | `registry/confidential_paths.py` — regex `^[a-zA-Z0-9_-]{1,128}$` + resolved path must stay under `storage/confidential/` |
| Batch ingest guard | `scripts/ingest_offline_reg.py` rejects `storage/confidential/**` |

**Verdict:** Path traversal risk **mitigated** for Phase A.

### Session cookies (Phase A0)

| Flag | Setting |
|------|---------|
| `httpOnly` | **true** (`registry/auth.py`) |
| `secure` | `SESSION_COOKIE_SECURE` env (default `false` for local dev; **set true in production HTTPS**) |
| `samesite` | `lax` (configurable) |
| Identity | Server-side `auth_sessions` table; cookie holds opaque token only |

**Review:** Production deploy must set `SESSION_COOKIE_SECURE=true` and `AUTH_SEED_PASSWORD` to a strong secret.

### Deserialization

- Harness/upload: `json.load` only on server-controlled quarantine/manifest paths.
- `coverage_expected.yaml`: uses existing registry loaders (no `yaml.load` unsafe path found in scanned modules).

### HIGH findings — **fixed**

| Issue | File | Fix |
|-------|------|-----|
| B324 MD5 “weak hash” | `scripts/diff_retrieval_chunks.py` | `usedforsecurity=False` (fingerprinting only) |
| B324 MD5 | `scripts/ingest_storage.py` | `usedforsecurity=False` |

### MEDIUM — flagged for review

| Issue | File | Risk |
|-------|------|------|
| B615 HuggingFace `from_pretrained` without revision pin | `vectorization/embedder.py` | Supply-chain drift; mitigated by pinned `EMBEDDING_MODEL` env + local cache |
| B608 SQL string format | `registry/sample_eval.py` | Low — parameterized |

### LOW — flagged

- B105 `changeme` default in `scripts/seed_auth_users.py` — **intentional dev default** with warning; override via `AUTH_SEED_PASSWORD`.
- B110 `try/except/pass` in eval/ingest scripts — non-auth paths.
- B404/B603 subprocess in RAGAS scripts — trusted internal commands only.

---

## Phase 4 — pip-audit

```
No known vulnerabilities found (requirements.txt)
```

No upgrades required for this pass. Re-run after dependency bumps.

---

## Phase 5 — frontend

### `tsc --noEmit`

- **Before:** 50+ errors from `styles` union type (`CSSProperties | function`).
- **After fix:** **0 errors** — extracted `statusPillStyle()` helper.

### eslint

- **Not configured** (`package.json` has no eslint devDependency). Recommend adding `eslint-config-next` in a follow-up.

### Login / session UI review (Phase A0 + A upload)

| Check | Status |
|-------|--------|
| Tokens in client storage | **Pass** — httpOnly cookie only; no localStorage JWT |
| `credentials: "include"` on API calls | **Yes** |
| 401 on upload without session | **Tested** (`test_user_upload_requires_auth`) |
| Auth failure UX | Login form shows error; upload shows API error text |
| Guest mode | Regulation chat only; harness/upload require login |

---

## Phase 6 — ponytail review (fixes applied)

| Finding | Decision |
|---------|----------|
| MD5 bandit HIGH | **Fix** with `usedforsecurity=False` — no new dependency |
| mypy None on parser | **Fix** with assert after guard — no weakened validation |
| TS styles | **Simplify** — extract function vs suppress |
| Mass ruff `--fix` | **Defer** — avoid 400-file churn; align with CLEANUP_AUDIT |
| Broad `except Exception` on chat | **Keep** — preserves fail-loud logging + HTTPException re-raise; document for narrowing later |
| B008 Depends | **Ignore** — framework idiom |

**Not done (deliberate):** No validation gates removed, no auth checks loosened, no silent catch-and-ignore on upload/harness paths.

---

## Phase A delivery checklist

| Criterion | Status |
|-----------|--------|
| `POST /api/v1/user-uploads` + session auth | Done |
| `GET /api/v1/clauses/search` search-and-select | Done |
| Storage `storage/confidential/uploads/{user_id}/{upload_id}/` | Done |
| Harness ingest + `owner_user_id` | Done |
| `source_kind=harness_test` citations | Done |
| Frontend upload + clause picker | Done |
| Synthetic PDF fixture + tests | Done |
| Cross-user upload isolation test | Done |
| `DELETE /api/v1/user-uploads/{id}` | Done |

### Runbook

```bash
python scripts/seed_auth_users.py
python scripts/create_harness_tables.py
python scripts/generate_synthetic_test_pdf.py   # tests/fixtures/synthetic_test_report.pdf
# Login as engineer_a, search clause "chest", select UN_R94#5.2.1.4, upload synthetic PDF
```

---

## Verification commands

```bash
USE_MOCK_EMBEDDINGS=true python -m pytest tests -q
python -c "from fastapi.testclient import TestClient; from app.main import app; print(TestClient(app).get('/api/v1/ready').json())"
cd frontend && npx tsc --noEmit
```

---

*Generated after Phase A implementation and static-analysis pass.*
