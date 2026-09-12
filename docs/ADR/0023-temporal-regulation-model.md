# ADR-0023 — Temporal regulation model: versions with validity windows and a lifecycle

Status: accepted · Date: 2026-09-11

## Decision
`regulations` (identity) → `regulation_versions` (consolidated text: label, series, revision, published_at, valid_from, valid_to, superseded_by, status) → `source_artifacts` (immutable bytes). Lifecycle `DISCOVERED … VERIFIED → ACTIVE`, terminal `SUPERSEDED | QUARANTINED | FAILED`; transitions validated in code. Current queries see `ACTIVE` versions whose window contains today; `as_of` queries see `ACTIVE|SUPERSEDED` whose window contains the date; `valid_from IS NULL` (supporting documents) is treated as always in force. Activation supersedes the previous ACTIVE version atomically and closes its window at the new `valid_from`. "Latest" therefore means *latest in force*, never latest downloaded.

## Consequences
Historical answers come from superseded text; a date before any ingested version abstains with `no_version_valid_on_date`. Change analysis diffs sections by path and content hash.

## Evidence
`tests/e2e/test_lifecycle_and_update.py`; `tests/integration/test_api_ask.py::test_ask_historical_uses_superseded_version`.
