# Incident response

## Severity
- **SEV1** — wrong regulatory content served (stale/wrong version, fabricated citation reaching users), data-class leakage, auth bypass.
- **SEV2** — API down or readiness failing, ingestion pipeline stuck, error rate > 2 %.
- **SEV3** — degraded quality (LLM outage → evidence-only, reranker degraded), latency SLO breach.

## First 15 minutes
1. Confirm with `/health/ready`, `/health/deps`, `/metrics` (`sa_http_requests_total`, `sa_answers_total`, `sa_citation_validation_failures_total`).
2. Pull the trace: every answer carries `trace_id`; `GET /api/v1/admin/traces/{trace_id}` (scope `audit:read`) shows candidates, evidence, validation and versions. Correlate with `X-Request-ID` in logs.
3. Contain:
   - wrong content → set the offending version `QUARANTINED` (`UPDATE regulation_versions …`); retrieval excludes it immediately (BM25 cache invalidates on the next generation change; restart workers to force).
   - auth/leakage → rotate API keys / revoke IdP client, redeploy with `AUTH_MODE` verified, review `query_traces.principal`.
   - LLM misbehaviour → set `LLM_PROVIDER=none` (evidence-only) and redeploy; validation failures are visible in metrics.
4. Communicate: status, affected regulations/versions, trace ids.

## Recovery
- Re-ingest the corrected source (`safety-assistant ingest <key> --force`), verify with the retrieval regression tests and a manual `/ask`.
- For database loss: RDS point-in-time restore, update secret, redeploy, `verify_registry.py`, spot-check `/api/v1/regulations`.

## Post-incident
- Add a regression case to `evals/datasets` or a test under `tests/security` reproducing the failure.
- Record the incident and the fix in `CHANGELOG.md`; update the threat model if a new class of failure appeared.
