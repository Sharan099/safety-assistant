# ADR-0024 — Citation contract with programmatic validation and explicit abstention

Status: accepted · Date: 2026-09-11

## Decision
Evidence is injected as `<evidence id="E1" regulation=… version=… section=… pages=… valid_from=… valid_to=…>`; the model must return `{answer, claims[{text, evidence_ids, kind}], warnings, insufficient_evidence}`. Code then checks that every evidence id exists and, for `REQUIREMENT` claims, that every number in the claim appears in the cited text (decimal-comma aware). Failing claims are dropped; if none survive the answer is withheld (`validation_failed`). A deterministic gate runs first: ambiguous query, no version valid on the as-of date, requested regulation not in corpus/evidence, no evidence → abstain; weak evidence → exactly one acronym-expansion rewrite. No pseudo-confidence numbers are emitted.

## Consequences
Modes: `GENERATED` (validated), `EVIDENCE_ONLY` (no/failed LLM, or provider not cleared for the data class), `ABSTAINED`. Every request writes a `query_traces` row.

## Evidence
`tests/unit/test_citations_and_gate.py`, `tests/security/test_adversarial.py`, `tests/integration/test_api_ask.py`.
