"""Answer service: retrieve → gate → (one corrective retry) → generate → validate → trace.

Modes:
- GENERATED      an LLM produced a schema-valid, citation-validated answer;
- EVIDENCE_ONLY  no LLM configured or the LLM failed; evidence and citations are
                 returned without synthesis (accepted degradation mode);
- ABSTAINED      the deterministic gate refused (scope/date/no evidence) or the
                 model itself reported insufficient evidence.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from safety_assistant.generation.citations import citation_views, validate_draft
from safety_assistant.generation.grounding import GateDecision, evaluate_gate
from safety_assistant.generation.prompts.grounded_v1 import PROMPT_VERSION, SYSTEM, build_user_message
from safety_assistant.generation.schemas import AnswerResponse, AnswerScope, GroundedDraft
from safety_assistant.persistence.models import QueryTrace
from safety_assistant.providers.llm import LLMError, LLMMessage, LLMProvider, get_llm_provider
from safety_assistant.retrieval import RetrievalResult, RetrievalService, ScopeFilter
from safety_assistant.security import injection_signals

log = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 2  # initial + one corrective rewrite
MAX_LLM_CALLS = 1


class AnswerService:
    def __init__(
        self,
        retrieval: RetrievalService | None = None,
        llm: LLMProvider | None = None,
        *,
        use_llm: bool = True,
    ) -> None:
        self.retrieval = retrieval or RetrievalService()
        self._llm = llm
        self._use_llm = use_llm

    @property
    def llm(self) -> LLMProvider | None:
        if not self._use_llm:
            return None
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    def answer(
        self,
        session: Session,
        query: str,
        *,
        scope: ScopeFilter | None = None,
        principal: str | None = None,
        scopes: list[str] | None = None,
        today: datetime.date | None = None,
        k: int | None = None,
    ) -> AnswerResponse:
        trace_id = uuid.uuid4().hex
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []
        plan: dict[str, Any] = {"rewrites": [], "llm_calls": 0}

        signals = injection_signals(query)
        if signals:
            warnings.append("prompt-injection pattern detected in question; instructions in it were ignored")
            plan["injection_signals"] = signals

        # --- retrieval with one bounded corrective retry -------------------------------
        current_query = query
        result: RetrievalResult | None = None
        decision: GateDecision | None = None
        for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
            t0 = time.perf_counter()
            result = self.retrieval.search(session, current_query, scope=scope, k=k, today=today)
            timings[f"retrieval_{attempt + 1}"] = _ms(t0)
            decision = evaluate_gate(
                session,
                current_query,
                result.query_scope,
                result.bundle.evidence,
                as_of=result.scope.as_of,
                retries_left=MAX_RETRIEVAL_ATTEMPTS - attempt - 1,
            )
            warnings.extend(decision.warnings)
            if decision.proceed and decision.rewrite:
                plan["rewrites"].append(decision.rewrite)
                current_query = decision.rewrite
                continue
            break
        assert result is not None and decision is not None
        plan.update(
            intent=result.query_scope.intent,
            scope=result.as_trace()["scope"],
            query_scope=result.as_trace()["query_scope"],
        )
        evidence = result.bundle.evidence
        answer_scope = AnswerScope(
            regulation_keys=sorted({e.regulation_key for e in evidence}) or list(result.scope.regulation_keys),
            as_of=result.scope.as_of,
            intent=result.query_scope.intent,
            historical=bool(result.scope.as_of or result.scope.include_superseded),
        )
        versions: dict[str, Any] = {**result.versions, "prompt_version": PROMPT_VERSION}

        if not decision.proceed:
            resp = self._response(
                trace_id,
                query,
                "ABSTAINED",
                None,
                [],
                evidence,
                warnings,
                decision.abstain_reason,
                decision.message,
                answer_scope,
                None,
                versions,
                timings,
                t_all,
            )
            self._persist(session, resp, result, plan, principal, scopes)
            return resp

        # --- generation -----------------------------------------------------------------
        llm = self.llm
        if llm is None:
            warnings.append("no LLM configured: evidence-only mode")
            resp = self._response(
                trace_id,
                query,
                "EVIDENCE_ONLY",
                None,
                [],
                evidence,
                warnings,
                None,
                None,
                answer_scope,
                None,
                versions,
                timings,
                t_all,
            )
            self._persist(session, resp, result, plan, principal, scopes)
            return resp

        scope_note = (
            f"intent={answer_scope.intent}; regulations={','.join(answer_scope.regulation_keys) or 'any'}; "
            f"as_of={answer_scope.as_of or 'current'}"
        )
        messages = [
            LLMMessage(role="system", content=SYSTEM),
            LLMMessage(role="user", content=build_user_message(query, evidence, scope_note)),
        ]
        t0 = time.perf_counter()
        try:
            llm_resp = llm.generate(messages, schema=GroundedDraft, temperature=0.0, max_tokens=1200)
            plan["llm_calls"] = 1
            versions.update(llm_model=llm_resp.model, llm_provider=llm_resp.provider)
            if llm_resp.usage:
                plan["tokens"] = llm_resp.usage
        except LLMError as exc:
            timings["llm"] = _ms(t0)
            warnings.append(f"generation unavailable ({type(exc).__name__}): evidence-only mode")
            log.warning("llm failure trace=%s err=%s", trace_id, exc)
            resp = self._response(
                trace_id,
                query,
                "EVIDENCE_ONLY",
                None,
                [],
                evidence,
                warnings,
                None,
                None,
                answer_scope,
                None,
                versions,
                timings,
                t_all,
            )
            self._persist(session, resp, result, plan, principal, scopes)
            return resp
        timings["llm"] = _ms(t0)
        draft: GroundedDraft = llm_resp.parsed  # type: ignore[assignment]
        warnings.extend(draft.warnings)

        if draft.insufficient_evidence or not draft.claims:
            resp = self._response(
                trace_id,
                query,
                "ABSTAINED",
                draft.answer,
                [],
                evidence,
                warnings,
                "weak_evidence",
                draft.answer,
                answer_scope,
                None,
                versions,
                timings,
                t_all,
            )
            self._persist(session, resp, result, plan, principal, scopes)
            return resp

        kept, report = validate_draft(draft, evidence)
        if report.dropped_claims:
            warnings.append(f"{report.dropped_claims} claim(s) removed: failed citation/numeric validation")
        if not kept:
            resp = self._response(
                trace_id,
                query,
                "ABSTAINED",
                None,
                [],
                evidence,
                warnings,
                "validation_failed",
                "The generated answer could not be verified against the evidence and was withheld.",
                answer_scope,
                report,
                versions,
                timings,
                t_all,
            )
            self._persist(session, resp, result, plan, principal, scopes)
            return resp

        resp = self._response(
            trace_id,
            query,
            "GENERATED",
            draft.answer,
            kept,
            evidence,
            warnings,
            None,
            None,
            answer_scope,
            report,
            versions,
            timings,
            t_all,
        )
        self._persist(session, resp, result, plan, principal, scopes)
        return resp

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _response(
        trace_id: str,
        query: str,
        mode: str,
        answer: str | None,
        claims: list[Any],
        evidence: list[Any],
        warnings: list[str],
        abstain_reason: str | None,
        message: str | None,
        scope: AnswerScope,
        report: Any,
        versions: dict[str, Any],
        timings: dict[str, float],
        t_all: float,
    ) -> AnswerResponse:
        used = {eid for c in claims for eid in c.evidence_ids} if claims else None
        timings["total"] = _ms(t_all)
        return AnswerResponse(
            trace_id=trace_id,
            query=query,
            mode=mode,
            answer=answer if mode != "ABSTAINED" else message,
            claims=claims,
            citations=citation_views(evidence, used),
            warnings=list(dict.fromkeys(warnings)),
            abstain_reason=abstain_reason,
            scope=scope,
            evidence=[e.model_dump(mode="json") for e in evidence],
            validation=report,
            versions=versions,
            latency_ms=timings,
        )

    @staticmethod
    def _persist(
        session: Session,
        resp: AnswerResponse,
        result: RetrievalResult,
        plan: dict[str, Any],
        principal: str | None,
        scopes: list[str] | None,
    ) -> None:
        try:
            plan = json.loads(json.dumps(plan, default=str))
            session.add(
                QueryTrace(
                    trace_id=resp.trace_id,
                    principal=principal,
                    scopes=scopes,
                    query=resp.query,
                    filters=plan.get("scope"),
                    plan=plan,
                    candidates=result.candidates,
                    evidence=[
                        {"evidence_id": c.evidence_id, "label": c.label, "section_path": c.section_path}
                        for c in resp.citations
                    ],
                    answer={
                        "mode": resp.mode,
                        "answer": resp.answer,
                        "claims": [c.model_dump() for c in resp.claims],
                        "abstain_reason": resp.abstain_reason,
                        "warnings": resp.warnings,
                    },
                    validation=resp.validation.model_dump() if resp.validation else None,
                    versions=resp.versions,
                    latency_ms=resp.latency_ms,
                    tokens=plan.get("tokens"),
                )
            )
            session.commit()
        except Exception:  # noqa: BLE001 — tracing must never fail the answer
            log.exception("failed to persist query trace %s", resp.trace_id)
            session.rollback()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
