"""Answer service: runs the bounded agent graph and turns its final state into an
`AnswerResponse`, persisting a `QueryTrace` for every request.

Modes:
- GENERATED      an LLM produced a schema-valid, citation-validated answer;
- EVIDENCE_ONLY  no LLM configured or the LLM failed; evidence and citations are
                 returned without synthesis (accepted degradation mode);
- ABSTAINED      the deterministic gate refused (scope/date/no evidence/budget) or
                 the model itself reported insufficient evidence.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from safety_assistant.agents.graph import RegulatoryAgent
from safety_assistant.agents.state import AgentState, Budget
from safety_assistant.config import get_settings
from safety_assistant.generation.citations import citation_views
from safety_assistant.generation.schemas import AnswerResponse, AnswerScope
from safety_assistant.persistence.models import QueryTrace
from safety_assistant.providers.llm import LLMProvider, get_llm_provider
from safety_assistant.retrieval import RetrievalService, ScopeFilter

log = logging.getLogger(__name__)


class AnswerService:
    def __init__(
        self,
        retrieval: RetrievalService | None = None,
        llm: LLMProvider | None = None,
        *,
        use_llm: bool = True,
        budget: Budget | None = None,
    ) -> None:
        self.retrieval = retrieval or RetrievalService()
        self._llm = llm
        self._use_llm = use_llm
        self.budget = budget

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
        agent = RegulatoryAgent(
            session,
            self.retrieval,
            self.llm,
            budget=self.budget,
            llm_data_classes=tuple(get_settings().llm_data_classes),
        )
        state = agent.run(query, scope=scope, k=k, today=today, trace_id=trace_id)
        resp = self._to_response(trace_id, query, state)
        self._persist(session, resp, state, principal, scopes)
        return resp

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _to_response(trace_id: str, query: str, s: AgentState) -> AnswerResponse:
        evidence = s.get("evidence", [])
        scope = s.get("scope") or s["base_scope"]
        draft = s.get("draft")
        mode = s.get("mode", "EVIDENCE_ONLY")
        claims = list(draft.claims) if (draft and mode == "GENERATED") else []
        used = {eid for c in claims for eid in c.evidence_ids} if claims else None
        answer = None
        if mode == "GENERATED" and draft:
            answer = draft.answer
        elif mode == "ABSTAINED":
            answer = s.get("message")
        retrieval = s.get("retrieval")
        return AnswerResponse(
            trace_id=trace_id,
            query=query,
            mode=mode,
            answer=answer,
            claims=claims,
            citations=citation_views(evidence, used),
            warnings=list(dict.fromkeys(s.get("warnings", []))),
            abstain_reason=s.get("abstain_reason"),
            scope=AnswerScope(
                regulation_keys=sorted({e.regulation_key for e in evidence}) or list(scope.regulation_keys),
                as_of=scope.as_of,
                intent=s.get("intent", "technical_qa"),
                historical=bool(scope.as_of or scope.include_superseded),
            ),
            evidence=[e.model_dump(mode="json") for e in evidence],
            validation=s.get("validation"),
            versions={**(retrieval.versions if retrieval else {}), **s.get("versions", {})},
            latency_ms=s.get("timings", {}),
        )

    @staticmethod
    def _persist(
        session: Session, resp: AnswerResponse, s: AgentState, principal: str | None, scopes: list[str] | None
    ) -> None:
        try:
            retrieval = s.get("retrieval")
            plan: dict[str, Any] = {
                "intent": s.get("intent"),
                "route": s.get("route"),
                "rewrites": s.get("rewrites", []),
                "retrieval_attempts": s.get("retrieval_attempts", 0),
                "llm_calls": s.get("llm_calls", 0),
                "tool_calls": s.get("tool_calls", 0),
                "scope": retrieval.as_trace()["scope"] if retrieval else None,
                "query_scope": retrieval.as_trace()["query_scope"] if retrieval else None,
                "extra_context": bool(s.get("extra_context")),
            }
            plan = json.loads(json.dumps(plan, default=str))
            candidates: list[dict[str, Any]] = []
            for r in s.get("sub_results") or ([retrieval] if retrieval else []):
                candidates.extend(r.candidates)
            session.add(
                QueryTrace(
                    trace_id=resp.trace_id,
                    principal=principal,
                    scopes=scopes,
                    query=resp.query,
                    filters=plan.get("scope"),
                    plan=plan,
                    candidates=candidates,
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
                    versions=json.loads(json.dumps(resp.versions, default=str)),
                    latency_ms=resp.latency_ms,
                    tokens=s.get("tokens"),
                )
            )
            session.commit()
        except Exception:  # noqa: BLE001 — tracing must never fail the answer
            log.exception("failed to persist query trace %s", resp.trace_id)
            session.rollback()
