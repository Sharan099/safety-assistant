"""Evidence/Hypothesis persistence — TRD.md Section 16 evidence chain:
Observation -> Calculation -> Documentary evidence -> Hypothesis ->
Supporting/contradicting evidence -> Engineer review -> Finding -> Decision.

The LLM (if available) only drafts the hypothesis *description* text, and
only from evidence already produced by packages/analysis — never invents
numbers, and its absence never blocks anything (TRD.md Section 30).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from packages.agent.llm import LLMMessage, LLMProvider, LLMUnavailableError
from packages.agent.state import InvestigationState
from packages.domain.investigation import Evidence, Hypothesis, HypothesisEvidenceLink, Investigation


def build_evidence(session: Session, investigation: Investigation, state: InvestigationState) -> list[Evidence]:
    rows: list[Evidence] = []

    for label, quality in (("A", state.get("quality_a")), ("B", state.get("quality_b"))):
        if quality is None:
            continue
        rows.append(
            Evidence(
                investigation_id=investigation.id,
                evidence_type="CALCULATED",
                source_type="run_quality_gate",
                content=f"Run {label} quality gate overall status: {quality['overall_status']}",
                value={"overall_status": quality["overall_status"]},
                calculation_version="run_quality_gate v0.1.0",
            )
        )

    for diff in state.get("configuration_diff") or []:
        if diff["change_status"] == "CHANGED":
            rows.append(
                Evidence(
                    investigation_id=investigation.id,
                    evidence_type="OBSERVED",
                    source_type="compare_configuration",
                    content=f"{diff['path']} changed: {diff['run_a_value']} -> {diff['run_b_value']}",
                    value=diff,
                )
            )

    for signal_name, result in (state.get("signal_results") or {}).items():
        divergence = result.get("divergence")
        if divergence:
            rows.append(
                Evidence(
                    investigation_id=investigation.id,
                    evidence_type="CALCULATED",
                    source_type="detect_first_divergence",
                    content=f"{signal_name} diverged at {divergence['time_ms']:.1f} ms",
                    value=divergence,
                    calculation_version=divergence["provenance"]["algorithm_version"],
                )
            )

    for k in state.get("knowledge_evidence") or []:
        rows.append(
            Evidence(
                investigation_id=investigation.id,
                evidence_type="DOCUMENTARY",
                source_type="knowledge_retrieval",
                source_locator={
                    "document_key": k["document_key"],
                    "page_start": k["page_start"],
                    "page_end": k["page_end"],
                },
                content=k["content"][:500],
            )
        )

    for h in state.get("historical_cases") or []:
        rows.append(
            Evidence(
                investigation_id=investigation.id,
                evidence_type="HISTORICAL",
                source_type="historical_retrieval",
                source_locator={"investigation_id": h["investigation_id"]},
                content=h["finding"],
            )
        )

    session.add_all(rows)
    session.flush()
    return rows


def _draft_description(question: str, evidence: list[Evidence], title: str, llm: LLMProvider | None) -> str:
    if llm is None:
        return title
    try:
        evidence_text = "\n".join(f"- {e.content}" for e in evidence) or "(no evidence recorded)"
        messages = [
            LLMMessage(
                role="system",
                content="You are an engineering-investigation assistant. Given a question and a list of "
                "deterministic evidence bullet points, write a two-sentence hypothesis description. "
                "Cite only facts present in the evidence list — never invent a number, a source, or a "
                "mechanism not listed.",
            ),
            LLMMessage(role="user", content=f"Question: {question}\nEvidence:\n{evidence_text}"),
        ]
        response = llm.complete(messages, max_tokens=200)
        return response.content
    except LLMUnavailableError:
        # Deterministic title stands alone — TRD.md Section 30.
        return title


def generate_hypothesis(
    session: Session,
    investigation: Investigation,
    state: InvestigationState,
    evidence: list[Evidence],
    *,
    llm: LLMProvider | None = None,
) -> Hypothesis:
    changed = [d for d in (state.get("configuration_diff") or []) if d["change_status"] == "CHANGED"]
    categories = sorted({d["path"].split(".")[0] for d in changed})

    if len(categories) == 1:
        title = f"{categories[0].capitalize()} configuration change is a leading contributor"
        status = "PROPOSED"
    elif categories:
        title = f"Multiple configuration categories changed ({', '.join(categories)}); isolation required"
        status = "PROPOSED"
    else:
        title = "No configuration-level mechanism identified"
        status = "INCONCLUSIVE"

    description = _draft_description(state.get("question", ""), evidence, title, llm)

    hypothesis = Hypothesis(
        investigation_id=investigation.id,
        title=title,
        description=description,
        status=status,
        confidence_basis="deterministic-configuration-diff" if categories else "none",
        affected_components={"categories": categories},
        created_by="agent",
    )
    session.add(hypothesis)
    session.flush()

    links: list[HypothesisEvidenceLink] = []
    for e in evidence:
        if e.source_type == "compare_configuration" and any(str(e.content).startswith(c) for c in categories):
            relationship = "SUPPORTS"
        elif e.source_type == "detect_first_divergence":
            relationship = "SUPPORTS"
        else:
            relationship = "CONTEXT"
        links.append(HypothesisEvidenceLink(hypothesis_id=hypothesis.id, evidence_id=e.id, relationship=relationship))
    session.add_all(links)
    session.flush()
    return hypothesis


def evidence_and_hypothesis(
    session: Session, investigation: Investigation, state: InvestigationState, *, llm: LLMProvider | None = None
) -> tuple[list[Evidence], Hypothesis]:
    evidence = build_evidence(session, investigation, state)
    hypothesis = generate_hypothesis(session, investigation, state, evidence, llm=llm)
    return evidence, hypothesis
