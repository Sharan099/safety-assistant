# PRD Addendum — Contextual Investigation Copilot

**Version:** 1.0  
**Status:** Proposed V1.1 Update

## 1. Purpose

The existing V1 already supports:

```text
Run A + Run B
→ Quality Gate
→ Comparability
→ Configuration Diff
→ Signal Analysis / Divergence
→ Evidence
→ Agent-drafted Hypothesis
→ Engineer Review
```

The missing capability is a **contextual conversational Copilot inside the active investigation**.

The Copilot is not a generic chatbot. It operates on the current investigation state and can use existing deterministic analysis, evidence, and knowledge-retrieval tools.

## 2. Product Goal

A passive-safety engineer must be able to remain inside one investigation and ask follow-up questions such as:

- Why is this the leading hypothesis?
- What evidence supports it?
- What evidence contradicts it?
- Which signal should I inspect next?
- Compare the crash pulse.
- What changed between the runs?
- Show the source for this claim.
- What evidence is missing?
- Find similar historical cases.
- What analysis should I perform next?
- I disagree with this hypothesis; investigate an alternative explanation.
- What controlled simulation should I run?

## 3. UX Change

Add a collapsible **Investigation Copilot** panel to the existing investigation workspace.

```text
┌──────────────┬──────────────────────────────┬───────────────┐
│ Investigation│ Analysis                     │ Copilot       │
│ Quality      │ Charts / Diff / Evidence     │               │
│ Comparability│                              │ Conversation  │
│ Configuration│                              │               │
│ Signals      │                              │ [Ask...]      │
│ Evidence     │                              │ [Send]        │
│ Hypotheses   │                              │               │
└──────────────┴──────────────────────────────┴───────────────┘
```

The existing workflow must remain intact.

## 4. Context

Every Copilot request is scoped to `investigation_id` and receives structured context:

```text
question
run_a
run_b
primary_metric
investigation_state
quality_results
comparability_results
configuration_diffs
selected_signals
signal_analysis_results
divergence_events
current_evidence
current_hypotheses
engineer_review_state
```

Do not send the entire database to the LLM.

## 5. Tool Use

Reuse the existing deterministic tools:

```text
get_investigation_context
get_evidence
get_hypotheses
run_quality_gate
assess_comparability
compare_global_response
compare_configuration
select_signal_plan
analyze_signal
detect_first_divergence
retrieve_knowledge
retrieve_historical_cases
create_hypothesis
evaluate_evidence
request_engineer_review
create_controlled_comparison_request
```

The LLM must call tools rather than perform numerical engineering calculations in free-form reasoning.

## 6. Example

Engineer:

> Why is the restraint configuration currently a leading contributor?

The Copilot should inspect the current investigation and explain the evidence. For the current SCN-001 investigation, the existing evidence includes the belt webbing revision change, force-limiter change, belt-force divergence at 29.6 ms, and chest-deflection divergence at 48.0 ms. These support a hypothesis but do not establish causality. fileciteturn4file0L64-L78

Engineer:

> What evidence contradicts this hypothesis?

The Copilot should distinguish:

```text
No direct contradiction found
```

from:

```text
Hypothesis proven
```

Engineer:

> I disagree. Investigate torso rotation as an alternative explanation.

The Copilot should create/update the alternative hypothesis, select relevant signals, execute analysis, gather evidence, and compare competing explanations.

## 7. Evidence Grounding

Important claims must reference one or more:

```text
CURRENT_INVESTIGATION_EVIDENCE
CALCULATED_RESULT
DOCUMENTARY_SOURCE
HISTORICAL_CASE
ENGINEER_INPUT
```

Claims should be classified internally as:

```text
FACT
CALCULATION
OBSERVATION
DOCUMENTARY_STATEMENT
HYPOTHESIS
INFERENCE
RECOMMENDATION
UNKNOWN
```

## 8. Knowledge/RAG Guard

The current investigation output demonstrates a retrieval-quality problem: documentary retrieval includes unrelated material such as an AES/Rijndael passage. fileciteturn4file0L80-L94

Therefore V1.1 must add:

```text
retrieve
→ relevance check
→ authority check
→ document/revision check
→ deduplicate
→ present
```

Irrelevant chunks must not be shown as evidence.

Documentary answers must expose:

```text
document
revision
page/section when available
authority
evidence/chunk ID
```

If the corpus cannot support the answer, say so.

## 9. Conversation Persistence

Persist conversation against the investigation:

```text
copilot_conversations
copilot_messages
copilot_tool_calls
```

Minimum message fields:

```text
conversation_id
investigation_id
role
content
created_at
tool_calls
evidence_refs
```

No global memory in V1.1.

## 10. API

Preferred endpoint:

```http
POST /api/v1/investigations/{investigation_id}/copilot/messages
```

Request:

```json
{
  "message": "Why is the restraint configuration currently the leading hypothesis?"
}
```

Response should expose:

```text
assistant message
evidence references
tool activity
suggested actions
unknowns
```

Follow existing project API conventions.

## 11. Guardrails

The Copilot must never:

- claim causality from temporal precedence alone;
- invent numerical results;
- invent regulations;
- invent solver behavior;
- hide contradictory evidence;
- treat LLM text as authoritative;
- override quality failures;
- automatically approve engineering conclusions;
- fabricate unavailable historical cases;
- cite irrelevant retrieved documents.

## 12. Engineer Review

The Copilot can recommend:

```text
LIKELY
PLAUSIBLE
SUPPORTED
PARTIALLY_SUPPORTED
INCONCLUSIVE
```

Final decisions remain engineer-controlled:

```text
ENGINEER_ACCEPTED
ENGINEER_REJECTED
ENGINEER_MODIFIED
REQUIRES_CONTROLLED_RERUN
INCONCLUSIVE
```

## 13. Acceptance Criteria

- Engineer can open an existing investigation and see Copilot.
- Engineer can type follow-up questions.
- Copilot automatically knows the active investigation.
- Copilot can answer from current evidence.
- Copilot can invoke deterministic tools.
- Copilot can retrieve documentary evidence.
- Copilot provides provenance.
- Copilot identifies unknowns/missing evidence.
- Copilot handles engineer disagreement.
- Copilot can request another analysis.
- Conversation is persisted per investigation.
- LLM failure does not break deterministic analysis.
- Irrelevant RAG chunks are rejected.
- Copilot cannot automatically become the final engineering decision.

## 14. Acceptance Conversation

Use:

```text
SCN-001-RUN-A
vs
SCN-001-RUN-B

Primary metric:
chest_deflection
```

Ask:

1. Why is the restraint configuration currently a leading contributor?
2. What evidence contradicts this hypothesis?
3. I disagree. Investigate torso rotation as an alternative explanation.
4. What should I analyze next?
5. Show me the source for your LS-DYNA-related claim.

Every answer must be grounded and traceable.

## 15. Success Definition

V1.1 succeeds when an engineer can stay inside one investigation and:

```text
ask
→ investigate
→ challenge
→ request analysis
→ retrieve evidence
→ compare hypotheses
→ choose next action
```

without leaving the investigation workspace.
