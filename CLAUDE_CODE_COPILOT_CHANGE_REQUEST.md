# Claude Code Change Request — Contextual Investigation Copilot

Implement the feature defined in `PRD_COPILOT_UPDATE.md`.

## Read first

Before editing:

```text
PRD.md
TRD.md
APP_FLOW.md
UI_UX_DESIGN_BRIEF.md
BACKEND_SCHEMA.md
IMPLEMENTATION_PLAN.md
ENVIRONMENT_SETUP.md
CLAUDE.md
PRD_COPILOT_UPDATE.md
```

Then inspect the actual current implementation and tests. Do not assume filenames or architecture.

## Critical constraints

1. Do NOT rebuild the application.
2. Do NOT replace the current investigation workflow.
3. Do NOT create a generic `/chat` page.
4. Add a contextual Copilot inside the existing investigation workspace.
5. Reuse the existing LangGraph, tools, evidence, retrieval, and LLM-provider architecture wherever possible.
6. Do not modify unrelated functionality.

Existing workflow must remain:

```text
Run A + Run B
→ Quality
→ Comparability
→ Configuration
→ Signal Analysis
→ Evidence
→ Hypothesis
→ Engineer Review
```

The Copilot operates on top of that state.

## Phase 1 — Inspect

Locate:

```text
investigation API
investigation state
LangGraph graph
agent tools
LLMProvider
knowledge retrieval
evidence models
hypothesis models
frontend investigation page
frontend API client
```

Run the existing tests before changes and record the baseline.

## Phase 2 — Investigation Context

Implement or reuse:

```python
build_investigation_context(investigation_id)
```

It must expose only relevant structured state:

```text
question
runs
primary metric
quality
comparability
configuration diff
signals
signal results
divergence events
evidence
hypotheses
review state
```

Prevent leakage from unrelated investigations.

## Phase 3 — Conversation Persistence

If absent, add:

```text
copilot_conversations
copilot_messages
copilot_tool_calls
```

Every conversation must belong to `investigation_id`.

Do not implement global memory.

Use an Alembic migration.

## Phase 4 — Copilot API

Add:

```http
POST /api/v1/investigations/{investigation_id}/copilot/messages
```

Request:

```json
{
  "message": "Why is the restraint configuration currently the leading hypothesis?"
}
```

Return a structured response containing:

```text
assistant message
evidence refs
tool activity
suggested actions
unknowns
```

Follow existing API conventions.

## Phase 5 — Agent Integration

Reuse the current LangGraph architecture.

Flow:

```text
user message
→ classify intent
→ choose tool(s)
→ execute tool(s)
→ collect evidence
→ grounded response
```

Possible intents:

```text
EXPLAIN_EVIDENCE
COMPARE_RUNS
ANALYZE_SIGNAL
ANALYZE_DIVERGENCE
RETRIEVE_KNOWLEDGE
RETRIEVE_HISTORY
CHALLENGE_HYPOTHESIS
REQUEST_NEXT_ANALYSIS
CONTROLLED_COMPARISON
GENERAL_INVESTIGATION_QUESTION
```

Do not create a second agent framework.

## Phase 6 — Tool Rules

Examples:

```text
"Compare the crash pulse"
→ compare_global_response

"When did belt force first diverge?"
→ detect_first_divergence

"What changed?"
→ compare_configuration

"Analyze chest acceleration"
→ analyze_signal

"What does LS-DYNA documentation say?"
→ retrieve_knowledge

"Find similar cases"
→ retrieve_historical_cases
```

The LLM must not calculate engineering values when deterministic tools exist.

## Phase 7 — Evidence Grounding

Every important factual response should reference evidence.

Expose:

```text
evidence_ids
source_ids
tool_calls
```

Allow the UI to open/inspect those references where the existing application supports it.

## Phase 8 — Fix RAG Relevance

The current output has shown irrelevant knowledge retrieval, including an unrelated AES/Rijndael passage.

Do not expose raw top-k chunks.

Implement:

```text
retrieval
→ metadata filtering
→ relevance threshold
→ authority validation
→ document/revision validation
→ deduplicate
→ evidence
```

Add regression tests for:

```text
relevant LS-DYNA query
relevant UN_R94 query
deliberately irrelevant retrieval result
```

The irrelevant result must be rejected.

## Phase 9 — Engineer Challenge

Support:

> I disagree. Investigate torso rotation as an alternative explanation.

Expected:

```text
engineer input
→ alternative hypothesis
→ relevant signal selection
→ deterministic analysis
→ evidence
→ comparison with current hypothesis
→ response
```

Do not automatically finalize the hypothesis.

## Phase 10 — Frontend

Add an `InvestigationCopilot` component to the existing investigation page.

Features:

```text
conversation history
message input
send button
loading state
tool activity
evidence references
suggested actions
error state
collapse/expand
```

Suggested actions:

```text
Why is this hypothesis leading?
Find contradictions
Analyze crash pulse
Analyze relevant signals
Show supporting evidence
Find documentation
Find similar cases
What evidence is missing?
Suggest next analysis
```

Do not build a separate chat page.

## Phase 11 — Failure Handling

If LLM provider fails:

```text
show Copilot unavailable
preserve deterministic investigation
allow retry
```

If a tool fails:

```text
report tool failure honestly
do not fabricate result
```

## Phase 12 — Tests

Add:

### Backend/API

```text
send message
invalid investigation
empty message
LLM failure
```

### Context

```text
correct investigation context
no cross-investigation leakage
```

### Agent

```text
tool selection
evidence grounding
hypothesis challenge
unknown handling
```

### RAG

```text
relevant retrieval
irrelevant chunk rejection
authority filtering
source provenance
```

### Frontend

```text
Copilot renders
message sends
loading state
tool activity
error state
evidence reference
```

Run the full existing test suite after changes.

## Phase 13 — Acceptance Test

Use:

```text
SCN-001-RUN-A
vs
SCN-001-RUN-B

Primary metric:
chest_deflection
```

Ask:

> Why is the restraint configuration currently a leading contributor?

Then:

> What evidence contradicts this hypothesis?

Then:

> I disagree. Investigate torso rotation as an alternative explanation.

Then:

> What should I analyze next?

Then:

> Show me the source for your LS-DYNA-related claim.

Verify every answer is grounded in current investigation evidence or a valid retrieved source.

## Do not implement

Do not add:

```text
multi-agent swarm
voice
web search
automatic solver execution
animation/VLM integration
historical-case generation
global long-term memory
new vector database
```

These are outside this change.

## Completion report

At the end, report:

```text
Files changed
Migration
New API endpoint
Frontend components
Agent changes
RAG changes
Tests added
Test results
Acceptance-test results
Known limitations
```

Explicitly mark:

```text
Copilot: PASS/FAIL
Evidence grounding: PASS/FAIL
RAG relevance: PASS/FAIL
Engineer challenge: PASS/FAIL
LLM failure isolation: PASS/FAIL
```

Do not claim completion unless the tests and acceptance flow pass.
