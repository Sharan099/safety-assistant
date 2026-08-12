# Passive Safety CAE Investigation Agent — Application Flow

**Version:** 2.0  
**Status:** V1 Application Contract

---

# 1. Product Flow

```text
Dashboard
 ↓
Project
 ↓
Runs
 ↓
New Investigation
 ↓
Select Run A / Run B
 ↓
Run Identity
 ↓
Quality Gate
 ↓
Comparability
 ↓
Global Crash Response
 ↓
Configuration Diff
 ↓
Metric-Specific Signal Analysis
 ↓
First Divergence
 ↓
Mechanism / Animation Review
 ↓
Historical Retrieval
 ↓
Knowledge Retrieval
 ↓
Hypothesis Generation
 ↓
Evidence Evaluation
 ↓
Engineer Review
 ↓
Decision
 ↓
Follow-up
```

This is not strictly linear.

The engineer can loop:

```text
Hypothesis
 ↓
Need more evidence
 ↓
Signal Analysis
 ↓
Hypothesis update
```

or:

```text
Hypothesis rejected
 ↓
Alternative mechanism
 ↓
New hypothesis
```

---

# 2. Application States

```text
DRAFT
RUNS_SELECTED
IDENTITY_CHECK
QUALITY_CHECK
COMPARABILITY_CHECK
GLOBAL_RESPONSE
CONFIGURATION_ANALYSIS
SIGNAL_ANALYSIS
MECHANISM_REVIEW
HYPOTHESIS_ANALYSIS
EVIDENCE_REVIEW
ENGINEER_REVIEW
DECISION
FOLLOW_UP
BLOCKED
CLOSED
```

---

# 3. Dashboard

Show:

```text
Active investigations
Recent runs
Quality warnings
Failed runs
Recent reports
Knowledge ingestion jobs
```

Primary CTA:

```text
New Investigation
```

---

# 4. New Investigation

Required:

```text
Run A
Run B
Engineering question
Primary metric
```

Optional:

```text
Physical test
Component focus
Additional signals
Event window
```

The user should not need to write a long AI prompt.

---

# 5. Run Identity

Display side-by-side:

```text
Property             Run A          Run B
Vehicle              ...
Model                ...
Revision             ...
Dummy                ...
Impact               ...
Speed                ...
Seat                 ...
Belt                 ...
Airbag               ...
Solver               ...
Version              ...
```

Statuses:

```text
SAME
CHANGED
UNKNOWN
```

---

# 6. Quality Gate

Show all checks:

```text
Solver termination
Errors/warnings
Final time
Time step
Mass scaling
Energy balance
Hourglass
Contact energy
Penetration
Failed elements
Output completeness
Signal availability
```

Overall:

```text
PASS
WARNING
FAIL
UNKNOWN
```

If FAIL:

```text
Continue analysis with warning
```

may be available, but final conclusions must be constrained.

---

# 7. Comparability

Show:

```text
Global crash pulse
Occupant response
Primary metric
Physical correlation
Causal isolation
```

For each:

```text
COMPARABLE
CONDITIONAL
NOT_COMPARABLE
NOT_ESTABLISHED
UNKNOWN
```

Each result has an explanation.

---

# 8. Global Crash Response

Show:

```text
Vehicle acceleration
Energy
Intrusion
Structural response
```

Use synchronized time cursors.

If global crash pulse differs materially, surface this before local occupant hypotheses.

---

# 9. Configuration Diff

Hierarchy:

```text
Vehicle
 ├── Body
 ├── Seat
 ├── Restraint
 │    ├── Belt
 │    ├── Retractor
 │    ├── Pretensioner
 │    └── Force limiter
 ├── Airbag
 ├── Dummy
 ├── Contacts
 └── Solver
```

Each change:

```text
old value
new value
source
classification
```

---

# 10. Signal Analysis

The application creates a metric-specific signal plan.

Example:

```text
Chest deflection
 ├── chest acceleration
 ├── chest velocity
 ├── belt force
 ├── pelvis acceleration
 ├── torso rotation
 ├── airbag
 └── crash pulse
```

The engineer can add/remove signals.

---

# 11. Signal Workspace

```text
┌───────────────────────────────────────────────┐
│ Signal                                        │
│ [Chest Deflection ▼]                         │
├───────────────────────────────────────────────┤
│                                               │
│              Time History                     │
│                                               │
├───────────────────────────────────────────────┤
│ Peak | Time-to-peak | Divergence | Correlation│
├───────────────────────────────────────────────┤
│ Event timeline                                │
└───────────────────────────────────────────────┘
```

---

# 12. First Divergence

Example:

```text
41.7 ms  Belt force
46.2 ms  Torso rotation
48.0 ms  Chest acceleration
51.0 ms  Chest deflection
```

Each event is independent.

Clicking an event updates:

```text
plots
animation
evidence
hypothesis context
```

---

# 13. Mechanism Review

If animation exists:

```text
Signal cursor
↕
Animation
↕
Event
↕
Component
```

The UI must clearly distinguish:

```text
Observed
Calculated
AI interpretation
```

---

# 14. Knowledge Retrieval

The engineer can inspect:

```text
Regulations
LS-DYNA documentation
PAM-CRASH documentation
ANSYS documentation
Internal documents
Historical cases
```

Current V1 sources are only those actually present/authorized.

---

# 15. Hypothesis Review

Example:

```text
Hypothesis:
Changed belt behaviour is a leading contributor.

Supporting:
✓ Belt force divergence
✓ Belt configuration changed

Contradicting:
⚠ Vehicle model revision changed

Missing:
? Controlled belt isolation
```

Status:

```text
PROPOSED
SUPPORTED
PARTIALLY_SUPPORTED
CONTRADICTED
REJECTED
INCONCLUSIVE
```

---

# 16. Engineer Review

Actions:

```text
Accept
Reject
Modify
Request another signal
Request another source
Add note
Request controlled comparison
Mark inconclusive
```

The agent must pause at this boundary when required.

---

# 17. Final Decision

Choices:

```text
ACCEPTED_EXPLANATION
PARTIALLY_SUPPORTED
INCONCLUSIVE
INVALID_COMPARISON
REQUIRES_CONTROLLED_RERUN
```

The engineer enters:

```text
decision
comment
follow-up
```

---

# 18. Report

Generate:

```text
Question
Runs
Quality
Comparability
Global response
Configuration differences
Signal analysis
Divergence events
Mechanism evidence
Historical evidence
Technical documentation
Hypotheses
Contradictions
Unknowns
Decision
Follow-up
Provenance
```

---

# 19. AI Interaction

Contextual actions:

```text
Explain this evidence
Why was this source retrieved?
Find contradictions
Analyse another signal
Find similar historical cases
What evidence is missing?
Draft investigation summary
```

No giant chatbot should dominate the UI.

---

# 20. Knowledge Ingestion Flow

```text
Upload/register
 ↓
Hash
 ↓
Inspect
 ↓
Extract
 ↓
Quality
 ↓
Structure
 ↓
Tables/figures/equations
 ↓
Chunk
 ↓
Embed
 ↓
Index
 ↓
READY
```

UI should expose progress.

---

# 21. Error Flow

If extraction fails:

```text
FAILED
→ show document
→ show failed step
→ show error
→ allow retry
```

If LLM fails:

```text
LLM UNAVAILABLE
→ preserve state
→ deterministic analysis remains accessible
→ retry
```

If source is missing:

```text
SOURCE NOT AVAILABLE
```

Never fabricate a result.

---

# 22. V1 Vertical Slice

```text
Run A
+
Run B
 ↓
Quality
 ↓
Comparability
 ↓
Configuration
 ↓
Chest-deflection signals
 ↓
Divergence
 ↓
Evidence
 ↓
Hypothesis
 ↓
Engineer review
```

Then add RAG and agent orchestration.
