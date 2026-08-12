# Passive Safety CAE Investigation Agent — UI/UX Design Brief

**Version:** 2.0  
**Status:** V1 UI Contract

---

# 1. Design Goal

Create a professional engineering workstation.

Mental model:

```text
CAE post-processing
+
engineering investigation
+
evidence management
+
AI copilot
```

Not:

```text
ChatGPT clone
```

---

# 2. Design Principles

1. Engineering-first.
2. Evidence visible.
3. Uncertainty explicit.
4. Numbers easy to compare.
5. Desktop-first.
6. Low visual noise.
7. AI contextual.
8. Engineer remains in control.
9. No decorative AI gimmicks.
10. Every important claim traceable.

---

# 3. Global Layout

```text
┌─────────────────────────────────────────────────────────┐
│ Header                                                  │
├──────────────┬──────────────────────────────────────────┤
│ Sidebar      │ Main workspace                           │
│              │                                          │
│ Dashboard    │                                          │
│ Projects     │                                          │
│ Runs         │                                          │
│ Investigations                                          │
│ Knowledge    │                                          │
│ Reports      │                                          │
│ Settings     │                                          │
└──────────────┴──────────────────────────────────────────┘
```

---

# 4. Investigation Layout

```text
┌────────────────────────────────────────────────────────────┐
│ Investigation: Run A vs Run B                             │
├──────────────┬─────────────────────────────┬───────────────┤
│ RUN A        │ Analysis                    │ RUN B         │
│              │                             │               │
│ Manifest     │ Chart / Diff / Analysis     │ Manifest      │
│ Quality      │                             │ Quality       │
│ Config       │                             │ Config        │
├──────────────┴─────────────────────────────┴───────────────┤
│ Investigation State                                        │
├────────────────────────────────────────────────────────────┤
│ Evidence / Hypothesis / AI assistant                       │
└────────────────────────────────────────────────────────────┘
```

---

# 5. Color Semantics

Use restrained technical colors.

Status must use:

```text
PASS
WARNING
FAIL
UNKNOWN
```

Never use color alone.

Example:

```text
✓ PASS
⚠ WARNING
✕ FAIL
? UNKNOWN
```

---

# 6. Typography

Prioritize:

- units
- numerical alignment
- readable tables
- signal labels
- timestamps
- source locators

Use tabular numerals for metrics where supported.

---

# 7. Dashboard

Cards:

```text
Open investigations
Recent runs
Quality warnings
Failed checks
Knowledge sources
Recent reports
```

---

# 8. Run Browser

Columns:

```text
Run ID
Vehicle
Model
Dummy
Impact
Solver
Quality
Created
```

Filters:

```text
project
vehicle
solver
dummy
impact
revision
quality
date
```

---

# 9. Run Manifest

Side-by-side comparison:

```text
Property             Run A          Run B
Impact speed         50 km/h        50 km/h
Dummy                THOR           THOR
Belt                 B-17           B-18
Airbag               A-12           A-12
Model                v12.3          v12.4
```

Clicking a property reveals its source.

---

# 10. Quality Screen

Table:

```text
Check                  Status      Value
Solver termination     PASS
Errors/warnings        PASS
Final time             PASS
Timestep               PASS
Mass scaling           WARNING
Energy                 PASS
Hourglass              PASS
Contact energy         WARNING
Penetration             FAIL
```

Show an explanation drawer.

---

# 11. Comparability Screen

Use a matrix:

```text
Dimension              Result
Global pulse           COMPARABLE
Occupant response      CONDITIONAL
Primary metric         COMPARABLE
Physical correlation   NOT_COMPARABLE
Causal isolation       NOT_ESTABLISHED
```

---

# 12. Configuration Diff

Use a tree plus detail pane.

```text
Restraint
 └── Belt
      └── Force limiter
           4000 N → 3500 N
```

Bad:

```text
parameter_147 changed
```

Good:

```text
Restraint > Belt > Force limiter changed
```

---

# 13. Signal Workspace

Controls:

```text
Run A
Run B
Overlay
Difference
Normalize
Zoom
Time window
```

Metric cards:

```text
Peak
Time-to-peak
First divergence
Correlation
Duration
Integral
```

---

# 14. Event Timeline

```text
41.7 ms  Belt force divergence
46.2 ms  Torso rotation divergence
48.0 ms  Chest acceleration divergence
51.0 ms  Chest deflection divergence
```

Event click:

```text
plot cursor
animation cursor
evidence
hypothesis context
```

---

# 15. Animation

```text
┌──────────────────────────┬───────────────────────────┐
│                          │ Signal plots              │
│ Animation                │                           │
│                          │                           │
│                          │                           │
├──────────────────────────┴───────────────────────────┤
│ 46.2 ms    Play  Pause  Step                         │
└──────────────────────────────────────────────────────┘
```

---

# 16. Evidence Card

```text
E-014
CALCULATED

Belt force diverged at 41.7 ms

Source:
SIM-0041 / belt_force

Algorithm:
first_divergence_detector v0.1.0

Parameters:
...
```

Click source.

---

# 17. Hypothesis Card

```text
H-001

Changed belt behaviour is a leading contributor.

Supporting evidence
✓ E-014
✓ E-018

Contradicting
⚠ E-021

Missing
? controlled isolation

Status:
PARTIALLY_SUPPORTED
```

Actions:

```text
Accept
Reject
Modify
Request analysis
```

---

# 18. AI Assistant

The AI panel should be compact and contextual.

Actions:

```text
Explain
Investigate
Compare
Retrieve
Find contradiction
Find missing evidence
Draft summary
```

The AI must always link important statements to evidence.

---

# 19. Knowledge Viewer

Display:

```text
Document
Revision
Page
Section
Source authority
Extracted content
Figure/table
```

Example:

```text
LS-DYNA R17 Theory Manual
Page 214
Section ...
Authority: OFFICIAL_DOCUMENTATION
```

---

# 20. Report

The report UI shows:

```text
Finding
Evidence
Contradictions
Unknowns
Decision
Follow-up
```

Avoid presenting an AI confidence percentage unless the metric is scientifically defined.

---

# 21. UX Safety Rules

Never:

- hide failed quality checks;
- present speculation as fact;
- show a source that was not retrieved;
- imply that a regulation was checked when it wasn't;
- make a final engineering decision automatically.

---

# 22. V1 Reusable Components

```text
RunManifest
StatusBadge
QualityCheckTable
ComparabilityMatrix
ConfigurationTree
DiffViewer
SignalChart
SignalMetricTable
EventTimeline
AnimationViewer
EvidenceCard
HypothesisCard
SourceViewer
InvestigationProgress
EngineerDecisionPanel
```
