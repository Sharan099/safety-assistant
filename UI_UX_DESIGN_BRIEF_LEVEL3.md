# Passive Safety CAE Investigation Agent — UI/UX Design Brief

**Version:** 1.1  
**Status:** Level-3 implementation contract  
**Primary user:** Passive Safety / Occupant Protection / Crash / CAE Engineer  
**Target:** Desktop engineering workstation, primarily 1280–1440px+ displays.

## 1. UX North Star

This is **not a chatbot with an engineering theme**. It is an engineering investigation workstation with an AI copilot embedded into the workflow.

The engineer should feel:

> “I am driving the investigation; the system is doing the tedious comparison, retrieval, analysis and evidence-gathering work.”

The interface must prioritize:
- fast access to runs, signals, models and evidence;
- high information density without clutter;
- traceability;
- direct access to original sources;
- keyboard/mouse efficiency;
- explicit separation of fact, calculation, inference and hypothesis.

## 2. Research Basis

The design direction combines:

### Human-factors / professional workstation research
NASA display guidance emphasizes consistency, situation awareness, reduced workload, grouping related controls, keeping critical information accessible, avoiding unnecessary navigation, and clear display/control relationships. These principles are used as human-factors references, not automotive regulatory requirements. citeturn0search0turn0search3turn0search8

NASA also recommends iterative human-in-the-loop usability evaluation and direct involvement of representative users. citeturn0search6turn0search8

### Figma design-system research
Figma's Pattern Library work highlights a shared design system as a source of truth for consistency and scalable component reuse. citeturn0search7

### Contemporary dashboard inspiration
Dribbble's engineering-dashboard collection provides visual references for dense engineering dashboards, simulation interfaces and AI-assisted technical workspaces. It is inspiration only; engineering usability takes priority over decorative trends. citeturn0search11

## 3. Overall Application Shell

Use a persistent desktop shell:

```text
┌────────────────────────────────────────────────────────────────────┐
│ Logo | Investigations | Runs | CAE Models | Knowledge | Reports   │
│      | Search / Command Palette                              User │
├──────────────┬─────────────────────────────────────────────────────┤
│ Investigation│                 WORKSPACE                          │
│ Navigator    │                                                     │
│              │                                                     │
│ Overview     │                                                     │
│ Quality      │                                                     │
│ Comparability│                                                     │
│ Configuration│                                                     │
│ Signals      │                                                     │
│ Divergence   │                                                     │
│ Mechanism    │                                                     │
│ Evidence     │                                                     │
│ Hypothesis   │                                                     │
│ Review       │                                                     │
├──────────────┴─────────────────────────────────────────────────────┤
│ Run A | Run B | Solver | Units | Data Quality | Agent Status      │
└────────────────────────────────────────────────────────────────────┘
```

The engineer must always know:
1. where they are;
2. what has been checked;
3. what remains;
4. what evidence exists.

## 4. Investigation Workspace

Default three-panel layout:

```text
Left 22%   = investigation/context
Center 55%  = analysis/evidence
Right 23%   = AI copilot
```

All panels are resizable.

The center analysis area gets the most space. The AI panel must never dominate the screen.

Example:

```text
┌──────────────────────────────────────────────────────────────────┐
│ Investigation: Chest Deflection Increase                         │
│ Run A: BASE_001      Run B: VAR_014       [Compare] [Ask AI]    │
├──────────────────────────────────────────────────────────────────┤
│ ✓ Quality   ✓ Comparable   ⚠ Configuration   ○ Mechanism         │
├───────────────────┬──────────────────────────────┬───────────────┤
│ RUN CONTEXT       │ SIGNAL / EVIDENCE            │ AI COPILOT    │
│ model             │                              │               │
│ solver            │ response chart              │ plan          │
│ scenario          │ configuration diff          │ findings      │
│ dummy             │ divergence marker           │ evidence      │
│ restraint         │                              │ hypothesis    │
├───────────────────┴──────────────────────────────┴───────────────┤
│ Evidence | Timeline | Configuration Diff | Tool Activity         │
└──────────────────────────────────────────────────────────────────┘
```

## 5. Investigation Navigation

Use a visible workflow:

```text
01 Context
02 Quality
03 Comparability
04 Configuration
05 Signals
06 Divergence
07 Mechanism
08 Evidence
09 Hypothesis
10 Review
```

Do not force engineers through a wizard. They can jump between completed stages.

## 6. AI Copilot

The copilot should look like a contextual engineering assistant, not a ChatGPT clone.

```text
┌──────────────────────────────┐
│ INVESTIGATION COPILOT        │
│                              │
│ QUESTION                     │
│ Why did chest deflection     │
│ increase?                    │
│                              │
│ PLAN                         │
│ ✓ Compare runs               │
│ ✓ Check configuration        │
│ ✓ Analyze chest signal       │
│ → Retrieve solver evidence   │
│                              │
│ FINDINGS                     │
│ ...                          │
│                              │
│ EVIDENCE                     │
│ [Run comparison]             │
│ [LS-DYNA p.184]              │
│ [Regulation p.42]            │
│                              │
│ [Review hypothesis]          │
└──────────────────────────────┘
```

Do not expose hidden chain-of-thought. Show only:
- tool name;
- purpose;
- status;
- concise result;
- source/evidence.

## 7. Agent Activity Trace

Example:

```text
✓ analyze_signal
  Chest deflection — first divergence at 82.4 ms

✓ retrieve_structured_cae
  2 CONTACT definitions differ

✓ retrieve_knowledge
  3 relevant sources

→ evaluate_hypothesis
```

This creates trust without exposing private reasoning.

## 8. Evidence Cards

Evidence must be visually distinct:

```text
┌──────────────────────────────────────────┐
│ AUTHORITATIVE SOURCE                     │
│ UN R94                                   │
│ Page 42 • Section 5.3                    │
│                                          │
│ Relevant requirement                     │
│ ...                                      │
│                                          │
│ [Open source] [Open page] [Use evidence]│
└──────────────────────────────────────────┘
```

Every evidence item should show:
- source type;
- authority;
- document;
- page/line;
- relevance;
- confidence where applicable.

## 9. Fact / AI / Engineering Trust Model

The UI must distinguish:

```text
SOURCE FACT
DETERMINISTIC RESULT
OBSERVATION
AI INFERENCE
HYPOTHESIS
ENGINEER DECISION
```

Example:

```text
✓ SOURCE FACT
UN R94, page 42

✓ CALCULATED
Peak chest deflection = 34.8 mm

◆ AI INFERENCE
Response change may be associated with...

? HYPOTHESIS
Requires restraint-force evidence.

✓ ENGINEER REVIEW
Accepted for further investigation
```

The engineer decision must visually carry more authority than an AI hypothesis.

## 10. Global Search

Search should cover:

```text
documents
regulations
solver keywords
CAE decks
parts
materials
contacts
signals
runs
investigations
```

Exact identifiers must work well:

```text
*MAT_024
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
*DATABASE_BINARY_D3PLOT
THOR-05F
HIC15
UN R94
Part 1042
```

Use a command palette (`Ctrl/Cmd + K`) for expert navigation.

## 11. Hybrid RAG Search UI

Normal users should not need to understand RRF.

Show results by useful category:

```text
[Exact keyword match]
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE

[Semantic match]
Automatic surface-to-surface contact...

[CAE model]
Frontal_Impact_18deg_40kph.key

[Regulation]
UN R94 — relevant section
```

Advanced users can expand:

```text
Retrieval Details

BM25
Dense
Structured
RRF
Reranker
```

This makes the Hybrid RAG observable without exposing implementation complexity to every user.

## 12. Run Selection

Use an engineering table rather than a generic file picker.

Columns:

```text
Run ID
Vehicle
Model version
Solver
Solver version
Scenario
Impact condition
Dummy
Restraint
Timestamp
Quality
```

Support:
- search;
- filtering;
- sorting;
- pinning;
- multi-select;
- compare.

## 13. Run Comparison

Show A/B side-by-side:

```text
                 RUN A              RUN B
Model            ...                ...
Solver           ...                ...
Mesh             ...                ...
Material         ...                ...
Contact          ...                ...
Boundary         ...                ...
Control          ...                ...
Output           ...                ...
```

Use explicit states:

```text
UNCHANGED
ADDED
REMOVED
MODIFIED
UNKNOWN
```

Never hide configuration changes behind an AI summary.

## 14. Signal Analysis

The signal view is a primary engineering workspace:

```text
┌──────────────────────────────────────────────────────────────┐
│ Signal: Chest Deflection                                     │
│ Run A ───── Run B ───── Reference                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                     signal plot                              │
│                                                              │
│        ▲                                                     │
│        │       A                                             │
│        │      / \                                            │
│        │     /   \                                           │
│        │    /     \______                                    │
│        │   /             B                                   │
│        └──────────────────────────────► time                 │
├──────────────────────────────────────────────────────────────┤
│ Peak | Time-to-peak | Divergence | Integral | Threshold      │
├──────────────────────────────────────────────────────────────┤
│ Cursor | Zoom | Pan | Align | Normalize | Export             │
└──────────────────────────────────────────────────────────────┘
```

Support:
- zoom;
- pan;
- cursor;
- multi-cursor;
- range selection;
- show/hide signals;
- alignment;
- normalization;
- units;
- thresholds;
- annotations;
- export.

A selected time range becomes context for the AI.

## 15. Divergence

When first divergence is detected:

```text
FIRST DIVERGENCE
82.4 ms

Chest Deflection
Run A: 31.2 mm
Run B: 34.8 mm

Potential related change:
Seatbelt force limiter
```

"Potential related change" must be labelled as association/hypothesis, never proven causality.

## 16. Configuration Diff

Use a code-diff-inspired CAE viewer:

```text
RUN A                         RUN B

*MAT_024                      *MAT_024
MID = 12                      MID = 12
RO = 7.85                     RO = 7.85

*CONTACT_...                  *CONTACT_...
SOFT = 1                      SOFT = 2
```

Use:
- added;
- removed;
- modified;
- unchanged.

## 17. CAE Model Explorer

Tree:

```text
Deck
├── Includes
├── Parts
├── Materials
├── Sections
├── Nodes
├── Elements
├── Contacts
├── Boundaries
├── Controls
└── Databases
```

Selecting an entity opens:
- metadata;
- raw source;
- line range;
- relationships;
- used-by;
- references.

## 18. Raw Source Viewer

Engineers must always be able to inspect the original source:

```text
Frontal_Impact_18deg_40kph.key

Line 2841–2852

*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
...
```

Show:
- source file;
- line numbers;
- include path;
- raw text;
- parsed interpretation.

Never hide source behind AI.

## 19. Document Viewer

Use:

```text
left   = page thumbnails
center = original document
right  = evidence / metadata
```

Clicking an AI citation should navigate directly to the cited page.

For extracted tables, provide:
- original page;
- structured table;
- extraction status;
- source coordinates;
- open-original action.

## 20. Engineer Review

AI hypothesis card:

```text
AI HYPOTHESIS

Chest deflection increased because the
restraint force-limiter configuration changed.

Evidence:
✓ Configuration diff
✓ Signal divergence
✓ Solver documentation

Contradicting evidence:
⚠ No direct restraint-force channel available

Confidence:
MEDIUM

[Accept]
[Reject]
[Needs more evidence]
[Edit hypothesis]
```

The decision is stored in the investigation.

## 21. Visual Style

Recommended:

> Dark engineering workstation theme by default.

Use layered surfaces rather than pure black everywhere:

```text
background
panel
elevated panel
selected
hover
border
```

Provide:
```text
Dark
Light
System
```

Use color sparingly and never as the sole indicator.

Semantic roles:

```text
neutral = normal
blue    = navigation/selection
green   = verified/completed
amber   = warning
red     = error/critical
purple  = AI/inference
```

Always pair color with text/icon.

## 22. Typography

Prefer:

```text
UI: Inter / system sans
Technical identifiers: JetBrains Mono / equivalent
```

Technical identifiers should be visually distinct:

```text
*MAT_024
Part 1042
Run_2026_014
```

## 23. Density

Design for:

> **information-dense, not cluttered.**

Use compact rows for:
- runs;
- signals;
- entities;
- evidence;
- metadata.

Avoid huge decorative cards.

## 24. Tables

Tables are primary engineering UI.

Support:
- column resize;
- sorting;
- filtering;
- pinning;
- copy;
- CSV export;
- column visibility;
- virtualization for large datasets.

## 25. Keyboard / Expert Interaction

Provide:

```text
/                 global search
Ctrl/Cmd + K      command palette
← / →             signal cursor
Shift + ← / →     larger cursor movement
Z                 zoom
R                 reset chart
E                 evidence
?                 shortcuts
```

Validate shortcuts against browser/OS conflicts.

## 26. Status / Loading

Never use unexplained spinners.

Prefer:

```text
Analyzing chest signal...
Checking 14 configuration entities...
Searching 3 knowledge sources...
Reranking 18 candidates...
```

Persistent status can show:

```text
● Analysis running
● Knowledge indexing 72%
✓ PDF extraction complete
⚠ 3 pages require review
```

Avoid consumer-style toast spam.

## 27. Errors

Bad:

```text
Something went wrong.
```

Good:

```text
Signal data unavailable

Run: BASE_014
Signal: Chest_Deflection

Reason:
Channel not found in available output data.

Try:
• inspect available channels
• choose another signal
• open output manifest
```

## 28. Responsive Behavior

Primary target:

```text
desktop >= 1440px
```

Minimum:

```text
1280px
```

At smaller widths:
- collapse navigator;
- collapse copilot;
- preserve center investigation workspace.

Do not turn the product into a mobile-first design.

## 29. Accessibility

Target WCAG 2.2 AA where practical:
- keyboard navigation;
- visible focus;
- screen-reader labels;
- contrast;
- non-color indicators;
- reduced motion;
- resizable panels.

## 30. Animation

Motion should communicate:
- state change;
- progress;
- selection;
- panel transition;
- simulation meaning.

Avoid decorative animation.

## 31. Design System

Build reusable components:

```text
Button
IconButton
CommandPalette
DataTable
StatusBadge
EvidenceCard
SourceBadge
Metric
SignalChart
ComparisonChart
DiffViewer
Timeline
Tree
DocumentViewer
AIActivity
ToolTrace
HypothesisCard
ReviewPanel
FilterBar
SearchBox
Panel
ResizablePanel
Modal
Drawer
Tooltip
```

Centralize design tokens:
- colors;
- spacing;
- radius;
- typography;
- borders;
- chart styles;
- status colors.

## 32. What NOT to Build

Avoid:
- generic ChatGPT landing page;
- giant chat window;
- AI avatar;
- excessive rounded cards;
- glassmorphism everywhere;
- animated gradients;
- marketing hero sections;
- huge empty metric cards;
- consumer mobile patterns.

The product should feel closer to:

```text
CAE workstation
engineering analysis environment
professional IDE
scientific visualization tool
```

than a consumer AI application.

## 33. Level-3 Usability Tasks

Test with passive-safety engineers where possible:

```text
T1 Select two runs
T2 Check comparability
T3 Find configuration difference
T4 Plot chest deflection
T5 Find first divergence
T6 Find exact LS-DYNA keyword
T7 Open cited regulation page
T8 Inspect retrieval evidence
T9 Review AI hypothesis
T10 Record engineer decision
```

Measure:
- task completion;
- time-on-task;
- errors;
- navigation steps;
- information findability;
- user confidence;
- perceived workload.

## 34. Final UX Acceptance

A passive-safety engineer must be able to:

1. Open an investigation.
2. Select Run A and Run B.
3. See quality/comparability.
4. Inspect configuration differences.
5. Compare signals.
6. Identify first divergence.
7. Inspect a CAE entity.
8. Search an exact LS-DYNA keyword.
9. Inspect BM25/dense/structured evidence.
10. Open the original cited source page.
11. Inspect AI tool activity.
12. Review a hypothesis.
13. Accept/reject/request more evidence.
14. Preserve the engineer's decision.

## 35. Core Principle

The UI should reduce engineering cognitive load without reducing engineering visibility.

The engineer should never have to trust the AI blindly.

The interface must make the chain visible:

```text
SOURCE
  ↓
DATA
  ↓
CALCULATION
  ↓
OBSERVATION
  ↓
EVIDENCE
  ↓
AI HYPOTHESIS
  ↓
ENGINEER DECISION
```
