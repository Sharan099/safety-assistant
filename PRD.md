# Passive Safety CAE Investigation Agent — Product Requirements Document

**Version:** 2.0  
**Status:** V1 Product Contract  
**Target users:** Passive Safety / Occupant Protection CAE Engineers  
**Primary development model:** Claude Code + Claude Sonnet  
**Development environment:** Local-first Windows workstation, 8 GB RAM  
**V1 focus:** Two-run crash-simulation investigation with evidence-backed AI assistance

---

## 1. Product Vision

Build a production-oriented AI-assisted engineering investigation application for passive-safety CAE engineers.

The product helps an engineer answer questions such as:

- Why did chest deflection increase between two simulations?
- What changed between Run A and Run B?
- Are the two runs actually comparable?
- When did the response first diverge?
- Which signals changed before the target metric changed?
- What physical or numerical mechanisms could explain the difference?
- Have we seen a similar historical case?
- What official solver/regulatory documentation is relevant?
- What evidence supports or contradicts each hypothesis?
- What follow-up analysis should the engineer perform?

The application accelerates investigation but does not replace engineering judgement.

---

# 2. Product Philosophy

The primary product is an **engineering investigation workstation**, not a chatbot.

The engineer's workflow is:

```text
Question
→ Run identity
→ Quality
→ Comparability
→ Global response
→ Configuration differences
→ Signal analysis
→ First divergence
→ Mechanism review
→ Hypotheses
→ Evidence
→ Engineer review
→ Decision / follow-up
```

The workflow is iterative. The engineer can request another signal, reject a hypothesis, inspect another source, or return to configuration analysis.

---

# 3. Target Audience

## Primary

- Passive Safety CAE Engineer
- Occupant Safety Engineer
- Crash Simulation Engineer
- Restraint System Engineer
- Vehicle Safety Simulation Engineer

## Typical environment

Engineers may work with:

- LS-DYNA
- PAM-CRASH
- ANSYS / Mechanical
- CAE post-processing tools
- time-history data
- animation
- model configuration
- crash-test correlation
- regulations
- internal engineering reports
- historical simulation investigations

V1 will support a solver-agnostic domain model with the first detailed implementation centered on LS-DYNA-compatible synthetic data and documentation.

---

# 4. User Problem

Current investigations often require the engineer to manually:

1. identify two simulation runs;
2. verify metadata;
3. inspect solver quality;
4. determine whether the runs are comparable;
5. compare model/configuration changes;
6. inspect global crash response;
7. inspect multiple time-history signals;
8. locate the first meaningful divergence;
9. inspect animation;
10. search historical investigations;
11. search technical documentation;
12. construct hypotheses;
13. gather supporting/contradicting evidence;
14. document the conclusion.

The application should reduce this investigation overhead while preserving engineering traceability.

---

# 5. Core Product Requirements

## PR-001 — Run comparison

The engineer can select:

```text
Run A
Run B
Primary metric
Engineering question
```

The system creates an immutable investigation context.

---

## PR-002 — Run identity

The application must expose:

- project
- vehicle
- model revision
- run ID
- parent run
- solver
- solver version
- dummy/HBM
- impact condition
- seat
- restraint configuration
- processing version
- result status

Unknown values must remain explicitly unknown.

---

## PR-003 — Quality gate

The application evaluates, where data is available:

- solver termination
- errors/warnings
- final simulation time
- timestep history
- mass scaling / added mass
- energy balance
- hourglass energy
- contact energy
- penetration
- failed/deleted elements
- rigid/tied contact issues
- result database completeness
- signal availability

Statuses:

```text
PASS
WARNING
FAIL
UNKNOWN
```

---

## PR-004 — Comparability

Quality and comparability are separate.

The system evaluates:

- global crash pulse
- occupant response
- primary metric
- physical-test correlation
- causal isolation

Possible results:

```text
COMPARABLE
CONDITIONAL
NOT_COMPARABLE
NOT_ESTABLISHED
UNKNOWN
```

---

## PR-005 — Global crash response

Before attributing an occupant response to a local component, compare relevant global behavior:

- vehicle acceleration/crash pulse
- barrier/load response where available
- energy
- intrusion
- structural deformation
- major load paths

---

## PR-006 — Configuration diff

The system compares CAE-relevant configuration:

- component identity/revision
- mesh
- element formulation
- material
- thickness
- sections
- welds/connectors
- contacts
- friction
- seat
- belt routing
- pretensioner
- force limiter
- airbag
- dummy/HBM
- solver controls
- mass scaling
- output definitions
- post-processing recipe

Changes must be classified:

```text
INTENTIONAL
DEPENDENCY
UNINTENTIONAL
UNKNOWN
```

---

## PR-007 — Metric-specific signal analysis

The system creates a signal plan based on the engineering question.

Example:

```text
Question: Why did chest deflection increase?

Primary:
Chest deflection

Related:
Chest acceleration
Chest velocity
Chest displacement
Shoulder-belt force
Lap-belt force
Pelvis acceleration
Torso rotation
Airbag pressure/deployment
Vehicle crash pulse
```

---

## PR-008 — First divergence

The system calculates first meaningful divergence separately for each signal.

Each event retains:

- signal
- timestamp
- algorithm
- algorithm version
- threshold
- analysis window
- filtering
- alignment
- source run
- source signal

The system must never infer the target metric's divergence from another signal's divergence without evidence.

---

## PR-009 — Mechanism review

Where animation is available, synchronize:

```text
signal timestamp
↕
animation timestamp
↕
event
↕
component
```

Animation supports mechanism inspection but does not establish causality by itself.

---

## PR-010 — Historical retrieval

Retrieve similar historical cases using:

- metric
- signal features
- waveform similarity
- divergence timing
- vehicle/model family
- dummy
- restraint configuration
- impact condition
- relevant component changes
- quality status

---

## PR-011 — Knowledge retrieval

The knowledge system uses source tiers:

```text
Tier 1 — Regulations
Tier 2 — Official solver/software documentation
Tier 3 — Approved internal engineering reports
Tier 4 — Historical investigations
Tier 5 — Synthetic CAE cases
Tier 6 — LLM reasoning
```

The LLM is not an authoritative knowledge source.

---

# 6. Current Knowledge Corpus

The current local folder contains the following documents, verified from the user's supplied screenshot:

```text
cc-PAM-Crash-Spec-Sheet
LS-DYNA_Manual_Theory_R17
LS-DYNA_Manual_Vol_I_R17
LS-DYNA_Manual_Vol_II_R17
LS-DYNA_Manual_Vol_III_R17
LS-DYNA_Users_Guide
ls-dyna-examples-manual
UN_R16
UN_R94
UN_R95
UN_R129
```

These are the only currently available source documents assumed by V1.

Not currently assumed available:

```text
LS-DYNA Database Manual
Licensed PAM-CRASH reference manuals
Additional ANSYS Mechanical documentation
Company internal reports
Real historical investigations
```

The product must not fabricate content from unavailable sources.

---

# 7. Source Governance

## Regulations

Use only legally obtained official/regulatory documents.

## Solver documentation

Prefer official vendor documentation.

## PAM-CRASH

Only documentation legally obtained by the user/company.

The public PAM-CRASH specification sheet is a reference document, not a substitute for a licensed technical manual.

## Internal documents

Only approved company material.

## Historical investigations

Only authorized engineering records.

## Synthetic

Explicitly labelled `SYNTHETIC`.

---

# 8. AI Requirements

AI should:

- plan investigations
- choose tools
- retrieve evidence
- synthesize evidence
- generate hypotheses
- identify missing evidence
- explain findings
- draft reports

AI must not silently:

- calculate numerical engineering metrics
- invent solver behavior
- invent regulatory limits
- invent evidence
- override quality failures
- claim causality without evidence
- approve engineering decisions

---

# 9. Free LLM Runtime Requirement

The project will initially integrate:

**FreeLLMAPI** as a configurable LLM gateway for development.

Repository supplied by the product owner:

`https://github.com/tashfeenahmed/freellmapi`

Important:

- free availability is not guaranteed;
- provider quotas/limits may change;
- API reliability must not be assumed;
- the application must use an LLM abstraction layer;
- the project must remain capable of switching providers.

Claude Sonnet remains the development/coding model used through Claude Code. The runtime LLM is a separate concern.

---

# 10. Claude Code Skill Requirement

Before implementation, the project should integrate/review:

**Ponytail**

Repository supplied by the product owner:

`https://github.com/DietrichGebert/ponytail`

Ponytail is treated as a development workflow/skill dependency, not as runtime application logic.

Claude Code must read the relevant Ponytail skill instructions before feature implementation where applicable.

If the repository is unavailable, incompatible, or unsuitable, Claude Code must report this rather than silently inventing an integration.

---

# 11. Primary User Experience

The primary workspace is:

```text
Investigation
├── Overview
├── Quality
├── Comparability
├── Crash Response
├── Configuration
├── Signals
├── Mechanism
├── Evidence
├── Hypotheses
├── Knowledge
├── Review
└── Report
```

Chat is contextual, not the primary navigation.

---

# 12. V1 Scope

V1 must implement:

```text
Synthetic Run A
+
Synthetic Run B
→
Run identity
→
Quality
→
Comparability
→
Configuration diff
→
Metric-specific signal analysis
→
First divergence
→
Evidence
→
Hypothesis
→
Engineer review
```

Then add:

```text
regulation/solver RAG
→ historical retrieval
→ LangGraph orchestration
→ full investigation workspace
```

---

# 13. V1 Synthetic Scenarios

Minimum benchmark:

```text
SCN-001 Belt revision
SCN-002 Pretensioner timing
SCN-003 Airbag deployment timing
SCN-004 Crash pulse change
SCN-005 Seat position
SCN-006 Dummy positioning
SCN-007 Contact/friction change
SCN-008 Signal-processing change
SCN-009 Model revision without intended physical change
SCN-010 Numerical-quality failure
```

Every scenario contains:

```text
Run A
Run B
known changed factor
expected signal changes
allowed conclusions
disallowed conclusions
```

---

# 14. Success Criteria

V1 is successful when an engineer can:

1. select two runs;
2. understand whether they are comparable;
3. see quality problems;
4. see configuration changes;
5. inspect relevant signals;
6. identify divergence events;
7. inspect evidence;
8. see competing hypotheses;
9. review technical/historical evidence;
10. make and record a decision;
11. reproduce the investigation later.

---

# 15. Out of Scope for V1

- autonomous commercial solver execution
- automatic simulation submission
- autonomous engineering approval
- full CFD platform
- full FEA platform
- automatic design optimization
- multi-agent swarm
- enterprise SSO
- cloud-scale infrastructure
- complete ANSYS corpus
- complete PAM-CRASH corpus
