# Claude Opus 5 Execution Strategy

## Goal

Use one orchestrated Claude Code session without using one giant context dump.

```text
lean root instructions
+ on-demand Skills
+ bounded subagents
+ persistent state files
+ deterministic hooks/tests
```

## Model

Use `claude-opus-5`.

Suggested effort policy:
- medium for normal repo implementation;
- high for security/schema/architecture conflicts;
- do not use max by default;
- for frontend, prefer browser/visual verification over increasing thinking indefinitely.

## Context efficiency

Do not paste PRD/TRD/UI/security/test plans into `CLAUDE.md`.

Root `CLAUDE.md` contains:
- identity;
- invariants;
- canonical doc paths;
- commands/gates;
- forbidden behaviors;
- definition of done.

Detailed procedures live in `.claude/skills/` and load only when relevant.

## Subagent budget

At discovery, use at most three concurrent read-only subagents:

```text
A frontend/product
B backend/data/security
C tests/cleanup
```

Default depth: one.

Implementation stays in the main agent unless work is genuinely independent.

## Persistent rebuild state

Maintain:

```text
docs/rebuild/STATE.md
docs/rebuild/FILE_LEDGER.md
docs/rebuild/DECISIONS.md
```

This is cheaper and more reliable than reconstructing decisions from a long chat.

## Phases

```text
A baseline/no edits
B architecture/design
C backend product foundation
D ingestion integration
E frontend rebuild
F tests/security/eval
G cleanup
H final verification/release report
```

## Token-saving rules

- search before opening large files;
- do not reread unchanged specs;
- reference canonical docs by path;
- load Skills only when needed;
- keep subagent reports bounded;
- summarize progress in STATE.md;
- avoid repeating requirements in narration;
- summarize test logs instead of dumping everything;
- do not send multiple agents to reread the full repo unless independent review requires it.

## Important distinction

“One Claude session” does not mean “one destructive command.”

The orchestrator can execute the full program in one session while still having checkpoint gates.

## Human/explicit decision gates

Stop before:
- deleting unclassified files;
- changing auth trust model;
- changing provider data policy;
- dropping/squashing migration history;
- weakening retrieval gates;
- automatically promoting uploads to authoritative scope;
- deleting eval baselines;
- adding major frameworks/services outside the TRD.
