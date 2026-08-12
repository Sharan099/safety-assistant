# ADR-0002: Adopt Ponytail as the Claude Code development workflow

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

`PRD.md` §10 and `TRD.md` §5 require inspecting
[DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) before
implementation and integrating it as a development-time workflow/skill, not
as runtime application logic — or reporting back if it's unavailable/unsuitable.

## Investigation

Ponytail is a Claude Code plugin (JavaScript, MIT license) distributed via
Claude Code's plugin marketplace mechanism. It enforces a 7-rung "decision
ladder" before writing new code (does it need to exist → reuse → stdlib →
platform feature → existing dependency → one line → minimum implementation),
with `/ponytail-review`, `/ponytail-audit`, `/ponytail-debt`, and
`/ponytail-gain` commands. It is compatible with this environment (Claude
Code CLI, Node.js on PATH) and required no code changes to adopt.

## Decision

Installed at user scope:

```
claude plugin marketplace add DietrichGebert/ponytail
claude plugin install ponytail@ponytail
```

Confirmed via `claude plugin list` → `ponytail@ponytail 4.9.0, enabled`.
It is a **development-time discipline only**; no runtime application code
depends on it (per TRD §5, it becomes a runtime dependency only if a concrete
runtime use case is discovered — none has been).

## Consequences

- New sessions apply the decision ladder automatically; this session applies
  it manually since the plugin loads on next restart.
- `/ponytail-review` should be run against non-trivial diffs before
  considering a change done.
