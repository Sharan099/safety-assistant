# ADR-0010: Level 3 uses the existing `Knowledge source/` folder, not `knowledge_source/`

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`PRD_LEVEL3.md` §5, `TRD_LEVEL3.md` §1/§3, and `LEVEL3_INSTRUCTIONS.md`
all name the immutable raw-source root as `knowledge_source/`. The repository's
actual immutable root — already referenced by the committed
`knowledge/00_registry/source_manifest.yaml` (`original_path` fields),
`packages/ingestion/pipeline.py`, `.gitignore`, and every V1 ADR — is
`Knowledge source/` (capitalized, with a literal space).

`LEVEL3_INSTRUCTIONS.md` §3 explicitly delegates this kind of
conflict: "If two existing documents disagree, identify the conflict and
choose the least disruptive solution consistent with Level 3 requirements."

## Decision

Keep `Knowledge source/` as the one immutable raw-source root. Level 3 code
(`scripts/profile_knowledge_sources.py`, archive inspector, manifest
extensions) treats `Knowledge source/` as what the Level-3 docs call
`knowledge_source/` — same folder, same immutability rules (PRD_LEVEL3.md §5:
never modify, rename, or rewrite originals), just the name already in use.

Renaming the folder would require rewriting every already-verified SHA-256
`original_path` in the manifest and touching working, tested ingestion code
for a purely cosmetic gain — the definition of "more disruptive than
necessary."

## Consequences

- All Level-3 code and docs written from this point on refer to
  `Knowledge source/` by its real name; comments quoting the Level-3 design
  docs' `knowledge_source/` spelling will note the mapping once, here.
- No file movement, no re-hashing, no manifest rewrite for existing V1
  entries.
