# ADR-0016: Recursive LS-DYNA include resolution, bounded by file count and file size

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md` §8 calls for moving from "main deck +
direct includes only" (the scope `scripts/ingest_level3_cae_decks.py`
originally used to stay within a smoke-test-era time budget) to full
recursive `*INCLUDE` traversal, with every reachable file
hashed/statused/kept traceable.

The real corpus makes the naive version of this genuinely risky: individual
component files run up to ~140 MB (Silverado's cabin/frame geometry). This
project has consistently treated "never load an entire large artifact into
RAM at once" as a hard constraint (`docs/ADR/0006`'s PDF page bound,
`docs/ADR/0011`'s dependency-size caution) — reading and lexing several
100+ MB text decks in one process, on an 8 GB machine, is the same risk
class.

## Decision

`packages/cae/lsdyna/resolve.py`'s `resolve_recursive()` does real
breadth-first transitive resolution — parse a deck, discover its includes,
parse those, repeat — reusing `include_graph.build_include_graph()`'s
existing RESOLVED/MISSING/CYCLE/DUPLICATE/AMBIGUOUS/OUTSIDE_ROOT logic over
the fully-assembled deck set rather than one level.

Two safety bounds, both real and load-bearing, not decorative:

1. `MAX_FILES` (1000): protects against a pathological/malformed include
   tree. Every real family in this corpus resolves far under it (Silverado,
   the largest, has 199 total `.k`/`.key` members).
2. `MAX_FILE_SIZE_BYTES` (20 MB): a file over this is recorded as *found*
   (`SKIPPED_TOO_LARGE`, a new `EdgeStatus` value) but never opened. Found
   a real, subtle correctness bug while implementing this: pre-filtering
   ambiguous/oversized candidates out of the deck set *before* calling
   `build_include_graph()` made that function unable to see them at all —
   it reported `MISSING` (0 matches) instead of the true `AMBIGUOUS`/
   `SKIPPED_TOO_LARGE` state, since its own ambiguity/status logic only
   looks at what's actually in the deck set it's given. Fixed by adding all
   same-named ambiguous candidates to the parsed set (so the real collision
   is visible) and by post-correcting `MISSING` edges that are actually
   oversized-and-skipped rather than genuinely absent.

`scripts/ingest_level3_cae_decks.py` re-runs the same 4 real vehicle/dummy
families with this deeper resolution. `persist_deck()`'s existing
idempotency (return the existing row unchanged for a known `deck_key`)
is correct for a same-parser re-run but wrong here — this is a *deeper*
resolution of the same logical deck — so the script clears the prior
(shallower) `CaeDeck` and its dependent rows first, matching the
regenerate-by-clearing precedent already established for the synthetic
benchmark (`docs/ADR/0008`).

## Consequences

- Structured search coverage over the 4 persisted deck families grows
  substantially — every file reachable within the size bound, not just
  direct includes of the main assembly deck.
- Some real component files remain unparsed (`SKIPPED_TOO_LARGE`) — visible
  in the graph and reported by the ingestion script, not silently absent.
  A future increase to `MAX_FILE_SIZE_BYTES` is a one-line, measured change
  once real memory headroom is confirmed, not a redesign.
- `EdgeStatus` gained `SKIPPED_TOO_LARGE`; `build_include_graph()` itself
  never produces it (it has no concept of file size) — only
  `resolve_recursive()`'s post-processing step does, by construction.
