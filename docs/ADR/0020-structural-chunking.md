# ADR-0020 — Structural chunking along the clause tree

Status: accepted · Date: 2026-09-11 · Supersedes: 0006 (fixed-window packing)

## Context
Regulations are hierarchical legal texts. The baseline packed paragraphs into ~400-word windows within regex-detected headings; it could not name the clause a chunk came from, and UNECE clause numbers (on their own line) were mostly not detected.

## Decision
`ingestion/normalize/structure.py` builds a clause tree (materialized paths such as `5.2.1.8`, `annex-3/1.4.3.5.2.1`) with annex scoping, a plausible-successor rule, TOC and footnote guards, definition typing and cross-reference extraction. `ingestion/chunk/structural.py` emits one chunk per clause, merges tiny consecutive siblings under the same parent (keeping each number inline, citation label as an exact range), splits oversized clauses on sentence boundaries with part labels, and emits table chunks that always carry headers. Chunk content starts with a version-independent context header so identical clauses across versions hash identically (embedding reuse). Target 120–500 estimated tokens.

## Alternatives rejected
- Fixed token windows: cannot produce exact citations.
- One chunk per top-level section: too coarse for numeric-threshold questions (a single clause is the answer unit).

## Consequences
Citations are exact and resolvable; the parser is regulation-aware with golden tests on real UNECE page text. Manuals/reports use a generic heading normalizer.

## Evidence
`tests/parser_golden`, `tests/unit/test_chunker.py`; R94: 415 sections, 63 definitions, 127 cross-references (100 resolved).
