"""Recursive `*INCLUDE` resolution — PASSIVE_SAFETY_LEVEL3_FINAL_FIX.md §8.

Moves from "main deck + direct includes only" (the Level-3 smoke-test-era
bound — `scripts/ingest_level3_cae_decks.py`'s original scope) to full
transitive traversal: parse a deck, find what it includes, parse those too,
repeat until no new files are discovered. Cycle/missing/ambiguous/
outside-root detection is unchanged — `build_include_graph()` already
handles an arbitrary deck set; this module's only job is assembling that
set completely instead of one level deep.

`MAX_FILES` is a safety bound, not a design limit: a real archive's total
reachable file count is finite and known ahead of time from the archive
manifest, so this only ever fires on a genuinely pathological deck (an
absurdly wide or malformed include tree), never on legitimate real corpus
content — the largest real family in this corpus (Silverado, 199 `.k`/`.key`
members total) resolves comfortably under it.

`MAX_FILE_SIZE_BYTES` is a separate, real constraint: this corpus contains
individual component files up to ~140 MB (e.g. Silverado's cabin geometry).
Reading and lexing one of those in full ("never load an entire large PDF
into RAM" applies equally to a 140 MB text deck) risks real memory pressure
on an 8 GB machine holding several such files at once. A file over the
bound is *found* (its edge still reports the real path) but not opened —
`SKIPPED_TOO_LARGE`, not silently missing and not silently parsed anyway.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from packages.cae.lsdyna.include_graph import IncludeGraph, build_include_graph
from packages.cae.lsdyna.models import ParsedDeck
from packages.cae.lsdyna.parser import parse_deck
from packages.ingestion.archives import ArchiveMember, read_member_bytes

MAX_FILES = 1000
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@dataclass
class RecursiveResolution:
    decks: dict[str, ParsedDeck]
    graph: IncludeGraph
    truncated: bool  # True if MAX_FILES was hit before the frontier emptied
    skipped_too_large: dict[str, int]  # relpath -> size_bytes, for files over MAX_FILE_SIZE_BYTES


def _basename(path_str: str) -> str:
    return pathlib.PurePosixPath(path_str.replace("\\", "/")).name


def resolve_recursive(
    archive_path: pathlib.Path,
    main_member: str,
    all_members: list[ArchiveMember],
    *,
    max_files: int = MAX_FILES,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> RecursiveResolution:
    """Parses `main_member` and every file it reaches transitively through
    `*INCLUDE` (and variants like `*INCLUDE_TRANSFORM`), resolving each
    target by basename against `all_members` — same resolution strategy as
    `include_graph.build_include_graph()`, applied breadth-first instead of
    once."""
    by_basename: dict[str, list[ArchiveMember]] = {}
    for m in all_members:
        by_basename.setdefault(_basename(m.path), []).append(m)

    decks: dict[str, ParsedDeck] = {}
    skipped_too_large: dict[str, int] = {}
    frontier = [main_member]
    truncated = False

    while frontier:
        if len(decks) >= max_files:
            truncated = True
            break

        relpath = frontier.pop(0)
        if relpath in decks or relpath in skipped_too_large:
            continue

        member = next((m for m in all_members if m.path == relpath), None)
        if member is not None and member.size_bytes > max_file_size_bytes:
            skipped_too_large[relpath] = member.size_bytes
            continue

        text = read_member_bytes(archive_path, relpath).decode("utf-8", errors="replace")
        deck = parse_deck(text, relpath)
        decks[relpath] = deck

        for include in deck.includes:
            if include.filename is None:
                continue
            candidates = [c for c in by_basename.get(_basename(include.filename), []) if not c.safety_issues]
            # All same-named candidates are queued (not just one), even
            # when there's more than one — build_include_graph() below
            # only reports AMBIGUOUS correctly if every same-named
            # candidate is actually present in `decks` for it to compare;
            # skipping the "losing" candidates here would make the edge
            # look MISSING instead (0 matches in decks) rather than
            # honestly AMBIGUOUS. build_include_graph() still never
            # resolves an ambiguous edge to any one of them.
            for candidate in candidates:
                if candidate.path not in decks and candidate.path not in skipped_too_large:
                    frontier.append(candidate.path)

    graph = build_include_graph(decks)
    if skipped_too_large:
        skipped_basenames = {_basename(p) for p in skipped_too_large}
        for edge in graph.edges:
            # A MISSING edge whose target is actually one of the
            # deliberately-skipped large files is not really missing —
            # correct the record rather than leave a false MISSING.
            if edge.status == "MISSING" and _basename(edge.target_as_written) in skipped_basenames:
                edge.status = "SKIPPED_TOO_LARGE"

    return RecursiveResolution(decks=decks, graph=graph, truncated=truncated, skipped_too_large=skipped_too_large)
