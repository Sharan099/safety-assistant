"""LS-DYNA `*INCLUDE` graph — TRD_LEVEL3.md §9, PRD_LEVEL3.md §12.

Operates on an already-parsed set of decks (`{source_file: ParsedDeck}`,
typically every `.k`/`.key`/`.inc` member of one archive or folder) rather
than touching the filesystem/archive itself — keeps this module pure and
easy to test; the smoke test wires it to real archive members via
`packages/ingestion/archives.py`.

Resolution is by basename match against the known deck set (the real NHTSA
corpus references includes by bare filename — e.g. `main.key` includes
`vehicle.k` — even when the files sit in different subdirectories inside
the same archive), not by literal relative-path resolution. A target that
resolves to more than one same-named file is ambiguous, not a silent guess:
it's reported, never picked arbitrarily.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Literal

from packages.cae.lsdyna.models import ParsedDeck

EdgeStatus = Literal["RESOLVED", "MISSING", "CYCLE", "DUPLICATE", "AMBIGUOUS", "OUTSIDE_ROOT"]


@dataclass
class IncludeEdge:
    parent: str
    target_as_written: str
    resolved_target: str | None  # a key into the deck set, or None
    status: EdgeStatus


@dataclass
class IncludeGraph:
    edges: list[IncludeEdge] = field(default_factory=list)
    deck_status: dict[str, str] = field(default_factory=dict)  # source_file -> COMPLETE | INCOMPLETE | UNRESOLVED


def _basename(path_str: str) -> str:
    return pathlib.PurePosixPath(path_str.replace("\\", "/")).name


def _escapes_root(path_str: str) -> bool:
    """Only an absolute path is flagged — real NHTSA decks routinely
    `*INCLUDE ../../00_INCLUDES/...` to reach a sibling folder inside the
    *same* archive/root, discovered by smoke-testing this against the real
    corpus (a genuine 39-include deck): every one of those legitimate
    references was being wrongly refused before this fix. That's ordinary,
    safe relative navigation, because resolution here is by basename match
    against an already-enumerated known-safe deck set (`build_include_graph`
    never reads the literal path off disk) — unlike
    `packages/ingestion/archives.py`'s `_normalize_member_path`, which *does*
    use the literal path to read/write and so still refuses `..` there. An
    absolute path is still flagged: it's a genuinely unusual reference worth
    surfacing to an engineer even though it's never followed for I/O here
    either."""
    p = path_str.replace("\\", "/")
    return p.startswith("/") or (len(p) > 1 and p[1] == ":")


def build_include_graph(decks: dict[str, ParsedDeck]) -> IncludeGraph:
    by_basename: dict[str, list[str]] = {}
    for key in decks:
        by_basename.setdefault(_basename(key), []).append(key)

    graph = IncludeGraph()
    seen_targets_by_parent: dict[str, set[str]] = {}

    for parent_key, deck in decks.items():
        seen = seen_targets_by_parent.setdefault(parent_key, set())
        for include in deck.includes:
            if include.filename is None:
                continue
            if _escapes_root(include.filename):
                seen.add(_basename(include.filename))
                graph.edges.append(
                    IncludeEdge(
                        parent=parent_key,
                        target_as_written=include.filename,
                        resolved_target=None,
                        status="OUTSIDE_ROOT",
                    )
                )
                continue

            target_basename = _basename(include.filename)
            matches = by_basename.get(target_basename, [])

            if target_basename in seen:
                status: EdgeStatus = "DUPLICATE"
                resolved = matches[0] if len(matches) == 1 else None
            elif len(matches) == 0:
                status = "MISSING"
                resolved = None
            elif len(matches) > 1:
                status = "AMBIGUOUS"
                resolved = None
            else:
                status = "RESOLVED"
                resolved = matches[0]

            seen.add(target_basename)
            graph.edges.append(
                IncludeEdge(
                    parent=parent_key, target_as_written=include.filename, resolved_target=resolved, status=status
                )
            )

    _mark_cycles(graph, decks)
    _mark_deck_status(graph, decks)
    return graph


def _mark_cycles(graph: IncludeGraph, decks: dict[str, ParsedDeck]) -> None:
    adjacency: dict[str, list[int]] = {}
    for i, edge in enumerate(graph.edges):
        if edge.resolved_target is not None:
            adjacency.setdefault(edge.parent, []).append(i)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(decks, WHITE)

    def dfs(node: str, path: list[int]) -> None:
        color[node] = GRAY
        for edge_index in adjacency.get(node, []):
            edge = graph.edges[edge_index]
            target = edge.resolved_target
            if target is None:
                continue
            if color.get(target) == GRAY:
                edge.status = "CYCLE"
                continue
            if color.get(target) == WHITE:
                dfs(target, [*path, edge_index])
        color[node] = BLACK

    for start in decks:
        if color[start] == WHITE:
            dfs(start, [])


def _mark_deck_status(graph: IncludeGraph, decks: dict[str, ParsedDeck]) -> None:
    """A deck is COMPLETE if every include it issues resolved cleanly;
    INCOMPLETE if at least one didn't (missing/ambiguous/cycle); decks with
    no includes at all are trivially COMPLETE (PRD_LEVEL3.md §12: "deck
    complete / deck incomplete / deck unresolved")."""
    edges_by_parent: dict[str, list[IncludeEdge]] = {}
    for edge in graph.edges:
        edges_by_parent.setdefault(edge.parent, []).append(edge)

    for key in decks:
        edges = edges_by_parent.get(key, [])
        if not edges:
            graph.deck_status[key] = "COMPLETE"
        elif all(e.status == "RESOLVED" for e in edges):
            graph.deck_status[key] = "COMPLETE"
        else:
            graph.deck_status[key] = "INCOMPLETE"
