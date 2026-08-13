"""Parsed LS-DYNA entities — TRD_LEVEL3.md §19's `cae_*` schema, minus
`cae_nodes`/`cae_elements` (deliberately absent from that schema; bulk
nodal/element geometry belongs in Parquet if ever needed, not relational
rows — see `packages/cae/lsdyna/keyword_registry.yaml`).

Every entity keeps its source `RawCard` — `parse_status` says how much of
the entity's fields could be confidently extracted, but the raw card is
always there regardless, so nothing is ever silently lost even on a
`PARTIAL` or `UNKNOWN` parse (Instructions §8: "never invent engineering
meaning").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from packages.cae.lsdyna.cards import RawCard

ParseStatus = Literal["PARSED", "PARTIAL", "UNKNOWN"]


@dataclass
class Part:
    part_id: int | None
    title: str | None
    section_id: int | None
    material_id: int | None
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class Section:
    section_id: int | None
    section_type: str | None  # keyword suffix, e.g. "SHELL", "SOLID", "BEAM"
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class Material:
    material_id: int | None
    mat_type: str | None  # keyword suffix, e.g. "024", "ELASTIC"
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class Contact:
    contact_type: str | None  # keyword suffix, e.g. "AUTOMATIC_SURFACE_TO_SURFACE"
    ssid: int | None  # slave/secondary set id — best-effort, see parser.py
    msid: int | None  # master/primary set id — best-effort
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class ControlCard:
    control_type: str | None  # keyword suffix, e.g. "TERMINATION", "TIMESTEP"
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class DatabaseOutput:
    database_type: str | None  # keyword suffix, e.g. "BINARY_D3PLOT", "NODOUT"
    dt: float | None  # output time interval — best-effort, first numeric field
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class IncludeStatement:
    filename: str | None  # exactly as written in the deck, not yet resolved
    parse_status: ParseStatus
    raw: RawCard


@dataclass
class GenericKeywordEntry:
    """Everything with no dedicated table (Instructions §8: still detected,
    still raw-preserved, never given invented structure)."""

    keyword: str
    root: str | None  # matched registry root, or None if genuinely unrecognized
    raw: RawCard


@dataclass
class ParsedDeck:
    source_file: str
    parts: list[Part] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)
    controls: list[ControlCard] = field(default_factory=list)
    databases: list[DatabaseOutput] = field(default_factory=list)
    includes: list[IncludeStatement] = field(default_factory=list)
    generic_keywords: list[GenericKeywordEntry] = field(default_factory=list)
    all_cards: list[RawCard] = field(default_factory=list)

    def unresolved_unknown_roots(self) -> set[str]:
        """Keyword roots this parser has never heard of at all — distinct
        from "known root, generic handling" (e.g. NODE)."""
        return {e.keyword for e in self.generic_keywords if e.root is None}
