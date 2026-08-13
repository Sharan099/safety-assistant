"""LS-DYNA parser — turns `RawCard`s into structured entities for the
`structured: true` keyword roots (`packages/cae/lsdyna/keyword_registry.yaml`),
per TRD_LEVEL3.md §7/§19.

Field extraction is deliberately conservative: `_split_fields` only trusts
comma-delimited or whitespace-delimited lines. A genuinely fixed-width line
where numeric fields abut with *no* separating space (rare, but the classic
old-format failure mode) is **not** guessed apart — a wrong guess would
fabricate an incorrect part/material/section id, which is worse than
honestly reporting `parse_status="PARTIAL"`/`"UNKNOWN"` with the raw card
still fully intact (Instructions §8: "never invent engineering meaning").
"""

from __future__ import annotations

from packages.cae.lsdyna.cards import RawCard
from packages.cae.lsdyna.lexer import tokenize
from packages.cae.lsdyna.models import (
    Contact,
    ControlCard,
    DatabaseOutput,
    GenericKeywordEntry,
    IncludeStatement,
    Material,
    ParsedDeck,
    ParseStatus,
    Part,
    Section,
)
from packages.cae.lsdyna.registry import registry_roots
from packages.cae.lsdyna.registry import root_for as _root_for


def _split_fields(line: str) -> list[str]:
    if "," in line:
        return [f.strip() for f in line.split(",")]
    stripped = line.strip()
    return stripped.split() if stripped else []


def _to_int(token: str | None) -> int | None:
    if token is None:
        return None
    try:
        return int(float(token))  # LS-DYNA sometimes writes IDs as "1.0"
    except ValueError:
        return None


def _to_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _first_nonempty_data_line(card: RawCard) -> str | None:
    for line in card.data_lines():
        if line.strip():
            return line
    return None


def _keyword_suffix(keyword: str, prefix: str) -> str | None:
    if keyword == prefix:
        return None
    if not keyword.startswith(prefix + "_"):
        return None
    suffix = keyword[len(prefix) + 1 :]
    if suffix.endswith("_TITLE"):
        suffix = suffix[: -len("_TITLE")]
    return suffix or None


def _parse_part(card: RawCard) -> Part:
    data_lines = [line for line in card.data_lines() if line.strip()]
    title = data_lines[0].strip() if data_lines else None
    fields = _split_fields(data_lines[1]) if len(data_lines) >= 2 else []
    pid = _to_int(fields[0]) if len(fields) >= 1 else None
    secid = _to_int(fields[1]) if len(fields) >= 2 else None
    mid = _to_int(fields[2]) if len(fields) >= 3 else None
    status: ParseStatus = (
        "PARSED"
        if pid is not None and secid is not None and mid is not None
        else "PARTIAL"
        if pid is not None
        else "UNKNOWN"
    )
    return Part(part_id=pid, title=title, section_id=secid, material_id=mid, parse_status=status, raw=card)


def _parse_section(card: RawCard) -> Section:
    section_type = _keyword_suffix(card.keyword, "SECTION")
    has_title = card.keyword.endswith("_TITLE")
    data_lines = [line for line in card.data_lines() if line.strip()]
    index = 1 if has_title else 0
    fields = _split_fields(data_lines[index]) if len(data_lines) > index else []
    secid = _to_int(fields[0]) if fields else None
    status: ParseStatus = "PARSED" if secid is not None else "PARTIAL" if section_type else "UNKNOWN"
    return Section(section_id=secid, section_type=section_type, parse_status=status, raw=card)


def _parse_material(card: RawCard) -> Material:
    mat_type = _keyword_suffix(card.keyword, "MAT")
    has_title = card.keyword.endswith("_TITLE")
    data_lines = [line for line in card.data_lines() if line.strip()]
    index = 1 if has_title else 0
    fields = _split_fields(data_lines[index]) if len(data_lines) > index else []
    mid = _to_int(fields[0]) if fields else None
    status: ParseStatus = "PARSED" if mid is not None else "PARTIAL" if mat_type else "UNKNOWN"
    return Material(material_id=mid, mat_type=mat_type, parse_status=status, raw=card)


def _parse_contact(card: RawCard) -> Contact:
    contact_type = _keyword_suffix(card.keyword, "CONTACT")
    line = _first_nonempty_data_line(card)
    fields = _split_fields(line) if line else []
    ssid = _to_int(fields[0]) if len(fields) >= 1 else None
    msid = _to_int(fields[1]) if len(fields) >= 2 else None
    status: ParseStatus = (
        "PARSED" if ssid is not None and msid is not None else "PARTIAL" if contact_type else "UNKNOWN"
    )
    return Contact(contact_type=contact_type, ssid=ssid, msid=msid, parse_status=status, raw=card)


def _parse_control(card: RawCard) -> ControlCard:
    control_type = _keyword_suffix(card.keyword, "CONTROL")
    status: ParseStatus = "PARSED" if control_type else "UNKNOWN"
    return ControlCard(control_type=control_type, parse_status=status, raw=card)


def _parse_database(card: RawCard) -> DatabaseOutput:
    database_type = _keyword_suffix(card.keyword, "DATABASE")
    line = _first_nonempty_data_line(card)
    fields = _split_fields(line) if line else []
    dt = _to_float(fields[0]) if fields else None
    status: ParseStatus = "PARSED" if database_type else "UNKNOWN"
    return DatabaseOutput(database_type=database_type, dt=dt, parse_status=status, raw=card)


def _parse_include(card: RawCard) -> IncludeStatement:
    line = _first_nonempty_data_line(card)
    filename = line.strip() if line else None
    status: ParseStatus = "PARSED" if filename else "UNKNOWN"
    return IncludeStatement(filename=filename, parse_status=status, raw=card)


_STRUCTURED_DISPATCH = {
    "PART": lambda card, deck: deck.parts.append(_parse_part(card)),
    "SECTION": lambda card, deck: deck.sections.append(_parse_section(card)),
    "MAT": lambda card, deck: deck.materials.append(_parse_material(card)),
    "CONTACT": lambda card, deck: deck.contacts.append(_parse_contact(card)),
    "CONTROL": lambda card, deck: deck.controls.append(_parse_control(card)),
    "DATABASE": lambda card, deck: deck.databases.append(_parse_database(card)),
    "INCLUDE": lambda card, deck: deck.includes.append(_parse_include(card)),
}


def parse_deck(text: str, source_file: str) -> ParsedDeck:
    cards = tokenize(text, source_file)
    deck = ParsedDeck(source_file=source_file, all_cards=cards)
    roots = registry_roots()

    for card in cards:
        root = _root_for(card.keyword, roots)
        dispatch = _STRUCTURED_DISPATCH.get(root) if root else None
        if dispatch is not None:
            dispatch(card, deck)
        else:
            deck.generic_keywords.append(GenericKeywordEntry(keyword=card.keyword, root=root, raw=card))

    return deck
