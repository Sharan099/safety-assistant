"""Deterministic query scope parsing: regulation identifiers, clause/annex
references, as-of dates. No LLM involved (CLAUDE.md §2.4)."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

_REG_RE = re.compile(
    r"\b(?:UN\s*|ECE\s*|UNECE\s*)?(?:R\s?|Regulation\s+(?:No\.?\s*)?)(?P<num>\d{1,3})\b(?!\.\d)", re.IGNORECASE
)
_CLAUSE_RE = re.compile(
    r"(?<![\w.])(?:§\s*|para(?:graph)?s?\.?\s+|clause\s+)?(?P<num>\d{1,2}(?:\.\d{1,3}){1,6})\.?(?![\w.]|\s*(?:mm|kn|kg|km|m/s|ms|g|%))",
    re.IGNORECASE,
)
_ANNEX_RE = re.compile(r"\bannex\s+(?P<num>\d{1,2}[A-Z]?)\b", re.IGNORECASE)
_YEAR_RE = re.compile(
    r"\b(?:as\s+of|as\s+at|in\s+force\s+(?:in|on)|effective\s+(?:in|on)|before|prior\s+to|until|in|during|back\s+in)\s+(?P<y>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(?P<d>(?:19|20)\d{2}-\d{2}-\d{2})\b")
_HISTORICAL_HINT_RE = re.compile(
    r"\b(previous|earlier|old(?:er)?|former|superseded|historical|before|prior|used to|originally)\b", re.IGNORECASE
)
_CHANGE_HINT_RE = re.compile(
    r"\b(chang(?:e|ed|es)|amend(?:ed|ment|ments)?|differ(?:s|ence|ences)?|compar(?:e|ed|ison)"
    r"|updated?|new in|what's new|revision)\b",
    re.IGNORECASE,
)
_DEFINITION_HINT_RE = re.compile(r"\b(defin(?:e|ed|ition)|what is (?:a|an|the)|meaning of|means)\b", re.IGNORECASE)


@dataclass
class QueryScope:
    regulation_keys: list[str] = field(default_factory=list)  # ["UN-R94"]
    clause_numbers: list[str] = field(default_factory=list)  # ["5.2.1.8"]
    annexes: list[str] = field(default_factory=list)  # ["3"]
    as_of: datetime.date | None = None
    historical_hint: bool = False
    change_hint: bool = False
    definition_hint: bool = False

    @property
    def has_exact_identifier(self) -> bool:
        return bool(self.clause_numbers or self.annexes)

    @property
    def intent(self) -> str:
        """exact_lookup | definition | historical | change_analysis | comparison | technical_qa"""
        if self.change_hint and (len(self.regulation_keys) > 1):
            return "comparison"
        if self.change_hint:
            return "change_analysis"
        if self.as_of or self.historical_hint:
            return "historical"
        if self.has_exact_identifier:
            return "exact_lookup"
        if self.definition_hint:
            return "definition"
        return "technical_qa"


def parse_query_scope(text: str, *, today: datetime.date | None = None) -> QueryScope:
    scope = QueryScope()
    for m in _REG_RE.finditer(text):
        key = f"UN-R{int(m.group('num'))}"
        if key not in scope.regulation_keys:
            scope.regulation_keys.append(key)
    for m in _CLAUSE_RE.finditer(text):
        num = m.group("num")
        # A bare "2.5" with no paragraph/clause cue and only one dot is more likely a number than a clause.
        if num.count(".") == 1 and not m.group(0).lower().lstrip().startswith(("para", "clause", "§")):
            continue
        if num not in scope.clause_numbers:
            scope.clause_numbers.append(num)
    for m in _ANNEX_RE.finditer(text):
        if m.group("num") not in scope.annexes:
            scope.annexes.append(m.group("num").upper())
    iso = _ISO_DATE_RE.search(text)
    year_match = None if iso else _YEAR_RE.search(text)
    if iso:
        scope.as_of = datetime.date.fromisoformat(iso.group("d"))
    elif year_match:
        year = int(year_match.group("y"))
        cue = year_match.group(0).lower()
        # "before 2021" → the day before that year starts; "in 2019" → end of that year.
        scope.as_of = (
            datetime.date(year - 1, 12, 31)
            if cue.startswith(("before", "prior", "until"))
            else datetime.date(year, 12, 31)
        )
        if today and scope.as_of > today:
            scope.as_of = today
    scope.historical_hint = bool(_HISTORICAL_HINT_RE.search(text))
    scope.change_hint = bool(_CHANGE_HINT_RE.search(text))
    scope.definition_hint = bool(_DEFINITION_HINT_RE.search(text))
    return scope
