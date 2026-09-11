"""Cover-page metadata for UNECE consolidated regulation texts.

Deterministic regexes over the first pages: document symbol, revision, and
the "Incorporating all valid text up to:" amendment list with entry-into-force
dates. Output is cross-checked against the human-reviewed registry; a
disagreement is recorded as an ingestion warning, never silently resolved.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

_SYMBOL_RE = re.compile(r"E/ECE/(?:TRANS/505|324)(?:/Rev\.\d+)?/Add\.\d+(?:/Rev\.\d+)?(?:/Amend\.\d+)?")
_REVISION_RE = re.compile(r"^\s*Revision\s+(\d+)\s*$", re.MULTILINE)
_AMEND_RE = re.compile(
    r"(?P<label>(?:Supplement\s+\d+\s+to\s+the\s+\d+\s+series\s+of\s+amendments"
    r"|\d+\s+series\s+of\s+amendments(?:\s+to\s+the\s+UN\s+Regulation)?"
    r"|Corrigendum\s+\d+\s+to\s+(?:the\s+\d+\s+series\s+of\s+amendments|Revision\s+\d+\s+of\s+the\s+Regulation)))"
    r"(?:\s*[–\-—])+\s*Date\s+of\s+entry\s+into\s+force:?\s*(?P<date>\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE,
)
_DOC_DATE_RE = re.compile(
    r"^\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\s*$",
    re.MULTILINE,
)
_SERIES_RE = re.compile(r"(\d{2})\s+series\s+of\s+amendments", re.IGNORECASE)


def _parse_date(text: str) -> datetime.date | None:
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.datetime.strptime(" ".join(text.split()), fmt).date()
        except ValueError:
            continue
    return None


@dataclass
class Amendment:
    label: str
    entry_into_force: datetime.date | None
    series: str | None


@dataclass
class CoverMetadata:
    document_symbol: str | None = None
    revision: str | None = None
    document_date: datetime.date | None = None
    amendments: list[Amendment] = field(default_factory=list)

    @property
    def latest_entry_into_force(self) -> datetime.date | None:
        dates = [a.entry_into_force for a in self.amendments if a.entry_into_force]
        return max(dates) if dates else None

    @property
    def latest_series(self) -> str | None:
        series = [a.series for a in self.amendments if a.series]
        return max(series) if series else None  # zero-padded two-digit strings sort correctly

    def to_json(self) -> list[dict[str, str | None]]:
        return [
            {
                "label": a.label,
                "entry_into_force": a.entry_into_force.isoformat() if a.entry_into_force else None,
                "series": a.series,
            }
            for a in self.amendments
        ]


def parse_cover(page_texts: list[str], *, max_pages: int = 3) -> CoverMetadata:
    text = "\n".join(page_texts[:max_pages])
    meta = CoverMetadata()
    if m := _SYMBOL_RE.search(text):
        meta.document_symbol = m.group(0)
    if m := _REVISION_RE.search(text):
        meta.revision = f"Rev.{m.group(1)}"
    seen: set[str] = set()
    for m in _AMEND_RE.finditer(text):
        label = " ".join(m.group("label").split())
        if label in seen:
            continue
        seen.add(label)
        s = _SERIES_RE.search(label)
        meta.amendments.append(
            Amendment(label=label, entry_into_force=_parse_date(m.group("date")), series=s.group(1) if s else None)
        )
    dates = [_parse_date(m.group(1)) for m in _DOC_DATE_RE.finditer(text)]
    real = [d for d in dates if d]
    if real:
        meta.document_date = max(real)
    return meta
