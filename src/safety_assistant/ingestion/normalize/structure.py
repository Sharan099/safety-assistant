"""Structural normalization: pages → section tree with paths, annex scope,
definitions, normative flag and cross-references.

Two strategies, chosen from the registry `kind`:

- ``normalize_regulation``  UNECE consolidated texts. Clause numbers appear on
  their own line ("5.2.1.8.") or lead a line; numbering restarts inside each
  Annex; page headers carry the document symbol, page number and "Annex N".
  A number only opens a clause when it is a *plausible successor* of the
  current position (child, sibling, or ancestor's sibling) — this is what keeps
  numeric values like "1.3" in running text from being mistaken for clauses.
- ``normalize_generic``     manuals/reports: numbered-heading + ALL-CAPS heuristic
  (baseline behaviour, kept).

Everything here is deterministic and unit-tested against real pages
(tests/parser_golden).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from safety_assistant.ingestion.parse.contract import ParsedPage

NORMALIZER_VERSION = "2.0.1"

# "5.2.1.8." or "6.3.1" alone on a line; a bare integer needs its dot ("3.") so page numbers never qualify.
_CLAUSE_ALONE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+|\d+(?=\.))\.?\s*$")
_CLAUSE_LEAD_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(\S.*)$")
# "6.3.1 Material" — dotted number without trailing dot followed by a capitalised title.
_CLAUSE_LEAD_NODOT_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+([A-Z].{1,80})$")
_ANNEX_RE = re.compile(r"^\s*Annex\s+(\d+[A-Z]?)\s*(?:[-–—]\s*(Appendix\s+\d+))?\s*$", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^\s*Appendix\s+(\d+)\s*$", re.IGNORECASE)
_SYMBOL_HEADER_RE = re.compile(r"^\s*E/ECE/(?:TRANS/505|324)\S*\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_DEFINITION_TITLE_RE = re.compile(r"\bdefinitions?\b", re.IGNORECASE)
_QUOTED_TERM_RE = re.compile(r"^\s*[\"“„']([^\"”“]+)[\"”“']\s+(?:means|is|are)\b", re.IGNORECASE)

# Cross references inside clause text.
_XREF_ANNEX_PARA_RE = re.compile(
    r"Annex\s+(?P<annex>\d+[A-Z]?)(?:\s*,\s*(?:Appendix\s+(?P<app>\d+)\s*,\s*)?(?:paragraphs?|paras?\.)\s+(?P<para>\d+(?:\.\d+)*)\.?)?",
    re.IGNORECASE,
)
_XREF_PARA_RE = re.compile(
    r"(?<!Annex\s)(?<!Annex\s\d,\s)(?:paragraphs?|paras?\.)\s+(?P<para>\d+(?:\.\d+)*)\.?(?P<ctx>\s+(?:above|below|of\s+this\s+Regulation|of\s+this\s+annex))?",
    re.IGNORECASE,
)
_XREF_REG_RE = re.compile(r"(?:UN\s+)?Regulation\s+No\.?\s*(?P<num>\d+)", re.IGNORECASE)

_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d*\s*$")
_TOC_MIN_LEADER_LINES = 3

_GENERIC_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z0-9 ,\-/()]{3,80})$")
_GENERIC_CAPS_HEADING = re.compile(r"^([A-Z][A-Z0-9 \-/]{6,80})$")


@dataclass
class NormSection:
    ordinal: int
    path: str
    parent_path: str | None
    section_number: str | None
    annex: str | None
    title: str | None
    kind: str  # CLAUSE | ANNEX | DEFINITION | FRONT_MATTER | HEADING
    depth: int
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def normative(self) -> bool | None:
        text = self.content
        if not text or self.kind in ("FRONT_MATTER", "DEFINITION"):
            return None
        if re.search(r"\bshall\b", text):
            return True
        if re.search(r"\b(should|may|recommended|for information)\b", text, re.IGNORECASE):
            return False
        return None


@dataclass
class CrossRef:
    from_path: str
    raw_text: str
    target_path: str
    target_regulation_key: str | None = None


@dataclass
class NormalizedDocument:
    sections: list[NormSection]
    cross_references: list[CrossRef]
    normalizer_version: str = NORMALIZER_VERSION

    @property
    def parsed_hash(self) -> str:
        """Fingerprint of structure + content (not of page layout noise)."""
        h = hashlib.sha256()
        for s in self.sections:
            h.update(f"{s.path}\x1f{s.kind}\x1f{s.content_sha256}\x1e".encode())
        h.update(NORMALIZER_VERSION.encode())
        return h.hexdigest()

    def by_path(self) -> dict[str, NormSection]:
        return {s.path: s for s in self.sections}


# --------------------------------------------------------------------------- helpers


def _components(number: str) -> list[int]:
    return [int(x) for x in number.split(".")]


def _plausible_successor(current: list[int] | None, candidate: list[int]) -> bool:
    """Accept a candidate clause number only if it continues the sequence."""
    if current is None:
        return (
            candidate in ([1], [0]) or len(candidate) == 1 and candidate[0] <= 3
        )  # first clause: "1." (tolerate 2./3.)
    # child: current + [1]
    if candidate == current + [1]:
        return True
    # sibling or ancestor's sibling: same prefix, last component +1
    for depth in range(len(current), 0, -1):
        prefix = current[:depth]
        if len(candidate) == depth and candidate[:-1] == prefix[:-1] and candidate[-1] == prefix[-1] + 1:
            return True
    return False


def _strip_running_header(lines: list[str], annex_state: dict[str, str | None]) -> list[str]:
    """Remove document-symbol lines, bare page numbers and 'Annex N' markers
    from the top of a page; record the annex the page belongs to."""
    out: list[str] = []
    header_zone = True
    for line in lines:
        s = line.strip()
        if header_zone:
            if not s or _SYMBOL_HEADER_RE.match(s) or _PAGE_NUMBER_RE.match(s):
                continue
            if m := _ANNEX_RE.match(s):
                annex_state["annex"] = f"Annex {m.group(1)}"
                annex_state["appendix"] = m.group(2)
                continue
            if m := _APPENDIX_RE.match(s):
                annex_state["appendix"] = f"Appendix {m.group(1)}"
                continue
            header_zone = False
        out.append(line)
    return out


def _scope_prefix(annex: str | None, appendix: str | None) -> str:
    if annex is None:
        return ""
    p = annex.lower().replace(" ", "-")
    if appendix:
        p += "/" + appendix.lower().replace(" ", "-")
    return p + "/"


# --------------------------------------------------------------------------- regulation


def normalize_regulation(pages: list[ParsedPage]) -> NormalizedDocument:
    sections: list[NormSection] = []
    ordinal = 0

    def new_section(**kw: object) -> NormSection:
        nonlocal ordinal
        s = NormSection(ordinal=ordinal, **kw)  # type: ignore[arg-type]
        ordinal += 1
        sections.append(s)
        return s

    first_page = pages[0].page_number if pages else 1
    current = new_section(
        path="front-matter",
        parent_path=None,
        section_number=None,
        annex=None,
        title="Front matter",
        kind="FRONT_MATTER",
        depth=0,
        page_start=first_page,
        page_end=first_page,
    )
    annex_state: dict[str, str | None] = {"annex": None, "appendix": None}
    scope_key: str | None = None  # which annex/appendix scope numbering belongs to
    current_number: list[int] | None = None
    in_definitions_top: str | None = None  # top-level path whose title mentions definitions
    pending_title_for: NormSection | None = None
    open_paths: dict[int, str] = {}  # depth -> path of the most recent section at that depth (within scope)

    for page in pages:
        raw_lines = page.text.splitlines()
        lines = _strip_running_header(raw_lines, annex_state)
        if sum(1 for ln in lines if _TOC_LINE_RE.search(ln)) >= _TOC_MIN_LEADER_LINES:
            # Table-of-contents page: dot leaders + page numbers. Kept as front
            # matter; its numbers must not open clauses or advance numbering.
            toc = sections[0]
            toc.lines.extend(ln.strip() for ln in lines if ln.strip())
            toc.page_end = page.page_number
            continue
        scope = _scope_prefix(annex_state["annex"], annex_state["appendix"])
        if scope != scope_key:
            # entering a new annex/appendix: numbering restarts, open an ANNEX node
            scope_key = scope
            current_number = None
            open_paths = {}
            in_definitions_top = None
            if scope:
                current = new_section(
                    path=scope.rstrip("/"),
                    parent_path=None,
                    section_number=None,
                    annex=annex_state["annex"],
                    title=(annex_state["annex"] or "")
                    + (f" {annex_state['appendix']}" if annex_state["appendix"] else ""),
                    kind="ANNEX",
                    depth=0,
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
                pending_title_for = current

        stripped = [ln.strip() for ln in lines if ln.strip()]
        for li, s in enumerate(stripped):
            m_alone = _CLAUSE_ALONE_RE.match(s)
            m_lead = None if m_alone else (_CLAUSE_LEAD_RE.match(s) or _CLAUSE_LEAD_NODOT_RE.match(s))
            match = m_alone or m_lead
            number = match.group(1) if match else None
            if number is not None and _plausible_successor(current_number, _components(number)):
                # Footnote markers ("3." alone) can look like a top-level clause. If the
                # next line continues the *current* sequence instead, this is not a clause.
                nxt = _CLAUSE_ALONE_RE.match(stripped[li + 1]) if li + 1 < len(stripped) else None
                if (
                    m_alone
                    and nxt
                    and _plausible_successor(current_number, _components(nxt.group(1)))
                    and not _plausible_successor(_components(number), _components(nxt.group(1)))
                ):
                    current.lines.append(s)
                    continue
                comps = _components(number)
                depth = len(comps)
                parent_path = open_paths.get(depth - 1) if depth > 1 else (scope.rstrip("/") or None)
                path = f"{scope}{number}"
                top_path = open_paths.get(1) if depth > 1 else path
                kind = "CLAUSE"
                if depth > 1 and in_definitions_top and top_path == in_definitions_top:
                    kind = "DEFINITION"
                current = new_section(
                    path=path,
                    parent_path=parent_path,
                    section_number=number + ".",
                    annex=annex_state["annex"],
                    title=None,
                    kind=kind,
                    depth=depth,
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
                current_number = comps
                open_paths = {d: p for d, p in open_paths.items() if d < depth}
                open_paths[depth] = path
                pending_title_for = current if depth == 1 else None
                if m_lead:
                    _consume_text(current, m_lead.group(2), pending_title_for is current)
                    if pending_title_for is current and current.title:
                        pending_title_for = None
                        if _DEFINITION_TITLE_RE.search(current.title):
                            in_definitions_top = path
                continue

            if (
                pending_title_for is current
                and current.title is None
                and len(s) <= 120
                and not s.endswith((".", ";", ":"))
            ):
                current.title = s
                pending_title_for = None
                if current.depth == 1 and _DEFINITION_TITLE_RE.search(s):
                    in_definitions_top = current.path
                continue
            if pending_title_for is None and current.title and not current.lines and s[:1].islower() and len(s) <= 80:
                current.title = (
                    f"{current.title} {s}"  # wrapped title line ("Requirements concerning the ... in the" + "vehicle")
                )
                continue
            pending_title_for = None
            current.lines.append(s)
            current.page_end = page.page_number

    # definition sections: title = the quoted term, or a short un-punctuated first line
    for sec in sections:
        if sec.kind == "DEFINITION" and sec.title is None and sec.lines:
            if m := _QUOTED_TERM_RE.match(sec.content):
                sec.title = m.group(1).strip()
            elif len(sec.lines[0]) <= 60 and not sec.lines[0].endswith((".", ";", ":", ",")):
                sec.title = sec.lines.pop(0)

    sections = [s for s in sections if s.lines or s.kind in ("ANNEX", "CLAUSE", "DEFINITION")]
    return NormalizedDocument(sections=sections, cross_references=_extract_cross_references(sections))


def _consume_text(section: NormSection, text: str, may_be_title: bool) -> None:
    if may_be_title and len(text) <= 120 and not text.endswith((".", ";", ":")):
        section.title = text
    else:
        section.lines.append(text)


def _extract_cross_references(sections: list[NormSection]) -> list[CrossRef]:
    refs: list[CrossRef] = []
    for sec in sections:
        text = sec.content
        if not text:
            continue
        own_scope = _scope_prefix(sec.annex, None)
        seen: set[tuple[str, str]] = set()
        for m in _XREF_ANNEX_PARA_RE.finditer(text):
            target = f"annex-{m.group('annex').lower()}"
            if m.group("app"):
                target += f"/appendix-{m.group('app')}"
            if m.group("para"):
                target += f"/{m.group('para')}"
            key = (m.group(0), target)
            if key not in seen:
                seen.add(key)
                refs.append(CrossRef(from_path=sec.path, raw_text=m.group(0).strip(), target_path=target))
        for m in _XREF_PARA_RE.finditer(text):
            ctx = (m.group("ctx") or "").strip().lower()
            scope = "" if "regulation" in ctx else own_scope
            target = f"{scope}{m.group('para')}"
            key = (m.group(0), target)
            if key not in seen and target != sec.path:
                seen.add(key)
                refs.append(CrossRef(from_path=sec.path, raw_text=m.group(0).strip(), target_path=target))
        for m in _XREF_REG_RE.finditer(text):
            reg = f"UN-R{int(m.group('num'))}"
            key = (m.group(0), reg)
            if key not in seen:
                seen.add(key)
                refs.append(
                    CrossRef(from_path=sec.path, raw_text=m.group(0).strip(), target_path="", target_regulation_key=reg)
                )
    return refs


# --------------------------------------------------------------------------- generic


def normalize_generic(pages: list[ParsedPage]) -> NormalizedDocument:
    sections: list[NormSection] = []
    ordinal = 0
    first_page = pages[0].page_number if pages else 1
    current = NormSection(
        ordinal=0,
        path="front-matter",
        parent_path=None,
        section_number=None,
        annex=None,
        title="Front matter",
        kind="FRONT_MATTER",
        depth=0,
        page_start=first_page,
        page_end=first_page,
    )
    seen_paths: dict[str, int] = {}
    for page in pages:
        for raw in page.text.splitlines():
            s = raw.strip()
            if not s or len(s) > 90:
                if s:
                    current.lines.append(s)
                    current.page_end = page.page_number
                continue
            heading: tuple[str, str | None] | None = None
            if m := _GENERIC_NUMBERED_HEADING.match(s):
                heading = (m.group(2).strip(), m.group(1))
            elif m := _GENERIC_CAPS_HEADING.match(s):
                heading = (m.group(1).strip(), None)
            if heading is None:
                current.lines.append(s)
                current.page_end = page.page_number
                continue
            current.page_end = page.page_number
            sections.append(current)
            ordinal += 1
            title, number = heading
            base = number or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
            n = seen_paths.get(base, 0)
            seen_paths[base] = n + 1
            path = base if n == 0 else f"{base}~{n}"
            depth = len(number.split(".")) if number else 1
            parent_path = None
            if number and depth > 1:
                parent_candidate = ".".join(number.split(".")[:-1])
                parent_path = parent_candidate if parent_candidate in seen_paths else None
            current = NormSection(
                ordinal=ordinal,
                path=path,
                parent_path=parent_path,
                section_number=number,
                annex=None,
                title=title,
                kind="HEADING",
                depth=depth,
                page_start=page.page_number,
                page_end=page.page_number,
            )
    sections.append(current)
    sections = [s for s in sections if s.lines or s.section_number]
    for i, sec in enumerate(sections):
        sec.ordinal = i
    return NormalizedDocument(sections=sections, cross_references=[])


def normalizer_config_hash() -> str:
    return hashlib.sha256(json.dumps({"normalizer": NORMALIZER_VERSION}).encode()).hexdigest()[:16]
