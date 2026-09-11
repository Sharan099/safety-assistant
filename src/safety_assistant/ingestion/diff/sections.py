"""Change-impact analysis between two versions of one regulation.

Deterministic, structure-aware: sections are matched by materialized path
(``5.2.1.8``, ``annex-3/1.3.1``) and compared by ``content_sha256``; changed
clauses get a compact unified text diff. No LLM is involved in deciding *what*
changed — a model may later explain the semantic impact of a given diff.
"""

from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from safety_assistant.persistence.models import Regulation, RegulationVersion, Section

ChangeKind = Literal["ADDED", "REMOVED", "CHANGED"]


@dataclass
class SectionChange:
    path: str
    kind: ChangeKind
    title: str | None
    normative_before: bool | None
    normative_after: bool | None
    page_before: int | None
    page_after: int | None
    diff: str = ""  # unified diff (CHANGED only), bounded

    @property
    def citation_before(self) -> str | None:
        return None if self.kind == "ADDED" else self.path

    @property
    def citation_after(self) -> str | None:
        return None if self.kind == "REMOVED" else self.path


@dataclass
class VersionDiff:
    regulation_key: str
    from_version: str
    to_version: str
    from_version_id: uuid.UUID
    to_version_id: uuid.UUID
    added: list[SectionChange] = field(default_factory=list)
    removed: list[SectionChange] = field(default_factory=list)
    changed: list[SectionChange] = field(default_factory=list)
    unchanged: int = 0

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)

    def summary(self) -> dict[str, object]:
        return {
            "regulation_key": self.regulation_key,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "added": [c.path for c in self.added],
            "removed": [c.path for c in self.removed],
            "changed": [c.path for c in self.changed],
            "unchanged": self.unchanged,
            "normative_changes": [c.path for c in self.changed + self.added if c.normative_after],
        }


MAX_DIFF_LINES = 60


def _sections(session: Session, version_id: uuid.UUID) -> dict[str, Section]:
    rows = session.scalars(select(Section).where(Section.version_id == version_id)).all()
    return {s.path: s for s in rows if s.kind != "FRONT_MATTER"}


def _unified(before: str, after: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), fromfile="before", tofile="after", lineterm="", n=1
        )
    )
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"… ({len(lines) - MAX_DIFF_LINES} more diff lines)"]
    return "\n".join(lines)


def diff_versions(session: Session, from_version: RegulationVersion, to_version: RegulationVersion) -> VersionDiff:
    if from_version.regulation_id != to_version.regulation_id:
        raise ValueError("versions belong to different regulations")
    reg = session.get(Regulation, from_version.regulation_id)
    assert reg is not None
    before, after = _sections(session, from_version.id), _sections(session, to_version.id)
    out = VersionDiff(
        regulation_key=reg.regulation_key,
        from_version=from_version.version_label,
        to_version=to_version.version_label,
        from_version_id=from_version.id,
        to_version_id=to_version.id,
    )
    for path in sorted(set(before) | set(after), key=_path_sort_key):
        b, a = before.get(path), after.get(path)
        if b is None and a is not None:
            out.added.append(SectionChange(path, "ADDED", a.title, None, a.normative, None, a.page_start))
        elif a is None and b is not None:
            out.removed.append(SectionChange(path, "REMOVED", b.title, b.normative, None, b.page_start, None))
        elif a is not None and b is not None:
            if a.content_sha256 == b.content_sha256:
                out.unchanged += 1
            else:
                out.changed.append(
                    SectionChange(
                        path,
                        "CHANGED",
                        a.title or b.title,
                        b.normative,
                        a.normative,
                        b.page_start,
                        a.page_start,
                        diff=_unified(b.content, a.content),
                    )  # fmt: skip
                )
    return out


def _path_sort_key(path: str) -> tuple[int, str, list[int]]:
    scope, _, num = path.rpartition("/")
    nums = [int(x) for x in num.split(".")] if num.replace(".", "").isdigit() else []
    return (1 if scope else 0, scope, nums)


def versions_of(session: Session, regulation_key: str) -> list[RegulationVersion]:
    """All non-quarantined versions of a regulation ordered by validity (oldest first)."""
    reg = session.scalar(select(Regulation).where(Regulation.regulation_key == regulation_key))
    if reg is None:
        return []
    rows = session.scalars(
        select(RegulationVersion).where(
            RegulationVersion.regulation_id == reg.id, RegulationVersion.status.in_(["ACTIVE", "SUPERSEDED"])
        )
    ).all()
    return sorted(rows, key=lambda v: (v.valid_from or v.published_at or v.created_at.date(), v.created_at))


def find_version(session: Session, regulation_key: str, label: str) -> RegulationVersion | None:
    for v in versions_of(session, regulation_key):
        if v.version_label == label or v.version_label.split(" ")[0] == label:
            return v
    return None
