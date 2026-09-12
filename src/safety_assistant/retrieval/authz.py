"""Document-level authorization predicate (ADR-0029 §4) — evaluated before any ranking.

One definition, two evaluators that must agree:
- `Authz.sql()` for the SQL scope statement (dense + exact legs, listings, evidence lookup);
- `Authz.allows()` for the in-memory BM25 pre-filter.
`tests/unit/test_authz_equivalence.py` drives both with the same cases.

No identity (`authz=None` in ScopeFilter) means: authoritative documents only, nothing that is
owned by a user or a workspace. That is the API-key / evaluation-script view.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, and_, false, or_

from safety_assistant.persistence.models import Regulation

SOURCE_SCOPES: tuple[str, ...] = ("AUTHORITATIVE_ORG", "WORKSPACE", "PRIVATE_USER")


@dataclass(frozen=True)
class DocumentRef:
    """The columns the predicate reads, as plain values (for the in-memory evaluator)."""

    document_id: uuid.UUID
    scope: str
    organization_id: uuid.UUID
    workspace_id: uuid.UUID | None
    owner_user_id: uuid.UUID | None
    archived: bool = False


@dataclass(frozen=True)
class Authz:
    """What one principal may see ∧ what they selected. Built server-side, never from client input."""

    user_id: uuid.UUID | None
    organization_ids: tuple[uuid.UUID, ...]
    workspace_ids: tuple[uuid.UUID, ...]
    # Selected subset of SOURCE_SCOPES (FR-CHAT-04). Never widens what membership allows.
    source_scopes: tuple[str, ...] = ("AUTHORITATIVE_ORG",)
    # Optional "ask this document" restriction; validated for visibility by the caller.
    document_ids: tuple[uuid.UUID, ...] = ()

    @classmethod
    def from_principal(
        cls,
        principal: Any,
        *,
        source_scopes: tuple[str, ...] = ("AUTHORITATIVE_ORG",),
        workspace_ids: tuple[uuid.UUID, ...] = (),
        document_ids: tuple[uuid.UUID, ...] = (),
    ) -> Authz:
        """Requested workspaces are intersected with memberships upstream (conversations.service);
        here an unknown workspace simply cannot match any row."""
        member = tuple(principal.workspace_ids)
        selected = tuple(w for w in workspace_ids if w in member) if workspace_ids else member
        return cls(
            user_id=principal.user_id,
            organization_ids=tuple(principal.organization_ids),
            workspace_ids=selected,
            source_scopes=tuple(s for s in source_scopes if s in SOURCE_SCOPES),
            document_ids=tuple(document_ids),
        )

    # ------------------------------------------------------------------ evaluators

    def sql(self) -> ColumnElement[bool]:
        legs: list[ColumnElement[bool]] = []
        if "AUTHORITATIVE_ORG" in self.source_scopes and self.organization_ids:
            legs.append(
                and_(Regulation.scope == "AUTHORITATIVE_ORG", Regulation.organization_id.in_(self.organization_ids))
            )
        if "WORKSPACE" in self.source_scopes and self.workspace_ids:
            legs.append(and_(Regulation.scope == "WORKSPACE", Regulation.workspace_id.in_(self.workspace_ids)))
        if "PRIVATE_USER" in self.source_scopes and self.user_id is not None:
            legs.append(and_(Regulation.scope == "PRIVATE_USER", Regulation.owner_user_id == self.user_id))
        pred: ColumnElement[bool] = or_(*legs) if legs else false()
        if self.document_ids:
            pred = and_(pred, Regulation.id.in_(self.document_ids))
        return and_(pred, Regulation.archived_at.is_(None))

    def allows(self, doc: DocumentRef) -> bool:
        if doc.archived:
            return False
        if self.document_ids and doc.document_id not in self.document_ids:
            return False
        if doc.scope not in self.source_scopes:
            return False
        if doc.scope == "AUTHORITATIVE_ORG":
            return doc.organization_id in self.organization_ids
        if doc.scope == "WORKSPACE":
            return doc.workspace_id is not None and doc.workspace_id in self.workspace_ids
        if doc.scope == "PRIVATE_USER":
            return self.user_id is not None and doc.owner_user_id == self.user_id
        return False


def anonymous_sql() -> ColumnElement[bool]:
    """No identity: authoritative documents of every organization, never user/workspace material."""
    return and_(Regulation.scope == "AUTHORITATIVE_ORG", Regulation.archived_at.is_(None))


def anonymous_allows(doc: DocumentRef) -> bool:
    return doc.scope == "AUTHORITATIVE_ORG" and not doc.archived


def sql_for_principal(principal: Any) -> ColumnElement[bool]:
    """Everything the principal may see (all three scopes) — for document listings and evidence lookup."""
    if getattr(principal, "user_id", None) is None:
        return anonymous_sql()
    return Authz.from_principal(principal, source_scopes=SOURCE_SCOPES).sql()
