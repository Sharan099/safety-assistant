"""Identity and tenancy: organizations, users, memberships, workspaces, preferences, audit (ADR-0029 §3)."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from safety_assistant.persistence.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin

# Organization-level roles (TRD). Scopes per role live in api/dependencies/auth.py.
ORG_ROLES = ("engineer", "knowledge_admin", "auditor", "org_admin")
WORKSPACE_ROLES = ("member", "owner")

# The single organization every fresh deployment starts with; the same UUID in the
# migration backfill and in `identity.service.default_organization` (idempotent).
DEFAULT_ORGANIZATION_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")


class Organization(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(Text, unique=True)


class User(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(Text, unique=True)
    display_name: Mapped[str] = mapped_column(Text)
    # ACTIVE | SUSPENDED
    status: Mapped[str] = mapped_column(Text, default="ACTIVE")
    # OIDC subject when provisioned from a token; NULL for CLI/dev-created users.
    external_subject: Mapped[str | None] = mapped_column(Text, unique=True)
    last_login_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class Membership(Base):
    __tablename__ = "memberships"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text)


class Workspace(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    __tablename__ = "workspaces"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_workspace_name_per_org"),)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    archived_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class WorkspaceMembership(Base):
    __tablename__ = "workspace_memberships"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, default="member")


class UserPreference(Base):
    """Explicit, user-visible settings only (06_SECURITY §C). Never regulatory memory."""

    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    default_workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL")
    )
    # concise | standard | detailed
    answer_density: Mapped[str] = mapped_column(Text, default="standard")
    preferred_language: Mapped[str] = mapped_column(Text, default="en")
    # light | dark | system
    ui_theme: Mapped[str] = mapped_column(Text, default="light")
    # The engineer's current project in their own words (vehicle category, mass, markets, programme).
    # Shown to the model as <project_context> data so "my vehicle" resolves; never evidence.
    project_context: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """Privileged / security-relevant actions (06_SECURITY "Audit")."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_org_created", "organization_id", "created_at"),)

    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    actor_subject: Mapped[str | None] = mapped_column(Text)  # api-key / OIDC subject when no user row
    organization_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    action: Mapped[str] = mapped_column(Text, index=True)
    resource_type: Mapped[str] = mapped_column(Text)
    resource_id: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
