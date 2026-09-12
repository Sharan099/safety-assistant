"""identity: organizations, users, memberships, workspaces, preferences, audit (ADR-0029 §3)

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-12

Additive. Downgrade drops the seven tables (and any identity data in them).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UUID = postgresql.UUID(as_uuid=True)
_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        'organizations',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('name', sa.Text(), nullable=False, unique=True),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
    )
    op.create_table(
        'users',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('email', sa.Text(), nullable=False, unique=True),
        sa.Column('display_name', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False, server_default='ACTIVE'),
        sa.Column('external_subject', sa.Text(), nullable=True, unique=True),
        sa.Column('last_login_at', _TS, nullable=True),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
    )
    op.create_table(
        'memberships',
        sa.Column('user_id', _UUID, sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('organization_id', _UUID, sa.ForeignKey('organizations.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('role', sa.Text(), nullable=False),
        sa.CheckConstraint("role IN ('engineer','knowledge_admin','auditor','org_admin')", name='ck_membership_role'),
    )
    op.create_table(
        'workspaces',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('organization_id', _UUID, sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('created_by', _UUID, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('archived_at', _TS, nullable=True),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('organization_id', 'name', name='uq_workspace_name_per_org'),
    )
    op.create_index('ix_workspaces_organization_id', 'workspaces', ['organization_id'])
    op.create_table(
        'workspace_memberships',
        sa.Column('workspace_id', _UUID, sa.ForeignKey('workspaces.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('user_id', _UUID, sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('role', sa.Text(), nullable=False, server_default='member'),
        sa.CheckConstraint("role IN ('member','owner')", name='ck_workspace_membership_role'),
    )
    op.create_table(
        'user_preferences',
        sa.Column('user_id', _UUID, sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('default_workspace_id', _UUID, sa.ForeignKey('workspaces.id', ondelete='SET NULL'), nullable=True),
        sa.Column('answer_density', sa.Text(), nullable=False, server_default='standard'),
        sa.Column('preferred_language', sa.Text(), nullable=False, server_default='en'),
        sa.Column('ui_theme', sa.Text(), nullable=False, server_default='light'),
        sa.Column('updated_at', _TS, nullable=False),
    )
    op.create_table(
        'audit_events',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('actor_user_id', _UUID, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('actor_subject', sa.Text(), nullable=True),
        sa.Column('organization_id', _UUID, sa.ForeignKey('organizations.id'), nullable=True),
        sa.Column('action', sa.Text(), nullable=False),
        sa.Column('resource_type', sa.Text(), nullable=False),
        sa.Column('resource_id', sa.Text(), nullable=True),
        sa.Column('request_id', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
    )
    op.create_index('ix_audit_events_action', 'audit_events', ['action'])
    op.create_index('ix_audit_events_org_created', 'audit_events', ['organization_id', 'created_at'])


def downgrade() -> None:
    for t in ('audit_events', 'user_preferences', 'workspace_memberships', 'workspaces', 'memberships', 'users', 'organizations'):
        op.drop_table(t)
