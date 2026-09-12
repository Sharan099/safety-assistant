"""document scope/ownership on regulations + ingestion_jobs queue (ADR-0029 §3/§5)

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-12

Additive except for one backfilled NOT NULL: every existing regulation becomes
AUTHORITATIVE_ORG in the default organization (created here if absent).
Downgrade drops ingestion_jobs and the five columns — upload scope metadata is lost.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UUID = postgresql.UUID(as_uuid=True)
_TS = sa.DateTime(timezone=True)
DEFAULT_ORG = '00000000-0000-4000-8000-000000000001'


def upgrade() -> None:
    op.execute(
        f"INSERT INTO organizations (id, name, created_at) VALUES ('{DEFAULT_ORG}', 'default', now()) "
        "ON CONFLICT (id) DO NOTHING"
    )
    op.add_column('regulations', sa.Column('scope', sa.Text(), nullable=False, server_default='AUTHORITATIVE_ORG'))
    op.add_column('regulations', sa.Column('organization_id', _UUID, sa.ForeignKey('organizations.id'), nullable=True))
    op.add_column('regulations', sa.Column('workspace_id', _UUID, sa.ForeignKey('workspaces.id'), nullable=True))
    op.add_column('regulations', sa.Column('owner_user_id', _UUID, sa.ForeignKey('users.id'), nullable=True))
    op.add_column('regulations', sa.Column('archived_at', _TS, nullable=True))
    op.execute(f"UPDATE regulations SET organization_id = '{DEFAULT_ORG}' WHERE organization_id IS NULL")
    op.alter_column('regulations', 'organization_id', nullable=False)
    op.create_check_constraint(
        'ck_regulations_workspace_scope', 'regulations', "scope <> 'WORKSPACE' OR workspace_id IS NOT NULL"
    )
    op.create_check_constraint(
        'ck_regulations_private_scope', 'regulations', "scope <> 'PRIVATE_USER' OR owner_user_id IS NOT NULL"
    )
    op.create_index('ix_regulations_org_scope', 'regulations', ['organization_id', 'scope'])
    op.create_index('ix_regulations_workspace_id', 'regulations', ['workspace_id'])
    op.create_index('ix_regulations_owner_user_id', 'regulations', ['owner_user_id'])

    op.create_table(
        'ingestion_jobs',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('version_id', _UUID, sa.ForeignKey('regulation_versions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.Text(), nullable=False, server_default='QUEUED'),
        sa.Column('stage', sa.Text(), nullable=False, server_default='UPLOADED'),
        sa.Column('attempt', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('run_after', _TS, nullable=False),
        sa.Column('locked_at', _TS, nullable=True),
        sa.Column('locked_by', sa.Text(), nullable=True),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('error_public_message', sa.Text(), nullable=True),
        sa.Column('error_internal_ref', _UUID, sa.ForeignKey('ingestion_runs.id'), nullable=True),
        sa.Column('requested_by_user_id', _UUID, sa.ForeignKey('users.id'), nullable=True),
        sa.Column('started_at', _TS, nullable=True),
        sa.Column('completed_at', _TS, nullable=True),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint(
            "status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED','QUARANTINED','CANCELLED')", name='ck_ingestion_job_status'
        ),
    )
    op.create_index('ix_ingestion_jobs_version_id', 'ingestion_jobs', ['version_id'])
    op.create_index('ix_ingestion_jobs_poll', 'ingestion_jobs', ['status', 'run_after'])
    op.create_index(
        'uq_ingestion_jobs_live_version',
        'ingestion_jobs',
        ['version_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('QUEUED','RUNNING')"),
    )


def downgrade() -> None:
    op.drop_table('ingestion_jobs')
    for ix in ('ix_regulations_owner_user_id', 'ix_regulations_workspace_id', 'ix_regulations_org_scope'):
        op.drop_index(ix, table_name='regulations')
    op.drop_constraint('ck_regulations_private_scope', 'regulations', type_='check')
    op.drop_constraint('ck_regulations_workspace_scope', 'regulations', type_='check')
    for col in ('archived_at', 'owner_user_id', 'workspace_id', 'organization_id', 'scope'):
        op.drop_column('regulations', col)
