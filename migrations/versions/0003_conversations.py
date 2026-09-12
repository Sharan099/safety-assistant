"""conversations, messages, message_citations (ADR-0029 §3)

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-12

Additive. Downgrade drops the three tables (conversation history is lost on downgrade).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UUID = postgresql.UUID(as_uuid=True)
_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('user_id', _UUID, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', _UUID, sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('workspace_id', _UUID, sa.ForeignKey('workspaces.id', ondelete='SET NULL'), nullable=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('title_locked', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('source_scope', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('updated_at', _TS, nullable=False),
        sa.Column('archived_at', _TS, nullable=True),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
    )
    op.create_index('ix_conversations_user_updated', 'conversations', ['user_id', 'updated_at'])
    op.create_table(
        'messages',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('conversation_id', _UUID, sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('answer_mode', sa.Text(), nullable=True),
        sa.Column('model', sa.Text(), nullable=True),
        sa.Column('provider', sa.Text(), nullable=True),
        sa.Column('trace_id', sa.Text(), nullable=True),
        sa.Column('warnings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', _TS, server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("role IN ('user','assistant')", name='ck_message_role'),
        sa.UniqueConstraint('conversation_id', 'ordinal', name='uq_message_ordinal_per_conversation'),
    )
    op.create_table(
        'message_citations',
        sa.Column('id', _UUID, primary_key=True),
        sa.Column('message_id', _UUID, sa.ForeignKey('messages.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_id', _UUID, sa.ForeignKey('chunks.id', ondelete='SET NULL'), nullable=True),
        sa.Column('version_id', _UUID, sa.ForeignKey('regulation_versions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('citation_order', sa.Integer(), nullable=False),
        sa.Column('citation_label', sa.Text(), nullable=False),
        sa.Column('regulation_key', sa.Text(), nullable=False),
        sa.Column('version_label', sa.Text(), nullable=False),
        sa.Column('section_path', sa.Text(), nullable=False),
        sa.Column('page_start', sa.Integer(), nullable=True),
        sa.Column('page_end', sa.Integer(), nullable=True),
        sa.Column('source_sha256', sa.Text(), nullable=False),
        sa.Column('quote_excerpt', sa.Text(), nullable=True),
        sa.Column('retrieval_rank', sa.Integer(), nullable=True),
    )
    op.create_index('ix_message_citations_message_id', 'message_citations', ['message_id'])


def downgrade() -> None:
    for t in ('message_citations', 'messages', 'conversations'):
        op.drop_table(t)
