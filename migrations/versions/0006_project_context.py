"""user_preferences.project_context (the engineer's stated project, shown to the model as data) and
messages.abstain_reason (why an assistant turn abstained, e.g. small_talk, so the UI can label it)

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-14

Additive, nullable. Downgrade drops the column (the text is lost).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0006'
down_revision: Union[str, Sequence[str], None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user_preferences', sa.Column('project_context', sa.Text(), nullable=True))
    op.add_column('messages', sa.Column('abstain_reason', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('messages', 'abstain_reason')
    op.drop_column('user_preferences', 'project_context')
