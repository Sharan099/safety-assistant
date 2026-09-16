"""users.password_hash (self-service email/password sign-up and sign-in) plus per-account lockout
counters (failed_login_attempts, locked_until) so a brute-force run against one account is bounded
without a shared cache. OIDC/CLI-provisioned users keep password_hash NULL — they sign in another way.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-16

Additive, nullable/defaulted. Downgrade drops the columns (any stored hashes are lost).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('password_hash', sa.Text(), nullable=True))
    op.add_column(
        'users', sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default='0')
    )
    op.add_column('users', sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'locked_until')
    op.drop_column('users', 'failed_login_attempts')
    op.drop_column('users', 'password_hash')
