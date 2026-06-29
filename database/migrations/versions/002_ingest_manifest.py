"""Alembic migration: ingest_log + source_manifest tables."""

from alembic import op
import sqlalchemy as sa

revision = "002_ingest_manifest"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingest_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=50), nullable=False),
        sa.Column("item", sa.String(length=512), nullable=False),
        sa.Column("outcome", sa.String(length=50), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingest_log_run_id", "ingest_log", ["run_id"])
    op.create_index("ix_ingest_log_timestamp", "ingest_log", ["timestamp"])

    op.create_table(
        "source_manifest",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("authority", sa.String(length=50), nullable=False),
        sa.Column("regulation_id", sa.String(length=100), nullable=False),
        sa.Column("content_text_hash", sa.String(length=64), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("version", sa.String(length=100), nullable=True),
        sa.Column("etag", sa.String(length=255), nullable=True),
        sa.Column("last_modified", sa.String(length=255), nullable=True),
        sa.Column("fetched_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_url"),
    )
    op.create_index("ix_source_manifest_authority", "source_manifest", ["authority"])
    op.create_index("ix_source_manifest_regulation_id", "source_manifest", ["regulation_id"])
    op.create_index("ix_source_manifest_content_text_hash", "source_manifest", ["content_text_hash"])
    op.create_index("ix_source_manifest_status", "source_manifest", ["status"])


def downgrade() -> None:
    op.drop_table("source_manifest")
    op.drop_table("ingest_log")
