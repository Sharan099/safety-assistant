"""Alembic migration: chunk metadata + document amendment audit fields."""

from alembic import op
import sqlalchemy as sa

revision = "003_chunk_metadata"
down_revision = "002_ingest_manifest"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.add_column("documents", sa.Column("amendment_from_text", sa.String(length=100), nullable=True))
    op.add_column("documents", sa.Column("amendment_from_filename", sa.String(length=100), nullable=True))
    op.add_column(
        "documents",
        sa.Column("amendment_mismatch", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("documents", sa.Column("content_text_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_documents_content_text_hash", "documents", ["content_text_hash"])

    op.add_column("chunks", sa.Column("chunk_type", sa.String(length=50), nullable=True))
    op.add_column("chunks", sa.Column("heading_path", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("parent_chunk_id", sa.Integer(), nullable=True))
    if not is_sqlite:
        op.create_foreign_key(
            "fk_chunks_parent_chunk_id",
            "chunks",
            "chunks",
            ["parent_chunk_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.drop_index("ix_chunks_content_hash", table_name="chunks")
    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
    if not is_sqlite:
        op.drop_constraint("fk_chunks_parent_chunk_id", "chunks", type_="foreignkey")
    op.drop_column("chunks", "parent_chunk_id")
    op.drop_column("chunks", "content_hash")
    op.drop_column("chunks", "heading_path")
    op.drop_column("chunks", "chunk_type")

    op.drop_index("ix_documents_content_text_hash", table_name="documents")
    op.drop_column("documents", "content_text_hash")
    op.drop_column("documents", "amendment_mismatch")
    op.drop_column("documents", "amendment_from_filename")
    op.drop_column("documents", "amendment_from_text")
