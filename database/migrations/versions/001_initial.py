"""Initial simplified schema for UNECE RAG v2."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if not is_sqlite:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "regulations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("regulation_code", sa.String(length=100), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("amendment", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("market", sa.String(length=50), nullable=True),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("local_file_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_regulations_regulation_code", "regulations", ["regulation_code"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("regulation_id", sa.Integer(), nullable=False),
        sa.Column("document_name", sa.String(length=255), nullable=False),
        sa.Column("document_type", sa.String(length=50), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column("upload_date", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["regulation_id"], ["regulations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_hash", "documents", ["hash"])

    embedding_type = sa.Text() if is_sqlite else Vector(384)
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(length=100), nullable=True),
        sa.Column("chunk_type", sa.String(length=50), nullable=True),
        sa.Column("embedding", embedding_type, nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunks_page_number", "chunks", ["page_number"])
    op.create_index("ix_chunks_section", "chunks", ["section"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("regulations")
