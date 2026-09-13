"""summary-augmented chunking: document_summaries, chunks.retrieval_text, per-representation embeddings

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-13

Additive. Existing embeddings become representation='content' (the baseline index) and keep
serving; the SAC index ('sac_v1') is populated separately by `safety-assistant reindex`.
The single HNSW index is replaced by one partial HNSW index per representation so a
query against one representation never post-filters the other's neighbours away.
Downgrade drops the SAC rows/columns and restores the original single index.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UUID = postgresql.UUID(as_uuid=True)
_HNSW = "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"


def upgrade() -> None:
    op.add_column('chunks', sa.Column('retrieval_text', sa.Text(), nullable=True))
    op.create_table(
        'document_summaries',
        sa.Column('id', _UUID, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('version_id', _UUID, sa.ForeignKey('regulation_versions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('cache_key', sa.Text(), nullable=False),
        sa.Column('content_sha256', sa.Text(), nullable=False),
        sa.Column('prompt_version', sa.Text(), nullable=False),
        sa.Column('model_name', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('version_id', 'cache_key', name='uq_document_summary_cache_key'),
    )
    op.create_index('ix_document_summaries_version_id', 'document_summaries', ['version_id'])

    op.add_column(
        'chunk_embeddings',
        sa.Column('representation', sa.Text(), nullable=False, server_default='content'),
    )
    op.drop_constraint('uq_embedding_per_chunk_model', 'chunk_embeddings', type_='unique')
    op.create_unique_constraint(
        'uq_embedding_per_chunk_model',
        'chunk_embeddings',
        ['chunk_id', 'model_name', 'model_version', 'representation'],
    )
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw_cosine")
    op.execute(
        f"CREATE INDEX ix_chunk_embeddings_hnsw_content ON chunk_embeddings {_HNSW} "
        "WHERE representation = 'content'"
    )
    op.execute(
        f"CREATE INDEX ix_chunk_embeddings_hnsw_sac ON chunk_embeddings {_HNSW} "
        "WHERE representation = 'sac_v1'"
    )
    op.execute(
        f"CREATE INDEX ix_chunk_embeddings_hnsw_sac_compact ON chunk_embeddings {_HNSW} "
        "WHERE representation = 'sac_v2'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw_sac_compact")
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw_sac")
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw_content")
    op.execute("DELETE FROM chunk_embeddings WHERE representation <> 'content'")
    op.drop_constraint('uq_embedding_per_chunk_model', 'chunk_embeddings', type_='unique')
    op.create_unique_constraint(
        'uq_embedding_per_chunk_model', 'chunk_embeddings', ['chunk_id', 'model_name', 'model_version']
    )
    op.drop_column('chunk_embeddings', 'representation')
    op.execute(f"CREATE INDEX ix_chunk_embeddings_hnsw_cosine ON chunk_embeddings {_HNSW}")
    op.drop_index('ix_document_summaries_version_id', table_name='document_summaries')
    op.drop_table('document_summaries')
    op.drop_column('chunks', 'retrieval_text')
