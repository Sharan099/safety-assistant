"""Initial migration

Revision ID: 001_initial
Revises: None
Create Date: 2026-06-24 20:30:00

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    # 1. Enable pgvector extension (Postgres only)
    if not is_sqlite:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Create regulations table
    op.create_table(
        'regulations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('regulation_code', sa.String(length=100), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),
        sa.Column('amendment', sa.String(length=100), nullable=True),
        sa.Column('revision', sa.String(length=100), nullable=True),
        sa.Column('supplement', sa.String(length=100), nullable=True),
        sa.Column('corrigendum', sa.String(length=100), nullable=True),
        sa.Column('publication_date', sa.Date(), nullable=True),
        sa.Column('effective_date', sa.Date(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True, server_default='ACTIVE'),
        sa.Column('market', sa.String(length=50), nullable=True),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('checksum', sa.String(length=64), nullable=True),
        sa.Column('local_file_path', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_regulations_regulation_code', 'regulations', ['regulation_code'], unique=False)
    op.create_index('ix_regulations_source_type', 'regulations', ['source_type'], unique=False)
    op.create_index('ix_regulations_status', 'regulations', ['status'], unique=False)
    op.create_index('ix_regulations_amendment', 'regulations', ['amendment'], unique=False)
    op.create_index('ix_regulations_publication_date', 'regulations', ['publication_date'], unique=False)
    op.create_index('ix_regulations_effective_date', 'regulations', ['effective_date'], unique=False)
    op.create_index('ix_regulations_market', 'regulations', ['market'], unique=False)
    op.create_index('ix_regulations_checksum', 'regulations', ['checksum'], unique=False)

    # 3. Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('regulation_id', sa.Integer(), nullable=False),
        sa.Column('document_name', sa.String(length=255), nullable=False),
        sa.Column('document_type', sa.String(length=50), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('hash', sa.String(length=64), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('upload_date', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['regulation_id'], ['regulations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_documents_hash', 'documents', ['hash'], unique=False)

    # 4. Create chunks table
    op.create_table(
        'chunks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('section', sa.String(length=100), nullable=True),
        sa.Column('paragraph', sa.String(length=100), nullable=True),
        sa.Column(
            'embedding',
            sa.JSON() if is_sqlite else Vector(768),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_chunks_page_number', 'chunks', ['page_number'], unique=False)
    op.create_index('ix_chunks_section', 'chunks', ['section'], unique=False)

    if not is_sqlite:
        # 5. Create Full-Text Search (GIN) index on chunk_text
        op.execute(
            "CREATE INDEX ix_chunks_chunk_text_fts ON chunks "
            "USING gin(to_tsvector('english', chunk_text));"
        )
        # 6. Create pgvector HNSW index on embedding
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops);"
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw;")
    op.execute("DROP INDEX IF EXISTS ix_chunks_chunk_text_fts;")
    op.drop_index('ix_chunks_section', table_name='chunks')
    op.drop_index('ix_chunks_page_number', table_name='chunks')
    op.drop_table('chunks')
    
    op.drop_index('ix_documents_hash', table_name='documents')
    op.drop_table('documents')
    
    op.drop_index('ix_regulations_checksum', table_name='regulations')
    op.drop_index('ix_regulations_market', table_name='regulations')
    op.drop_index('ix_regulations_effective_date', table_name='regulations')
    op.drop_index('ix_regulations_publication_date', table_name='regulations')
    op.drop_index('ix_regulations_amendment', table_name='regulations')
    op.drop_index('ix_regulations_status', table_name='regulations')
    op.drop_index('ix_regulations_source_type', table_name='regulations')
    op.drop_index('ix_regulations_regulation_code', table_name='regulations')
    op.drop_table('regulations')
    
    op.execute("DROP EXTENSION IF EXISTS vector;")
