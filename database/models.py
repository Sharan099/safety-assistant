from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey, Text, Index, func, Boolean, Float
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.types import TypeDecorator, TEXT
import json
import datetime

from registry.embedding_config import EMBEDDING_DIMENSION

Base = declarative_base()

class SafeVector(TypeDecorator):
    impl = TEXT
    cache_ok = True
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        else:
            return dialect.type_descriptor(TEXT())
            
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == 'postgresql':
            return value
        return json.dumps(value)
        
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == 'postgresql':
            return value
        return json.loads(value)

class Regulation(Base):
    __tablename__ = 'regulations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    regulation_code = Column(String(100), nullable=False, index=True)  # e.g., R95, FMVSS 214
    title = Column(Text, nullable=False)
    source_type = Column(String(50), nullable=False, index=True)  # UNECE, Euro NCAP, FMVSS, NHTSA, IIHS, INTERNAL
    amendment = Column(String(100), index=True)  # e.g., 05 Series
    revision = Column(String(100))
    supplement = Column(String(100))
    corrigendum = Column(String(100))
    publication_date = Column(Date, index=True)
    effective_date = Column(Date, index=True)
    status = Column(String(50), default="ACTIVE", index=True)  # ACTIVE, SUPERSEDED, DRAFT
    market = Column(String(50), index=True)  # EU, US, CN, GLOBAL
    source_url = Column(Text)
    checksum = Column(String(64), index=True)  # SHA-256 hash of the source document
    local_file_path = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="regulation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Regulation(code={self.regulation_code}, amendment={self.amendment}, status={self.status})>"


class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    regulation_id = Column(Integer, ForeignKey('regulations.id', ondelete='CASCADE'), nullable=False)
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)  # PDF, HTML, etc.
    source_url = Column(Text)
    file_path = Column(Text, nullable=False)
    hash = Column(String(64), nullable=False, index=True)  # SHA-256 of document content
    page_count = Column(Integer)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)

    # Graph Lineage Fields
    document_role = Column(String(50), index=True)  # BASE_REGULATION, AMENDMENT, SUPPLEMENT, CORRIGENDUM, TECHNICAL_BULLETIN
    is_complete_regulation = Column(Boolean, default=True, nullable=False)

    parent_document_id = Column(Integer, ForeignKey('documents.id', ondelete='SET NULL'))
    supersedes_document_id = Column(Integer, ForeignKey('documents.id', ondelete='SET NULL'))
    applies_to_document_id = Column(Integer, ForeignKey('documents.id', ondelete='SET NULL'))

    revision_number = Column(Integer)
    series_number = Column(Integer)
    supplement_number = Column(Integer)
    corrigendum_number = Column(Integer)

    # Amendment audit (FR-16)
    amendment_from_text = Column(String(100))
    amendment_from_filename = Column(String(100))
    amendment_mismatch = Column(Boolean, default=False, nullable=False)
    content_text_hash = Column(String(64), index=True)

    # Relationships
    regulation = relationship("Regulation", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    # Self-referential graph mappings
    parent = relationship("Document", remote_side=[id], foreign_keys=[parent_document_id], backref="children")
    superseded_doc = relationship("Document", remote_side=[id], foreign_keys=[supersedes_document_id], backref="superseded_by")
    applied_doc = relationship("Document", remote_side=[id], foreign_keys=[applies_to_document_id], backref="applied_by")

    def __repr__(self):
        return f"<Document(name={self.document_name}, role={self.document_role}, pages={self.page_count})>"


class KnowledgeSection(Base):
    __tablename__ = 'knowledge_sections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    regulation_id = Column(Integer, ForeignKey('regulations.id', ondelete='CASCADE'), nullable=False)
    section_number = Column(String(100), nullable=False)
    section_title = Column(String(255))
    text_content = Column(Text, nullable=False)
    effective_document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    provenance_chain = Column(Text)

    regulation = relationship("Regulation")
    effective_document = relationship("Document")

    def __repr__(self):
        return f"<KnowledgeSection(reg={self.regulation.regulation_code}, sec={self.section_number})>"


class IngestLog(Base):
    __tablename__ = 'ingest_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    stage = Column(String(50), nullable=False)
    item = Column(String(512), nullable=False)
    outcome = Column(String(50), nullable=False)
    reason = Column(Text)
    duration_ms = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class SourceManifest(Base):
    __tablename__ = 'source_manifest'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(Text, nullable=False, unique=True)
    authority = Column(String(50), nullable=False, index=True)
    regulation_id = Column(String(100), nullable=False, index=True)
    content_text_hash = Column(String(64), nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    status = Column(String(50), default="active", index=True)
    version = Column(String(100))
    etag = Column(String(255))
    last_modified = Column(String(255))
    fetched_at = Column(DateTime)


class Chunk(Base):
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    knowledge_section_id = Column(Integer, ForeignKey('knowledge_sections.id', ondelete='SET NULL'))
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, index=True)
    section = Column(String(100), index=True)  # e.g. 5.2.1
    paragraph = Column(String(100))
    chunk_type = Column(String(50), index=True)  # section / clause / paragraph / table
    heading_path = Column(Text)
    content_hash = Column(String(64), index=True)
    parent_chunk_id = Column(Integer, ForeignKey('chunks.id', ondelete='SET NULL'))
    provenance = Column(Text)  # JSON lineage metadata
    embedding = Column(SafeVector(EMBEDDING_DIMENSION))

    document = relationship("Document", back_populates="chunks")
    knowledge_section = relationship("KnowledgeSection")
    parent_chunk = relationship(
        "Chunk",
        remote_side=[id],
        foreign_keys=[parent_chunk_id],
        backref="child_chunks",
    )

    def __repr__(self):
        return f"<Chunk(doc_id={self.document_id}, index={self.chunk_index}, page={self.page_number})>"


from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import Function

@compiles(Function, "sqlite")
def compile_function_sqlite(element, compiler, **kw):
    if element.name.lower() == 'to_tsvector':
        args = list(element.clauses)
        if len(args) > 1:
            return compiler.process(args[1], **kw)
        return ""
    return compiler.visit_function(element, **kw)

# Add GIN index for full-text search
# We use coalesce to ensure that if chunk_text is null (though it's nullable=False), it handles it safely
Index(
    'ix_chunks_chunk_text_fts',
    func.to_tsvector('english', Chunk.chunk_text),
    postgresql_using='gin'
)

# Add HNSW index for pgvector cosine distance search
Index(
    'ix_chunks_embedding_hnsw',
    Chunk.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_cosine_ops'}
)


class User(Base):
    __tablename__ = 'users'

    user_id = Column(String(100), primary_key=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username={self.username})>"


class AuthSession(Base):
    __tablename__ = 'auth_sessions'

    id = Column(String(64), primary_key=True)
    user_id = Column(String(100), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User")

    def __repr__(self):
        return f"<AuthSession(user_id={self.user_id}, expires_at={self.expires_at})>"


class Test(Base):
    __tablename__ = 'tests'

    id = Column(Integer, primary_key=True, autoincrement=True)
    test_id = Column(String(100), nullable=False, unique=True, index=True)
    program = Column(String(100), nullable=False, index=True)
    date = Column(Date, nullable=False)
    test_type = Column(String(50), nullable=False)
    impact_mode = Column(String(50), nullable=False)
    dummy = Column(String(100))
    setup_revision = Column(String(50))
    signed_off_by = Column(String(100))
    confidential_tier = Column(Boolean, default=True, nullable=False)
    owner_user_id = Column(String(100), ForeignKey('users.user_id', ondelete='SET NULL'), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    results = relationship("TestResult", back_populates="test", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Test(id={self.test_id}, program={self.program}, type={self.test_type})>"


class TestResult(Base):
    __tablename__ = 'test_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    test_id = Column(String(100), ForeignKey('tests.test_id', ondelete='CASCADE'), nullable=False)
    channel = Column(String(100))
    filter_class = Column(String(50))
    peak_value = Column(Float)
    injury_criterion = Column(String(50))
    value = Column(Float)
    pass_fail = Column(String(10))
    linked_regulation_clause = Column(String(255), index=True)  # e.g. UN_R94#5.3

    test = relationship("Test", back_populates="results")

    def __repr__(self):
        return f"<TestResult(test_id={self.test_id}, criterion={self.injury_criterion}, value={self.value}, result={self.pass_fail})>"


class TestAuditLog(Base):
    __tablename__ = 'test_audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    resource = Column(String(255), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    model_used = Column(String(100))
    details = Column(Text)

    def __repr__(self):
        return f"<TestAuditLog(user={self.user_id}, resource={self.resource}, action={self.action})>"


class UserUpload(Base):
    __tablename__ = 'user_uploads'

    id = Column(String(36), primary_key=True)
    user_id = Column(String(100), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    upload_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    confidential_tier = Column(Boolean, default=True, nullable=False)
    test_id = Column(String(100), index=True)
    linked_regulation_clause = Column(String(255))
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<UserUpload(id={self.id}, user_id={self.user_id}, status={self.status})>"
