import datetime
import json

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import TypeDecorator

from core.embedder import EMBEDDING_DIMENSION

Base = declarative_base()


class SafeVector(TypeDecorator):
    impl = Text
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.loads(value)


class Regulation(Base):
    __tablename__ = "regulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regulation_code = Column(String(100), nullable=False, index=True)
    title = Column(Text, nullable=False)
    source_type = Column(String(50), nullable=False, default="UNECE")
    amendment = Column(String(100))
    status = Column(String(50), default="ACTIVE")
    market = Column(String(50), default="GLOBAL")
    checksum = Column(String(64))
    local_file_path = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="regulation", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regulation_id = Column(Integer, ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False)
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False, default="PDF")
    file_path = Column(Text, nullable=False)
    hash = Column(String(64), nullable=False, index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)

    regulation = relationship("Regulation", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, index=True)
    section = Column(String(100), index=True)
    chunk_type = Column(String(50), default="adaptive")
    embedding = Column(SafeVector(EMBEDDING_DIMENSION))

    document = relationship("Document", back_populates="chunks")
