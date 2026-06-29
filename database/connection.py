import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Fetch database URLs. If the environment variable DATABASE_URL is set, use it.
# Otherwise, default to SQLite database "safety_registry.db" in the root directory.
SYNC_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///safety_registry.db"
)

# Convert sync schema to async scheme if needed
if SYNC_DATABASE_URL.startswith("sqlite://"):
    ASYNC_DATABASE_URL = SYNC_DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
else:
    ASYNC_DATABASE_URL = SYNC_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Sync DB setup
if SYNC_DATABASE_URL.startswith("sqlite://"):
    engine = create_engine(SYNC_DATABASE_URL)
else:
    engine = create_engine(SYNC_DATABASE_URL, pool_size=10, max_overflow=20)
    
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async DB setup
if ASYNC_DATABASE_URL.startswith("sqlite+aiosqlite://"):
    async_engine = create_async_engine(ASYNC_DATABASE_URL)
else:
    async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_size=10, max_overflow=20)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

def get_db():
    """Dependency for retrieving a synchronous database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    """Dependency for retrieving an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
