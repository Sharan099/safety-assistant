import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App General Settings
    APP_NAME: str = "Automotive Safety Regulation Registry"
    API_PREFIX: str = "/api/v1"
    
    # Paths
    ROOT_DIR: Path = Path(__file__).resolve().parents[1]
    UPLOAD_DIR: Path = ROOT_DIR / "data" / "uploads"
    DOWNLOAD_DIR: Path = ROOT_DIR / "data" / "downloads"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgrespassword@localhost:5432/safety_registry"
    )
    
    # Models (pinned to live safety_registry.db — see docs/EMBEDDING_EVIDENCE.md)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # Structure-aware chunking — default OFF until full-corpus re-chunk is approved.
    STRUCTURE_CHUNKING: bool = os.getenv("STRUCTURE_CHUNKING", "false").lower() == "true"
    
    # Redis & Task queue
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_URL: str = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")
    
    # Observability
    ENABLE_OTEL_TRACING: bool = os.getenv("ENABLE_OTEL_TRACING", "false").lower() == "true"
    ENABLE_PROMETHEUS: bool = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"

    # Session auth (Phase A0)
    SESSION_TTL_DAYS: int = int(os.getenv("SESSION_TTL_DAYS", "7"))
    SESSION_COOKIE_SECURE: bool = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    SESSION_COOKIE_SAMESITE: str = os.getenv("SESSION_COOKIE_SAMESITE", "lax")

    # Confidential user uploads (Phase A)
    CONFIDENTIAL_UPLOAD_ROOT: Path = ROOT_DIR / "storage" / "confidential" / "uploads"

    class Config:
        env_file = ".env"
        extra = "allow"

# Instantiate configurations
settings = Settings()

# Ensure target directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.DOWNLOAD_DIR, exist_ok=True)
os.makedirs(settings.CONFIDENTIAL_UPLOAD_ROOT, exist_ok=True)
