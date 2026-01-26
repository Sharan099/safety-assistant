"""
Configuration for Safety Copilot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
REGULATIONS_DIR = BASE_DIR / "data" / "regulations"  # Single folder for all regulations and PDFs
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"

# Legacy support (for backward compatibility, but regulations folder is primary)
DATA_DIR = BASE_DIR / "data"  # Legacy - not used for RAG
DOCUMENTS_DIR = BASE_DIR / "documents"  # Legacy - not used for RAG

# Ensure directories exist (with proper error handling for read-only filesystems)
try:
    REGULATIONS_DIR.mkdir(parents=True, exist_ok=True)  # Primary folder for all regulations
except (OSError, PermissionError):
    # In read-only filesystems (like Streamlit Cloud), this is OK - directories will be created when needed
    pass

try:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass

try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic", "openai", or "local"
# Default model - use claude-3-5-sonnet-20241022 or claude-3-sonnet-20240229
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG Configuration (Hierarchical chunking strategy for regulations)
CHUNK_SIZE = 500  # 300-600 tokens range, using 500 as middle
CHUNK_OVERLAP = 50  # Reduced overlap for regulations
TOP_K_DENSE = 10  # Dense retrieval top-K
TOP_K_KEYWORD = 10  # Keyword retrieval top-K
TOP_K_FINAL = 5  # Final results after re-ranking
TOP_K_RETRIEVAL = 8  # Legacy support
SIMILARITY_THRESHOLD = 0.3  # Lowered to catch more relevant results

# Domain Classification
DOMAINS = [
    "Functional Safety",
    "Cybersecurity",
    "ADAS",
    "Driver Monitoring",
    "Software Update",
    "Validation",
    "Passive Safety",
    "General Safety"
]

# Passive Safety Metadata
TEST_TYPES = ["Frontal", "Side", "Pole", "Pedestrian", "Post-Crash"]
METRICS = ["HIC", "Chest_Deflection", "Tibia_Index", "Intrusion"]
DUMMY_TYPES = ["Hybrid-III", "WorldSID", "THOR-M"]

# Confidence Scoring
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.6
CONFIDENCE_LOW = 0.4

# Safety Guardrails
REFUSE_KEYWORDS = [
    "legal interpretation",
    "legal advice",
    "approve",
    "approval",
    "certify",
    "certification",
    "guarantee",
    "warranty",
    "liability"
]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

