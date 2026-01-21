"""
Configuration for Safety Copilot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"  # Professional data structure
REGULATIONS_DIR = DATA_DIR / "regulations"  # Base regulations folder (loaded at startup)
DOCUMENTS_DIR = BASE_DIR / "documents"  # Legacy support
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
BASE_INDEX_DIR = VECTOR_STORE_DIR / "base_index"  # Base regulations index
USER_INDEX_DIR = VECTOR_STORE_DIR / "user_index"  # User-uploaded documents index
LOGS_DIR = BASE_DIR / "logs"

# Professional data structure subdirectories
UNECE_DIR = DATA_DIR / "unece_regulations"
NHTSA_DIR = DATA_DIR / "nhtsa_guidelines"
FUNCTIONAL_SAFETY_DIR = DATA_DIR / "functional_safety_concepts"
VALIDATION_DIR = DATA_DIR / "validation_testing"
PASSIVE_SAFETY_DIR = DATA_DIR / "passive_safety"
PASSIVE_SAFETY_REGULATIONS_DIR = PASSIVE_SAFETY_DIR / "regulations"
PASSIVE_SAFETY_NCAP_DIR = PASSIVE_SAFETY_DIR / "ncap_protocols"
PASSIVE_SAFETY_FUNDAMENTALS_DIR = PASSIVE_SAFETY_DIR / "fundamentals_training"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
REGULATIONS_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
BASE_INDEX_DIR.mkdir(exist_ok=True)
USER_INDEX_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
UNECE_DIR.mkdir(exist_ok=True)
NHTSA_DIR.mkdir(exist_ok=True)
FUNCTIONAL_SAFETY_DIR.mkdir(exist_ok=True)
VALIDATION_DIR.mkdir(exist_ok=True)
PASSIVE_SAFETY_DIR.mkdir(exist_ok=True)
PASSIVE_SAFETY_REGULATIONS_DIR.mkdir(exist_ok=True)
PASSIVE_SAFETY_NCAP_DIR.mkdir(exist_ok=True)
PASSIVE_SAFETY_FUNDAMENTALS_DIR.mkdir(exist_ok=True)

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

