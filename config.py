"""
config.py — Passive Safety Regulation Hybrid RAG

Shared configuration for paths, models, retrieval, and prompts.
Models are unchanged from the original system:
- LLM:        Groq llama-3.3-70b-versatile
- Embeddings: sentence-transformers/BAAI/bge-base-en-v1.5
- Reranker:   BAAI/bge-reranker-base (see backend settings)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = OUTPUT_DIR / ".cache"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# OUTPUT FILES (retrieval artifacts)
# ─────────────────────────────────────────────

CHUNKS_FILE = OUTPUT_DIR / "regulation_chunks.json"
EMBEDDINGS_FILE = OUTPUT_DIR / "regulation_embeddings.json"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
INGEST_MANIFEST = OUTPUT_DIR / "ingest_manifest.json"
PAGE_IMAGE_CACHE = OUTPUT_DIR / "page_cache"

MARKDOWN_DIR.mkdir(exist_ok=True)
PAGE_IMAGE_CACHE.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# INGESTION / CHUNKING (Docling + hierarchical)
# ─────────────────────────────────────────────

# Target words per leaf chunk (hierarchical splitter)
HIER_CHUNK_WORDS = int(os.getenv("HIER_CHUNK_WORDS", "180"))
HIER_CHUNK_OVERLAP = int(os.getenv("HIER_CHUNK_OVERLAP", "40"))
HIER_MIN_CHUNK_WORDS = int(os.getenv("HIER_MIN_CHUNK_WORDS", "35"))

# Docling OCR for scanned PDFs
DOCLING_OCR = os.getenv("DOCLING_OCR", "true").lower() == "true"
DOCLING_FORCE_FULL_PAGE_OCR = os.getenv("DOCLING_FORCE_FULL_PAGE_OCR", "true").lower() == "true"

# OCR engine selection: "paddle" (default, low-memory) | "docling" | "pymupdf"
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").lower()
# Paddle implementation: auto | paddle | rapidocr (PP-OCR ONNX on Windows)
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").lower()

# PaddleOCR low-memory settings
OCR_DPI = int(os.getenv("OCR_DPI", "150"))          # lower DPI -> less RAM
OCR_BATCH_PAGES = int(os.getenv("OCR_BATCH_PAGES", "4"))  # pages per batch
OCR_LANG = os.getenv("OCR_LANG", "en")
# Skip OCR for pages that already have an embedded text layer
OCR_SKIP_TEXT_PAGES = os.getenv("OCR_SKIP_TEXT_PAGES", "true").lower() == "true"
# Set OCR_FORCE_ALL=true to OCR every page (pure scans with no/garbage text layer)
OCR_FORCE_ALL = os.getenv("OCR_FORCE_ALL", "false").lower() == "true"
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "200"))

# Legacy flat chunking (PyMuPDF path — kept for reference)
CHUNK_SIZE = 220
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 50

# ─────────────────────────────────────────────
# MODELS (unchanged)
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "2"))
EMBEDDING_DIMENSION = 768  # BAAI/bge-base-en-v1.5

# ─────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────

LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 800

# ─────────────────────────────────────────────
# VECTOR / RETRIEVAL CONFIG
# ─────────────────────────────────────────────

TOP_K_VECTOR = 8
VECTOR_SCORE_THRESHOLD = 0.35
TOP_K_CHUNKS = 6
MAX_CONTEXT_TOKENS = 4000

# ─────────────────────────────────────────────
# REGULATIONS
# ─────────────────────────────────────────────

SUPPORTED_REGULATIONS = [
    "UN_R14",
    "UN_R16",
    "UN_R17",
    "UN_R94",
    "UN_R95",
    "UN_R135",
    "UN_R137",
    "FMVSS",
    "EURO_NCAP",
    "ISO",
    "CAE_REFERENCE",
    "SAFETY_REFERENCE",
]

REGULATION_DESCRIPTIONS = {
    "UN_R14": "Safety belt anchorages",
    "UN_R16": "Safety belts and restraint systems",
    "UN_R17": "Seats, seat strength and head restraints",
    "UN_R94": "Frontal impact occupant protection",
    "UN_R95": "Side impact occupant protection",
    "UN_R135": "Pole side impact occupant protection",
    "UN_R137": "Full width frontal impact occupant protection",
    "FMVSS": "Federal Motor Vehicle Safety Standards",
    "EURO_NCAP": "Consumer crashworthiness assessment",
    "ISO": "International engineering standards",
    "CAE_REFERENCE": "Crashworthiness and CAE engineering references",
    "SAFETY_REFERENCE": "Vehicle safety engineering references",
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert passive safety and homologation assistant.

You answer ONLY using retrieved regulation context.

Never hallucinate regulations.

Always prioritize:
- geometry requirements
- test procedures
- injury criteria
- homologation logic
- compliance requirements
- restraint systems
- load requirements
- crashworthiness engineering

Structure responses clearly.

If information is missing, say:
'Information not found in regulations.'
"""

# ─────────────────────────────────────────────
# OPTIONAL: extraction API key (offline only)
# ─────────────────────────────────────────────

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

VERBOSE = True


if __name__ == "__main__":
    print("Passive Safety Regulation Hybrid RAG")
    print(f"DATA_DIR        : {DATA_DIR}")
    print(f"OUTPUT_DIR      : {OUTPUT_DIR}")
    print(f"GROQ_MODEL      : {GROQ_MODEL}")
    print(f"EMBEDDING_MODEL : {EMBEDDING_MODEL}")
