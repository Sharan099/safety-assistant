"""
config.py — FINAL Optimized Passive Safety Regulation GraphRAG

Production-ready configuration for:
- UN Regulations
- FMVSS
- Euro NCAP
- ISO Standards
- CAE References
- Safety Engineering Manuals

Designed for:
- Passive Safety
- Occupant Protection
- Crashworthiness
- Homologation Engineering
- Regulation GraphRAG
- Neo4j Knowledge Graphs
- Groq Llama Extraction

Author:
Sharan — Passive Safety GraphRAG
"""

import os

from pathlib import Path

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
# OUTPUT FILES
# ─────────────────────────────────────────────

CHUNKS_FILE = (
    OUTPUT_DIR / "regulation_chunks.json"
)

ENTITIES_FILE = (
    OUTPUT_DIR / "regulation_kg.json"
)

EMBEDDINGS_FILE = (
    OUTPUT_DIR / "regulation_embeddings.json"
)

GRAPH_FILE = (
    OUTPUT_DIR / "regulation_graph.json"
)

COMMUNITY_FILE = (
    OUTPUT_DIR / "regulation_communities.json"
)

PROGRESS_FILE = (
    OUTPUT_DIR / "extract_progress.json"
)

# ─────────────────────────────────────────────
# DEVELOPMENT MODE
# ─────────────────────────────────────────────

DEV_MODE = True

DEV_MAX_CHUNKS = 50

# limit huge regulations during development
MAX_FMVSS_PAGES = 250

# ─────────────────────────────────────────────
# CHUNKING CONFIG
# ─────────────────────────────────────────────

# optimized for Groq free tier
CHUNK_SIZE = 220

CHUNK_OVERLAP = 50

MIN_CHUNK_LEN = 50

# ─────────────────────────────────────────────
# EXTRACTION CONFIG
# ─────────────────────────────────────────────

# FAST development model
EXTRACTION_MODEL = (
    "zai-glm-4.7"
)

GROQ_MODEL = "llama-3.3-70b-versatile"

# HIGH QUALITY final extraction
FINAL_EXTRACTION_MODEL = (
    "zai-glm-4.7"
)

MAX_EXTRACTION_RETRIES = 4

LLM_TEMPERATURE = 0

# avoid Groq TPM limits
REQUEST_DELAY_SECONDS = 4

MAX_TOKENS = 4096

# ─────────────────────────────────────────────
# EMBEDDING CONFIG
# ─────────────────────────────────────────────

# excellent technical retrieval model
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_BATCH = 16

EMBEDDING_DIMENSION = 1024



MAX_ENTITY_PER_CHUNK = 15

MAX_RELATIONSHIPS_PER_CHUNK = 12

EXTRACTION_TEMPERATURE = 0

MAX_EXTRACTION_TOKENS = 1200

# ─────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────

TOP_K_VECTOR = 8

VECTOR_SCORE_THRESHOLD = 0.45

# ─────────────────────────────────────────────
# GRAPH CONFIG
# ─────────────────────────────────────────────

MAX_GRAPH_NODES = 60

MAX_GRAPH_RELATIONSHIPS = 80

GRAPH_HOPS = 2

# ─────────────────────────────────────────────
# COMMUNITY DETECTION
# ─────────────────────────────────────────────

COMMUNITY_RESOLUTION = 1.0

COMMUNITY_MIN_SIZE = 3

# ─────────────────────────────────────────────
# RETRIEVAL CONFIG
# ─────────────────────────────────────────────

TOP_K_CHUNKS = 6

TOP_K_RELATIONSHIPS = 10

TOP_K_COMMUNITIES = 3

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

    "SAFETY_REFERENCE"
]

# ─────────────────────────────────────────────
# REGULATION DESCRIPTIONS
# ─────────────────────────────────────────────

REGULATION_DESCRIPTIONS = {

    "UN_R14":
        "Safety belt anchorages",

    "UN_R16":
        "Safety belts and restraint systems",

    "UN_R17":
        "Seats, seat strength and head restraints",

    "UN_R94":
        "Frontal impact occupant protection",

    "UN_R95":
        "Side impact occupant protection",

    "UN_R135":
        "Pole side impact occupant protection",

    "UN_R137":
        "Full width frontal impact occupant protection",

    "FMVSS":
        "Federal Motor Vehicle Safety Standards",

    "EURO_NCAP":
        "Consumer crashworthiness assessment",

    "ISO":
        "International engineering standards",

    "CAE_REFERENCE":
        "Crashworthiness and CAE engineering references",

    "SAFETY_REFERENCE":
        "Vehicle safety engineering references"
}

# ─────────────────────────────────────────────
# IMPORTANT SECTION KEYWORDS
# ─────────────────────────────────────────────

IMPORTANT_SECTION_KEYWORDS = [

    "requirements",

    "test procedure",

    "dynamic test",

    "static test",

    "geometry",

    "load",

    "injury",

    "criteria",

    "anchorage",

    "belt",

    "compliance",

    "approval",

    "impact",

    "restraint",

    "occupant"
]

# ─────────────────────────────────────────────
# HIGH VALUE ENTITIES
# ─────────────────────────────────────────────

HIGH_VALUE_ENTITY_TYPES = [

    "Requirement",

    "ComplianceRule",

    "ApprovalRequirement",

    "GeometryConstraint",

    "AngleRequirement",

    "DistanceRequirement",

    "TestProcedure",

    "StaticTest",

    "DynamicTest",

    "TestLoad",

    "VehicleCategory",

    "BeltAnchorage",

    "SafetyBelt",

    "InjuryCriterion",

    "Measurement",

    "FailureMode"
]

# ─────────────────────────────────────────────
# ENTITY PRIORITY
# ─────────────────────────────────────────────

NODE_PRIORITY = {

    "Requirement": 1.00,

    "ComplianceRule": 0.99,

    "ApprovalRequirement": 0.98,

    "GeometryConstraint": 0.97,

    "AngleRequirement": 0.96,

    "DistanceRequirement": 0.95,

    "TestProcedure": 0.94,

    "StaticTest": 0.93,

    "DynamicTest": 0.93,

    "TestLoad": 0.92,

    "InjuryCriterion": 0.91,

    "VehicleCategory": 0.90,

    "BeltAnchorage": 0.89,

    "SafetyBelt": 0.88,

    "Measurement": 0.85,

    "ValidationResult": 0.82,

    "FailureMode": 0.80,

    "Chunk": 0.50,

    "Community": 0.45
}

# ─────────────────────────────────────────────
# NEO4J CONFIG
# ─────────────────────────────────────────────

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://localhost:7687"
)

NEO4J_USER = os.getenv(
    "NEO4J_USER",
    "neo4j"
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "safety2024"
)

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
)

NEO4J_VECTOR_INDEX = (
    "regulation_vector_index"
)

NEO4J_VECTOR_DIMENSION = (
    EMBEDDING_DIMENSION
)

# ─────────────────────────────────────────────
# GROQ CONFIG
# ─────────────────────────────────────────────

CEREBRAS_API_KEY = os.getenv(
    "CEREBRAS_API_KEY"
)

CEREBRAS_BASE_URL = (
    "https://api.cerebras.ai/v1"
)

# ── Claude API (answers) ───────────────────────────────────────────────────────
# claude-haiku-4-5-20251001  → fast (2-3s), cheap  ← default
# claude-sonnet-4-6          → better quality (4-6s), more expensive
CLAUDE_MODEL       = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
LLM_MAX_TOKENS     = 800
LLM_MAX_TEMPERATURE    = 0.2

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
# EXTRACTION SYSTEM PROMPT
# ─────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """
You are a passive safety regulation extraction engine.

Extract:
- entities
- relationships
- geometry constraints
- test procedures
- load requirements
- homologation rules
- compliance requirements
- restraint systems
- injury criteria

Return ONLY valid JSON.

No markdown.
No explanations.
No commentary.

Every entity should connect to another entity.
"""

# ─────────────────────────────────────────────
# DEBUGGING
# ─────────────────────────────────────────────

VERBOSE = True

SAVE_RAW_LLM_OUTPUT = False

ENABLE_JSON_REPAIR = True

# ─────────────────────────────────────────────
# PIPELINE ORDER
# ─────────────────────────────────────────────

PIPELINE_STEPS = [

    "chunker.py",

    "extractor.py",

    "embedder.py",

    "graph/builder.py"
]

# ─────────────────────────────────────────────
# FINAL STARTUP CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n===================================")

    print("Passive Safety Regulation GraphRAG")

    print("===================================\n")

    print(f"DATA_DIR        : {DATA_DIR}")

    print(f"OUTPUT_DIR      : {OUTPUT_DIR}")

    print(f"EXTRACTION_MODEL: {EXTRACTION_MODEL}")

    print(f"EMBEDDING_MODEL : {EMBEDDING_MODEL}")

    print(f"DEV_MODE        : {DEV_MODE}")

    print("\nSupported Regulations:")

    for reg in SUPPORTED_REGULATIONS:

        print(f"  - {reg}")

    print("\nReady.")