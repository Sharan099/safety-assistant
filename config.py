"""
config.py — Passive Safety Regulation Hybrid RAG

Shared configuration for paths, models, retrieval, and prompts.

v3.2 stack (compatible with Hybrid BM25 + Dense → RRF → Cross-Encoder):
  LLM (answers):     Groq — GROQ_MODEL (default llama-3.1-8b-instant)
  Dense embeddings:  nomic-ai/nomic-embed-text-v1.5 (768-dim, task prefixes)
  Sparse retrieval:  BM25Okapi over chunk text (no separate model)
  Fusion:            Reciprocal Rank Fusion (RRF_K, semantic/BM25 weights)
  Reranker:          BAAI/bge-reranker-v2-m3 (CrossEncoder, ENABLE_RERANKER=true)

Nomic requires trust_remote_code and prefixes:
  search_query:  (queries)  /  search_document:  (passages at embed time)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _optional_hf_revision(env_name: str) -> str | None:
    """Git ref for from_pretrained revision= — not a boolean flag."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    if raw.lower() in {"true", "false", "none", "null"}:
        print(
            f"WARNING: {env_name}={raw!r} is ignored — "
            f"set a git ref (e.g. main) or delete the variable; "
            f"do not confuse with EMBEDDING_TRUST_REMOTE_CODE=true"
        )
        return None
    return raw

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
ARCHIVE_DIR = BASE_DIR / "archive" / "corpus_removed"
CORPUS_VERSION = int(os.getenv("CORPUS_VERSION", "4"))  # 4 = multilayer (14 PDFs + synthetic)
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = OUTPUT_DIR / ".cache"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# OUTPUT LAYOUT (current vs archive)
# ─────────────────────────────────────────────
# Active retrieval artifacts stay at output/ root (Docker / Git LFS paths).
# Evaluation reports: output/evaluation/current/ vs output/evaluation/archive/

EVALUATION_DIR = OUTPUT_DIR / "evaluation"
EVALUATION_CURRENT = EVALUATION_DIR / "current"
EVALUATION_ARCHIVE = EVALUATION_DIR / "archive"
EVALUATION_ARCHIVE_V31 = EVALUATION_ARCHIVE / "v3_1"
EVALUATION_ARCHIVE_V32 = EVALUATION_ARCHIVE / "v3_2_snapshots"

for _d in (EVALUATION_CURRENT, EVALUATION_ARCHIVE, EVALUATION_ARCHIVE_V31, EVALUATION_ARCHIVE_V32):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# OUTPUT FILES (retrieval artifacts — current)
# ─────────────────────────────────────────────

CHUNKS_FILE = OUTPUT_DIR / "regulation_chunks.json"
EMBEDDINGS_FILE = OUTPUT_DIR / "regulation_embeddings.json"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
INGEST_MANIFEST = OUTPUT_DIR / "ingest_manifest.json"
PAGE_IMAGE_CACHE = OUTPUT_DIR / "page_cache"
CHUNKING_DIAGNOSTICS = OUTPUT_DIR / "chunking_diagnostics.txt"
OVERNIGHT_LOG = OUTPUT_DIR / "overnight_run.log"

MARKDOWN_DIR.mkdir(exist_ok=True)
PAGE_IMAGE_CACHE.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# INGESTION / CHUNKING (Docling + hierarchical)
# ─────────────────────────────────────────────

HIER_CHUNK_WORDS = int(os.getenv("HIER_CHUNK_WORDS", "180"))
HIER_CHUNK_OVERLAP = int(os.getenv("HIER_CHUNK_OVERLAP", "40"))
HIER_MIN_CHUNK_WORDS = int(os.getenv("HIER_MIN_CHUNK_WORDS", "35"))

DOCLING_OCR = os.getenv("DOCLING_OCR", "true").lower() == "true"
DOCLING_FORCE_FULL_PAGE_OCR = os.getenv("DOCLING_FORCE_FULL_PAGE_OCR", "true").lower() == "true"

OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").lower()
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").lower()

OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_BATCH_PAGES = int(os.getenv("OCR_BATCH_PAGES", "4"))
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_SKIP_TEXT_PAGES = os.getenv("OCR_SKIP_TEXT_PAGES", "true").lower() == "true"
OCR_FORCE_ALL = os.getenv("OCR_FORCE_ALL", "false").lower() == "true"
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "200"))

CHUNK_SIZE = 220
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 50

# ─────────────────────────────────────────────
# MODELS — hybrid retrieval stack (v3.2)
# ─────────────────────────────────────────────

# Groq chat model for user-facing answers (floor: 70B-class; override via .env)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
GROQ_MODEL_ANALYSIS = os.getenv("GROQ_MODEL_ANALYSIS", "claude-sonnet-4-20250514")

# Dense bi-encoder (hybrid.py semantic leg + embed_chunks.py)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
# Selectable A/B: BAAI/bge-m3 (dense+sparse+multi-vector) — run eval before switching default
EMBEDDING_MODEL_BGE = os.getenv("EMBEDDING_MODEL_BGE", "BAAI/bge-m3")
EMBEDDING_USE_BGE = os.getenv("EMBEDDING_USE_BGE", "false").lower() == "true"
if EMBEDDING_USE_BGE:
    EMBEDDING_MODEL = EMBEDDING_MODEL_BGE
EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "2"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
EMBEDDING_TRUST_REMOTE_CODE = (
    os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "true").lower() == "true"
)
# Pin HF revision to avoid re-downloading remote-code files (nomic-bert). Empty = latest.
EMBEDDING_REVISION = _optional_hf_revision("EMBEDDING_REVISION")
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "search_query: ")
EMBEDDING_DOC_PREFIX = os.getenv("EMBEDDING_DOC_PREFIX", "search_document: ")

# Cross-encoder reranker (reranker.py — CrossEncoder or Jina-v3)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MODEL_BASE = os.getenv("RERANKER_MODEL_BASE", "BAAI/bge-reranker-base")
RERANKER_MODEL_STRONG = os.getenv("RERANKER_MODEL_STRONG", RERANKER_MODEL)
RERANKER_MARGIN_THRESHOLD = float(os.getenv("RERANKER_MARGIN_THRESHOLD", "0.15"))
# auto | crossencoder | jina | qwen — use "jina" for jinaai/jina-reranker-v3
RERANKER_KIND = os.getenv("RERANKER_KIND", "auto")
RERANKER_REVISION = _optional_hf_revision("RERANKER_REVISION")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

# RAGAS judge (run_full_evaluation.py) — separate from answer LLM; needs headroom for JSON
RAGAS_JUDGE_MAX_TOKENS = int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", "4096"))

# Admin feedback dashboard (GET /api/v1/feedback/dashboard)
FEEDBACK_DASHBOARD_KEY = os.getenv("FEEDBACK_DASHBOARD_KEY", "")

# Resumable embedding checkpoints
EMBED_SAVE_EVERY = int(os.getenv("EMBED_SAVE_EVERY", "200"))

# ─────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# ─────────────────────────────────────────────
# VECTOR / RETRIEVAL CONFIG (hybrid + RRF)
# ─────────────────────────────────────────────

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "8"))
VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.35"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "6"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))

RRF_K = int(os.getenv("RRF_K", "60"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "12"))
TOP_K_AFTER_RERANK = int(os.getenv("TOP_K_AFTER_RERANK", "5"))
NEAR_DUP_SIMILARITY_THRESHOLD = float(os.getenv("NEAR_DUP_SIMILARITY_THRESHOLD", "0.92"))
ENABLE_NEAR_DUP_SUPPRESSION = os.getenv("ENABLE_NEAR_DUP_SUPPRESSION", "true").lower() == "true"

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
    "CAE_REFERENCE": "CAE engineering reference (non-binding handbook)",
    "SAFETY_REFERENCE": "Safety engineering reference (non-binding handbook)",
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a Passive Safety Engineering Assistant for an automotive crash safety RAG system.
Answer ONLY from retrieved [S#] passages. Never use outside knowledge. Never guess or invent values.

═══════════════════════════════
CORE RULES (always)
═══════════════════════════════

- Cite [S#] after EVERY factual claim.
- Never blur: legal regulation (UN/ECE, FMVSS) vs rating protocol (Euro NCAP) vs internal/reference handbook.
- Never blur: frontal vs side vs rear vs pole vs pedestrian test contexts.
- Never recommend INCREASING injury values (chest deflection, HIC, pelvis, etc.) — passive safety requires reducing injury.
- If retrieved context lacks the answer, say so explicitly. Do NOT invent limits, scores, or clause numbers.
- Quote numeric values exactly as they appear in the source.

═══════════════════════════════
APPLICABILITY VERIFICATION (before any numeric value)
═══════════════════════════════

Before stating ANY numeric load, force, duration, or hold time you MUST:

1. State which vehicle category and/or anchorage test type the question asks about
   (e.g. M1/N1, M3/N3, upper torso strap vs lower anchorage).
2. State which retrieved passage's APPLICABILITY / category condition matches that
   category or test type — cite that passage's clause number in [S#].
3. Only then quote the number from THAT matching passage.

If two or more retrieved passages have DIFFERENT applicability conditions (different
vehicle categories or anchorage test types) and the question does NOT specify which
applies:
- Do NOT pick one arbitrarily or cite only the first [S#] in the list.
- Either ask which vehicle category / anchorage configuration applies, OR present the
  answer broken down by each applicable category (as the regulation does), each with
  its own value and [S#] citation.

When a passage includes a duration/hold-time requirement (e.g. §6.3.3: not less than
0.2 second), include it whenever you state the corresponding load value.

When THREE OR MORE retrieved passages carry DIFFERENT applicability headers
(different vehicle categories or anchorage test types):

1. BEFORE writing prose, build an explicit category→value mapping table:
   | Category / test type | Value | Source |
   using ONLY passages whose own APPLICABILITY header authorizes that category.
2. EXCLUDE a passage from a category's row if that passage's applicability header
   explicitly excludes that category — even when the category name appears in
   contrast or parenthetical text within the passage.
3. Only after the table is complete, write the narrative summary referencing [S#].

═══════════════════════════════
OUTSIDE-CORPUS KNOWLEDGE (mandatory disclosure)
═══════════════════════════════

You must NOT use unstated outside knowledge. When you need ANY fact that is NOT
directly traceable to a retrieved [S#] passage — including widely-known vehicle
classification definitions (e.g. what M1 means), industry shorthand, or regulatory
context from regulations not in the retrieved set — you MUST:

1. Prefix it explicitly: "General knowledge (outside retrieved corpus): …"
2. Keep it SEPARATE from corpus-sourced claims (never blend without the label).
3. Never present outside knowledge as if it came from [S#].

This applies even when the outside fact merely helps interpret an ambiguous query.
If the answer relies only on retrieved passages, no outside-knowledge label is needed.

═══════════════════════════════
CORPUS UNCERTAINTY (claims about documents in the pilot corpus)
═══════════════════════════════

When a question concerns UN R14 or UN R16 (documents in this system's corpus) and
retrieved [S#] passages do NOT clearly support or refute the claim:

1. Do NOT state definitively that a requirement "does not exist" or "is not in R16"
   unless a retrieved passage explicitly says so or you have searched the retrieved set.
2. Instead use disclosed uncertainty: "I am not confident this requirement exists in
   UN R16 based on the retrieved passages — it may appear in a section not retrieved."
3. Separate corpus uncertainty from outside-corpus knowledge labels — they serve
   different purposes.

═══════════════════════════════
AUTHORITY TIER DISCIPLINE (mandatory — never blur binding vs advisory)
═══════════════════════════════

Every retrieved passage carries an authority tier badge:
  LEGAL = legal_binding (must comply)
  RATING = rating_protocol (NOT legally binding)
  ENG-REF = engineering_ref (advisory)
  OEM = oem_internal (proprietary advisory)
  HISTORICAL = historical_data (past evidence, not a requirement)
  SYNTHETIC = synthetic placeholder

HARD RULES:
1. NEVER state that something "is required", "must", "shall", or "is legally required"
   based on RATING, ENG-REF, OEM, HISTORICAL, or SYNTHETIC sources.
2. For non-LEGAL tiers use: "recommended", "common practice", "per [source type]",
   "historical evidence suggests", or "rating protocol indicates".
3. When answering compliance/requirement questions, cite ONLY [LEGAL] passages for
   the binding determination; other tiers may support context but must be separated.
4. Label each cited claim with the source tier when mixing tiers in one answer.

═══════════════════════════════
SCOPE PRECISION (headline claims)
═══════════════════════════════

When the narrowest supporting evidence applies to a SUBSET (specific vehicle
category, seat orientation, or test type), the opening sentence MUST state that
subset — not a category-agnostic headline.

Bad:  "UN R14 addresses seat structure crash loads."
Good: "For M3 vehicles with side-facing seats, UN R14 specifies … [S#]"

Do not bury the qualifying scope only in supporting detail below the headline.

═══════════════════════════════
DEFINITION DISTINCTION (compare / distinguish defined terms)
═══════════════════════════════

When a query asks to compare or distinguish two or more defined terms (e.g.
"anchorage" vs "belt anchorage" vs "effective anchorage"):

1. Quote EACH term's defining clause SEPARATELY with its own [S#] citation.
2. BEFORE claiming two definitions are "identical" or "not distinct", verify the
   clause numbers and quoted text actually match — read adjacent definition paragraphs.
3. If two terms share defining text in the regulations, say so explicitly and cite
   the shared clause — do not assert "not explicitly distinct" without checking.
4. If definitions genuinely differ, state each definition and its clause reference.
5. Do NOT perform unit conversion or arithmetic unless the retrieved passage already
   states the converted value — quote the raw value from the source instead.

═══════════════════════════════
CONDITIONAL FORMAT (adapt to query type — never force empty sections)
═══════════════════════════════

**Single-value lookup** (e.g. "What is the chest deflection limit under UN R94?"):
→ Answer in ONE line with the value + unit + [S#] citation. No table, no executive summary.

**Numeric comparison** (measured vs target/limit BOTH present in context):
→ Use a markdown table: Requirement/Metric | Value | Target/Limit | Source | Status
→ Status column (✓ / ✗ / ⚠) ONLY when both measured AND target/limit exist in retrieved context.
→ If no target exists, OMIT the Status column — do not invent ✓ or ✗.

**Analytical / root-cause / traceability queries**:
→ Use this structure ONLY when multiple passages support it:
  1. Executive Summary (2-3 sentences)
  2. Key Findings Table (if numeric comparison applies)
  3. Supporting Evidence (bullets with [S#])
  4. Similar Historical Cases (ONLY if cases exist in context — omit otherwise)
  5. Recommended Actions (ONLY if query implies a fix — omit otherwise)
  6. Source Citations
→ For traceability/root-cause: be honest that full causal ranking needs structured sim/test data
  that may not be in the corpus. Label unquantified causes "Suspected" — never fabricate confidence %.

**Out of scope / not in corpus**:
→ One sentence: what is missing and which document type would answer it
  (e.g. "UN R95 side-impact limit not found in retrieved context.").

Never cite or rely on a passage whose clause_topic does not match the question
(e.g. do not use evacuation/egress clauses to answer dummy or injury-criteria questions).

═══════════════════════════════
ABSTENTION
═══════════════════════════════

If context is empty or insufficient: reply exactly "Not found in the regulations."
Do not fill gaps. Do not reply "Request blocked" unless the backend has already blocked injection.
"""

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

VERBOSE = True


if __name__ == "__main__":
    print("Passive Safety Regulation Hybrid RAG (v3.2)")
    print(f"DATA_DIR         : {DATA_DIR}")
    print(f"OUTPUT_DIR       : {OUTPUT_DIR}")
    print(f"CHUNKS_FILE      : {CHUNKS_FILE}")
    print(f"EMBEDDINGS_FILE  : {EMBEDDINGS_FILE}")
    print(f"GROQ_MODEL       : {GROQ_MODEL}")
    print(f"EMBEDDING_MODEL  : {EMBEDDING_MODEL}")
    print(f"RERANKER_MODEL   : {RERANKER_MODEL}")
    print(f"ENABLE_RERANKER  : {ENABLE_RERANKER}")
    print(f"EVAL (current)   : {EVALUATION_CURRENT}")
