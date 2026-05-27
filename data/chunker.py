"""
chunker.py — Passive Safety Regulation Chunker

Production-ready GraphRAG chunking pipeline for:
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

Author:
Sharan — Passive Safety GraphRAG
"""

import json
import re
import sys
import hashlib

from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# STDOUT
# ─────────────────────────────────────────────

sys.stdout.reconfigure(line_buffering=True)

def p(msg):
    print(msg, flush=True)

# ─────────────────────────────────────────────
# PDF ENGINE
# ─────────────────────────────────────────────

try:
    import fitz

except ImportError:

    p("ERROR: pip install pymupdf")

    sys.exit(1)

# ─────────────────────────────────────────────
# PATH CONFIG
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

CHUNKS_FILE = (
    OUTPUT_DIR / "regulation_chunks.json"
)

# ─────────────────────────────────────────────
# CHUNK CONFIG
# ─────────────────────────────────────────────

# optimized for Groq free tier
CHUNK_SIZE = 250

CHUNK_OVERLAP = 60

MIN_CHUNK_WORDS = 60

MAX_CHUNKS_PER_DOC = None

# ─────────────────────────────────────────────
# REGEX
# ─────────────────────────────────────────────

TEST_RE = re.compile(
    r"\b(test|testing|dynamic test|static test)\b",
    re.I
)

LOAD_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:daN|kN|N)\b",
    re.I
)

ANGLE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:degree|degrees|°)\b",
    re.I
)

DISTANCE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m)\b",
    re.I
)

VEHICLE_RE = re.compile(
    r"\b(M1|M2|M3|N1|N2|N3)\b"
)

REQ_RE = re.compile(
    r"\b(shall|must|required|requirement)\b",
    re.I
)

BELT_RE = re.compile(
    r"\b(anchorages?|seat.?belt|retractor|pretensioner)\b",
    re.I
)

INJURY_RE = re.compile(
    r"\b(HIC|chest deflection|femur force|neck force|thorax)\b",
    re.I
)

SECTION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\s+(.+)$",
    re.MULTILINE
)

ANNEX_RE = re.compile(
    r"^\s*(Annex\s+\d+.*?)$",
    re.MULTILINE | re.I
)

# ─────────────────────────────────────────────
# CLEAN FILENAMES
# ─────────────────────────────────────────────

def sanitize_filename(name: str) -> str:

    name = re.sub(
        r"\.pdf\.pdf$",
        ".pdf",
        name,
        flags=re.I
    )

    return name

# ─────────────────────────────────────────────
# REGULATION DETECTION
# ─────────────────────────────────────────────

def detect_regulation_type(filename: str) -> str:

    fname = filename.lower()

    # UN regulations
    if "r14" in fname:
        return "UN_R14"

    if "r16" in fname:
        return "UN_R16"

    if "r17" in fname:
        return "UN_R17"

    if "r94" in fname:
        return "UN_R94"

    if "r95" in fname:
        return "UN_R95"

    if "r135" in fname:
        return "UN_R135"

    if "r137" in fname:
        return "UN_R137"

    # FMVSS
    if "fmvss" in fname or "571" in fname:
        return "FMVSS"

    # Euro NCAP
    if "euro_ncap" in fname:
        return "EURO_NCAP"

    if "euroncap" in fname:
        return "EURO_NCAP"

    # ISO
    if "iso" in fname:
        return "ISO"

    # CAE docs
    if "cae" in fname:
        return "CAE_REFERENCE"

    # Safety docs
    if "safety_companion" in fname:
        return "SAFETY_REFERENCE"

    return "UNKNOWN"

# ─────────────────────────────────────────────
# FEATURE DETECTION
# ─────────────────────────────────────────────

def detect_features(text: str) -> dict:

    return {

        "has_test_procedure":
            bool(TEST_RE.search(text)),

        "has_loads":
            bool(LOAD_RE.search(text)),

        "has_angles":
            bool(ANGLE_RE.search(text)),

        "has_distances":
            bool(DISTANCE_RE.search(text)),

        "has_vehicle_classes":
            bool(VEHICLE_RE.search(text)),

        "has_requirements":
            bool(REQ_RE.search(text)),

        "has_belt_system":
            bool(BELT_RE.search(text)),

        "has_injury_criteria":
            bool(INJURY_RE.search(text)),

        "load_count":
            len(LOAD_RE.findall(text)),

        "angle_count":
            len(ANGLE_RE.findall(text)),

        "distance_count":
            len(DISTANCE_RE.findall(text)),

        "vehicle_count":
            len(VEHICLE_RE.findall(text))
    }

# ─────────────────────────────────────────────
# HASH
# ─────────────────────────────────────────────

def short_hash(text: str) -> str:

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()[:10]

# ─────────────────────────────────────────────
# SECTION EXTRACTION
# ─────────────────────────────────────────────

def extract_sections(doc):

    toc = doc.get_toc(simple=True)

    sections = []

    if not toc:

        sections.append({

            "section_id":
                "SEC-0001",

            "title":
                "Main Content",

            "level":
                1,

            "page_start":
                0,

            "page_end":
                len(doc) - 1
        })

        return sections

    for idx, item in enumerate(toc):

        level, title, page_1 = item

        page_end = len(doc) - 1

        for j in range(idx + 1, len(toc)):

            if toc[j][0] <= level:

                page_end = max(
                    0,
                    toc[j][2] - 2
                )

                break

        sections.append({

            "section_id":
                f"SEC-{idx+1:04d}",

            "title":
                title.strip(),

            "level":
                level,

            "page_start":
                max(0, page_1 - 1),

            "page_end":
                page_end
        })

    return sections

# ─────────────────────────────────────────────
# PAGE → SECTION
# ─────────────────────────────────────────────

def page_to_section(sections, page_idx):

    best = None

    for sec in sections:

        if sec["page_start"] <= page_idx <= sec["page_end"]:

            if best is None:

                best = sec

            elif sec["level"] > best["level"]:

                best = sec

    return best or sections[0]

# ─────────────────────────────────────────────
# CLEAN TEXT
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = text.strip()

    return text

# ─────────────────────────────────────────────
# SMART CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, metadata: dict):

    words = text.split()

    total = len(words)

    chunks = []

    pos = 0
    seq = 0

    while pos < total:

        end = min(
            pos + CHUNK_SIZE,
            total
        )

        chunk_words = words[pos:end]

        if len(chunk_words) < MIN_CHUNK_WORDS:

            break

        chunk_text = " ".join(chunk_words)

        chunk_text = clean_text(chunk_text)

        features = detect_features(
            chunk_text
        )

        chunk_id = (
            f"{metadata['regulation']}"
            f"-{metadata['section_id']}"
            f"-C{seq+1:03d}"
        )

        chunks.append({

            # identifiers
            "chunk_id":
                chunk_id,

            "chunk_hash":
                short_hash(chunk_text),

            # regulation
            "regulation":
                metadata["regulation"],

            "pdf_name":
                metadata["pdf_name"],

            # hierarchy
            "section_id":
                metadata["section_id"],

            "section_title":
                metadata["section_title"],

            "section_level":
                metadata["section_level"],

            # page tracking
            "page_start":
                metadata["page_start"],

            "page_end":
                metadata["page_end"],

            # content
            "text":
                chunk_text,

            "word_count":
                len(chunk_words),

            "chunk_seq":
                seq,

            # feature metadata
            **features
        })

        seq += 1

        pos += (
            CHUNK_SIZE -
            CHUNK_OVERLAP
        )

    return chunks

# ─────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────

def process_pdf(pdf_path: Path):

    pdf_name = sanitize_filename(
        pdf_path.name
    )

    p(f"\nProcessing: {pdf_name}")

    regulation = detect_regulation_type(
        pdf_name
    )

    doc = fitz.open(str(pdf_path))

    p(f"  Pages        : {len(doc)}")

    p(f"  Regulation   : {regulation}")

    sections = extract_sections(doc)

    p(f"  Sections     : {len(sections)}")

    page_texts = {}

    for page_idx, page in enumerate(doc):

        text = page.get_text(
            "text"
        ).strip()

        if text:

            page_texts[page_idx] = text

    doc.close()

    # aggregate by section
    section_data = {}

    for page_idx, text in page_texts.items():

        sec = page_to_section(
            sections,
            page_idx
        )

        sid = sec["section_id"]

        if sid not in section_data:

            section_data[sid] = {

                **sec,

                "pages": [],

                "full_text": ""
            }

        section_data[sid]["pages"].append(
            page_idx + 1
        )

        section_data[sid]["full_text"] += (
            "\n" + text
        )

    all_chunks = []

    global_seq = 0

    for sid, sdata in section_data.items():

        full_text = clean_text(
            sdata["full_text"]
        )

        if len(full_text.split()) < MIN_CHUNK_WORDS:

            continue

        pages = sdata["pages"]

        metadata = {

            "regulation":
                regulation,

            "pdf_name":
                pdf_name,

            "section_id":
                sid,

            "section_title":
                sdata["title"],

            "section_level":
                sdata["level"],

            "page_start":
                pages[0],

            "page_end":
                pages[-1]
        }

        chunks = chunk_text(
            full_text,
            metadata
        )

        for chunk in chunks:

            chunk["global_seq"] = global_seq

            global_seq += 1

            all_chunks.append(chunk)

            if (
                MAX_CHUNKS_PER_DOC
                and
                len(all_chunks)
                >= MAX_CHUNKS_PER_DOC
            ):
                break

    p(f"  Chunks       : {len(all_chunks)}")

    return all_chunks

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def process_all_pdfs():

    pdfs = sorted(
        DATA_DIR.rglob("*.pdf")
    )

    if not pdfs:

        p("ERROR: No PDFs found")

        sys.exit(1)

    p(f"\nFound {len(pdfs)} PDFs")

    all_chunks = []

    regulation_stats = defaultdict(int)

    for pdf in pdfs:

        try:

            chunks = process_pdf(pdf)

            all_chunks.extend(chunks)

            reg = detect_regulation_type(
                pdf.name
            )

            regulation_stats[reg] += len(chunks)

        except Exception as e:

            p(f"ERROR: {pdf.name}")

            p(str(e))

    # statistics
    word_counts = [

        c["word_count"]

        for c in all_chunks
    ]

    p("\n────────────────────────────────")

    p("FINAL STATISTICS")

    p("────────────────────────────────")

    p(f"Total chunks : {len(all_chunks)}")

    if word_counts:

        avg_words = (
            sum(word_counts)
            /
            len(word_counts)
        )

        p(
            f"Average words/chunk : "
            f"{avg_words:.0f}"
        )

    p("\nChunks by regulation:")

    for reg, count in sorted(
        regulation_stats.items()
    ):

        p(f"  {reg:<20} {count}")

    # feature statistics
    feature_keys = [

        "has_test_procedure",

        "has_loads",

        "has_angles",

        "has_distances",

        "has_requirements",

        "has_belt_system",

        "has_injury_criteria"
    ]

    p("\nFeature statistics:")

    for key in feature_keys:

        count = sum(

            1 for c in all_chunks

            if c.get(key)
        )

        p(f"  {key:<25} {count}")

    # save
    dataset = {

        "total_chunks":
            len(all_chunks),

        "chunks":
            all_chunks
    }

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    p(f"\nSaved → {CHUNKS_FILE}")

# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":

    process_all_pdfs()