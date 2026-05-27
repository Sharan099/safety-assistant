"""
embedder.py — Passive Safety Regulation Embedding Pipeline
Production Optimized Version

Designed for:
- UN R14 / R16 / R94 / R95 / R137
- FMVSS
- Euro NCAP
- ISO Standards
- Occupant Protection
- Crashworthiness
- Homologation Engineering

Features:
- Low RAM embedding pipeline
- Incremental checkpoint saving
- Resume support
- Safe batch processing
- CPU-friendly settings
- GraphRAG-ready vector store

Author:
Sharan — Passive Safety GraphRAG
"""

import json
import sys
import gc
import traceback

from pathlib import Path
from collections import defaultdict

# ───────────────────────────────────────────────────────────────
# STDOUT
# ───────────────────────────────────────────────────────────────

sys.stdout.reconfigure(line_buffering=True)

def p(msg):
    print(msg, flush=True)

# ───────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output")

CHUNKS_FILE = OUTPUT_DIR / "regulation_chunks.json"

ENTITIES_FILE = OUTPUT_DIR / "regulation_kg.json"

EMBEDDINGS_FILE = OUTPUT_DIR / "regulation_embeddings.json"

CHECKPOINT_FILE = OUTPUT_DIR / "embedding_checkpoint.json"

# SMALL + FAST + LOW RAM
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# VERY IMPORTANT FOR LAPTOPS
EMBEDDING_BATCH = 2

# save every N vectors
SAVE_INTERVAL = 50

# ───────────────────────────────────────────────────────────────
# MODEL LOADING
# ───────────────────────────────────────────────────────────────
def load_model():

    p("Importing sentence-transformers...")

    try:

        from sentence_transformers import (
            SentenceTransformer
        )

    except Exception as e:

        p(f"IMPORT ERROR: {e}")

        traceback.print_exc()

        sys.exit(1)

    p("Loading embedding model...")

    try:

        model = SentenceTransformer(

            EMBEDDING_MODEL
        )

        model = model.cpu()

        p("Model loaded successfully")

        dim = (
            model
            .get_sentence_embedding_dimension()
        )

        p(f"Embedding dimension: {dim}")

        return model

    except Exception as e:

        p(f"MODEL LOAD ERROR: {e}")

        traceback.print_exc()

        sys.exit(1)

# ───────────────────────────────────────────────────────────────
# CHECKPOINT SUPPORT
# ───────────────────────────────────────────────────────────────

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return {}, {}

    try:

        with open(
            CHECKPOINT_FILE,
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        embeddings = data.get(
            "embeddings",
            {}
        )

        metadata = data.get(
            "metadata",
            {}
        )

        p(
            f"Resume mode → "
            f"{len(embeddings)} vectors already embedded"
        )

        return embeddings, metadata

    except Exception:

        traceback.print_exc()

        return {}, {}

def save_checkpoint(

    embeddings,

    metadata
):

    dataset = {

        "embeddings": embeddings,

        "metadata": metadata
    }

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False
        )

    p(
        f"Checkpoint saved → "
        f"{len(embeddings)} vectors"
    )

# ───────────────────────────────────────────────────────────────
# SAFE ENCODING
# ───────────────────────────────────────────────────────────────

def encode_text(

    model,

    text
):

    vector = model.encode(

        [text],

        batch_size=1,

        show_progress_bar=False,

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    return vector[0].tolist()

# ───────────────────────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────────────────────

def load_chunks():

    if not CHUNKS_FILE.exists():

        p(
            f"ERROR: {CHUNKS_FILE} not found"
        )

        sys.exit(1)

    with open(
        CHUNKS_FILE,
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    chunks = data.get("chunks", [])

    p(f"Chunks loaded: {len(chunks)}")

    return chunks

def load_entities():

    if not ENTITIES_FILE.exists():

        p(
            f"ERROR: {ENTITIES_FILE} not found"
        )

        sys.exit(1)

    with open(
        ENTITIES_FILE,
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    entities = data.get("entities", [])

    p(f"Entities loaded: {len(entities)}")

    return entities

# ───────────────────────────────────────────────────────────────
# CHUNK EMBEDDING TEXT
# ───────────────────────────────────────────────────────────────

def build_chunk_embedding_text(chunk):

    regulation = chunk.get(
        "regulation",
        "UNKNOWN"
    )

    section = chunk.get(
        "section_title",
        ""
    )

    text = chunk.get(
        "text",
        ""
    )

    feature_tokens = []

    if chunk.get("has_test_procedure"):
        feature_tokens.append("TEST_PROCEDURE")

    if chunk.get("has_loads"):
        feature_tokens.append("LOAD_REQUIREMENTS")

    if chunk.get("has_angles"):
        feature_tokens.append("ANGLE_CONSTRAINTS")

    if chunk.get("has_distances"):
        feature_tokens.append("DISTANCE_CONSTRAINTS")

    if chunk.get("has_requirements"):
        feature_tokens.append("COMPLIANCE_REQUIREMENTS")

    if chunk.get("has_belt_system"):
        feature_tokens.append("RESTRAINT_SYSTEMS")

    if chunk.get("has_injury_criteria"):
        feature_tokens.append("INJURY_CRITERIA")

    features = " ".join(feature_tokens)

    return f"""
[REGULATION {regulation}]
[SECTION {section}]
[FEATURES {features}]

{text}
""".strip()

# ───────────────────────────────────────────────────────────────
# ENTITY EMBEDDING TEXT
# ───────────────────────────────────────────────────────────────

def build_entity_embedding_text(entity):

    entity_type = entity.get(
        "type",
        "Unknown"
    )

    entity_id = entity.get(
        "id",
        ""
    )

    regulation = entity.get(
        "regulation",
        ""
    )

    fields = []

    important_fields = [

        "name",
        "title",
        "description",

        "requirement_text",
        "rule_text",

        "parameter_name",

        "condition",

        "test_name",
        "test_type",

        "load_value",
        "unit",

        "min_value",
        "max_value",

        "value",

        "regulation_reference"
    ]

    for field in important_fields:

        value = entity.get(field)

        if value is not None:

            fields.append(str(value))

    combined = " ".join(fields)

    return f"""
[ENTITY_TYPE {entity_type}]
[ENTITY_ID {entity_id}]
[REGULATION {regulation}]

{combined}
""".strip()

# ───────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────────────────────

def main():

    chunks = load_chunks()

    entities = load_entities()

    model = load_model()

    embeddings, metadata = load_checkpoint()

    processed = set(embeddings.keys())

    regulation_stats = defaultdict(int)

    total_processed = 0

    # ─────────────────────────────────────────────
    # CHUNKS
    # ─────────────────────────────────────────────

    p("\nEmbedding chunks...")

    for idx, chunk in enumerate(chunks, 1):

        chunk_id = chunk["chunk_id"]

        if chunk_id in processed:

            continue

        try:

            text = build_chunk_embedding_text(
                chunk
            )

            vector = encode_text(
                model,
                text
            )

            embeddings[chunk_id] = vector

            metadata[chunk_id] = {

                "type": "chunk",

                "regulation":
                    chunk.get("regulation"),

                "section":
                    chunk.get("section_title"),

                "page_start":
                    chunk.get("page_start"),

                "page_end":
                    chunk.get("page_end")
            }

            total_processed += 1

            regulation = chunk.get(
                "regulation",
                "UNKNOWN"
            )

            regulation_stats[
                regulation
            ] += 1

            p(
                f"[{idx}/{len(chunks)}] "
                f"{chunk_id}"
            )

            if total_processed % SAVE_INTERVAL == 0:

                save_checkpoint(

                    embeddings,

                    metadata
                )

                gc.collect()

        except Exception:

            traceback.print_exc()

    # ─────────────────────────────────────────────
    # ENTITIES
    # ─────────────────────────────────────────────

    p("\nEmbedding entities...")

    for idx, entity in enumerate(entities, 1):

        entity_id = entity.get("id")

        if not entity_id:
            continue

        if entity_id in processed:

            continue

        try:

            text = build_entity_embedding_text(
                entity
            )

            vector = encode_text(
                model,
                text
            )

            embeddings[entity_id] = vector

            metadata[entity_id] = {

                "type": "entity",

                "entity_type":
                    entity.get("type"),

                "regulation":
                    entity.get("regulation"),

                "reference":
                    entity.get(
                        "regulation_reference"
                    )
            }

            total_processed += 1

            p(
                f"[{idx}/{len(entities)}] "
                f"{entity_id}"
            )

            if total_processed % SAVE_INTERVAL == 0:

                save_checkpoint(

                    embeddings,

                    metadata
                )

                gc.collect()

        except Exception:

            traceback.print_exc()

    # ─────────────────────────────────────────────
    # FINAL SAVE
    # ─────────────────────────────────────────────

    dataset = {

        "embeddings": embeddings,

        "metadata": metadata,

        "summary": {

            "total_vectors":
                len(embeddings),

            "embedding_model":
                EMBEDDING_MODEL,

            "regulation_distribution":
                dict(regulation_stats)
        }
    }

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False
        )

    p("\n======================================")
    p("EMBEDDING COMPLETE")
    p("======================================")

    p(
        f"Total vectors : "
        f"{len(embeddings)}"
    )

    p(f"\nSaved → {EMBEDDINGS_FILE}")

    p(
        "\nNext step: "
        "python graph/builder.py"
    )

# ───────────────────────────────────────────────────────────────
# ENTRY
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    main()