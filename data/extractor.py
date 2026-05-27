"""
extractor.py — Production Passive Safety Regulation KG Extractor
Cerebras Edition

Production-ready GraphRAG extraction pipeline for:
- UN Regulations
- FMVSS
- Euro NCAP
- ISO Standards
- Occupant Protection
- Crashworthiness
- Homologation Engineering

Features:
- Cerebras API extraction
- Automatic checkpoint saving
- Resume support
- Smart rate-limit handling
- Incremental KG persistence
- Deterministic extraction
- JSON repair
- Regulation-aware prompting

Author:
Sharan — Passive Safety GraphRAG
"""

import json
import os
import re
import sys
import time
import traceback

from openai import OpenAI

from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# STDOUT
# ─────────────────────────────────────────────

sys.stdout.reconfigure(line_buffering=True)

def p(msg):
    print(msg, flush=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

CHUNKS_FILE = (
    OUTPUT_DIR / "regulation_chunks.json"
)

KG_FILE = (
    OUTPUT_DIR / "regulation_kg.json"
)

PROGRESS_FILE = (
    OUTPUT_DIR / "extract_progress.json"
)

# Cerebras model
MODEL_NAME = "llama3.1-8b"

MAX_RETRIES = 5

TEMPERATURE = 0

REQUEST_DELAY_SECONDS = 1

MAX_ENTITY_PER_CHUNK = 10

MAX_RELATIONSHIPS_PER_CHUNK = 8

MAX_CHUNKS_PER_RUN = None

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an expert passive safety regulation extraction engine.

IMPORTANT:
- Extract maximum {MAX_ENTITY_PER_CHUNK} entities
- Extract maximum {MAX_RELATIONSHIPS_PER_CHUNK} relationships
- Prioritize engineering relevance
- Ignore duplicate measurements
- Ignore repeated legal wording
- Preserve regulation references
- Preserve engineering semantics
- Use deterministic IDs

Extract:
- Requirement
- ComplianceRule
- TestProcedure
- DynamicTest
- StaticTest
- InjuryCriterion
- BeltAnchorage
- SafetyBelt
- GeometryConstraint
- DistanceRequirement
- Measurement
- TestLoad
- FailureMode

Return ONLY valid JSON.

No markdown.
No explanations.
No commentary.

Always extract relationships.
Every entity should connect to another entity.
"""

# ─────────────────────────────────────────────
# REGULATION CONTEXT
# ─────────────────────────────────────────────

REGULATION_CONTEXT = {

    "UN_R14":
        "Safety belt anchorages",

    "UN_R16":
        "Safety belts and restraint systems",

    "UN_R17":
        "Seats and head restraints",

    "UN_R94":
        "Frontal impact occupant protection",

    "UN_R95":
        "Side impact occupant protection",

    "UN_R135":
        "Pole side impact occupant protection",

    "UN_R137":
        "Full width frontal impact occupant protection",

    "FMVSS":
        "Federal motor vehicle safety standards",

    "EURO_NCAP":
        "Crashworthiness rating protocol",

    "ISO":
        "International engineering standards"
}

# ─────────────────────────────────────────────
# FEATURE HINTS
# ─────────────────────────────────────────────

def build_feature_hints(chunk):

    hints = []

    if chunk.get("has_test_procedure"):

        hints.append("""
Extract:
- TestProcedure
- DynamicTest
- StaticTest
- TESTS relationships
- VALIDATES relationships
""")

    if chunk.get("has_loads"):

        hints.append("""
Extract:
- TestLoad
- LOADS relationships
""")

    if chunk.get("has_angles"):

        hints.append("""
Extract:
- GeometryConstraint
- AngleRequirement
""")

    if chunk.get("has_distances"):

        hints.append("""
Extract:
- DistanceRequirement
- Measurement
""")

    if chunk.get("has_requirements"):

        hints.append("""
Extract:
- Requirement
- ComplianceRule
""")

    if chunk.get("has_belt_system"):

        hints.append("""
Extract:
- BeltAnchorage
- SafetyBelt
- RESTRAINS relationships
""")

    if chunk.get("has_injury_criteria"):

        hints.append("""
Extract:
- InjuryCriterion
- FailureMode
""")

    return "\n".join(hints)

# ─────────────────────────────────────────────
# SCHEMA PROMPT
# ─────────────────────────────────────────────

def build_schema_prompt():

    return """
Return JSON EXACTLY like this:

{
  "entities": [
    {
      "id": "REQ-5_4_2-ALPHA1",
      "type": "GeometryConstraint",
      "parameter_name": "alpha_1",
      "min_value": 30,
      "max_value": 80,
      "unit": "degree",
      "regulation_reference": "5.4.2.1"
    }
  ],

  "relationships": [
    {
      "source": "TEST-6_4_1",
      "target": "LOAD-1350",
      "type": "REQUIRES"
    }
  ]
}
"""

# ─────────────────────────────────────────────
# BUILD PROMPT
# ─────────────────────────────────────────────

def build_prompt(chunk):

    regulation = chunk.get(
        "regulation",
        "UNKNOWN"
    )

    context = REGULATION_CONTEXT.get(
        regulation,
        "vehicle safety regulation"
    )

    hints = build_feature_hints(chunk)

    schema = build_schema_prompt()

    return f"""
DOCUMENT TYPE:
{regulation}

DOMAIN:
{context}

SECTION:
{chunk.get('section_title','')}

PAGES:
{chunk.get('page_start','?')}–{chunk.get('page_end','?')}

FEATURES:
{hints}

TEXT:
================================================

{chunk.get('text','')}

================================================

Extract only the MOST important engineering entities and relationships.

RULES:

1. Preserve regulation references
2. Use deterministic IDs
3. Every entity should connect to another entity
4. Prioritize safety engineering meaning

{schema}
"""

# ─────────────────────────────────────────────
# JSON REPAIR
# ─────────────────────────────────────────────

def salvage_json(raw):

    if not raw:
        return None

    clean = re.sub(
        r"```[a-z]*",
        "",
        raw,
        flags=re.I
    )

    clean = clean.strip("` \n")

    start = clean.find("{")

    if start == -1:
        return None

    clean = clean[start:]

    try:
        return json.loads(clean)

    except Exception:
        pass

    last = clean.rfind("}")

    if last != -1:

        candidate = clean[:last+1]

        try:
            return json.loads(candidate)

        except Exception:
            pass

    return None

# ─────────────────────────────────────────────
# LOAD CHUNKS
# ─────────────────────────────────────────────

def load_chunks():

    if not CHUNKS_FILE.exists():

        p("ERROR: regulation_chunks.json missing")

        sys.exit(1)

    with open(
        CHUNKS_FILE,
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    chunks = data.get(
        "chunks",
        []
    )

    p(f"Loaded {len(chunks)} chunks")

    return chunks

# ─────────────────────────────────────────────
# ENTITY REGISTRY
# ─────────────────────────────────────────────

class EntityRegistry:

    def __init__(self):

        self.entities = {}

        self.relationships = []

        self.by_type = defaultdict(int)

    def add_entity(self, entity, chunk_id):

        eid = entity.get("id")

        if not eid:
            return

        if eid not in self.entities:

            entity["source_chunks"] = [chunk_id]

            self.entities[eid] = entity

            self.by_type[
                entity.get(
                    "type",
                    "Unknown"
                )
            ] += 1

        else:

            if chunk_id not in self.entities[eid]["source_chunks"]:

                self.entities[eid]["source_chunks"].append(
                    chunk_id
                )

    def add_relationship(self, rel):

        if rel not in self.relationships:

            self.relationships.append(rel)

    def dataset(self):

        return {

            "entities":
                list(self.entities.values()),

            "relationships":
                self.relationships,

            "summary": {

                "total_entities":
                    len(self.entities),

                "total_relationships":
                    len(self.relationships),

                "by_type":
                    dict(self.by_type)
            }
        }

# ─────────────────────────────────────────────
# CHECKPOINT SAVE
# ─────────────────────────────────────────────

def save_checkpoint(

    registry,

    processed_chunks
):

    dataset = registry.dataset()

    dataset["processed_chunks"] = list(
        processed_chunks
    )

    tmp_file = KG_FILE.with_suffix(".tmp")

    with open(
        tmp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    tmp_file.replace(KG_FILE)

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump({

            "processed_chunks":
                list(processed_chunks),

            "last_update":
                time.time()

        }, f, indent=2)

    p(
        f"Checkpoint saved → "
        f"{len(processed_chunks)} chunks"
    )

# ─────────────────────────────────────────────
# RESUME SUPPORT
# ─────────────────────────────────────────────

def load_previous_progress():

    registry = EntityRegistry()

    processed_chunks = set()

    if KG_FILE.exists():

        try:

            with open(
                KG_FILE,
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            for ent in data.get(
                "entities",
                []
            ):

                eid = ent.get("id")

                if eid:

                    registry.entities[eid] = ent

                    registry.by_type[
                        ent.get(
                            "type",
                            "Unknown"
                        )
                    ] += 1

                    for cid in ent.get(
                        "source_chunks",
                        []
                    ):

                        processed_chunks.add(cid)

            registry.relationships = data.get(
                "relationships",
                []
            )

            p(
                f"Resume mode → "
                f"{len(processed_chunks)} chunks already processed"
            )

        except Exception:

            traceback.print_exc()

            p(
                "WARNING: Resume failed"
            )

    return registry, processed_chunks

# ─────────────────────────────────────────────
# EXTRACT SINGLE CHUNK
# ─────────────────────────────────────────────

def extract_chunk(client, chunk):

    chunk_id = chunk["chunk_id"]

    p(f"Extracting: {chunk_id}")

    prompt = build_prompt(chunk)

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                temperature=TEMPERATURE,

                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            raw = (
                response
                .choices[0]
                .message
                .content
            )

            parsed = salvage_json(raw)

            if not parsed:

                p(
                    f"JSON parse failed: "
                    f"{chunk_id}"
                )

                continue

            entities = parsed.get(
                "entities",
                []
            )

            rels = parsed.get(
                "relationships",
                []
            )

            p(
                f"→ {len(entities)} entities | "
                f"{len(rels)} rels"
            )

            return parsed

        except Exception as e:

            msg = str(e)

            p(
                f"Retry {attempt+1}: {msg}"
            )

            if "429" in msg:

                wait_time = 30

                p(
                    f"Rate limit hit → sleeping {wait_time}s"
                )

                time.sleep(wait_time)

            else:

                time.sleep(5)

    return None

# ─────────────────────────────────────────────
# EXTRACTION PIPELINE
# ─────────────────────────────────────────────

def extract_all(client, chunks):

    registry, processed_chunks = (
        load_previous_progress()
    )

    remaining = [

        c for c in chunks

        if c["chunk_id"]
        not in processed_chunks
    ]

    if MAX_CHUNKS_PER_RUN:

        remaining = remaining[:MAX_CHUNKS_PER_RUN]

    p(
        f"Remaining chunks: "
        f"{len(remaining)}"
    )

    for idx, chunk in enumerate(
        remaining,
        1
    ):

        chunk_id = chunk["chunk_id"]

        p(
            f"\n[{idx}/{len(remaining)}] "
            f"{chunk_id}"
        )

        result = extract_chunk(
            client,
            chunk
        )

        if not result:

            p(
                f"Skipped: {chunk_id}"
            )

            continue

        for ent in result.get(
            "entities",
            []
        ):

            registry.add_entity(
                ent,
                chunk_id
            )

        for rel in result.get(
            "relationships",
            []
        ):

            registry.add_relationship(rel)

        processed_chunks.add(
            chunk_id
        )

        save_checkpoint(

            registry,

            processed_chunks
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    return registry.dataset()

# ─────────────────────────────────────────────
# SAVE FINAL DATASET
# ─────────────────────────────────────────────

def save_dataset(dataset):

    with open(
        KG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    summary = dataset["summary"]

    p("\n===================================")

    p("EXTRACTION COMPLETE")

    p("===================================")

    p(
        f"Entities      : "
        f"{summary['total_entities']}"
    )

    p(
        f"Relationships : "
        f"{summary['total_relationships']}"
    )

    p("\nEntity Types:")

    for etype, count in sorted(

        summary["by_type"].items(),

        key=lambda x: -x[1]
    ):

        p(f"  {etype:<25} {count}")

    p(f"\nSaved → {KG_FILE}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    api_key = os.getenv(
        "CEREBRAS_API_KEY"
    )

    if not api_key:

        p(
            "ERROR: CEREBRAS_API_KEY missing"
        )

        sys.exit(1)

    client = OpenAI(

        api_key=api_key,

        base_url="https://api.cerebras.ai/v1"
    )

    chunks = load_chunks()

    dataset = extract_all(
        client,
        chunks
    )

    save_dataset(dataset)

# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":

    main()