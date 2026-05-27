"""
builder.py — Passive Safety Regulation Graph Builder

Production-ready Neo4j GraphRAG builder for:
- UN Regulations
- FMVSS
- Euro NCAP
- ISO Standards
- Occupant Protection
- Crashworthiness
- Homologation Engineering

Features:
- Neo4j vector index
- Semantic chunk graph
- Regulation hierarchy
- Hybrid GraphRAG retrieval
- Entity relationships
- Chunk adjacency graph
- Regulation-aware KG
- Vector retrieval
- Keyword retrieval
- Graph neighborhood traversal

Author:
Sharan — Passive Safety GraphRAG
"""

import json
import sys

from pathlib import Path

from loguru import logger

# ─────────────────────────────────────────────
# IMPORT CONFIG
# ─────────────────────────────────────────────

sys.path.insert(
    0,
    str(Path(__file__).parent.parent)
)

from config import (

    CHUNKS_FILE,

    ENTITIES_FILE,

    EMBEDDINGS_FILE,

    NEO4J_URI,

    NEO4J_USER,

    NEO4J_PASSWORD,

    NEO4J_DATABASE,

    NEO4J_VECTOR_INDEX,

    NEO4J_VECTOR_DIMENSION,

    MAX_GRAPH_RELATIONSHIPS
)

# ─────────────────────────────────────────────
# NEO4J
# ─────────────────────────────────────────────

try:

    from neo4j import GraphDatabase

except ImportError:

    print("ERROR: pip install neo4j")

    sys.exit(1)

# ─────────────────────────────────────────────
# SANITIZE
# ─────────────────────────────────────────────

def sanitize(v):

    if isinstance(
        v,
        (str, int, float, bool)
    ) or v is None:

        return v

    if isinstance(v, dict):

        return json.dumps(v)

    if isinstance(v, list):

        return [
            sanitize(i)
            for i in v
        ]

    return str(v)

def sanitize_props(props: dict):

    return {

        k: sanitize(v)

        for k, v in props.items()

        if v is not None
        and v != ""
    }

# ─────────────────────────────────────────────
# ENTITY LABELS
# ─────────────────────────────────────────────

ENTITY_LABELS = {

    "Requirement":
        "Requirement",

    "ComplianceRule":
        "ComplianceRule",

    "ApprovalRequirement":
        "ApprovalRequirement",

    "GeometryConstraint":
        "GeometryConstraint",

    "AngleRequirement":
        "AngleRequirement",

    "DistanceRequirement":
        "DistanceRequirement",

    "TestProcedure":
        "TestProcedure",

    "StaticTest":
        "StaticTest",

    "DynamicTest":
        "DynamicTest",

    "TestLoad":
        "TestLoad",

    "VehicleCategory":
        "VehicleCategory",

    "BeltAnchorage":
        "BeltAnchorage",

    "SafetyBelt":
        "SafetyBelt",

    "InjuryCriterion":
        "InjuryCriterion",

    "FailureMode":
        "FailureMode",

    "Measurement":
        "Measurement"
}

# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

class RegulationGraphBuilder:

    def __init__(self):

        self.driver = GraphDatabase.driver(

            NEO4J_URI,

            auth=(
                NEO4J_USER,
                NEO4J_PASSWORD
            )
        )

        logger.info(
            f"Connected → {NEO4J_URI}"
        )

    # ─────────────────────────────

    def close(self):

        self.driver.close()

    # ─────────────────────────────
    # CONSTRAINTS
    # ─────────────────────────────

    def create_constraints(self):

        constraints = [

            """
            CREATE CONSTRAINT regulation_id
            IF NOT EXISTS
            FOR (n:Regulation)
            REQUIRE n.id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT section_id
            IF NOT EXISTS
            FOR (n:DocumentSection)
            REQUIRE n.id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT chunk_id
            IF NOT EXISTS
            FOR (n:Chunk)
            REQUIRE n.id IS UNIQUE
            """
        ]

        for label in ENTITY_LABELS.values():

            constraints.append(

                f"""
                CREATE CONSTRAINT {label.lower()}_id
                IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.id IS UNIQUE
                """
            )

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for q in constraints:

                try:

                    s.run(q)

                except Exception as e:

                    logger.warning(e)

        logger.info(
            "Constraints ready"
        )

    # ─────────────────────────────
    # VECTOR INDEX
    # ─────────────────────────────

    def create_vector_index(self):

        q = f"""
        CREATE VECTOR INDEX
        {NEO4J_VECTOR_INDEX}
        IF NOT EXISTS
        FOR (n:Chunk)
        ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`:
                    {NEO4J_VECTOR_DIMENSION},

                `vector.similarity_function`:
                    'cosine'
            }}
        }}
        """

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            try:

                s.run(q)

            except Exception as e:

                logger.warning(e)

        logger.info(
            "Vector index ready"
        )

    # ─────────────────────────────
    # CLEAR GRAPH
    # ─────────────────────────────

    def clear(self):

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            s.run(
                "MATCH (n) DETACH DELETE n"
            )

        logger.info(
            "Graph cleared"
        )

    # ─────────────────────────────
    # LOAD REGULATIONS
    # ─────────────────────────────

    def load_regulations(self, chunks):

        regulations = {}

        for chunk in chunks:

            reg = chunk.get(
                "regulation",
                "UNKNOWN"
            )

            if reg not in regulations:

                regulations[reg] = {

                    "id": reg,

                    "name": reg
                }

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for reg in regulations.values():

                s.run(
                    """
                    MERGE (r:Regulation {id:$id})
                    SET r += $props
                    """,

                    id=reg["id"],

                    props=sanitize_props(reg)
                )

        logger.info(
            f"Regulations loaded: "
            f"{len(regulations)}"
        )

    # ─────────────────────────────
    # LOAD SECTIONS
    # ─────────────────────────────

    def load_sections(self, chunks):

        sections = {}

        for chunk in chunks:

            sid = chunk.get(
                "section_id",
                ""
            )

            if not sid:

                continue

            if sid not in sections:

                sections[sid] = {

                    "id":
                        sid,

                    "title":
                        chunk.get(
                            "section_title",
                            ""
                        ),

                    "regulation":
                        chunk.get(
                            "regulation",
                            ""
                        )
                }

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for sec in sections.values():

                s.run(
                    """
                    MERGE (s:DocumentSection {id:$id})
                    SET s += $props
                    """,

                    id=sec["id"],

                    props=sanitize_props(sec)
                )

                s.run(
                    """
                    MATCH (r:Regulation {id:$rid})

                    MATCH (s:DocumentSection {id:$sid})

                    MERGE (r)-[:HAS_SECTION]->(s)
                    """,

                    rid=sec["regulation"],

                    sid=sec["id"]
                )

        logger.info(
            f"Sections loaded: "
            f"{len(sections)}"
        )

    # ─────────────────────────────
    # LOAD CHUNKS
    # ─────────────────────────────

    def load_chunks(
        self,
        chunks,
        embeddings
    ):

        if "embeddings" in embeddings:

            embeddings = embeddings["embeddings"]

        count = 0

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for chunk in chunks:

                cid = chunk.get(
                    "chunk_id"
                )

                if not cid:

                    continue

                props = sanitize_props({

                    "id":
                        cid,

                    "text":
                        chunk.get(
                            "text",
                            ""
                        ),

                    "regulation":
                        chunk.get(
                            "regulation",
                            ""
                        ),

                    "section_title":
                        chunk.get(
                            "section_title",
                            ""
                        ),

                    "page_start":
                        chunk.get(
                            "page_start"
                        ),

                    "page_end":
                        chunk.get(
                            "page_end"
                        ),

                    "word_count":
                        chunk.get(
                            "word_count"
                        ),

                    "global_seq":
                        chunk.get(
                            "global_seq"
                        )
                })

                emb = embeddings.get(cid)

                if emb:

                    props["embedding"] = emb

                s.run(
                    """
                    MERGE (c:Chunk {id:$id})
                    SET c += $props
                    """,

                    id=cid,

                    props=props
                )

                sid = chunk.get(
                    "section_id"
                )

                if sid:

                    s.run(
                        """
                        MATCH (c:Chunk {id:$cid})

                        MATCH (s:DocumentSection {id:$sid})

                        MERGE (s)-[:HAS_CHUNK]->(c)
                        """,

                        cid=cid,

                        sid=sid
                    )

                count += 1

        sorted_chunks = sorted(

            chunks,

            key=lambda x:
                x.get(
                    "global_seq",
                    0
                )
        )

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for i in range(
                len(sorted_chunks) - 1
            ):

                a = sorted_chunks[i]

                b = sorted_chunks[i+1]

                s.run(
                    """
                    MATCH (a:Chunk {id:$a})

                    MATCH (b:Chunk {id:$b})

                    MERGE (a)-[:NEXT_CHUNK]->(b)
                    """,

                    a=a["chunk_id"],

                    b=b["chunk_id"]
                )

        logger.info(
            f"Chunks loaded: {count}"
        )

    # ─────────────────────────────
    # LOAD ENTITIES
    # ─────────────────────────────

    def load_entities(
        self,
        entities,
        embeddings
    ):

        if "embeddings" in embeddings:

            embeddings = embeddings["embeddings"]

        count = 0

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for ent in entities:

                eid = ent.get(
                    "id",
                    ""
                ).strip()

                etype = ent.get(
                    "type",
                    ""
                )

                if not eid:

                    continue

                label = ENTITY_LABELS.get(
                    etype,
                    "Entity"
                )

                props = sanitize_props({

                    k: v

                    for k, v in ent.items()

                    if k not in (
                        "type",
                        "source_chunks"
                    )
                })

                props["id"] = eid

                emb = embeddings.get(eid)

                if emb:

                    props["embedding"] = emb

                s.run(

                    f"""
                    MERGE (e:{label} {{id:$id}})
                    SET e += $props
                    """,

                    id=eid,

                    props=props
                )

                source_chunks = ent.get(
                    "source_chunks",
                    []
                )

                for cid in source_chunks:

                    s.run(
                        """
                        MATCH (c:Chunk {id:$cid})

                        MATCH (e {id:$eid})

                        MERGE (c)-[:CONTAINS]->(e)
                        """,

                        cid=cid,

                        eid=eid
                    )

                count += 1

        logger.info(
            f"Entities loaded: {count}"
        )

    # ─────────────────────────────
    # LOAD RELATIONSHIPS
    # ─────────────────────────────

    def load_relationships(
        self,
        relationships
    ):

        count = 0

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for rel in relationships[
                :MAX_GRAPH_RELATIONSHIPS
            ]:

                src = rel.get(
                    "source",
                    ""
                ).strip()

                tgt = rel.get(
                    "target",
                    ""
                ).strip()

                rtype = rel.get(
                    "type",
                    "RELATED_TO"
                )

                rtype = (
                    rtype
                    .upper()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                if not src or not tgt:

                    continue

                try:

                    s.run(

                        f"""
                        MATCH (a {{id:$src}})

                        MATCH (b {{id:$tgt}})

                        MERGE (a)-[:{rtype}]->(b)
                        """,

                        src=src,

                        tgt=tgt
                    )

                    count += 1

                except Exception as e:

                    logger.warning(
                        f"{src} -> {tgt}: {e}"
                    )

        logger.info(
            f"Relationships loaded: "
            f"{count}"
        )

    # ─────────────────────────────
    # KEYWORD SEARCH
    # ─────────────────────────────

    def keyword_search(

        self,

        query: str,

        limit: int = 10
    ):

        cypher = """
        MATCH (n)

        WHERE

            (
                n.text IS NOT NULL
                AND
                toLower(n.text)
                CONTAINS toLower($query)
            )

            OR

            (
                n.name IS NOT NULL
                AND
                toLower(n.name)
                CONTAINS toLower($query)
            )

        RETURN

            n.id AS id,

            labels(n) AS labels,

            coalesce(
                n.name,
                n.text,
                n.id
            ) AS name,

            n.text AS text,

            n.regulation AS regulation

        LIMIT $limit
        """

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            result = s.run(

                cypher,
                {
                    "query":query,

                    "limit":limit
                }
            )

            return [

                dict(r)

                for r in result
            ]

    # ─────────────────────────────
    # VECTOR SEARCH
    # ─────────────────────────────

    def vector_search(

        self,

        embedding,

        limit: int = 8
    ):

        cypher = f"""
        CALL db.index.vector.queryNodes(

            '{NEO4J_VECTOR_INDEX}',

            $limit,

            $embedding
        )

        YIELD node, score

        RETURN

            node.id AS id,

            node.text AS text,

            node.section_title
                AS section_title,

            node.regulation
                AS regulation,

            score

        ORDER BY score DESC
        """

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            result = s.run(

                cypher,

                embedding=embedding,

                limit=limit
            )

            return [

                dict(r)

                for r in result
            ]

    # ─────────────────────────────
    # GRAPH CONTEXT
    # ─────────────────────────────

    def get_context(

        self,

        node_id: str,

        hops: int = 2
    ):

        cypher = f"""
        MATCH (root {{id:$id}})

        OPTIONAL MATCH path =
            (root)-[*1..{hops}]-(related)

        RETURN

            root,

            collect(DISTINCT related)
                AS related_nodes,

            collect(DISTINCT relationships(path))
                AS rels
        """

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            rec = s.run(

                cypher,

                id=node_id
            ).single()

            if not rec:

                return {

                    "node": {},

                    "related": [],

                    "rels": []
                }

            root = rec["root"]

            related_nodes = (
                rec["related_nodes"]
                or []
            )

            rel_lists = (
                rec["rels"]
                or []
            )

            rels = []

            for rel_group in rel_lists:

                for r in rel_group:

                    rels.append({

                        "source":
                            r.start_node.get("id"),

                        "target":
                            r.end_node.get("id"),

                        "type":
                            r.type
                    })

            def node_to_dict(n):

                if not n:

                    return {}

                props = dict(n)

                props["labels"] = list(n.labels)

                return props

            return {

                "node":
                    node_to_dict(root),

                "related": [

                    node_to_dict(n)

                    for n in related_nodes

                    if n
                ],

                "rels":
                    rels
            }

    # ─────────────────────────────
    # GRAPH STATS
    # ─────────────────────────────

    def stats(self):

        with self.driver.session(
            database=NEO4J_DATABASE
        ) as s:

            nodes = s.run(
                """
                MATCH (n)
                RETURN count(n) AS c
                """
            ).single()["c"]

            rels = s.run(
                """
                MATCH ()-[r]->()
                RETURN count(r) AS c
                """
            ).single()["c"]

        return {

            "nodes":
                nodes,

            "relationships":
                rels
        }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_graph():

    for f in [

        CHUNKS_FILE,

        ENTITIES_FILE,

        EMBEDDINGS_FILE
    ]:

        if not f.exists():

            logger.error(
                f"Missing file: {f}"
            )

            return

    with open(
        CHUNKS_FILE,
        encoding="utf-8"
    ) as f:

        chunk_data = json.load(f)

    with open(
        ENTITIES_FILE,
        encoding="utf-8"
    ) as f:

        entity_data = json.load(f)

    with open(
        EMBEDDINGS_FILE,
        encoding="utf-8"
    ) as f:

        embeddings = json.load(f)

    chunks = chunk_data.get(
        "chunks",
        []
    )

    entities = entity_data.get(
        "entities",
        []
    )

    relationships = entity_data.get(
        "relationships",
        []
    )

    logger.info(
        f"Input: "
        f"{len(chunks)} chunks | "
        f"{len(entities)} entities | "
        f"{len(relationships)} rels"
    )

    builder = RegulationGraphBuilder()

    try:

        builder.clear()

        builder.create_constraints()

        builder.create_vector_index()

        builder.load_regulations(
            chunks
        )

        builder.load_sections(
            chunks
        )

        builder.load_chunks(
            chunks,
            embeddings
        )

        builder.load_entities(
            entities,
            embeddings
        )

        builder.load_relationships(
            relationships
        )

        stats = builder.stats()

        logger.info(
            f"Graph complete → "
            f"{stats['nodes']} nodes | "
            f"{stats['relationships']} rels"
        )

    finally:

        builder.close()

# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":

    build_graph()