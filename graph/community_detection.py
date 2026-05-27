"""
community.py — Passive Safety Regulation Community Detection

Production-ready GraphRAG community pipeline for:
- UN Regulations
- FMVSS
- Euro NCAP
- ISO Standards
- Occupant Protection
- Crashworthiness
- Homologation Engineering

Features:
- Louvain community detection
- Regulation-aware semantic clustering
- Neo4j community graph
- Groq-powered summaries
- Regulation KG aggregation

Author:
Sharan — Passive Safety GraphRAG
"""

import json
import os
import sys
import time

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

    ENTITIES_FILE,

    COMMUNITY_FILE,

    COMMUNITY_RESOLUTION,

    COMMUNITY_MIN_SIZE,

    NEO4J_URI,

    NEO4J_USER,

    NEO4J_PASSWORD,

    NEO4J_DATABASE,

    CEREBRAS_API_KEY,

    CEREBRAS_BASE_URL,

    EXTRACTION_MODEL
)

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

try:

    import networkx as nx

except ImportError:

    print("ERROR: pip install networkx")

    sys.exit(1)

try:

    import community as community_louvain

except ImportError:

    print("ERROR: pip install python-louvain")

    sys.exit(1)

try:

    from neo4j import GraphDatabase

except ImportError:

    print("ERROR: pip install neo4j")

    sys.exit(1)

try:

    from openai import OpenAI

except ImportError:

    print("ERROR: pip install openai")

    sys.exit(1)

# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────

def build_nx_graph(
    entities,
    relationships
):

    EXCLUDE_TYPES = {

        "Chunk",

        "DocumentSection",

        "Measurement"
    }

    G = nx.Graph()

    # nodes
    for ent in entities:

        eid = ent.get("id","")

        etype = ent.get("type","")

        if not eid:

            continue

        if etype in EXCLUDE_TYPES:

            continue

        name = (

            ent.get("name","")

            or

            ent.get("text","")

            or

            eid
        )

        regulation = ent.get(
            "regulation",
            "UNKNOWN"
        )

        G.add_node(

            eid,

            type=etype,

            name=name,

            regulation=regulation
        )

    # edges
    entity_ids = set(G.nodes)

    for rel in relationships:

        src = rel.get("source","")

        tgt = rel.get("target","")

        rtype = rel.get("type","")

        if src in entity_ids and tgt in entity_ids:

            G.add_edge(

                src,

                tgt,

                type=rtype
            )

    logger.info(

        f"NetworkX graph → "
        f"{G.number_of_nodes()} nodes | "
        f"{G.number_of_edges()} edges"
    )

    return G

# ─────────────────────────────────────────────
# LOUVAIN
# ─────────────────────────────────────────────

def run_louvain(G):

    if G.number_of_nodes() == 0:

        return {}

    partition = community_louvain.best_partition(

        G,

        resolution=COMMUNITY_RESOLUTION
    )

    n_comm = len(set(partition.values()))

    logger.info(

        f"Louvain → "
        f"{n_comm} communities"
    )

    return partition

# ─────────────────────────────────────────────
# COMMUNITY SUMMARY
# ─────────────────────────────────────────────

def summarize_community(

    client,

    comm_id,

    members
):

    by_type = {}

    for m in members:

        t = m.get(

            "type",

            "Unknown"
        )

        by_type.setdefault(
            t,
            []
        ).append(m)

    lines = []

    for etype, items in sorted(
        by_type.items()
    ):

        names = [

            i.get("name","")

            or

            i.get("text","")

            or

            i.get("id","")

            for i in items[:10]
        ]

        lines.append(

            f"{etype} "
            f"({len(items)}): "
            f"{', '.join(names)}"
        )

    member_text = "\n".join(lines)

    prompt = f"""
The following entities belong to a passive safety regulation cluster:

{member_text}

Write a concise technical summary describing:
- what regulation topic this cluster represents
- what crashworthiness or occupant protection area it covers
- what homologation engineer would use this for

Maximum 4 sentences.
"""

    for attempt in range(3):

        try:

            resp = client.chat.completions.create(

                model=EXTRACTION_MODEL,

                temperature=0,

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            "You are a passive safety regulation expert."
                    },

                    {
                        "role":
                            "user",

                        "content":
                            prompt
                    }
                ]
            )

            return (
                resp
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception as e:

            wait = 10 * (attempt + 1)

            logger.warning(
                f"Retry {attempt+1}: {e}"
            )

            time.sleep(wait)

    return (

        f"Community {comm_id} "
        f"contains {len(members)} "
        f"regulation entities."
    )

# ─────────────────────────────────────────────
# WRITE TO NEO4J
# ─────────────────────────────────────────────

def write_to_neo4j(communities):

    driver = GraphDatabase.driver(

        NEO4J_URI,

        auth=(
            NEO4J_USER,
            NEO4J_PASSWORD
        )
    )

    try:

        with driver.session(
            database=NEO4J_DATABASE
        ) as s:

            for comm in communities:

                props = {

                    "id":
                        comm["id"],

                    "size":
                        comm["size"],

                    "regulations":
                        ",".join(
                            comm["regulations"]
                        ),

                    "summary":
                        comm["summary"],

                    "member_ids":
                        ",".join(
                            comm["member_ids"][:50]
                        )
                }

                s.run(
                    """
                    MERGE (c:Community {id:$id})

                    SET c += $props
                    """,

                    id=comm["id"],

                    props=props
                )

                # link entities
                for mid in comm["member_ids"]:

                    try:

                        s.run(
                            """
                            MATCH (e {id:$eid})

                            MATCH (c:Community {id:$cid})

                            MERGE (e)-[:MEMBER_OF]->(c)
                            """,

                            eid=mid,

                            cid=comm["id"]
                        )

                    except Exception:

                        pass

        logger.info(
            f"Wrote "
            f"{len(communities)} "
            f"communities to Neo4j"
        )

    finally:

        driver.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def detect_communities():

    if not ENTITIES_FILE.exists():

        logger.error(
            f"{ENTITIES_FILE} missing"
        )

        return

    with open(
        ENTITIES_FILE,
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    entities = data.get(
        "entities",
        []
    )

    relationships = data.get(
        "relationships",
        []
    )

    # graph
    G = build_nx_graph(
        entities,
        relationships
    )

    partition = run_louvain(G)

    if not partition:

        logger.warning(
            "No communities detected"
        )

        return

    # group members
    comm_members = {}

    ent_map = {

        e["id"]: e

        for e in entities
    }

    for node_id, comm_id in partition.items():

        comm_members.setdefault(
            comm_id,
            []
        )

        if node_id in ent_map:

            comm_members[
                comm_id
            ].append(
                ent_map[node_id]
            )

    # filter
    valid_comms = {

        cid: members

        for cid, members in
        comm_members.items()

        if len(members)
        >= COMMUNITY_MIN_SIZE
    }

    logger.info(

        f"Valid communities "
        f"(size>={COMMUNITY_MIN_SIZE}) → "
        f"{len(valid_comms)}"
    )

    # Groq client
    client = OpenAI(

        api_key=CEREBRAS_API_KEY,

        base_url=CEREBRAS_BASE_URL
    )

    output = []

    sorted_comms = sorted(

        valid_comms.items(),

        key=lambda x: -len(x[1])
    )

    for i, (comm_id, members) in enumerate(

        sorted_comms,

        1
    ):

        regulations = sorted(list(set(

            m.get(
                "regulation",
                "UNKNOWN"
            )

            for m in members
        )))

        logger.info(

            f"[{i}/{len(sorted_comms)}] "
            f"Community {comm_id} "
            f"({len(members)} entities)"
        )

        summary = summarize_community(

            client,

            comm_id,

            members
        )

        output.append({

            "id":
                f"COMM-{comm_id:04d}",

            "size":
                len(members),

            "regulations":
                regulations,

            "summary":
                summary,

            "member_ids": [

                m.get("id","")

                for m in members
            ]
        })

    # save
    with open(
        COMMUNITY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            output,

            f,

            indent=2,

            ensure_ascii=False
        )

    logger.info(
        f"Saved → {COMMUNITY_FILE}"
    )

    # Neo4j
    write_to_neo4j(output)

    logger.info(
        "Community detection complete"
    )

# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":

    detect_communities()