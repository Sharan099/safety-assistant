"""
retrieval/retriever.py — Passive Safety Regulation Hybrid RAG Retriever

FINAL OPTIMIZED VERSION
- regulation-aware retrieval
- BM25 retrieval
- local vector retrieval
- regulation filtering
- lightweight architecture
- no noisy graph traversal
- fast + grounded responses

Author:
Sharan — Passive Safety RAG
"""

import json
import re
import sys
import time

from pathlib import Path

from loguru import logger

sys.path.insert(
    0,
    str(Path(__file__).parent.parent)
)

from config import (

    CHUNKS_FILE,

    EMBEDDINGS_FILE,

    COMMUNITY_FILE,

    TOP_K_VECTOR,

    TOP_K_CHUNKS,

    TOP_K_COMMUNITIES,

    VECTOR_SCORE_THRESHOLD,

    EMBEDDING_MODEL
)

# =========================================================
# IMPORTANT TERMS
# =========================================================

IMPORTANT_TERMS = [

    "un r14",
    "un r16",
    "un r94",
    "un r95",
    "un r137",

    "fmvss",

    "seat belt",
    "belt anchorage",
    "anchorage",

    "injury criterion",
    "injury criteria",

    "dynamic test",
    "static test",

    "geometry",
    "occupant",
    "restraint",

    "test load",
    "load",

    "crashworthiness",
]

# =========================================================
# REGULATION MAP
# =========================================================

REG_MAP = {

    "un r14": "UN_R14",

    "un r16": "UN_R16",

    "un r17": "UN_R17",

    "un r94": "UN_R94",

    "un r95": "UN_R95",

    "un r137": "UN_R137",

    "fmvss": "FMVSS",
}

# =========================================================
# RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(self, graph_builder=None):

        self.graph = graph_builder

        self.chunks = []

        self.embeddings = {}

        self.communities = []

        self._embedding_model = None

        self._load_artifacts()

    # =====================================================
    # LOAD FILES
    # =====================================================

    def _load_artifacts(self):

        # ---------------------------------------------
        # chunks
        # ---------------------------------------------

        if CHUNKS_FILE.exists():

            with open(
                CHUNKS_FILE,
                encoding="utf-8"
            ) as f:

                self.chunks = json.load(f).get(
                    "chunks",
                    []
                )

            logger.info(
                f"Chunks loaded → "
                f"{len(self.chunks)}"
            )

        # ---------------------------------------------
        # embeddings
        # ---------------------------------------------

        if EMBEDDINGS_FILE.exists():

            with open(
                EMBEDDINGS_FILE,
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                self.embeddings = data.get(
                    "embeddings",
                    {}
                )

            logger.info(
                f"Embeddings loaded → "
                f"{len(self.embeddings)}"
            )

        # ---------------------------------------------
        # communities
        # ---------------------------------------------

        if COMMUNITY_FILE.exists():

            with open(
                COMMUNITY_FILE,
                encoding="utf-8"
            ) as f:

                self.communities = json.load(f)

            logger.info(
                f"Communities loaded → "
                f"{len(self.communities)}"
            )

    # =====================================================
    # EMBEDDING MODEL
    # =====================================================

    def _get_embedding_model(self):

        if self._embedding_model is None:

            try:

                from sentence_transformers import (
                    SentenceTransformer
                )

                self._embedding_model = (
                    SentenceTransformer(
                        EMBEDDING_MODEL
                    )
                )

                logger.info(
                    f"Embedding model loaded → "
                    f"{EMBEDDING_MODEL}"
                )

            except Exception as e:

                logger.warning(
                    f"Embedding model failed: {e}"
                )

        return self._embedding_model

    # =====================================================
    # QUERY KEYWORDS
    # =====================================================

    def _extract_keywords(self, query):

        query_lower = query.lower()

        keywords = []

        for term in IMPORTANT_TERMS:

            if term in query_lower:

                keywords.append(term)

        if not keywords:

            keywords = [

                w

                for w in query_lower.split()

                if len(w) > 4
            ]

        return keywords

    # =====================================================
    # DETECT REGULATIONS
    # =====================================================

    def _detect_regulations(self, query):

        query_lower = query.lower()

        target_regs = []

        for k, v in REG_MAP.items():

            if k in query_lower:

                target_regs.append(v)

        return target_regs

    # =====================================================
    # FILTER CHUNKS
    # =====================================================

    def _filter_chunks(

        self,

        target_regs
    ):

        if not target_regs:

            return self.chunks

        filtered = [

            c

            for c in self.chunks

            if c.get(
                "regulation",
                ""
            ) in target_regs
        ]

        logger.info(
            f"Filtered chunks → "
            f"{len(filtered)}"
        )

        return filtered

    # =====================================================
    # VECTOR SEARCH
    # =====================================================

    def _vector_path(

        self,

        query,

        chunks
    ):

        model = self._get_embedding_model()

        if model is None:
            return []

        if not self.embeddings:
            return []

        try:

            query_vec = model.encode(
                [query],
                convert_to_numpy=True
            )[0]

        except Exception as e:

            logger.warning(
                f"Encoding failed: {e}"
            )

            return []

        try:

            import numpy as np

            q = np.array(query_vec)

            scored = []

            for chunk in chunks:

                cid = chunk.get(
                    "chunk_id",
                    ""
                )

                emb = self.embeddings.get(cid)

                if emb is None:
                    continue

                c = np.array(emb)

                if len(c) != len(q):
                    continue

                denom = (
                    np.linalg.norm(q)
                    *
                    np.linalg.norm(c)
                )

                if denom == 0:
                    continue

                sim = float(
                    np.dot(q, c) / denom
                )

                if sim >= VECTOR_SCORE_THRESHOLD:

                    scored.append(
                        (sim, chunk)
                    )

            scored.sort(
                key=lambda x: -x[0]
            )

            return [

                {

                    "score":
                        sim,

                    "text":
                        c.get(
                            "text",
                            ""
                        ),

                    "title":
                        c.get(
                            "section_title",
                            ""
                        ),

                    "regulation":
                        c.get(
                            "regulation",
                            ""
                        ),

                    "source":
                        "vector",

                    "id":
                        c.get(
                            "chunk_id",
                            ""
                        )
                }

                for sim, c in
                scored[:TOP_K_VECTOR]
            ]

        except Exception as e:

            logger.warning(
                f"Vector search failed: {e}"
            )

            return []

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    def _bm25_path(

        self,

        query,

        chunks
    ):

        try:

            from rank_bm25 import BM25Okapi

        except Exception:

            logger.warning(
                "pip install rank-bm25"
            )

            return []

        tokenized = []

        valid_chunks = []

        for c in chunks:

            txt = (

                c.get(
                    "section_title",
                    ""
                )

                + " "

                +

                c.get(
                    "text",
                    ""
                )
            )

            toks = re.sub(

                r"[^a-z0-9]",

                " ",

                txt.lower()

            ).split()

            if toks:

                tokenized.append(toks)

                valid_chunks.append(c)

        if not tokenized:
            return []

        bm25 = BM25Okapi(tokenized)

        tokens = re.sub(

            r"[^a-z0-9]",

            " ",

            query.lower()

        ).split()

        scores = bm25.get_scores(tokens)

        # ---------------------------------------------
        # BOOSTING
        # ---------------------------------------------

        BOOST_TERMS = [

            "load",
            "force",
            "strength",
            "dynamic",
            "static",
            "test",
            "approval",
            "anchorage",
        ]

        boosted = []

        for i, score in enumerate(scores):

            txt = (

                valid_chunks[i].get(
                    "text",
                    ""
                )

                + " "

                +

                valid_chunks[i].get(
                    "section_title",
                    ""
                )

            ).lower()

            boost = 1.0

            for bt in BOOST_TERMS:

                if (

                    bt in query.lower()

                    and

                    bt in txt
                ):

                    boost += 0.4

            boosted.append(
                score * boost
            )

        ranked = sorted(

            enumerate(boosted),

            key=lambda x: -x[1]
        )

        return [

            {

                "score":
                    float(s),

                "text":
                    valid_chunks[i].get(
                        "text",
                        ""
                    ),

                "title":
                    valid_chunks[i].get(
                        "section_title",
                        ""
                    ),

                "regulation":
                    valid_chunks[i].get(
                        "regulation",
                        ""
                    ),

                "source":
                    "bm25",

                "id":
                    valid_chunks[i].get(
                        "chunk_id",
                        ""
                    )
            }

            for i, s in
            ranked[:TOP_K_CHUNKS]

            if s > 0
        ]

    # =====================================================
    # COMMUNITY SEARCH
    # =====================================================

    def _community_path(self, query):

        if not self.communities:
            return []

        query_low = query.lower()

        scored = []

        for comm in self.communities:

            summary = (

                comm.get(
                    "summary",
                    ""
                )

                + " "

                +

                " ".join(
                    comm.get(
                        "regulations",
                        []
                    )
                )
            )

            score = sum(

                1

                for w in query_low.split()

                if len(w) > 4
                and w in summary.lower()
            )

            if score > 0:

                scored.append(
                    (score, comm)
                )

        scored.sort(
            key=lambda x: -x[0]
        )

        return [

            {

                "score":
                    float(s),

                "text":
                    c.get(
                        "summary",
                        ""
                    ),

                "title":
                    f"Community "
                    f"{c.get('id','')}",

                "regulation":
                    ",".join(
                        c.get(
                            "regulations",
                            []
                        )
                    ),

                "source":
                    "community",

                "id":
                    c.get(
                        "id",
                        ""
                    )
            }

            for s, c in
            scored[:TOP_K_COMMUNITIES]
        ]

    # =====================================================
    # CONTEXT COMPRESSION
    # =====================================================

    def _clean_text(self, text):

        import re

        if not text:
            return ""

    # ---------------------------------------------
    # remove broken markdown hashes
    # ---------------------------------------------

        text = re.sub(
            r"\s*#{2,}\s*",
            ".",
            text
        )

    # ---------------------------------------------
    # fix regulation numbering
    # ---------------------------------------------

        text = re.sub(
            r"(\d)\s*\.\s*#+\s*(\d)",
            r"\1.\2",
            text
        )

        text = re.sub(
            r"(\d)\s*#+\s*(\d)",
            r"\1.\2",
            text
        )

    # ---------------------------------------------
    # remove duplicate periods
    # ---------------------------------------------

        text = re.sub(
            r"\.{2,}",
            ".",
            text
        )

    # ---------------------------------------------
    # normalize spaces
    # ---------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        )

    # ---------------------------------------------
    # clean bullets
    # ---------------------------------------------

        text = text.replace(
            "•",
            "-"
        )

    # ---------------------------------------------
    # clean references
    # ---------------------------------------------

        text = re.sub(
            r"\(\s*([^)]+)\s*\)",
            r"(\1)",
            text
        )

        return text.strip()


    def _compress(self, results):

        lines = []

        for r in results:

            clean_text = self._clean_text(

                r.get(
                    "text",
                    ""
                )
            )

            clean_title = self._clean_text(

                r.get(
                    "title",
                    ""
                )
            )

            lines.append(

                f"\n=== {r.get('regulation','REG')} ===\n"

                f"{clean_title}\n\n"

                f"{clean_text[:700]}"
            )

        return "\n".join(lines)


    # =====================================================
    # RETRIEVE
    # =====================================================

    def retrieve(self, query):

        t0 = time.time()

        # ---------------------------------------------
        # detect regulations
        # ---------------------------------------------

        target_regs = self._detect_regulations(
            query
        )

        # ---------------------------------------------
        # filter chunks
        # ---------------------------------------------

        filtered_chunks = self._filter_chunks(
            target_regs
        )

        # ---------------------------------------------
        # vector
        # ---------------------------------------------

        vector_results = self._vector_path(

            query,

            filtered_chunks
        )

        # ---------------------------------------------
        # bm25
        # ---------------------------------------------

        bm25_results = self._bm25_path(

            query,

            filtered_chunks
        )

        # ---------------------------------------------
        # communities
        # ---------------------------------------------

        community_results = (
            self._community_path(query)
        )

        # ---------------------------------------------
        # merge
        # ---------------------------------------------

        all_results = (

            vector_results

            +

            bm25_results

            +

            community_results
        )

        # ---------------------------------------------
        # deduplicate
        # ---------------------------------------------

        dedup = {}

        for r in all_results:

            rid = r.get("id")

            if not rid:
                continue

            if rid not in dedup:

                dedup[rid] = r

            else:

                if r["score"] > dedup[rid]["score"]:

                    dedup[rid] = r

        merged = sorted(

            dedup.values(),

            key=lambda x:
                -x.get("score",0)
        )

        # ---------------------------------------------
        # top results
        # ---------------------------------------------

        top_results = merged[:8]

        # ---------------------------------------------
        # context
        # ---------------------------------------------

        context = self._compress(
            top_results
        )

        timing = {

            "total_ms":

                round(
                    (time.time()-t0)*1000,
                    1
                ),

            "vector_results":
                len(vector_results),

            "bm25_results":
                len(bm25_results),

            "community_results":
                len(community_results),
        }

        logger.info(

            f"Retrieve → "

            f"{timing['vector_results']} vector | "

            f"{timing['bm25_results']} bm25 | "

            f"{timing['community_results']} community | "

            f"{timing['total_ms']} ms"
        )

        return {

            "query":
                query,

            "context":
                context,

            "entities":
                [],

            "graph_viz":
                {},

            "timing":
                timing
        }