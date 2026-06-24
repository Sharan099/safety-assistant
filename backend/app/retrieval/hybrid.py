"""
Hybrid retriever: dense (Nomic) + BM25 with Reciprocal Rank Fusion.
Vectorized cosine search over precomputed embeddings.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from backend.app.core.settings import (
    BM25_WEIGHT,
    CHUNKS_FILE,
    CLUSTER_CHUNKS_PER_REG,
    COMPARISON_CHUNKS_PER_REG,
    EMBEDDINGS_FILE,
    ENABLE_CLUSTER_RETRIEVAL,
    ENABLE_COMPARISON_RETRIEVAL,
    ENABLE_HARD_METADATA_FILTER,
    ENABLE_METADATA_FILTER,
    ENABLE_MULTI_QUERY,
    ENABLE_NEAR_DUP_SUPPRESSION,
    ENABLE_PARENT_CHILD,
    ENABLE_QUERY_EXPANSION,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_REVISION,
    EMBEDDING_TRUST_REMOTE_CODE,
    METADATA_BOOST,
    NEAR_DUP_SIMILARITY_THRESHOLD,
    RRF_K,
    SEMANTIC_WEIGHT,
    TOP_K_CHUNKS,
    TOP_K_RETRIEVE,
    TOP_K_VECTOR,
    VECTOR_SCORE_THRESHOLD,
)
from backend.app.core.document_registry import (
    cluster_boost_for_regulation,
    cluster_member_codes,
    detect_regulations_in_query,
    is_indexed_regulation,
    regulation_matches_corpus,
)
from backend.app.retrieval.query_expansion import (
    expand_query,
    generate_multi_queries,
    is_comparison_query,
)
from backend.app.retrieval.query_decomposition import decompose_query
from backend.app.retrieval.query_intent import (
    chunk_passes_intent_filter,
    detect_query_intent,
)
from ingestion.embed_chunks import _embedding_prefixes

REG_MAP = {
    "un r14": "UN_R14",
    "un r16": "UN_R16",
    "un r17": "UN_R17",
    "un r94": "UN_R94",
    "un r95": "UN_R95",
    "un r135": "UN_R135",
    "un r137": "UN_R137",
    "fmvss": "FMVSS",
    "euro ncap": "EURO_NCAP",
    "nhtsa ncap": "NCAP_NHTSA",
}

_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def _load_json_artifact(path: Path, *, label: str) -> dict:
    """Load a JSON artifact; fail clearly if Git LFS was not pulled."""
    raw = path.read_text(encoding="utf-8")
    if raw.lstrip().startswith(_LFS_POINTER_PREFIX):
        raise ValueError(
            f"{label} ({path.name}) is a Git LFS pointer, not the real file. "
            "Enable Git LFS on Railway or rebuild the Docker image (Dockerfile "
            "runs git lfs pull for output/regulation_embeddings.json)."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} ({path.name}) is not valid JSON: {exc}") from exc


class HybridRetriever:
    def __init__(
        self,
        *,
        chunks_file: Path | None = None,
        embeddings_file: Path | None = None,
    ) -> None:
        self._chunks_file = chunks_file or CHUNKS_FILE
        self._embeddings_file = embeddings_file or EMBEDDINGS_FILE
        self.chunks: list[dict] = []
        self.embeddings: dict[str, list[float]] = {}
        self._chunk_by_id: dict[str, dict] = {}
        self._model = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self._emb_matrix: np.ndarray | None = None
        self._emb_norms: np.ndarray | None = None
        self._matrix_chunk_ids: list[str] = []
        self._chunk_id_to_emb_idx: dict[str, int] = {}
        self._semantic_disabled = (
            os.getenv("DISABLE_SEMANTIC", "false").lower() == "true"
        )
        self._load()

    def reload(self) -> None:
        """Reload chunk/embedding artifacts (after session ingest completes)."""
        self.chunks = []
        self.embeddings = {}
        self._chunk_by_id = {}
        self._bm25 = None
        self._bm25_chunks = []
        self._emb_matrix = None
        self._load()

    @staticmethod
    def _eligible_chunks(chunks: list[dict]) -> list[dict]:
        """Exclude uploads pending authority-tier confirmation."""
        return [c for c in chunks if c.get("tier_confirmed", True) is not False]

    def _load(self) -> None:
        if self._chunks_file.exists():
            data = _load_json_artifact(self._chunks_file, label="Chunk index")
            self.chunks = self._eligible_chunks(data.get("chunks", []))
            self._chunk_by_id = {
                c.get("chunk_id", ""): c for c in self.chunks if c.get("chunk_id")
            }
            logger.info(f"Loaded {len(self.chunks)} chunks")

        if self._embeddings_file.exists():
            try:
                data = _load_json_artifact(self._embeddings_file, label="Embedding index")
                self.embeddings = data.get("embeddings", {})
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
            except ValueError as exc:
                logger.error(f"{exc} — semantic search disabled; BM25-only fallback")

        self._build_bm25_index()
        self._build_vector_index()

    def _build_bm25_index(self) -> None:
        tokenized = []
        valid = []
        for c in self.chunks:
            txt = (
                f"{c.get('heading_path', '')} "
                f"{c.get('section_title', '')} "
                f"{c.get('text', '')}"
            )
            toks = re.sub(r"[^a-z0-9]", " ", txt.lower()).split()
            if toks:
                tokenized.append(toks)
                valid.append(c)
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_chunks = valid

    def _build_vector_index(self) -> None:
        vectors: list[list[float]] = []
        ids: list[str] = []
        for cid, emb in self.embeddings.items():
            if cid in self._chunk_by_id and emb:
                vectors.append(emb)
                ids.append(cid)
        if not vectors:
            return
        mat = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        self._emb_matrix = mat
        self._emb_norms = mat / norms
        self._matrix_chunk_ids = ids
        self._chunk_id_to_emb_idx = {cid: i for i, cid in enumerate(ids)}
        logger.info(f"Vector index ready: {len(ids)} vectors")

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model (first query may take ~30-60s): {EMBEDDING_MODEL}")
                st_kwargs: dict = {"trust_remote_code": EMBEDDING_TRUST_REMOTE_CODE}
                if EMBEDDING_REVISION:
                    st_kwargs["revision"] = EMBEDDING_REVISION
                self._model = SentenceTransformer(EMBEDDING_MODEL, **st_kwargs)
                logger.info("Embedding model ready")
            except Exception as exc:
                logger.error(f"Embedding model failed: {exc}")
                raise
        return self._model

    def warmup(self) -> None:
        """Load embedding model and run one encode (avoids delay on first chat)."""
        if self._emb_norms is None:
            logger.warning("No vector index — check output/regulation_embeddings.json")
            return
        model = self._get_model()
        model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)
        logger.info("Embedding model warmed up")

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single string with the shared embedding model.

        Exposed so the gateway's semantic cache can REUSE this exact embedding
        model (nomic-embed-text-v1.5) rather than introducing a second one.
        Uses the query prefix so cached prompts share the query embedding space.
        """
        model = self._get_model()
        return model.encode(
            [_embedding_prefixes()[0] + text], convert_to_numpy=True, show_progress_bar=False
        )[0]

    def detect_regs(self, query: str) -> list[str]:
        """Public wrapper around regulation detection (used for cache scoping)."""
        return self._detect_regs(query)

    def _detect_regs(self, query: str) -> list[str]:
        """Detect regulation codes via document registry aliases."""
        return detect_regulations_in_query(query)

    def _chunk_reg_codes(self, chunk: dict) -> set[str]:
        codes: set[str] = set()
        for key in ("regulation", "doc_id"):
            v = chunk.get(key)
            if v:
                codes.add(str(v).upper().replace(" ", "_"))
        return codes

    def _allowed_ids_for_reg(self, reg_code: str) -> set[str]:
        allowed: set[str] = set()
        for cid, c in self._chunk_by_id.items():
            chunk_regs = self._chunk_reg_codes(c)
            if any(regulation_matches_corpus(reg_code, r) for r in chunk_regs):
                allowed.add(cid)
        return allowed

    def _retrieve_hybrid_core(
        self,
        query: str,
        allowed_ids: set[str] | None,
        intent,
    ) -> tuple[list[dict], list[str], int, int]:
        """Run expansion → multi-query → fusion → metadata boost for one allowed set."""
        exp = expand_query(query) if ENABLE_QUERY_EXPANSION else None
        intent_flags = exp.intent_flags if exp else []

        if ENABLE_MULTI_QUERY:
            queries = generate_multi_queries(query, exp)
        elif exp is not None:
            queries = [exp.expanded]
        else:
            queries = [query]

        per_query: list[list[dict]] = []
        semantic_total = 0
        bm25_total = 0
        for q in queries:
            semantic = self._semantic_search(q, allowed_ids)
            bm25 = self._bm25_search(q, allowed_ids)
            semantic_total += len(semantic)
            bm25_total += len(bm25)
            per_query.append(self._rrf_fusion(semantic, bm25, k=RRF_K))

        fused = (
            self._multi_query_fusion(per_query, k=RRF_K)
            if len(per_query) > 1
            else per_query[0]
        )
        fused = self._apply_metadata_boost(fused, intent_flags, intent)
        fused = self._hard_filter_docs(fused, intent)
        return fused, intent_flags, semantic_total, bm25_total

    def _retrieve_balanced_regs(
        self,
        query: str,
        reg_codes: list[str],
        intent,
        per_reg_k: int,
        tag_key: str = "comparison_reg",
        *,
        pool_cap: int | None = None,
    ) -> tuple[list[dict], list[str], int, int]:
        cap = pool_cap or TOP_K_RETRIEVE
        merged: list[dict] = []
        seen: set[str] = set()
        intent_flags: list[str] = []
        semantic_total = 0
        bm25_total = 0
        from dataclasses import replace
        from backend.app.core.document_registry import get_document_meta

        for reg in reg_codes:
            allowed = self._allowed_ids_for_reg(reg)
            if not allowed:
                continue
            reg_meta = get_document_meta(reg)
            reg_intent = intent
            if reg_meta.impact_mode not in ("general", "belt", "seat"):
                reg_intent = replace(intent, test_type=reg_meta.impact_mode)
            fused, flags, sem_n, bm_n = self._retrieve_hybrid_core(
                query, allowed, reg_intent
            )
            semantic_total += sem_n
            bm25_total += bm_n
            for f in flags:
                if f not in intent_flags:
                    intent_flags.append(f)
            for doc in fused[:per_reg_k]:
                if doc["id"] not in seen:
                    seen.add(doc["id"])
                    doc = doc.copy()
                    doc[tag_key] = reg
                    merged.append(doc)

        if len(merged) < cap:
            global_allowed = self._filter_chunk_ids(reg_codes, intent)
            extra, flags, sem_n, bm_n = self._retrieve_hybrid_core(
                query, global_allowed, intent
            )
            semantic_total += sem_n
            bm25_total += bm_n
            for f in flags:
                if f not in intent_flags:
                    intent_flags.append(f)
            for doc in extra:
                if doc["id"] not in seen and len(merged) < cap:
                    seen.add(doc["id"])
                    merged.append(doc)

        merged.sort(key=lambda x: -x.get("score", 0.0))
        return merged[:cap], intent_flags, semantic_total, bm25_total

    def _retrieve_cluster(
        self,
        query: str,
        cluster_name: str,
        intent,
        *,
        per_reg_k: int | None = None,
        pool_cap: int | None = None,
    ) -> tuple[list[dict], list[str], int, int]:
        """Per-member retrieval for requirement clusters (e.g. belt R14+R16+R17)."""
        members = list(cluster_member_codes(cluster_name))
        per_k = per_reg_k if per_reg_k is not None else max(1, CLUSTER_CHUNKS_PER_REG)
        return self._retrieve_balanced_regs(
            query, members, intent, per_k, tag_key="cluster_reg", pool_cap=pool_cap
        )

    def _retrieve_multi_reg(
        self,
        query: str,
        named_regs: list[str],
        intent,
        *,
        per_reg_k: int | None = None,
        pool_cap: int | None = None,
    ) -> tuple[list[dict], list[str], int, int]:
        """Per-regulation sub-retrieval: guarantee chunks from each named regulation."""
        per_k = per_reg_k if per_reg_k is not None else max(1, COMPARISON_CHUNKS_PER_REG)
        return self._retrieve_balanced_regs(
            query, named_regs, intent, per_k, tag_key="comparison_reg", pool_cap=pool_cap
        )

    def _intent_for_regulation(self, intent, reg_code: str):
        from dataclasses import replace
        from backend.app.core.document_registry import get_document_meta
        from backend.app.retrieval.query_intent import DOC_TYPE_RATING

        if reg_code == "EURO_NCAP":
            return replace(
                intent,
                doc_type_intent=DOC_TYPE_RATING,
                value_type_intent="rating_threshold",
                exclude_doc_types=[
                    d for d in intent.exclude_doc_types if d != DOC_TYPE_RATING
                ],
                exclude_value_types=[
                    v for v in intent.exclude_value_types if v != "rating_threshold"
                ],
                binding_authority_only=False,
                compliance_determination=False,
            )

        meta = get_document_meta(reg_code)
        if meta.impact_mode not in ("general", "belt", "seat"):
            return replace(intent, test_type=meta.impact_mode)
        if reg_code in ("UN_R14", "UN_R16") and intent.test_type in (None, "general"):
            return replace(intent, test_type="belt")
        if reg_code == "UN_R17":
            return replace(intent, test_type="seat")
        return intent

    def _filter_chunk_ids(
        self,
        regs: list[str],
        intent=None,
    ) -> set[str] | None:
        if intent and getattr(intent, "_ghost_query", False):
            return set()
        if not regs and not (intent and ENABLE_HARD_METADATA_FILTER):
            return None
        allowed: set[str] = set()
        for cid, c in self._chunk_by_id.items():
            if regs:
                chunk_regs = self._chunk_reg_codes(c)
                if not any(
                    regulation_matches_corpus(r, cr) for r in regs for cr in chunk_regs
                ):
                    continue
            if intent and ENABLE_HARD_METADATA_FILTER:
                if not chunk_passes_intent_filter(c, intent):
                    continue
            allowed.add(cid)
        return allowed if allowed else None

    def _hard_filter_docs(self, docs: list[dict], intent) -> list[dict]:
        """Remove chunks that violate query intent (post-fusion safety net)."""
        if not ENABLE_HARD_METADATA_FILTER or intent is None:
            return docs
        return [
            d for d in docs
            if chunk_passes_intent_filter(self._chunk_by_id.get(d["id"], {}), intent)
        ]

    def _semantic_search(self, query: str, allowed_ids: set[str] | None) -> list[dict]:
        if (
            self._semantic_disabled
            or self._emb_norms is None
            or not self._matrix_chunk_ids
        ):
            return []

        try:
            model = self._get_model()
            # Nomic requires the "search_query: " task prefix on queries.
            q = model.encode(
                [_embedding_prefixes()[0] + query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            qn = q / max(float(np.linalg.norm(q)), 1e-9)
            sims = self._emb_norms @ qn

            if allowed_ids is not None:
                mask = np.array(
                    [cid in allowed_ids for cid in self._matrix_chunk_ids],
                    dtype=bool,
                )
                sims = np.where(mask, sims, -1.0)

            top_idx = np.argpartition(sims, -TOP_K_VECTOR)[-TOP_K_VECTOR :]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

            results = []
            for i in top_idx:
                sim = float(sims[i])
                if sim < VECTOR_SCORE_THRESHOLD:
                    continue
                cid = self._matrix_chunk_ids[i]
                c = self._chunk_by_id[cid]
                results.append(
                    {
                        "id": cid,
                        "score": sim,
                        "semantic_score": sim,
                        "text": c.get("text", ""),
                        "title": c.get("section_title", ""),
                        "heading_path": c.get("heading_path", ""),
                        "regulation": c.get("regulation", ""),
                        "chunk_type": c.get("chunk_type", ""),
                        "parent_id": c.get("parent_id"),
                        "doc_type": c.get("doc_type"),
                        "test_type": c.get("test_type"),
                        "value_type": c.get("value_type"),
                        "doc_id": c.get("doc_id"),
                        "clause": c.get("clause") or c.get("clause_number"),
                        "clause_topic": c.get("clause_topic", "general"),
                        "is_synthetic": c.get("is_synthetic", False),
                        "source": "semantic",
                    }
                )
            return results[:TOP_K_VECTOR]
        except Exception as exc:
            logger.warning(f"Semantic search disabled after error: {exc}")
            self._semantic_disabled = True
            return []

    def _bm25_search(self, query: str, allowed_ids: set[str] | None) -> list[dict]:
        if self._bm25 is None:
            return []

        q_low = query.lower()
        tokens = re.sub(r"[^a-z0-9]", " ", q_low).split()
        if any(term in q_low for term in ("strength", "load", "force", "withstand")):
            tokens.extend(["test", "load", "force", "dan", "traction", "tractive"])
        if any(term in q_low for term in ("chest", "deflection", "thorax", "thcc", "hic", "injury")):
            tokens.extend(["thorax", "compression", "deflection", "mm", "criterion", "shall", "exceed"])
        all_scores = self._bm25.get_scores(tokens)

        ranked = []
        for i, score in enumerate(all_scores):
            chunk = self._bm25_chunks[i]
            cid = chunk.get("chunk_id", "")
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if score > 0:
                text = (
                    f"{chunk.get('heading_path', '')} "
                    f"{chunk.get('section_title', '')} "
                    f"{chunk.get('text', '')}"
                ).lower()
                boost = 1.0

                if any(term in q_low for term in ("strength", "load", "force", "withstand")):
                    if "test load" in text:
                        boost += 2.5
                    if "tractive force" in text:
                        boost += 2.0
                    if "dan" in text:
                        boost += 1.5
                    if "6.4." in text:
                        boost += 1.0
                    if "test in configuration" in text:
                        boost += 0.8

                    # De-prioritize contents / admin sections for requirements questions.
                    if "contents page" in text or "application for approval" in text:
                        boost *= 0.35
                    if "production definitively discontinued" in text:
                        boost *= 0.25
                    if "as from" in text and "contracting parties" in text:
                        boost *= 0.25

                if any(term in q_low for term in ("chest", "deflection", "thorax", "thcc")):
                    if "thorax compression" in text or "thcc" in text:
                        boost += 3.0
                    if "chest deflection" in text:
                        boost += 2.5
                    if "shall not exceed" in text and "mm" in text:
                        boost += 2.0

                # Numeric / unit queries — keyword path must compete with dense search.
                if any(
                    term in q_low
                    for term in (
                        "km/h", "kph", "km h", "mph", "overlap", "angle", "degree",
                        "speed", "velocity", "mm", "percent", "%", "g ", " m/s",
                    )
                ) or re.search(r"\d+(?:[.,]\d+)?\s*(?:%|km/h|kph|mm|°|deg|g)\b", q_low):
                    for num in re.findall(r"\d+(?:[.,]\d+)?", query):
                        if num.replace(",", ".") in text or num in text:
                            boost += 2.0
                    if "km/h" in q_low and ("km/h" in text or "km h" in text):
                        boost += 2.5
                    if "overlap" in q_low and "overlap" in text:
                        boost += 2.0
                    if "context:" in text[:200]:
                        boost += 0.5

                ranked.append((float(score) * boost, i))

        ranked.sort(key=lambda x: -x[0])
        ranked = ranked[:TOP_K_CHUNKS]

        return [
            {
                "id": self._bm25_chunks[i].get("chunk_id", ""),
                "score": float(s),
                "bm25_score": float(s),
                "text": self._bm25_chunks[i].get("text", ""),
                "title": self._bm25_chunks[i].get("section_title", ""),
                "heading_path": self._bm25_chunks[i].get("heading_path", ""),
                "regulation": self._bm25_chunks[i].get("regulation", ""),
                "chunk_type": self._bm25_chunks[i].get("chunk_type", ""),
                "parent_id": self._bm25_chunks[i].get("parent_id"),
                "doc_type": self._bm25_chunks[i].get("doc_type"),
                "test_type": self._bm25_chunks[i].get("test_type"),
                "value_type": self._bm25_chunks[i].get("value_type"),
                "clause_topic": self._bm25_chunks[i].get("clause_topic", "general"),
                "is_synthetic": self._bm25_chunks[i].get("is_synthetic", False),
                "source": "bm25",
            }
            for s, i in ranked
        ]

    @staticmethod
    def _rrf_fusion(
        semantic: list[dict],
        bm25: list[dict],
        k: int = 60,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, doc in enumerate(semantic):
            did = doc["id"]
            scores[did] = scores.get(did, 0) + SEMANTIC_WEIGHT / (k + rank + 1)
            docs[did] = doc

        for rank, doc in enumerate(bm25):
            did = doc["id"]
            scores[did] = scores.get(did, 0) + BM25_WEIGHT / (k + rank + 1)
            docs.setdefault(did, doc)

        merged = []
        for did, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
            d = docs[did].copy()
            d["rrf_score"] = rrf_score
            d["score"] = rrf_score
            merged.append(d)
        return merged[:TOP_K_RETRIEVE]

    @staticmethod
    def _multi_query_fusion(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
        """Fuse ranked lists from several query variants with RRF."""
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}
        for ranked in result_lists:
            for rank, doc in enumerate(ranked):
                did = doc["id"]
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
                docs.setdefault(did, doc)
        merged = []
        for did, sc in sorted(scores.items(), key=lambda x: -x[1]):
            d = docs[did].copy()
            d["mq_score"] = sc
            d["score"] = sc
            merged.append(d)
        return merged

    def _apply_metadata_boost(
        self,
        docs: list[dict],
        intent_flags: list[str],
        intent=None,
    ) -> list[dict]:
        """
        Metadata filtering stage: boost chunks whose metadata flags match the
        query intent and de-prioritize admin / contents / section-preview chunks.
        Soft boost (not a hard filter) so we never zero-out the candidate set.
        """
        if not ENABLE_METADATA_FILTER:
            return docs
        for d in docs:
            chunk = self._chunk_by_id.get(d["id"], {})
            mult = 1.0
            for flag in intent_flags:
                if chunk.get(flag):
                    mult += METADATA_BOOST
            # Prefer concrete paragraph content over section previews.
            if chunk.get("chunk_type") == "section":
                mult *= 0.85
            if chunk.get("doc_type") == "reference":
                mult *= 0.6
            reg = chunk.get("regulation") or chunk.get("doc_id") or ""
            if intent and intent.requirement_cluster:
                mult *= cluster_boost_for_regulation(reg, intent.requirement_cluster)
            if not is_indexed_regulation(reg) and reg and reg not in ("EURO_NCAP", "NCAP_NHTSA") and not str(reg).startswith("NCAP_"):
                mult *= 0.15
            text_low = (chunk.get("text", "") or "").lower()
            if "contents page" in text_low or "application for approval" in text_low:
                mult *= 0.5
            d["score"] = d.get("score", 0.0) * mult
            d["metadata_mult"] = round(mult, 3)
        docs.sort(key=lambda x: -x.get("score", 0.0))
        return docs

    def _expand_parent_child(self, docs: list[dict]) -> list[dict]:
        """
        Parent-Child retrieval: precise child (paragraph) chunks are matched, then
        enriched with their parent section context. Near-duplicate section/paragraph
        variants of the same content are de-duplicated.
        """
        if not ENABLE_PARENT_CHILD:
            return docs

        seen_hashes: set[str] = set()
        deduped: list[dict] = []
        for d in docs:
            chunk = self._chunk_by_id.get(d["id"], {})
            # Normalize body (drop "# heading" / "[heading]" markers) so a section
            # preview and its child paragraph of the same clause collapse to one.
            body = re.sub(r"^[#\[].*?\]?\n+", "", (d.get("text", "") or ""), count=1)
            body_key = re.sub(r"[^a-z0-9]", "", body.lower())[:160]
            h = body_key or chunk.get("chunk_hash") or d.get("id", "")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            parent_id = chunk.get("parent_id")
            if parent_id and parent_id in self._chunk_by_id:
                parent = self._chunk_by_id[parent_id]
                d["parent_id"] = parent_id
                d["parent_heading"] = parent.get("heading_path", "")
                # Attach concise parent context for the reranker/LLM.
                parent_text = (parent.get("text", "") or "").strip()
                if parent_text:
                    d["parent_context"] = parent_text[:600]
            deduped.append(d)
        return deduped

    def _chunk_embedding(self, chunk_id: str) -> np.ndarray | None:
        if self._emb_norms is None:
            return None
        idx = self._chunk_id_to_emb_idx.get(chunk_id)
        if idx is None:
            return None
        return self._emb_norms[idx]

    def _suppress_near_duplicates(
        self,
        docs: list[dict],
        *,
        pool: list[dict] | None = None,
        threshold: float | None = None,
        target_k: int | None = None,
    ) -> list[dict]:
        """
        Drop near-duplicate candidates (high cosine on stored embeddings).
        Backfills from pool so distinct chunks fill freed slots.
        """
        if not ENABLE_NEAR_DUP_SUPPRESSION or len(docs) < 2:
            return docs
        if self._emb_norms is None:
            return docs

        thr = threshold if threshold is not None else NEAR_DUP_SIMILARITY_THRESHOLD
        k = target_k or len(docs)
        seen_ids = {d["id"] for d in docs}
        candidates = list(docs)
        if pool:
            for d in pool:
                if d["id"] not in seen_ids:
                    candidates.append(d)
                    seen_ids.add(d["id"])
        candidates.sort(key=lambda x: -x.get("score", 0.0))

        kept: list[dict] = []
        kept_vecs: list[np.ndarray] = []
        for d in candidates:
            if len(kept) >= k:
                break
            vec = self._chunk_embedding(d["id"])
            if vec is not None and kept_vecs:
                sims = [float(np.dot(vec, kv)) for kv in kept_vecs]
                if max(sims) >= thr:
                    continue
            kept.append(d)
            if vec is not None:
                kept_vecs.append(vec)
        return kept if kept else docs

    def _attach_related_clauses(self, docs: list[dict], max_extra: int = 2) -> list[dict]:
        """Pull linked duration/applicability chunks (e.g. §6.3.3 with §6.4 loads)."""
        if not docs:
            return docs
        present = {d["id"] for d in docs}
        extras: list[dict] = []
        for d in docs:
            chunk = self._chunk_by_id.get(d["id"], {})
            related = chunk.get("related_clause_ids") or []
            if not related:
                continue
            for rel_clause in related:
                if len(extras) >= max_extra:
                    break
                rel_prefix = str(rel_clause).rstrip(".")
                for cid, c in self._chunk_by_id.items():
                    if cid in present or any(e["id"] == cid for e in extras):
                        continue
                    cn = (c.get("clause_number") or "").rstrip(".")
                    if not cn.startswith(rel_prefix):
                        continue
                    if rel_prefix == "6.3.3" and not c.get("has_duration_requirement"):
                        if "0.2 second" not in (c.get("text") or "").lower():
                            continue
                    extras.append({
                        "id": cid,
                        "text": c.get("text", ""),
                        "score": d.get("score", 0.0) * 0.92,
                        "semantic_score": d.get("semantic_score"),
                        "related_to": d["id"],
                        "related_clause": rel_clause,
                    })
                    break
        if not extras:
            return docs
        merged = list(docs)
        for e in extras:
            if e["id"] not in present:
                merged.append(e)
                present.add(e["id"])
        merged.sort(key=lambda x: -x.get("score", 0.0))
        return merged

    def retrieve(self, query: str, mode: str | None = None) -> dict[str, Any]:
        from backend.app.core.modes import get_mode
        from backend.app.retrieval.mode_filter import (
            chunk_passes_mode_filter,
            mode_soft_boost,
            resolve_mode_config,
        )
        from backend.app.retrieval.applicability_boost import applicability_soft_boost
        from backend.app.retrieval.query_breadth import (
            assess_query_breadth,
            effective_fusion_pool_k,
            effective_retrieval_k,
        )

        q_low = query.lower()
        breadth = assess_query_breadth(query)
        fusion_cap = effective_fusion_pool_k(breadth, default_k=TOP_K_RETRIEVE)
        t0 = time.perf_counter()

        intent = detect_query_intent(query)
        named_regs = list(dict.fromkeys(
            self._detect_regs(query) + intent.regulation_codes
        ))
        mode_cfg = resolve_mode_config(
            mode,
            query=query,
            named_regs=named_regs,
            doc_type_intent=intent.doc_type_intent,
        )
        if mode is None and "ncap" in q_low and "euro" not in q_low and "euroncap" not in q_low:
            mode_cfg = get_mode("knowledge_reuse")
        decomp = decompose_query(query)

        use_comparison = (
            ENABLE_COMPARISON_RETRIEVAL
            and len(named_regs) >= 2
            and (is_comparison_query(query) or decomp.is_comparative)
        )
        use_cluster = (
            ENABLE_CLUSTER_RETRIEVAL
            and intent.requirement_cluster
            and len(cluster_member_codes(intent.requirement_cluster)) >= 2
            and not named_regs
        )
        if "ncap" in q_low and "euro" not in q_low and "euroncap" not in q_low:
            use_cluster = False

        if use_comparison:
            per_reg_k = max(1, COMPARISON_CHUNKS_PER_REG)
            if breadth.is_broad:
                per_reg_k = max(per_reg_k, fusion_cap // max(len(named_regs), 1))
            fused, intent_flags, semantic_total, bm25_total = self._retrieve_multi_reg(
                query,
                named_regs,
                intent,
                per_reg_k=per_reg_k,
                pool_cap=fusion_cap,
            )
            queries = [query]
            use_cluster = False
        elif use_cluster:
            per_reg_k = max(1, CLUSTER_CHUNKS_PER_REG)
            if breadth.is_broad:
                members = list(cluster_member_codes(intent.requirement_cluster))
                per_reg_k = max(per_reg_k, fusion_cap // max(len(members), 1))
            fused, intent_flags, semantic_total, bm25_total = self._retrieve_cluster(
                query,
                intent.requirement_cluster,
                intent,
                per_reg_k=per_reg_k,
                pool_cap=fusion_cap,
            )
            queries = [query]
        else:
            exp = expand_query(query) if ENABLE_QUERY_EXPANSION else None
            intent_flags = exp.intent_flags if exp else []
            if ENABLE_MULTI_QUERY:
                queries = generate_multi_queries(query, exp)
            elif exp is not None:
                queries = [exp.expanded]
            else:
                queries = [query]

            if len(decomp.sub_queries) > 1 and not use_comparison:
                queries = decomp.sub_queries
                merged_docs: dict[str, dict] = {}
                semantic_total = 0
                bm25_total = 0
                for sq in decomp.sub_queries:
                    sq_regs = list(dict.fromkeys(
                        self._detect_regs(sq) + detect_regulations_in_query(sq)
                    )) or named_regs
                    allowed = self._filter_chunk_ids(sq_regs, intent)
                    part, flags, sem_t, bm25_t = self._retrieve_hybrid_core(
                        sq, allowed, intent
                    )
                    semantic_total += sem_t
                    bm25_total += bm25_t
                    for f in flags:
                        if f not in intent_flags:
                            intent_flags.append(f)
                    for d in part:
                        cid = d["id"]
                        prev = merged_docs.get(cid)
                        if prev is None or d.get("score", 0) > prev.get("score", 0):
                            merged_docs[cid] = d
                fused = sorted(merged_docs.values(), key=lambda x: -x.get("score", 0.0))
            else:
                active_intent = intent
                if len(named_regs) == 1:
                    active_intent = self._intent_for_regulation(intent, named_regs[0])
                allowed = self._filter_chunk_ids(named_regs, active_intent)
                fused, intent_flags, semantic_total, bm25_total = self._retrieve_hybrid_core(
                    query, allowed, active_intent
                )
            fusion_cap = effective_fusion_pool_k(
                breadth, default_k=TOP_K_RETRIEVE
            )
            fused = fused[:fusion_cap]

        pre_dedup_pool = list(fused)
        target_k = effective_retrieval_k(
            mode_cfg.retrieval_k or TOP_K_RETRIEVE,
            breadth,
            default_pool=TOP_K_RETRIEVE,
        )

        # Mode hard filters
        fused = [
            d for d in fused
            if chunk_passes_mode_filter(self._chunk_by_id.get(d["id"], d), mode_cfg)
        ]
        for d in fused:
            chunk = self._chunk_by_id.get(d["id"], d)
            mult = mode_soft_boost(chunk, mode_cfg, query)
            mult *= applicability_soft_boost(chunk, query)
            if mult != 1.0:
                d["score"] = d.get("score", 0.0) * mult
        fused.sort(key=lambda x: -x.get("score", 0.0))

        dedup_input = fused if breadth.is_broad else fused[:target_k]
        dup_thr = NEAR_DUP_SIMILARITY_THRESHOLD
        if breadth.is_broad:
            dup_thr = min(0.98, dup_thr + 0.04)
        fused = self._suppress_near_duplicates(
            dedup_input,
            pool=pre_dedup_pool,
            target_k=target_k,
            threshold=dup_thr,
        )
        fused = self._attach_related_clauses(fused)

        # Parent-Child Retrieval (dedupe + attach parent context)
        fused = self._expand_parent_child(fused)

        # Best raw semantic similarity in the candidate set (grounding signal).
        sem_scores = [
            d["semantic_score"] for d in fused if d.get("semantic_score") is not None
        ]
        top_semantic_score = max(sem_scores) if sem_scores else None

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"Retrieve done: queries={len(queries)} semantic={semantic_total} "
            f"bm25={bm25_total} fused={len(fused)} intent={intent_flags} "
            f"comparison={use_comparison} cluster={use_cluster} regs={named_regs} "
            f"broad={breadth.is_broad} target_k={target_k} in {latency_ms}ms"
        )
        return {
            "query": query,
            "queries": queries,
            "documents": fused,
            "semantic_count": semantic_total,
            "bm25_count": bm25_total,
            "intent_flags": intent_flags,
            "comparison_retrieval": use_comparison,
            "cluster_retrieval": use_cluster,
            "requirement_cluster": intent.requirement_cluster,
            "named_regulations": named_regs,
            "query_intent": {
                "test_type": intent.test_type,
                "region": intent.region,
                "doc_type_intent": intent.doc_type_intent,
                "value_type_intent": intent.value_type_intent,
            },
            "query_breadth": {
                "is_broad": breadth.is_broad,
                "signals": list(breadth.signals),
                "retrieval_k": target_k,
                "rerank_k": breadth.rerank_k if breadth.is_broad else None,
                "fusion_pool_k": fusion_cap if not use_comparison and not use_cluster else target_k,
            },
            "top_semantic_score": top_semantic_score,
            "latency_ms": latency_ms,
            "mode": mode_cfg.name,
        }
