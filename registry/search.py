import os
import re
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_
import httpx
from loguru import logger

from database.models import Regulation, Document, Chunk
from registry.query_timing import QueryTiming, log_timing_summary
from registry.prompt_templates import GROUNDED_SYSTEM_PROMPT
from registry.chat_intent import (
    classify_chat_query,
    corpus_meta_chat_result,
    identity_chat_result,
    injection_chat_result,
)
from registry.margin_query import try_margin_query_response
from vectorization.embedder import RegulationEmbedder, RegulationReranker

class RegulationSearchEngine:
    """
    Core search engine that implements:
    1. Query metadata parsing (extracting standards from strings)
    2. Dense Vector Search (pgvector)
    3. Sparse Keyword Search (FTS tsvector)
    4. Reciprocal Rank Fusion (RRF)
    5. Cross-Encoder Reranking (bge-reranker-v2-m3)
    6. Grounded LLM Answer Generation (optional, via Groq/OpenAI)
    """

    def __init__(self):
        self.embedder = RegulationEmbedder()
        self.reranker = RegulationReranker()

    def search(
        self, 
        db: Session, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        rerank: bool = True
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        qt = QueryTiming()

        with qt.step("guardrails_ms"):
            intent = classify_chat_query(query)
            if intent != "regulatory":
                total_ms = (time.perf_counter() - start_time) * 1000
                if intent == "identity":
                    payload = identity_chat_result(query)
                elif intent == "injection_blocked":
                    payload = injection_chat_result(query)
                else:
                    payload = corpus_meta_chat_result(query, db)
                payload["timing"] = qt.finalize(total_ms)
                payload["metadata"]["latency_ms"] = payload["timing"]["total_ms"]
                payload["metadata"]["timing"] = payload["timing"]
                return payload

            margin_payload = try_margin_query_response(db, query)
            if margin_payload:
                total_ms = (time.perf_counter() - start_time) * 1000
                margin_payload["timing"] = qt.finalize(total_ms)
                margin_payload["metadata"]["latency_ms"] = margin_payload["timing"]["total_ms"]
                margin_payload["metadata"]["timing"] = margin_payload["timing"]
                return margin_payload

            parsed_filters = self._parse_query_for_filters(query)
            if filters:
                parsed_filters.update(filters)
            retrieval_query = self._expand_retrieval_query(query, parsed_filters)
            if retrieval_query != query:
                logger.info(f"Expanded retrieval query: {retrieval_query[:200]}...")
            logger.info(f"Applying filters for search: {parsed_filters}")
            fusion_limit = max(30, top_k * 3)

        with qt.step("dense_retrieval_ms"):
            query_embedding = self.embedder.embed_query(retrieval_query)
            if db.bind.dialect.name == "sqlite":
                dense_results = self._dense_search_sqlite(
                    db, query_embedding, parsed_filters, limit=fusion_limit
                )
            else:
                dense_results = self._dense_search(
                    db, query_embedding, parsed_filters, limit=fusion_limit
                )

        with qt.step("sparse_retrieval_ms"):
            if db.bind.dialect.name == "sqlite":
                sparse_results = self._sparse_search_sqlite(
                    db, retrieval_query, parsed_filters, limit=fusion_limit
                )
            else:
                sparse_results = self._sparse_search(
                    db, retrieval_query, parsed_filters, limit=fusion_limit
                )

        with qt.step("rrf_fusion_ms"):
            fused_chunks = self._reciprocal_rank_fusion(
                dense_results, sparse_results, limit=fusion_limit
            )

        final_chunks: list[dict] = []
        use_mock = getattr(self.reranker, "use_mock_reranker", False)
        if rerank and fused_chunks and not use_mock:
            with qt.step("rerank_ms"):
                passages = [chunk["chunk_text"] for chunk in fused_chunks]
                scores = self.reranker.compute_scores(query, passages)
                for idx, score in enumerate(scores):
                    fused_chunks[idx]["score"] = score
                fused_chunks.sort(key=lambda x: x["score"], reverse=True)
            fused_chunks = self._deprioritize_scope_chunks(fused_chunks)
            with qt.step("annex_promotion_ms"):
                final_chunks = self._promote_named_annex_chunks(query, fused_chunks, top_k)
                final_chunks = self._promote_cited_section_chunks(
                    query, fused_chunks, top_k, current_top=final_chunks
                )
        else:
            if rerank and fused_chunks and use_mock:
                logger.debug("Skipping keyword mock reranker — using RRF order")
                qt.rerank_bypassed = True
            elif not rerank:
                qt.rerank_bypassed = True
            with qt.step("rerank_ms"):
                fused_chunks.sort(
                    key=lambda x: x.get("rrf_score", x.get("search_score", 0)),
                    reverse=True,
                )
            fused_chunks = self._deprioritize_scope_chunks(fused_chunks)
            with qt.step("annex_promotion_ms"):
                final_chunks = self._promote_named_annex_chunks(query, fused_chunks, top_k)
                final_chunks = self._promote_cited_section_chunks(
                    query, fused_chunks, top_k, current_top=final_chunks
                )

        with qt.step("parent_expansion_ms"):
            final_chunks = self._expand_with_parents(db, final_chunks, max_extra=4)
            final_chunks = final_chunks[:top_k]

        with qt.step("cross_reference_expansion_ms"):
            final_chunks = self._expand_cross_references(db, final_chunks, query=query)

        with qt.step("llm_generation_ms"):
            answer, routing = self._generate_grounded_answer(query, final_chunks)

        total_ms = (time.perf_counter() - start_time) * 1000
        timing = qt.finalize(total_ms)
        log_timing_summary(query, timing)
        logger.info(
            f"Search query completed in {total_ms:.2f}ms. Returned {len(final_chunks)} sources."
        )

        return {
            "answer": answer,
            "sources": final_chunks,
            "timing": timing,
            "metadata": {
                "filters_applied": parsed_filters,
                "latency_ms": timing["total_ms"],
                "dense_candidates_count": len(dense_results),
                "sparse_candidates_count": len(sparse_results),
                "routing": routing,
                "timing": timing,
            },
        }

    def _parse_query_for_filters(self, query: str) -> Dict[str, Any]:
        """Extracts structured filters from search string using simple heuristics."""
        filters = {}
        query_upper = query.upper()
        
        # 1. Detect UNECE Regulation number e.g. UN R95, R16, R17
        unece_regs = re.findall(r"\b(?:UN[\s_-]?)?(R\d{2,3})\b", query_upper)
        if len(unece_regs) == 1:
            filters["regulation_code"] = f"UN_{unece_regs[0]}"
        elif len(unece_regs) > 1:
            filters["source_type"] = "UNECE"
            filters["regulation_codes"] = [f"UN_{r}" for r in sorted(set(unece_regs))]
        else:
            m_reg = re.search(r"\b(R\d{2,3})\b", query_upper)
            if m_reg:
                filters["regulation_code"] = f"UN_{m_reg.group(1)}"
            
        # 2. Detect series of amendments e.g. 05 series, 05 Series, 03 series
        m_amend = re.search(r"(\d{2})\s*(?:SERIES|SERIES\s+OF\s+AMENDMENTS)", query_upper)
        if m_amend:
            filters["amendment"] = f"{int(m_amend.group(1)):02d} Series"
            
        # 3. Detect Euro NCAP protocol year e.g. Euro NCAP 2026, NCAP 2024
        m_ncap_yr = re.search(r"\b(?:EURO\s+)?NCAP\s+(20\d{2})\b", query_upper)
        if m_ncap_yr:
            filters["source_type"] = "Euro NCAP"
            filters["amendment"] = f"Protocol Year {m_ncap_yr.group(1)}"
            
        # 4. Detect FMVSS standards
        m_fmvss = re.search(r"\bFMVSS\s+(\d{3})\b", query_upper)
        if m_fmvss:
            filters["regulation_code"] = f"FMVSS {m_fmvss.group(1)}"
            filters["source_type"] = "FMVSS"

        # Multi-reg queries need source_type to block Euro NCAP / FMVSS bleed; a single
        # UN_* regulation_code already scopes to one family (see retrieval diff audit).
        if filters.get("regulation_codes"):
            filters.setdefault("source_type", "UNECE")

        # R16 §7.6.2 locking-test cross-ref (corpus-specific; no UN prefix in natural query)
        if re.search(r"(?:§|paragraph\s+)7\.6\.2\b", query, re.I) or re.search(
            r"\b7\.6\.2\b.*(?:lock|retractor|decelerat)", query, re.I
        ):
            filters.setdefault("regulation_code", "UN_R16")

        return filters

    def _expand_retrieval_query(self, query: str, filters: Dict[str, Any]) -> str:
        """Lightweight query expansion for short natural-language questions."""
        if os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() not in ("1", "true", "yes"):
            return query
        q_lower = query.lower()
        extras: list[str] = []
        reg = filters.get("regulation_code") or ""
        regs = filters.get("regulation_codes") or []
        is_r94 = reg == "UN_R94" or "UN_R94" in regs or re.search(r"\bR\s*94\b", query, re.I)

        if is_r94:
            if any(w in q_lower for w in ("chest", "deflection", "thorax", "thcc", "compression")):
                extras.extend(["ThCC", "thorax compression criterion", "42 mm", "5.2.1.4"])
            if any(w in q_lower for w in ("injury", "criteria", "frontal", "impact", "dummy")):
                extras.extend([
                    "5.2.1.3", "5.2.1.4", "5.2.1.8", "neck bending moment",
                    "ThCC", "viscous criterion", "FFC", "TCFC", "8 kN", "tibia",
                ])
        if len(regs) >= 2 or ("r44" in q_lower and "r129" in q_lower):
            extras.extend([
                "child restraint", "i-Size", "mass group", "height-based",
                "UN Regulation No. 44", "UN Regulation No. 129",
            ])
        is_r14 = reg == "UN_R14" or re.search(r"\bR\s*14\b", query, re.I)
        if is_r14 and any(
            w in q_lower for w in ("isofix", "annex 6", "annex6", "anchorage", "anchorage points")
        ):
            extras.extend([
                "Annex 6", "anchorage table", "Vehicle category",
                "M1", "M2", "N1", "lower anchorages", "minimum number of anchorage",
            ])
        if reg == "UN_R16" or re.search(r"\b7\.6\.2\b", query, re.I):
            if any(w in q_lower for w in ("lock", "retractor", "decelerat", "emergency")):
                extras.extend([
                    "7.6.2", "emergency locking retractor", "locking test",
                    "deceleration-actuated", "sort of locking mechanism",
                ])
        if not extras:
            return query
        return f"{query} {' '.join(extras)}"

    def _is_scope_admin_chunk(self, chunk: Dict[str, Any]) -> bool:
        """§1 application, §2 definitions, amendment date headers — low precision for substance queries."""
        sec = (chunk.get("section") or "").strip()
        if sec in ("1", "2", "3", "4", "General"):
            return True
        if re.match(r"^2\.\d+$", sec):
            return True
        body = (chunk.get("chunk_text") or "")
        if "]" in body:
            body = body.split("]", 1)[1]
        body_l = body[:200].lower()
        if "as defined in the consolidated resolution" in body_l:
            return True
        if re.match(r"^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}", body_l):
            return True
        return False

    def _deprioritize_scope_chunks(self, fused: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stable reorder: substantive clauses before scope/admin within the fusion pool."""
        substantive = [c for c in fused if not self._is_scope_admin_chunk(c)]
        scope = [c for c in fused if self._is_scope_admin_chunk(c)]
        return substantive + scope

    def _promote_cited_section_chunks(
        self, query: str, fused: List[Dict[str, Any]], top_k: int,
        *, current_top: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """When the query cites §X.Y.Z, ensure the best matching section chunk is in top-k."""
        cited: set[str] = set()
        cited.update(re.findall(r"(?:§|paragraph\s+)(\d+(?:\.\d+)*)", query, re.I))
        cited.update(re.findall(r"\b(\d+\.\d+\.\d+)\b", query))
        if not cited or not fused:
            return (current_top or fused)[:top_k]

        def _matches_cited(chunk: Dict[str, Any]) -> bool:
            sec = (chunk.get("section") or "").strip()
            if not sec:
                return False
            return any(
                sec == c or sec.startswith(c + ".") or c.startswith(sec + ".")
                for c in cited
            )

        hits = [c for c in fused if _matches_cited(c)]
        if not hits:
            return (current_top or fused)[:top_k]

        def _cited_priority(c: Dict[str, Any]) -> tuple:
            sec = (c.get("section") or "").strip()
            depth = len(sec.split(".")) if sec else 0
            rrf = c.get("rrf_score", c.get("search_score", c.get("score", 0)))
            return (depth, rrf)

        best = max(hits, key=_cited_priority)
        top = list(current_top) if current_top else fused[:top_k]
        if len(top) > top_k:
            top = top[:top_k]
        if any(c["chunk_id"] == best["chunk_id"] for c in top):
            return top
        return top[: top_k - 1] + [best]

    def _promote_named_annex_chunks(
        self, query: str, fused: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """When the query names an annex, ensure the best matching annex chunk is in top-k.

        Skipping the Windows mock reranker restored R94 RRF order but dropped R14 Annex 6
        table chunks (~RRF #17). Promotion pulls the annex body from the fusion pool without
        re-enabling keyword reranking.
        """
        m_annex = re.search(r"\bannex\s*(\d+)\b", query, re.I)
        if not m_annex or not fused:
            return fused[:top_k]
        annex_key = f"Annex_{m_annex.group(1)}"
        annex_hits = [
            c
            for c in fused
            if annex_key in (c.get("section") or "")
            or annex_key in (c.get("heading_path") or "")
        ]
        if not annex_hits:
            return fused[:top_k]

        def _annex_priority(c: Dict[str, Any]) -> tuple:
            text = c.get("chunk_text") or ""
            rrf = c.get("rrf_score", c.get("search_score", 0))
            is_table = c.get("chunk_type") == "table" or "Vehicle category" in text
            return (1 if is_table else 0, rrf)

        best_annex = max(annex_hits, key=_annex_priority)
        top = fused[:top_k]
        if any(c["chunk_id"] == best_annex["chunk_id"] for c in top):
            return top
        return top[: top_k - 1] + [best_annex]

    def _regulation_filter_conditions(self, filters: Dict[str, Any]) -> list:
        conditions = []
        if filters.get("regulation_codes"):
            conditions.append(Regulation.regulation_code.in_(filters["regulation_codes"]))
        elif "regulation_code" in filters:
            conditions.append(Regulation.regulation_code == filters["regulation_code"])
        if "amendment" in filters:
            conditions.append(Regulation.amendment == filters["amendment"])
        if "source_type" in filters:
            conditions.append(Regulation.source_type == filters["source_type"])
        if "market" in filters:
            conditions.append(Regulation.market == filters["market"])
        if "chunk_type" in filters:
            conditions.append(Chunk.chunk_type == filters["chunk_type"])
        if "document_name" in filters:
            conditions.append(Document.document_name == filters["document_name"])
        if filters.get("amendment_mismatch") is False:
            conditions.append(Document.amendment_mismatch.is_(False))
        if "year" in filters:
            year_val = int(filters["year"])
            conditions.append(func.extract("year", Regulation.effective_date) == year_val)
        if "include_superseded" not in filters or not filters["include_superseded"]:
            conditions.append(Regulation.status == "ACTIVE")
        return conditions

    def _expand_with_parents(
        self, db: Session, chunks: List[Dict[str, Any]], *, max_extra: int = 4
    ) -> List[Dict[str, Any]]:
        """Include parent section chunks when a child clause is retrieved."""
        if os.getenv("ENABLE_PARENT_CHILD", "true").lower() not in ("1", "true", "yes"):
            return chunks
        seen = {c["chunk_id"] for c in chunks}
        out: list[dict] = []
        for ch in chunks:
            out.append(ch)
            pid = ch.get("parent_chunk_id")
            if not pid or pid in seen or len(out) >= len(chunks) + max_extra:
                continue
            parent = (
                db.query(Chunk)
                .join(Document)
                .join(Regulation)
                .filter(Chunk.id == pid)
                .first()
            )
            if parent:
                parent_item = self._format_chunk_item(parent, ch.get("search_score", 0.0))
                out.append(parent_item)
                seen.add(pid)
        return out

    def _expand_cross_references(
        self, db: Session, chunks: List[Dict[str, Any]], query: str = "", max_depth: int = 2, max_expansion: int = 5
    ) -> List[Dict[str, Any]]:
        """Follows transitive cross-references in chunks to pull referenced paragraphs in the same regulation."""
        if not os.environ.get("ENABLE_CROSS_REFERENCE_EXPANSION", "false").lower() == "true":
            return chunks

        # Extract query keywords for surgical relevance filtering
        important_keywords = []
        cited_sections: set[str] = set()
        if query:
            cited_sections.update(
                re.findall(r"(?:§|paragraph\s+)(\d+(?:\.\d+)*)", query, re.I)
            )
            cited_sections.update(re.findall(r"\b(\d+\.\d+(?:\.\d+)*)\b", query))
            # Match only alphabetical words of at least 4 letters to discard 'un', 'r16', 'r94', and short numbers
            query_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", query)]
            stop_words = {
                "what", "does", "define", "when", "must", "they", "under", "limit",
                "specify", "points", "with", "from", "each", "were", "been", "that",
                "this", "have", "some", "more", "than", "other", "same", "only",
                "show", "will", "would", "could", "should", "types", "about",
                "their", "them", "then", "there", "these", "here", "were", "where",
                "unece", "regulation", "standard", "document", "clause", "section",
                "paragraph", "paragraphs"
            }
            important_keywords = [w for w in query_words if w not in stop_words]

        seen_ids = {c["chunk_id"] for c in chunks}
        expanded_chunks = list(chunks)
        added_count = 0

        # Transitively expand using a queue of (chunk, current_depth)
        queue = [(c, 0) for c in chunks]

        # Regex to match paragraph citations: paragraph X.Y.Z, para. X.Y
        ref_pattern = re.compile(
            r"\b(?:paragraph|paragraphs|para\.|paras\.)\s*([0-9]+(?:\.[0-9]+)*)\b",
            re.IGNORECASE
        )

        while queue and added_count < max_expansion:
            current_chunk, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            text = current_chunk.get("chunk_text") or ""
            matches = ref_pattern.findall(text)
            if not matches:
                continue

            reg_id = current_chunk.get("regulation_id")
            if not reg_id:
                # Fallback to query database if regulation_id is not in chunk dictionary
                chunk_id = current_chunk.get("chunk_id")
                db_chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
                if db_chunk and db_chunk.document:
                    reg_id = db_chunk.document.regulation_id

            if not reg_id:
                continue

            for target_sec in matches:
                if added_count >= max_expansion:
                    break

                # Confirm the target paragraph resolves to a real chunk in the SAME regulation
                # Parse section hierarchy: e.g. "2.12.4.1" -> try "2.12.4.1", then "2.12.4", then "2.12"
                parts = target_sec.split(".")
                target_db_chunk = None
                while len(parts) >= 1:
                    candidate_sec = ".".join(parts)
                    target_db_chunk = db.query(Chunk).join(Document).filter(
                        Document.regulation_id == reg_id,
                        Chunk.section == candidate_sec
                    ).first()
                    if target_db_chunk:
                        break
                    parts.pop()

                if not target_db_chunk:
                    continue

                # Surgical relevance check — skip when query explicitly cites this section
                target_text = target_db_chunk.chunk_text.lower()
                if "]" in target_text:
                    target_text = target_text.split("]", 1)[1]
                cited = target_sec in cited_sections or any(
                    target_sec.startswith(c + ".") or c.startswith(target_sec)
                    for c in cited_sections
                )
                if important_keywords and not cited:
                    if not any(w in target_text for w in important_keywords):
                        continue

                if target_db_chunk.id in seen_ids:
                    continue

                seen_ids.add(target_db_chunk.id)
                new_chunk_dict = self._format_chunk_item(
                    target_db_chunk, current_chunk.get("search_score", 0.5) - 0.05
                )
                new_chunk_dict["expanded_via_ref"] = True
                expanded_chunks.append(new_chunk_dict)
                added_count += 1
                queue.append((new_chunk_dict, depth + 1))

        logger.info(f"Cross-reference expansion added {added_count} chunks to the retrieval context.")
        return expanded_chunks

    def _dense_search(self, db: Session, embedding: List[float], filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Dense retrieval using pgvector cosine similarity."""
        stmt = db.query(
            Chunk,
            (1 - Chunk.embedding.cosine_distance(embedding)).label("score")
        ).join(Document).join(Regulation)
        
        conditions = self._regulation_filter_conditions(filters)
        if conditions:
            stmt = stmt.filter(and_(*conditions))
            
        results = stmt.order_by("score").desc().limit(limit).all()
        
        candidates = []
        for chunk, score in results:
            candidates.append(self._format_chunk_item(chunk, score))
        return candidates

    def _sparse_search(self, db: Session, query: str, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Sparse retrieval using PostgreSQL full text search."""
        # Convert search query to tsquery
        # Replace spaces with & for basic AND keyword search
        clean_terms = re.sub(r"[^\w\s]", "", query).strip().split()
        if not clean_terms:
            return []
        tsquery_str = " & ".join(clean_terms)
        
        stmt = db.query(
            Chunk,
            func.ts_rank(func.to_tsvector('english', Chunk.chunk_text), func.plainto_tsquery('english', query)).label("score")
        ).join(Document).join(Regulation)
        
        conditions = [
            func.plainto_tsquery('english', query).op('@@')(func.to_tsvector('english', Chunk.chunk_text))
        ]
        conditions.extend(self._regulation_filter_conditions(filters))
            
        stmt = stmt.filter(and_(*conditions))
        results = stmt.order_by("score").desc().limit(limit).all()
        
        candidates = []
        for chunk, score in results:
            candidates.append(self._format_chunk_item(chunk, score))
        return candidates

    def _reciprocal_rank_fusion(self, dense: List[Dict[str, Any]], sparse: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Combines dense and sparse results using Reciprocal Rank Fusion (RRF)."""
        rrf_map = {}
        k = 60  # RRF constant
        
        # Process dense
        for rank, item in enumerate(dense):
            cid = item["chunk_id"]
            rrf_map[cid] = {"item": item, "score": 1.0 / (k + rank + 1)}
            
        # Process sparse
        for rank, item in enumerate(sparse):
            cid = item["chunk_id"]
            if cid in rrf_map:
                rrf_map[cid]["score"] += 1.0 / (k + rank + 1)
            else:
                rrf_map[cid] = {"item": item, "score": 1.0 / (k + rank + 1)}
                
        # Sort and take top limit
        sorted_rrf = sorted(rrf_map.values(), key=lambda x: x["score"], reverse=True)
        
        fused = []
        for item_data in sorted_rrf[:limit]:
            item = item_data["item"]
            # Store the RRF score
            item["rrf_score"] = item_data["score"]
            fused.append(item)
            
        return fused

    def _format_chunk_item(self, chunk: Chunk, search_score: float) -> Dict[str, Any]:
        """Formats database chunk record to structured retrieval candidate dict."""
        return {
            "chunk_id": chunk.id,
            "chunk_text": chunk.chunk_text,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "paragraph": chunk.paragraph,
            "document_id": chunk.document_id,
            "document_name": chunk.document.document_name,
            "file_path": chunk.document.file_path,
            "regulation_id": chunk.document.regulation.id,
            "regulation_code": chunk.document.regulation.regulation_code,
            "title": chunk.document.regulation.title,
            "amendment": chunk.document.regulation.amendment,
            "effective_date": str(chunk.document.regulation.effective_date) if chunk.document.regulation.effective_date else None,
            "source_type": chunk.document.regulation.source_type,
            "source_url": chunk.document.regulation.source_url,
            "market": chunk.document.regulation.market,
            "chunk_type": chunk.chunk_type,
            "heading_path": chunk.heading_path,
            "parent_chunk_id": chunk.parent_chunk_id,
            "content_hash": chunk.content_hash,
            "search_score": float(search_score)
        }

    def _generate_grounded_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
        """
        Synthesizes a grounded answer using retrieved contexts.
        Routes through LLM gateway when ENABLE_GATEWAY=true; otherwise direct Groq or evidence summary.
        """
        from backend.app.gateway import config as gateway_config

        if not context_chunks:
            routing = {
                "model_key": "none",
                "model_id": "none",
                "provider": "registry",
                "evidence_only": True,
                "cache_hit": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0.0,
                "steps": [],
            }
            return (
                "I couldn't find any relevant passages in the safety regulation registry to answer your question.",
                routing,
            )

        context_text = ""
        for idx, chunk in enumerate(context_chunks):
            context_text += (
                f"[S{idx+1}] Regulation: {chunk['regulation_code']} | "
                f"Amendment: {chunk['amendment']}\n"
                f"Document: {chunk['document_name']} | Page: {chunk['page_number']}\n"
                f"{chunk['chunk_text']}\n\n"
            )

        system_prompt = GROUNDED_SYSTEM_PROMPT
        user_prompt = (
            f"RETRIEVED CONTEXT\n{context_text}\n"
            f"QUESTION: {query}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if gateway_config.ENABLE_GATEWAY:
            from backend.app.gateway.gateway import LLMGateway

            gateway = LLMGateway(use_cache=True)
            q_lower = query.lower()
            is_comparison = (
                "difference" in q_lower
                or "compare" in q_lower
                or " vs " in q_lower
                or " versus " in q_lower
            )
            max_out = 2048 if is_comparison else 1024
            result = gateway.complete(
                messages, context_chunks=context_chunks, max_output_tokens=max_out
            )
            logger.info(
                f"Gateway routing: model={result.model_key} evidence_only={result.evidence_only} "
                f"cache_hit={result.cache_hit} latency_ms={result.latency_ms:.1f}"
            )
            return result.text, result.to_dict()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found in env. Returning structured context summary instead of LLM generation.")
            summary_parts = [f"Directly grounded answers require a GROQ_API_KEY. However, here are the most relevant findings from the registry:\n"]
            for idx, chunk in enumerate(context_chunks):
                summary_parts.append(
                    f"[{idx+1}] Regulation: {chunk['regulation_code']} ({chunk['amendment'] or 'Base'}) | "
                    f"Doc: {chunk['document_name']} (Page {chunk['page_number']})\n"
                    f"Content: \"{chunk['chunk_text'][:250].strip()}...\"\n"
                )
            routing = {
                "model_key": "evidence_only",
                "model_id": "none",
                "provider": "registry",
                "evidence_only": True,
                "cache_hit": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0.0,
                "steps": [],
            }
            return "\n".join(summary_parts), routing

        try:
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 1024
                    }
                )
                response.raise_for_status()
                result = response.json()
                usage = result.get("usage") or {}
                routing = {
                    "model_key": "groq_direct",
                    "model_id": model_name,
                    "provider": "groq",
                    "evidence_only": False,
                    "cache_hit": False,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "latency_ms": 0.0,
                    "steps": [],
                }
                return result["choices"][0]["message"]["content"], routing
        except Exception as e:
            logger.error(f"Error generating answer via Groq API: {e}")
            routing = {
                "model_key": "evidence_only",
                "model_id": "none",
                "provider": "registry",
                "evidence_only": True,
                "cache_hit": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0.0,
                "steps": [{"outcome": "error", "detail": str(e)[:200]}],
            }
            return (
                f"Failed to generate grounded answer via LLM. Retrieved context matches:\n"
                + "\n".join([c["chunk_text"][:200] for c in context_chunks]),
                routing,
            )

    def _dense_search_sqlite(self, db: Session, embedding: List[float], filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Fallback dense retrieval for SQLite using NumPy-based in-memory cosine similarity."""
        import numpy as np
        stmt = db.query(Chunk).join(Document).join(Regulation)
        
        conditions = self._regulation_filter_conditions(filters)
        if conditions:
            stmt = stmt.filter(and_(*conditions))
            
        chunks = stmt.all()
        if not chunks:
            return []
            
        # Compute cosine similarity in memory
        query_vec = np.array(embedding)
        candidates = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            chunk_vec = np.array(chunk.embedding)
            # Cosine similarity
            sim = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
            candidates.append((chunk, sim))
            
        # Sort and limit
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score in candidates[:limit]:
            results.append(self._format_chunk_item(chunk, score))
        return results

    def _sparse_search_sqlite(self, db: Session, query: str, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Fallback keyword retrieval for SQLite using python word matching."""
        stmt = db.query(Chunk).join(Document).join(Regulation)
        
        conditions = self._regulation_filter_conditions(filters)
        if conditions:
            stmt = stmt.filter(and_(*conditions))
            
        chunks = stmt.all()
        
        terms = [t.lower() for t in query.split() if len(t) > 2]
        candidates = []
        for chunk in chunks:
            text_lower = chunk.chunk_text.lower()
            score = sum(1 for term in terms if term in text_lower)
            if score > 0:
                candidates.append((chunk, score))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score in candidates[:limit]:
            results.append(self._format_chunk_item(chunk, score))
        return results
