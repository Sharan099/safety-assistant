"""Structure-aware chunking on clause/annex boundaries (FR-20…FR-22)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from loguru import logger

from ingestion.clause_splitter import merge_subclause_blocks, split_body_by_clauses
from ingestion.table_chunker import extract_tables_from_body
from parser.structure_extract import pages_to_structured_blocks
from registry.text_normalize import content_text_hash

DEFAULT_MAX_CLAUSE = 1800
DEFAULT_OVERLAP = 200


class StructureAwareChunker:
    def __init__(
        self,
        max_clause_chars: int = DEFAULT_MAX_CLAUSE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.max_clause_chars = max_clause_chars
        self.overlap = overlap

    def chunk_document(
        self,
        pages: list[dict[str, Any]],
        regulation_metadata: dict[str, Any],
        document_name: str,
    ) -> list[dict[str, Any]]:
        blocks = pages_to_structured_blocks(pages)
        chunks: list[dict[str, Any]] = []
        chunk_idx = 0
        local_id_map: dict[str, int] = {}

        reg_code = regulation_metadata.get("regulation_code", "UNKNOWN")
        amendment = regulation_metadata.get("amendment") or "Base"
        source_type = regulation_metadata.get("source_type", "INTERNAL")

        for block in blocks:
            section_local = f"sec_{block['section_id']}_{chunk_idx}"
            section_text = block["text"]
            if not section_text.strip():
                continue

            if block.get("block_type") == "clause":
                clause_blocks = [
                    (block["section_id"], "", section_text, block.get("page_start")),
                ]
                section_chunk = None
            else:
                section_chunk = self._make_chunk(
                    text=section_text[: min(len(section_text), self.max_clause_chars)],
                    chunk_index=chunk_idx,
                    chunk_type="section",
                    section_id=block["section_id"],
                    heading_path=block["heading_path"],
                    page_number=block.get("page_start"),
                    regulation_metadata=regulation_metadata,
                    document_name=document_name,
                    local_id=section_local,
                    parent_local_id=None,
                )
                clause_blocks = merge_subclause_blocks(
                    split_body_by_clauses(
                        section_text,
                        page_number=block.get("page_start"),
                    )
                )

            if section_chunk is not None:
                chunks.append(section_chunk)
                local_id_map[section_local] = chunk_idx
                chunk_idx += 1

            for clause_num, clause_title, clause_body, clause_page in clause_blocks:
                if not clause_body.strip():
                    continue

                clause_local = f"cl_{clause_num or 'general'}_{chunk_idx}"
                remainder, tables = extract_tables_from_body(
                    clause_body,
                    clause_number=clause_num,
                    section_title=clause_title,
                )

                for part_idx, part_text in enumerate(
                    self._split_oversized(remainder or clause_body)
                ):
                    parent = section_local if section_chunk is not None else None
                    ctype = "clause" if clause_num else "paragraph"
                    cl_id = clause_local if part_idx == 0 else f"{clause_local}_p{part_idx}"
                    chunk = self._make_chunk(
                        text=part_text,
                        chunk_index=chunk_idx,
                        chunk_type=ctype,
                        section_id=clause_num or block["section_id"],
                        heading_path=f"{block['heading_path']} > {clause_title}".strip(" >"),
                        page_number=clause_page or block.get("page_start"),
                        regulation_metadata=regulation_metadata,
                        document_name=document_name,
                        local_id=cl_id,
                        parent_local_id=parent,
                    )
                    chunks.append(chunk)
                    local_id_map[cl_id] = chunk_idx
                    chunk_idx += 1

                for table in tables:
                    tbl_local = table.table_id
                    tbl_text = table.markdown
                    if table.preamble:
                        tbl_text = f"{table.preamble}\n\n{tbl_text}"
                    if table.summary:
                        tbl_text = f"{table.summary}\n\n{tbl_text}"
                    chunk = self._make_chunk(
                        text=tbl_text,
                        chunk_index=chunk_idx,
                        chunk_type="table",
                        section_id=clause_num or block["section_id"],
                        heading_path=f"{block['heading_path']} > {table.summary or 'Table'}",
                        page_number=clause_page or block.get("page_start"),
                        regulation_metadata=regulation_metadata,
                        document_name=document_name,
                        local_id=tbl_local,
                        parent_local_id=clause_local if clause_num else (section_local if section_chunk is not None else None),
                    )
                    chunks.append(chunk)
                    local_id_map[tbl_local] = chunk_idx
                    chunk_idx += 1

        for ch in chunks:
            parent_local = ch.pop("parent_local_id", None)
            ch.pop("local_id", None)
            if parent_local and parent_local in local_id_map:
                ch["parent_chunk_index"] = local_id_map[parent_local]

        logger.info(
            "Structure chunker produced %s chunks for %s (%s)",
            len(chunks),
            document_name,
            reg_code,
        )
        return chunks

    def _split_oversized(self, text: str) -> list[str]:
        if len(text) <= self.max_clause_chars:
            return [text] if text.strip() else []
        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_clause_chars)
            parts.append(text[start:end])
            if end >= len(text):
                break
            start += self.max_clause_chars - self.overlap
        return parts

    def _make_chunk(
        self,
        *,
        text: str,
        chunk_index: int,
        chunk_type: str,
        section_id: str | None,
        heading_path: str,
        page_number: int | None,
        regulation_metadata: dict,
        document_name: str,
        local_id: str,
        parent_local_id: str | None,
    ) -> dict[str, Any]:
        reg_code = regulation_metadata.get("regulation_code", "UNKNOWN")
        amendment = regulation_metadata.get("amendment") or "Base"
        source_type = regulation_metadata.get("source_type", "INTERNAL")
        header = (
            f"[Source: {source_type} | Reg: {reg_code} | Amendment: {amendment} | "
            f"Doc: {document_name} | Page: {page_number} | Section: {section_id} | "
            f"Type: {chunk_type}]\n\n"
        )
        body = text.strip()
        full_text = header + body
        return {
            "chunk_text": full_text,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "section": section_id,
            "paragraph": section_id if chunk_type == "clause" else None,
            "chunk_type": chunk_type,
            "heading_path": heading_path,
            "content_hash": content_text_hash(body),
            "local_id": local_id,
            "parent_local_id": parent_local_id,
            "metadata": {
                "authority": source_type,
                "regulation_id": reg_code,
                "amendment": amendment,
                "document": document_name,
                "chunk_type": chunk_type,
                "heading_path": heading_path,
                "section_id": section_id,
                "content_hash": content_text_hash(body),
            },
        }
