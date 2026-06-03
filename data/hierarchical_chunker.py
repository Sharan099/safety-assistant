"""
hierarchical_chunker.py — Hierarchical chunking from Docling Markdown.

Structure:
  document -> section (heading) -> paragraph chunks (with overlap)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CHUNKS_FILE,
    HIER_CHUNK_OVERLAP,
    HIER_CHUNK_WORDS,
    HIER_MIN_CHUNK_WORDS,
    MARKDOWN_DIR,
)

sys.stdout.reconfigure(line_buffering=True)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CLAUSE_RE = re.compile(
    r"^(?:[-•*]\s*)?(\d+(?:\.\d+)+)[\.\s:_-]+(.+?)\s*$",
    re.I,
)
FRONTMATTER_RE = re.compile(r"^---\s*$")

TEST_RE = re.compile(r"\b(test|testing|dynamic test|static test)\b", re.I)
LOAD_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:daN|kN|N)\b", re.I)
ANGLE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:degree|degrees|°)\b", re.I)
DISTANCE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m)\b", re.I)
VEHICLE_RE = re.compile(r"\b(M1|M2|M3|N1|N2|N3)\b")
REQ_RE = re.compile(r"\b(shall|must|required|requirement)\b", re.I)
BELT_RE = re.compile(r"\b(anchorages?|seat.?belt|retractor|pretensioner)\b", re.I)
INJURY_RE = re.compile(r"\b(HIC|chest deflection|femur force|neck force|thorax)\b", re.I)


def p(msg: str) -> None:
    print(msg, flush=True)


def detect_regulation_type(filename: str) -> str:
    fname = filename.lower()
    mapping = {
        "UN_R14": ["un_r14", "r14.pdf", "_r14", "regulation no. 14", "add.13"],
        "UN_R16": ["un_r16", "r16.pdf", "_r16", "regulation no. 16", "ece-r-16"],
        "UN_R17": ["un_r17", "r17"],
        "UN_R94": ["un_r94", "r94"],
        "UN_R95": ["un_r95", "r95"],
        "UN_R135": ["un_r135", "r135"],
        "UN_R137": ["un_r137", "r137"],
        "FMVSS": ["fmvss", "571"],
        "EURO_NCAP": ["euro-ncap", "euroncap", "ncap"],
        "ISO": ["iso"],
        "CAE_REFERENCE": ["cae"],
        "SAFETY_REFERENCE": ["safety", "passive"],
    }
    for reg, keys in mapping.items():
        if any(k in fname for k in keys):
            return reg
    return "SAFETY_REFERENCE"


def detect_features(text: str) -> dict:
    return {
        "has_test_procedure": bool(TEST_RE.search(text)),
        "has_loads": bool(LOAD_RE.search(text)),
        "has_angles": bool(ANGLE_RE.search(text)),
        "has_distances": bool(DISTANCE_RE.search(text)),
        "has_vehicle_classes": bool(VEHICLE_RE.search(text)),
        "has_requirements": bool(REQ_RE.search(text)),
        "has_belt_system": bool(BELT_RE.search(text)),
        "has_injury_criteria": bool(INJURY_RE.search(text)),
        "load_count": len(LOAD_RE.findall(text)),
        "angle_count": len(ANGLE_RE.findall(text)),
        "distance_count": len(DISTANCE_RE.findall(text)),
        "vehicle_count": len(VEHICLE_RE.findall(text)),
    }


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def _parse_frontmatter(lines: list[str]) -> tuple[dict, list[str]]:
    meta: dict = {}
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return meta, lines
    body = []
    in_fm = True
    for line in lines[1:]:
        if in_fm and FRONTMATTER_RE.match(line):
            in_fm = False
            continue
        if in_fm and ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        elif not in_fm:
            body.append(line)
    return meta, body if body else lines


@dataclass
class SectionNode:
    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if text.strip() else []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        piece = " ".join(words[start:end]).strip()
        if len(piece.split()) >= HIER_MIN_CHUNK_WORDS or start == 0:
            chunks.append(piece)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def _parse_markdown_sections(md_text: str) -> list[SectionNode]:
    lines = md_text.splitlines()
    _, body_lines = _parse_frontmatter(lines)

    root = SectionNode(level=0, title="ROOT")
    stack: list[SectionNode] = [root]
    current = root

    for line in body_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--"):
            continue

        m = HEADING_RE.match(stripped)
        cm = None if m else CLAUSE_RE.match(stripped)

        if m or cm:
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
            else:
                clause_num = cm.group(1)
                level = min(6, clause_num.count(".") + 1)
                title = f"{clause_num} {cm.group(2).strip()[:100]}"

            node = SectionNode(level=level, title=title)
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
            current = node
        else:
            current.lines.append(line)

    def flatten(node: SectionNode, path: list[str]) -> list[tuple[list[str], str, int, str]]:
        out: list[tuple[list[str], str, int, str]] = []
        path = path + [node.title] if node.title != "ROOT" else path
        if node.lines and node.title != "ROOT":
            body = "\n".join(node.lines).strip()
            if body:
                out.append((path, node.title, node.level, body))
        for child in node.children:
            out.extend(flatten(child, path))
        return out

    return flatten(root, [])


def _file_slug(md_path: Path) -> str:
    slug = md_path.stem.upper().replace("-", "_").replace(" ", "_")
    return slug[:24]


def _make_chunk(
    *,
    regulation: str,
    file_slug: str,
    pdf_name: str,
    markdown_file: str,
    chunk_type: str,
    parent_id: str | None,
    heading_path: str,
    section_title: str,
    section_level: int,
    text: str,
    seq: int,
    section_idx: int,
) -> dict:
    features = detect_features(text)
    chunk_id = f"{regulation}-{file_slug}-H{section_idx:04d}-C{seq:03d}"
    return {
        "chunk_id": chunk_id,
        "chunk_hash": short_hash(text),
        "regulation": regulation,
        "pdf_name": pdf_name,
        "markdown_file": markdown_file,
        "chunk_type": chunk_type,
        "parent_id": parent_id,
        "heading_path": heading_path,
        "section_id": f"{regulation}-{file_slug}-H{section_idx:04d}",
        "section_title": section_title,
        "section_level": section_level,
        "text": text,
        "word_count": len(text.split()),
        "chunk_seq": seq,
        **features,
    }


def chunk_markdown_file(md_path: Path) -> list[dict]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    meta, _ = _parse_frontmatter(text.splitlines())
    pdf_name = meta.get("source_pdf", md_path.stem + ".pdf")
    regulation = meta.get("regulation") or detect_regulation_type(pdf_name)
    file_slug = _file_slug(md_path)

    sections = _parse_markdown_sections(text)
    all_chunks: list[dict] = []
    global_seq = 0

    for sec_idx, (path, title, level, body) in enumerate(sections, 1):
        heading_path = " > ".join(path)
        section_chunk_id = f"{regulation}-{file_slug}-H{sec_idx:04d}-SEC"

        # Section-level parent chunk (heading + short preview)
        preview_words = body.split()[:120]
        preview = " ".join(preview_words)
        section_text = f"# {heading_path}\n\n{preview}"
        section_chunk = _make_chunk(
            regulation=regulation,
            file_slug=file_slug,
            pdf_name=pdf_name,
            markdown_file=md_path.name,
            chunk_type="section",
            parent_id=None,
            heading_path=heading_path,
            section_title=title,
            section_level=level,
            text=section_text,
            seq=0,
            section_idx=sec_idx,
        )
        section_chunk["chunk_id"] = section_chunk_id
        all_chunks.append(section_chunk)

        # Leaf paragraph chunks under section
        leaf_parts = _split_words(body, HIER_CHUNK_WORDS, HIER_CHUNK_OVERLAP)
        for i, part in enumerate(leaf_parts, 1):
            leaf_text = f"[{heading_path}]\n\n{part}"
            chunk = _make_chunk(
                regulation=regulation,
                file_slug=file_slug,
                pdf_name=pdf_name,
                markdown_file=md_path.name,
                chunk_type="paragraph",
                parent_id=section_chunk_id,
                heading_path=heading_path,
                section_title=title,
                section_level=level,
                text=leaf_text,
                seq=i,
                section_idx=sec_idx,
            )
            chunk["global_seq"] = global_seq
            global_seq += 1
            all_chunks.append(chunk)

    return all_chunks


def run(only_regs: list[str] | None = None) -> dict:
    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    if only_regs:
        allowed = {x.upper() for x in only_regs}
        md_files = [
            m
            for m in md_files
            if m.stem.upper().replace("-", "_") in allowed
            or detect_regulation_type(m.name).upper() in allowed
        ]
        # Prefer exact stems (UN_R14.md) over alias files (ECE-R-16-Regulation.md)
        exact = [m for m in md_files if m.stem.upper().replace("-", "_") in allowed]
        if exact:
            md_files = exact
    if not md_files:
        p(f"No markdown files in {MARKDOWN_DIR}. Run data/docling_converter.py first.")
        sys.exit(1)

    all_chunks: list[dict] = []
    stats: dict[str, int] = {}

    for idx, md_path in enumerate(md_files, 1):
        chunks = chunk_markdown_file(md_path)
        all_chunks.extend(chunks)
        reg = chunks[0]["regulation"] if chunks else "UNKNOWN"
        stats[reg] = stats.get(reg, 0) + len(chunks)
        p(f"[{idx}/{len(md_files)}] {md_path.name} -> {len(chunks)} chunks")

    dataset = {
        "pipeline": "docling_hierarchical",
        "total_chunks": len(all_chunks),
        "source_markdown_files": len(md_files),
        "chunks": all_chunks,
        "stats_by_regulation": stats,
    }

    CHUNKS_FILE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    p(f"\nSaved {len(all_chunks)} hierarchical chunks -> {CHUNKS_FILE}")
    return dataset


if __name__ == "__main__":
    run()
