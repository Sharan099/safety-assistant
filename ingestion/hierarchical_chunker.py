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
from collections import defaultdict
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
# Docling sometimes emits "6.4.1 3." instead of "6.4.1.3."
CLAUSE_SPACED_RE = re.compile(
    r"^(?:[-•*]\s*)?((?:\d+\.)+\d+)\s+(\d+)\.?\s*(.*)$",
    re.I,
)
TEXT_SECTION_RE = re.compile(
    r"^(Section|Part|Annex|Article|Appendix)\s+([IVXLCDM]+|\d+)\s*[:.]?\s*(.*)$",
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
        "UN_R16": ["un_r16", "r16.pdf", "_r16", "regulation no. 16"],
        "UN_R17": ["un_r17", "r17.pdf", "_r17"],
        "UN_R94": ["un_r94", "r94.pdf", "_r94"],
        "UN_R95": ["un_r95", "r95.pdf", "_r95"],
        "UN_R135": ["un_r135", "r135.pdf", "_r135"],
        "UN_R137": ["un_r137", "r137.pdf", "_r137"],
        "UN_R127": ["un_r127", "r127", "_r127"],
        "FMVSS": ["fmvss", "571", "208"],
        "EURO_NCAP": ["euro-ncap", "euroncap", "euro_ncap", "ncap"],
        "CAE_REFERENCE": ["cae_companion", "cae-companion", "cae companion"],
        "SAFETY_REFERENCE": ["safety_companion", "safety-companion", "safety companion"],
        "PROG_X_FT_001": ["prog_x_ft_001", "ft-prog-x-001"],
        "PROG_X_FT_002": ["prog_x_ft_002", "ft-prog-x-002"],
        "PROG_X_CAE_001": ["prog_x_cae_001"],
        "PROG_X_CAE_002": ["prog_x_cae_002"],
        "PROG_X_RCA_001": ["prog_x_rca", "rca-prog-x"],
        "PROG_X_DR": ["prog_x_dr", "design_review"],
        "PROG_X_STATUS": ["prog_x_status", "project_status"],
        "NCAP": ["ncap_"],
    }
    for reg, keys in mapping.items():
        if any(k in fname for k in keys):
            return reg
    if fname.startswith("ncap_"):
        return "NCAP"
    if "cae" in fname:
        return "CAE_REFERENCE"
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
    clause_number: str | None = None
    heading_source: str | None = None
    lines: list[str] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries: '. ', '? ', '! ', or newline."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text.split()) <= size:
        return [text]

    sentences = _split_sentences(text)
    from ingestion.applicability_enrichment import bond_load_duration_sentences

    sentences = bond_load_duration_sentences(sentences)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def finalize() -> str | None:
        nonlocal current, current_words
        if not current:
            return None
        chunk = " ".join(current).strip()
        current = []
        current_words = 0
        return chunk or None

    def overlap_prefix(prev_chunk: str) -> list[str]:
        if overlap <= 0:
            return []
        tail_words = prev_chunk.split()[-overlap:]
        if not tail_words:
            return []
        return [" ".join(tail_words)]

    for sent in sentences:
        sent_words = len(sent.split())

        # Dense single-sentence clause — keep whole, never mid-sentence split.
        if sent_words > size:
            finished = finalize()
            if finished and (len(finished.split()) >= HIER_MIN_CHUNK_WORDS or not chunks):
                chunks.append(finished)
            chunks.append(sent)
            continue

        if current_words + sent_words > size and current:
            finished = finalize()
            if finished:
                if len(finished.split()) >= HIER_MIN_CHUNK_WORDS or not chunks:
                    chunks.append(finished)
                current = overlap_prefix(finished)
                current_words = len(current[0].split()) if current else 0

        current.append(sent)
        current_words += sent_words

    finished = finalize()
    if finished and (len(finished.split()) >= HIER_MIN_CHUNK_WORDS or not chunks):
        chunks.append(finished)

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
        csm = None if (m or cm) else CLAUSE_SPACED_RE.match(stripped)
        tm = None if (m or cm or csm) else TEXT_SECTION_RE.match(stripped)

        if m or cm or csm or tm:
            clause_num: str | None = None
            heading_source: str | None = None
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
            elif cm:
                clause_num = cm.group(1)
                level = min(6, clause_num.count(".") + 1)
                title = f"{clause_num} {cm.group(2).strip()[:100]}"
            elif csm:
                clause_num = f"{csm.group(1).rstrip('.')}.{csm.group(2)}"
                level = min(6, clause_num.count(".") + 1)
                tail = csm.group(3).strip()[:100]
                title = f"{clause_num} {tail}".strip()
            else:
                title = f"{tm.group(1)} {tm.group(2)}: {tm.group(3)}".strip(": ").strip()
                level = 2
                heading_source = "text_section"

            node = SectionNode(
                level=level,
                title=title,
                clause_number=clause_num,
                heading_source=heading_source,
            )
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
            current = node
        else:
            current.lines.append(line)

    def flatten(
        node: SectionNode, path: list[str]
    ) -> list[tuple[list[str], str, int, str, str | None, str | None]]:
        out: list[tuple[list[str], str, int, str, str | None, str | None]] = []
        path = path + [node.title] if node.title != "ROOT" else path
        if node.lines and node.title != "ROOT":
            body = "\n".join(node.lines).strip()
            if body:
                out.append(
                    (
                        path,
                        node.title,
                        node.level,
                        body,
                        node.clause_number,
                        node.heading_source,
                    )
                )
        for child in node.children:
            out.extend(flatten(child, path))
        return out

    return flatten(root, [])


def _file_slug(md_path: Path) -> str:
    """Unique per markdown file — avoids collisions when stems share a 24-char prefix."""
    stem = md_path.stem.upper().replace("-", "_").replace(" ", "_")
    digest = hashlib.sha1(md_path.name.encode("utf-8")).hexdigest()[:8]
    prefix = stem[:20]
    return f"{prefix}_{digest}"


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
    clause_number: str | None = None,
) -> dict:
    from ingestion.metadata_classifier import classify_chunk

    features = detect_features(text)
    chunk_id = f"{regulation}-{file_slug}-H{section_idx:04d}-C{seq:03d}"
    meta = classify_chunk(
        regulation=regulation,
        pdf_name=pdf_name,
        text=text,
        clause_number=clause_number,
        heading_path=heading_path,
        section_title=section_title,
    )
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
        "clause_number": clause_number,
        "text": text,
        "word_count": len(text.split()),
        "chunk_seq": seq,
        **features,
        **meta,
    }


def _prepend_chunk_header(
    text: str,
    *,
    doc_id: str,
    revision: str | None,
    clause_number: str | None,
    section_title: str,
) -> str:
    rev = revision or "unknown"
    clause = clause_number or section_title or "n/a"
    header = f"[{doc_id} | {rev} | {clause}]\n"
    if header.strip() in text[:80]:
        return text
    return header + text


def _normalize_clause_number(clause_number: str | None, section_title: str) -> str | None:
    """Fix Docling headings like '6.4.1 3.' -> clause 6.4.1.3."""
    spaced = re.match(r"^(\d+(?:\.\d+)+)\s+(\d+)\.?\s*", section_title or "")
    if spaced:
        return f"{spaced.group(1).rstrip('.')}.{spaced.group(2)}"
    return clause_number


def chunk_markdown_file(md_path: Path) -> list[dict]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    meta, _ = _parse_frontmatter(text.splitlines())
    pdf_name = meta.get("source_pdf", md_path.stem + ".pdf")
    regulation = meta.get("regulation") or detect_regulation_type(pdf_name)
    if regulation == "SAFETY_REFERENCE" and meta.get("regulation"):
        regulation = meta["regulation"]
    file_slug = _file_slug(md_path)

    from ingestion.event_chunker import chunk_event_document, is_event_document

    if is_event_document(md_path, meta, regulation):
        def _mk(**kwargs):
            c = _make_chunk(**kwargs)
            c["text"] = _prepend_chunk_header(
                c["text"],
                doc_id=regulation,
                revision=meta.get("revision", "synthetic"),
                clause_number=kwargs.get("clause_number"),
                section_title=kwargs.get("section_title", ""),
            )
            return c

        events = chunk_event_document(
            md_path, text, meta,
            make_chunk=_mk,
            regulation=regulation,
            file_slug=file_slug,
            pdf_name=pdf_name,
        )
        if events:
            return events

    sections = _parse_markdown_sections(text)
    from ingestion.applicability_enrichment import (
        enrich_section_body,
        extract_r14_duration_snippet,
        prepend_applicability_to_chunk_text,
    )

    duration_snippet = (
        extract_r14_duration_snippet(sections) if regulation == "UN_R14" else None
    )
    all_chunks: list[dict] = []
    global_seq = 0

    from ingestion.annex_table_enrichment import enrich_annex_chunks

    for sec_idx, (path, title, level, body, clause_number, heading_source) in enumerate(
        sections, 1
    ):
        enriched_body, applicability_meta = enrich_section_body(
            regulation=regulation,
            clause_number=_normalize_clause_number(clause_number, title),
            section_title=title,
            body=body,
            duration_snippet=duration_snippet,
        )
        clause_number = _normalize_clause_number(clause_number, title)
        body = enriched_body
        heading_path = " > ".join(path)
        section_chunk_id = f"{regulation}-{file_slug}-H{sec_idx:04d}-SEC"

        body_word_count = len(body.split())
        is_truncated_parent = body_word_count > HIER_CHUNK_WORDS * 3
        if is_truncated_parent:
            section_body = " ".join(body.split()[:120])
        else:
            section_body = body
        section_text = f"# {heading_path}\n\n{section_body}"
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
            clause_number=clause_number,
        )
        section_chunk["chunk_id"] = section_chunk_id
        section_chunk["is_truncated_parent"] = is_truncated_parent
        if heading_source:
            section_chunk["heading_source"] = heading_source
        if applicability_meta:
            section_chunk.update(applicability_meta)
        all_chunks.append(section_chunk)

        # Leaf paragraph chunks under section
        leaf_parts = _split_words(body, HIER_CHUNK_WORDS, HIER_CHUNK_OVERLAP)
        for i, part in enumerate(leaf_parts, 1):
            leaf_part = part
            if applicability_meta and "APPLICABILITY:" not in part[:120]:
                leaf_part = prepend_applicability_to_chunk_text(
                    part, applicability_meta, duration_snippet
                )
            leaf_text = f"[{heading_path}]\n\n{leaf_part}"
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
                text=_prepend_chunk_header(
                    leaf_text,
                    doc_id=regulation,
                    revision=meta.get("revision"),
                    clause_number=clause_number,
                    section_title=title,
                ),
                seq=i,
                section_idx=sec_idx,
                clause_number=clause_number,
            )
            if applicability_meta:
                chunk.update(applicability_meta)
            chunk["global_seq"] = global_seq
            global_seq += 1
            all_chunks.append(chunk)

        # Self-sufficient combined-context chunk for §6.4 load clauses (angle + duration inline).
        if regulation == "UN_R14" and clause_number and str(clause_number).startswith("6.4"):
            from ingestion.clause_dependencies import build_denormalized_block

            denorm = build_denormalized_block(clause_number, regulation)
            if denorm:
                combined_body = f"{denorm}\n\n{body}"
                combined_text = f"[{heading_path}]\n\n{combined_body}"
                combined = _make_chunk(
                    regulation=regulation,
                    file_slug=file_slug,
                    pdf_name=pdf_name,
                    markdown_file=md_path.name,
                    chunk_type="combined_context",
                    parent_id=section_chunk_id,
                    heading_path=heading_path,
                    section_title=title,
                    section_level=level,
                    text=_prepend_chunk_header(
                        combined_text,
                        doc_id=regulation,
                        revision=meta.get("revision"),
                        clause_number=clause_number,
                        section_title=title,
                    ),
                    seq=0,
                    section_idx=sec_idx,
                    clause_number=clause_number,
                )
                if applicability_meta:
                    combined.update(applicability_meta)
                combined["global_seq"] = global_seq
                global_seq += 1
                all_chunks.append(combined)

    from ingestion.table_structure_enrichment import enrich_table_chunks

    return enrich_table_chunks(enrich_annex_chunks(all_chunks, md_path), md_path)


def run(only_regs: list[str] | None = None) -> dict:
    before_total = 0
    before_with_clause = 0
    if CHUNKS_FILE.exists():
        try:
            prev = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
            prev_chunks = prev.get("chunks", [])
            before_total = len(prev_chunks)
            before_with_clause = sum(1 for c in prev_chunks if c.get("clause_number"))
        except (json.JSONDecodeError, OSError):
            pass

    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    for corpus_sub in ("synthetic", "historical"):
        src_dir = Path(__file__).resolve().parents[1] / "data" / "corpus" / corpus_sub
        if not src_dir.is_dir():
            continue
        pattern = "NCAP_*.md" if corpus_sub == "historical" else "*.md"
        for src_path in sorted(src_dir.glob(pattern)):
            dest = MARKDOWN_DIR / src_path.name
            if not dest.exists():
                dest.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        if corpus_sub == "historical":
            md_files = sorted(set(md_files) | set(MARKDOWN_DIR.glob("NCAP_*.md")))
        else:
            md_files = sorted(set(md_files) | set(MARKDOWN_DIR.glob("PROG_X_*.md")))
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
        p(f"No markdown files in {MARKDOWN_DIR}. Run ingestion/docling_converter.py first.")
        sys.exit(1)

    all_chunks: list[dict] = []
    stats: dict[str, int] = {}

    for idx, md_path in enumerate(md_files, 1):
        chunks = chunk_markdown_file(md_path)
        all_chunks.extend(chunks)
        reg = chunks[0]["regulation"] if chunks else "UNKNOWN"
        stats[reg] = stats.get(reg, 0) + len(chunks)
        p(f"[{idx}/{len(md_files)}] {md_path.name} -> {len(chunks)} chunks")

    chunk_ids = [c["chunk_id"] for c in all_chunks if (c.get("text") or "").strip()]
    unique_ids = set(chunk_ids)
    if len(chunk_ids) != len(unique_ids):
        from collections import Counter

        dupes = [cid for cid, n in Counter(chunk_ids).items() if n > 1]
        p(f"ERROR: {len(chunk_ids) - len(unique_ids)} duplicate chunk_id(s) — aborting save")
        for cid in dupes[:10]:
            files = sorted(
                {c["markdown_file"] for c in all_chunks if c.get("chunk_id") == cid}
            )
            p(f"  {cid} <- {files}")
        sys.exit(1)

    dataset = {
        "pipeline": "docling_hierarchical",
        "total_chunks": len(all_chunks),
        "unique_chunk_ids": len(unique_ids),
        "source_markdown_files": len(md_files),
        "chunks": all_chunks,
        "stats_by_regulation": stats,
    }

    CHUNKS_FILE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    p(f"\nSaved {len(all_chunks)} hierarchical chunks -> {CHUNKS_FILE}")

    with_clause = sum(1 for c in all_chunks if c.get("clause_number"))
    truncated_parents = sum(
        1
        for c in all_chunks
        if c.get("chunk_type") == "section" and c.get("is_truncated_parent")
    )
    text_section_by_reg: dict[str, int] = defaultdict(int)
    for c in all_chunks:
        if c.get("chunk_type") == "section" and c.get("heading_source") == "text_section":
            text_section_by_reg[c.get("regulation") or "UNKNOWN"] += 1

    p(f"Sanity check: total_chunks={len(all_chunks)}")
    p(f"  chunks with clause_number: {with_clause}")
    p(f"  section chunks is_truncated_parent=true: {truncated_parents}")

    p("\nBefore/after comparison (TEXT_SECTION_RE):")
    p(f"  total chunks:        {before_total:>6} -> {len(all_chunks):>6}  (delta {len(all_chunks) - before_total:+d})")
    p(
        f"  with clause_number:  {before_with_clause:>6} -> {with_clause:>6}  "
        f"(delta {with_clause - before_with_clause:+d})"
    )
    text_section_total = sum(text_section_by_reg.values())
    p(f"  text_section sections (new): {text_section_total}")
    if text_section_by_reg:
        p("  text_section sections by regulation:")
        for reg, count in sorted(text_section_by_reg.items(), key=lambda x: x[1], reverse=True):
            p(f"    {reg:<22} {count:>5}")

    return dataset


if __name__ == "__main__":
    run()
