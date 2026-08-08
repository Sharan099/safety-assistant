"""Prepend a one-line context header to each chunk before embedding."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from ingestion.models import Chunk

_LIMIT_ROLE_RE = re.compile(
    r"(?ix)("
    r"shall\s+not\s+exceed|"
    r"less\s+than\s+or\s+equal|"
    r"performance\s+criteria|"
    r"injury\s+criter(?:ion|ia)|"
    r"no\s+(?:liquid\s+)?electrolyte\s+leakage|"
    r"rib\s+deflection\s+criterion|"
    r"soft\s+tissue\s+criterion|"
    r"head\s+performance\s+criterion"
    r")"
)
_PROCEDURE_ROLE_RE = re.compile(
    r"(?ix)("
    r"procedure\s+for\s+calculat|"
    r"is\s+calculated\s+as|"
    r"peak\s+viscous\s+response|"
    r"test\s+procedure|"
    r"measurements?\s+to\s+be\s+made|"
    r"the\s+mobile\s+deformable\s+barrier|"
    r"dummy\s+shall\s+be|"
    r"instrumentation|"
    r"calibrat"
    r")"
)
_DEFINITION_ROLE_RE = re.compile(
    r'(?ix)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"[^"]+"\s+means\b'
)
_ADMIN_ROLE_RE = re.compile(
    r"(?ix)("
    r"approval\s+mark|"
    r"arrangements?\s+of\s+(?:the\s+)?approval|"
    r"communication\b|"
    r"application\s+for\s+approval|"
    r"conformity\s+of\s+production"
    r")"
)


def classify_clause_role(text: str, *, section_number: str = "") -> str:
    """Deterministic ROLE label for limit vs procedure vs admin disambiguation."""
    body = (text or "").strip()
    sec = (section_number or "").strip()
    if _DEFINITION_ROLE_RE.search(body) or (
        re.match(r"(?i)^2\.", sec) and "means" in body.lower()[:120]
    ):
        return "DEFINITION"
    if _ADMIN_ROLE_RE.search(body) or re.match(r"(?i)^(preamble|annex\s*[12])$", sec):
        return "APPROVAL/ADMINISTRATIVE"
    if _LIMIT_ROLE_RE.search(body):
        return "PERFORMANCE LIMIT"
    if _PROCEDURE_ROLE_RE.search(body) or re.match(r"(?i)^annex\s*4", sec):
        # Annex 4 often mixes limits + procedure; prefer procedure when no limit language.
        if not _LIMIT_ROLE_RE.search(body):
            return "TEST PROCEDURE"
    if re.match(r"(?i)^5\.", sec):
        return "PERFORMANCE REQUIREMENT"
    return "REQUIREMENT"


def context_header(chunk: Chunk) -> str:
    """Cheap, deterministic context line for embedding / retrieval."""
    section = chunk.section_number or "?"
    title = chunk.section_title or ""
    title_bit = f" {title}" if title else ""
    role = classify_clause_role(chunk.text or "", section_number=section)
    return (
        f"From {chunk.regulation_id} {chunk.revision} "
        f"§{section}{title_bit} — This is the {role} for this topic:"
    ).strip()


def enrich_chunk(chunk: Chunk) -> Chunk:
    """Return a copy with ``enriched_text`` = header + body."""
    header = context_header(chunk)
    enriched = f"{header}\n{chunk.text}".strip()
    return chunk.model_copy(update={"enriched_text": enriched})


def enrich_chunks(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Batch-enrich chunks (pure string work — no model calls)."""
    return [enrich_chunk(c) for c in chunks]


def iter_embed_texts(chunks: Iterable[Chunk]) -> list[str]:
    """Texts to send to the embedder (enriched when present)."""
    return [(c.enriched_text or c.text) for c in chunks]


def role_prefixed_text(text: str, *, section_number: str = "") -> str:
    """Prefix ROLE for rerank-time disambiguation without re-embedding."""
    role = classify_clause_role(text, section_number=section_number)
    body = (text or "").strip()
    return f"[ROLE: {role}]\n{body}"
