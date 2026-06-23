"""Typed cross-regulation reference edges extracted from corpus text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Seed edges — extend as new regulations are indexed.
_SEED_EDGES: list[tuple[str, str, str, str]] = [
    ("UN_R16", "UN_R14", "belt_anchorage", "R16 references R14 belt anchorage requirements"),
    ("UN_R14", "UN_R94", "sled_test_alternative", "R14 §7.1 sled-test alternative references R94"),
    ("UN_R14", "UN_R16", "restraint_system", "R14 cross-ref restraint system (R16)"),
]

_REF_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("UN_R14", re.compile(r"\b(?:UN\s*)?R(?:egulation\s*)?\.?\s*14\b", re.I)),
    ("UN_R16", re.compile(r"\b(?:UN\s*)?R(?:egulation\s*)?\.?\s*16\b", re.I)),
    ("UN_R94", re.compile(r"\b(?:UN\s*)?R(?:egulation\s*)?\.?\s*94\b", re.I)),
    ("UN_R95", re.compile(r"\b(?:UN\s*)?R(?:egulation\s*)?\.?\s*95\b", re.I)),
    ("UN_R137", re.compile(r"\b(?:UN\s*)?R(?:egulation\s*)?\.?\s*137\b", re.I)),
    ("FMVSS", re.compile(r"\bFMVSS\b", re.I)),
]


@dataclass(frozen=True)
class CrossReferenceEdge:
    source_reg: str
    target_reg: str
    edge_type: str
    note: str


def seed_edges() -> list[CrossReferenceEdge]:
    return [
        CrossReferenceEdge(s, t, et, n) for s, t, et, n in _SEED_EDGES
    ]


def detect_cross_refs_in_text(text: str, source_reg: str) -> list[CrossReferenceEdge]:
    """Detect in-text references from source_reg to other regulations."""
    found: list[CrossReferenceEdge] = []
    for target, pat in _REF_PATTERNS:
        if target == source_reg:
            continue
        if pat.search(text):
            found.append(CrossReferenceEdge(
                source_reg=source_reg,
                target_reg=target,
                edge_type="in_text_reference",
                note=f"{source_reg} text references {target}",
            ))
    return found


def enrich_chunk_cross_refs(chunk: dict[str, Any]) -> dict[str, Any]:
    reg = chunk.get("regulation") or chunk.get("doc_id") or ""
    text = chunk.get("text") or ""
    edges = detect_cross_refs_in_text(text, reg)
    if edges:
        chunk = dict(chunk)
        chunk["cross_references"] = [
            {"target": e.target_reg, "type": e.edge_type, "note": e.note}
            for e in edges
        ]
    return chunk
