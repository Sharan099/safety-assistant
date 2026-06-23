"""Pre-group retrieved chunks by applicability for broad-k prompt injection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.app.graph.prompt_budget import strip_chunk_boilerplate
from backend.app.retrieval.citations import enrich_doc_provenance


def _group_key(doc: dict[str, Any]) -> str:
    cats = doc.get("applies_to_category") or []
    if isinstance(cats, str):
        cats = [cats]
    cat_part = ",".join(sorted(cats)) if cats else "GENERAL"
    test = doc.get("anchorage_test_type") or doc.get("applicable_test_configuration") or "any"
    seat = doc.get("seat_position") or "any"
    return f"{cat_part}|{test}|{seat}"


def _group_label(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    if doc.get("applicability_display"):
        parts.append(str(doc["applicability_display"]))
    elif doc.get("applies_to_category"):
        cats = doc["applies_to_category"]
        if isinstance(cats, list):
            parts.append(", ".join(cats))
        else:
            parts.append(str(cats))
    if doc.get("anchorage_test_label"):
        parts.append(str(doc["anchorage_test_label"]))
    elif doc.get("applicable_test_configuration"):
        parts.append(str(doc["applicable_test_configuration"]).replace("_", " "))
    if doc.get("seat_position") and doc["seat_position"] != "any":
        parts.append(f"seat: {doc['seat_position']}")
    return " — ".join(parts) if parts else "General / unspecified applicability"


def group_documents_by_applicability(
    documents: list[dict],
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Cluster docs by category + test configuration metadata."""
    buckets: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for d in documents:
        enriched = enrich_doc_provenance(d, chunk_lookup)
        key = _group_key(enriched)
        buckets[key].append((enriched, d))

    groups: list[dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        items = buckets[key]
        label = _group_label(items[0][0])
        groups.append({
            "key": key,
            "label": label,
            "documents": [raw for _, raw in items],
            "enriched": [e for e, _ in items],
        })
    return groups


def format_grouped_context(
    documents: list[dict],
    citations: list[dict],
    *,
    char_cap: int,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build prompt context with applicability-group headers when k is large."""
    cite_by_marker = {c["marker"]: c for c in citations}
    doc_cite_pairs = list(zip(documents, citations))
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for d, c in doc_cite_pairs:
        enriched = enrich_doc_provenance(d, chunk_lookup)
        groups[_group_key(enriched)].append((enriched, c))

    if len(groups) <= 1:
        return ""

    parts: list[str] = []
    for gi, (key, items) in enumerate(sorted(groups.items()), 1):
        label = _group_label(items[0][0])
        header = f"--- APPLICABILITY GROUP G{gi}: {label} ---"
        blocks: list[str] = []
        for enriched, c in items:
            raw = strip_chunk_boilerplate(enriched.get("text", "") or "")
            text = raw[:char_cap]
            blocks.append(
                f"[{c['marker']}] {c['label']} ({c['doc_type_label']})\n{text}"
            )
        if blocks:
            parts.append(header + "\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


def should_use_grouped_context(
    documents: list[dict],
    *,
    breadth_label: str | None = None,
    min_docs: int = 9,
) -> bool:
    if len(documents) >= min_docs:
        return True
    if breadth_label in ("broad", "very_broad"):
        return len(documents) >= 6
    keys = {_group_key(enrich_doc_provenance(d, {})) for d in documents}
    return len(keys) >= 3 and len(documents) >= 5
