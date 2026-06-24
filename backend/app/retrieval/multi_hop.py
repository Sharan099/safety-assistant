"""
Multi-hop cross-document retrieval — chain facts across uploaded doc types.

Example: "Does our crash report meet the chest-deflection limit in the regulation?"
  Hop 1 → legal/regulation doc (binding requirement)
  Hop 2 → test_report doc (measured value)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.query_decomposition import decompose_cross_document

_COMPARE_RE = re.compile(
    r"\b(does|do|did|meet|comply|compliance|compare|versus|vs\.?|against|exceed|within)\b",
    re.I,
)
_MEASURED_RE = re.compile(
    r"\b(measured|our|crash report|test report|result|actual|observed|vehicle)\b",
    re.I,
)
_REQUIREMENT_RE = re.compile(
    r"\b(limit|requirement|regulation|shall|must|standard|protocol|binding)\b",
    re.I,
)


@dataclass
class RetrievalHop:
    hop_id: int
    label: str
    query: str
    target_doc_types: list[str] = field(default_factory=list)
    authority_role: str = "advisory"  # binding | measured | advisory
    documents: list[dict] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""


def is_cross_document_query(query: str) -> bool:
    """True when the question likely chains regulation + test/measurement sources."""
    q = query.lower()
    if not _COMPARE_RE.search(query):
        return False
    has_measured = bool(_MEASURED_RE.search(query))
    has_requirement = bool(_REQUIREMENT_RE.search(query))
    decomp = decompose_cross_document(query)
    return (has_measured and has_requirement) or len(decomp.hops) >= 2


def _filter_by_doc_types(docs: list[dict], doc_types: list[str]) -> list[dict]:
    if not doc_types:
        return docs
    allowed = {d.lower() for d in doc_types}
    filtered = [d for d in docs if (d.get("doc_type") or "").lower() in allowed]
    return filtered or docs


def retrieve_multi_hop(
    retriever: HybridRetriever,
    query: str,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Run per-hop retrieval and merge with hop provenance for audit UI."""
    decomp = decompose_cross_document(query)
    t0 = time.perf_counter()
    hops: list[RetrievalHop] = []
    all_docs: list[dict] = []
    seen: set[str] = set()

    for hop_def in decomp.hops:
        hop = RetrievalHop(
            hop_id=hop_def.hop_id,
            label=hop_def.label,
            query=hop_def.query,
            target_doc_types=hop_def.target_doc_types,
            authority_role=hop_def.authority_role,
        )
        result = retriever.retrieve(hop_def.query, mode=mode)
        docs = _filter_by_doc_types(result.get("documents", []), hop_def.target_doc_types)
        if not docs:
            hop.abstained = True
            hop.abstain_reason = (
                f"No supporting passages found for {hop_def.label} "
                f"in uploaded documents."
            )
        else:
            hop.documents = docs[:5]
            for d in docs:
                cid = d.get("id", "")
                if cid and cid not in seen:
                    seen.add(cid)
                    enriched = {**d, "hop_id": hop.hop_id, "hop_label": hop.label}
                    enriched["authority_role"] = hop_def.authority_role
                    all_docs.append(enriched)

        hops.append(hop)

    any_abstain = any(h.abstained for h in hops)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "documents": all_docs,
        "semantic_count": sum(len(h.documents) for h in hops),
        "bm25_count": 0,
        "latency_ms": latency_ms,
        "queries": [h.query for h in hops],
        "intent_flags": ["multi_hop"],
        "query_breadth": {"label": "multi_hop", "multi_hop": True},
        "multi_hop": {
            "hops": [
                {
                    "hop_id": h.hop_id,
                    "label": h.label,
                    "query": h.query,
                    "target_doc_types": h.target_doc_types,
                    "authority_role": h.authority_role,
                    "abstained": h.abstained,
                    "abstain_reason": h.abstain_reason,
                    "document_count": len(h.documents),
                    "citations_preview": [
                        {
                            "id": d.get("id"),
                            "source": d.get("source"),
                            "doc_type": d.get("doc_type"),
                        }
                        for d in h.documents[:3]
                    ],
                }
                for h in hops
            ],
            "any_abstain": any_abstain,
        },
    }
