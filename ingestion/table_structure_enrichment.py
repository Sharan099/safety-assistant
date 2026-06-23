"""Convert linearized regulation tables into structured markdown table chunks."""

from __future__ import annotations

import re
from pathlib import Path

_ANNEX6_PIPE = """\
| Vehicle category | Front outboard | Front centre | Other front | Rearward front | Other rearward | Side facing |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | 3 | 3 | 3 | 3 | 2 | - |
| M2 ≤ 3.5 t | 3 | 3 | 3 | 3 | 2 | - |
| M2 > 3.5 t | 3  | 3 or 2 ╬ | 3 or 2 ╬ | 3 or 2 ╬ | 2 | - |
| M3 | 3  | 3 or 2 ╬ | 3 or 2 ╬ | 3 or 2 ╬ | 2 | 2 |
| N1 | 3 | 3 or 2 Ø | 3 or 2 * | 2 | 2 | - |
| N2 & N3 | 3 | 2 | 3 or 2 * | 2 | 2 | - |

Symbols: Ø=§5.3.3 passageway; *=§5.3.4 windscreen; ╬=§5.3.5 reference zone; =§5.3.7 upper deck.
"""

_ANNEX6_APP1_PIPE = """\
| Seat | M1 buckle side (α2) | M1 other side (α1) | M1 angle constant | Other than M1 buckle | Other than M1 other | Other than M1 constant |
| --- | --- | --- | --- | --- | --- | --- |
| Front* | 45°–80° | 30°–80° | 50°–70° | 30°–80° | 30°–80° | 50°–70° |
| Rear | 45°–80° | 30°–80° | 50°–70° | 30°–80° | 30°–80° | 50°–70° |
| Centre | 45°–80° | 30°–80° | 50°–70° | 30°–80° | 30°–80° | 50°–70° |
"""

_R16_ABRASION_PIPE = """\
| Procedure | Load (daN) | Frequency (Hz) | Cycles | Shift (mm) |
| --- | --- | --- | --- | --- |
| Procedure 1 | 2.5 | 0.5 | 5,000 | 300 ± 20 |
| Procedure 2 | 0.5 | 0.5 | 45,000 | 300 ± 20 |
| Procedure 3* | 0 to 5 | 0.5 | 45,000 | - |
"""


def _is_table_chunk_candidate(chunk: dict) -> tuple[str, str] | None:
    title = (chunk.get("section_title") or chunk.get("clause") or "").strip()
    reg = chunk.get("regulation", "")
    text = chunk.get("text") or ""
    if reg == "UN_R14" and title == "Annex 6" and "M1" in text:
        return ("annex6", _ANNEX6_PIPE)
    if reg == "UN_R14" and "Annex 6" in title and "Appendix 1" in title:
        return ("annex6_app1", _ANNEX6_APP1_PIPE)
    if reg == "UN_R16" and "7.4.1.6.3" in text and "Procedure 1" in text:
        return ("abrasion", _R16_ABRASION_PIPE)
    return None


def enrich_table_chunks(chunks: list[dict], md_path: Path | None = None) -> list[dict]:
    """Emit atomic table chunks with chunk_type=table and pipe markdown body."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in chunks:
        hit = _is_table_chunk_candidate(c)
        if hit and hit[0] not in seen:
            seen.add(hit[0])
            table_id, pipe_body = hit
            tbl = dict(c)
            header = tbl.get("text", "").split("\n\n")[0]
            tbl["text"] = f"{header}\n\n{pipe_body}"
            tbl["chunk_type"] = "table"
            tbl["word_count"] = len(tbl["text"].split())
            tbl["table_structured"] = True
            tbl["table_id"] = table_id
            tbl["has_vehicle_classes"] = True
            tbl["has_requirements"] = True
            out.append(tbl)
            continue
        out.append(c)
    return out
