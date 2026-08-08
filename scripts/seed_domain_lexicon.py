#!/usr/bin/env python3
"""Harvest acronym + concept candidates from Docling JSON for human review.

Never auto-merges into ``config/acronyms.json`` or ``config/design_components.json``.
Writes a review bundle under ``eval/results/domain_lexicon_seed.json``.

Usage:
  .venv/Scripts/python.exe scripts/seed_domain_lexicon.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.acronyms import extract_acronyms_from_text  # noqa: E402

DOCLING = ROOT / "data" / "docling"
OUT = ROOT / "eval" / "results" / "domain_lexicon_seed.json"

REG_FILES = {
    "UN-ECE-R94": "UN_R94.docling.json",
    "UN-ECE-R95": "UN_R95.docling.json",
    "UN-ECE-R16": "UN_R16.docling.json",
    "UN-ECE-R129": "UN_R129.docling.json",
}

# Reject obvious harvest noise before writing the review bundle.
_REJECT_ACRONYM = re.compile(
    r"(?i)^(mm|kg|fn|e\d|u\d|v\d|ub|vb|tc|th|te|ri|pu|ptfe|unece|"
    r"assembled|carrycots|inboard|outboard|integral|sitting|rotational|"
    r"static|with|four|six|es|ies|tilt|curved|metres|kg)$"
)

_CONCEPT_PROBES = [
    "door",
    "intrusion",
    "passenger compartment",
    "safety-belt",
    "safety belt",
    "anchorage",
    "REESS",
    "electrolyte",
    "seat-back",
    "H point",
    "mobile deformable barrier",
    "side impact",
    "Emergency Locking Retractor",
    "reminder",
    "geometry of the belt",
]


def _texts_from_docling(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    snippets = re.findall(
        r'"text"\s*:\s*"((?:\\.|[^"\\]){20,800})"',
        raw[:8_000_000],
    )
    out: list[str] = []
    for s in snippets:
        try:
            out.append(json.loads(f'"{s}"'))
        except json.JSONDecodeError:
            out.append(s.replace("\\n", " "))
    return "\n".join(out)


def main() -> int:
    harvest: dict[str, dict] = {}
    concepts: dict[str, list[str]] = {}
    for rid, fname in REG_FILES.items():
        path = DOCLING / fname
        if not path.exists():
            print(f"skip missing {path}")
            continue
        body = _texts_from_docling(path)
        found = extract_acronyms_from_text(body)
        kept = {
            k: v
            for k, v in found.items()
            if not _REJECT_ACRONYM.match(k) and len(k) >= 2 and len(v) >= 4
        }
        harvest[rid] = kept
        hits = [p for p in _CONCEPT_PROBES if re.search(re.escape(p), body, re.I)]
        concepts[rid] = hits
        print(f"{rid}: harvested={len(found)} kept={len(kept)} concepts={hits}")

    bundle = {
        "_meta": {
            "purpose": "Review-only seed for Fix 5 acronyms + DESIGN_IMPLICATION concepts",
            "do_not_auto_merge": True,
            "targets": [
                "config/acronyms.json",
                "config/design_components.json",
            ],
        },
        "acronym_candidates_by_regulation": harvest,
        "concept_phrase_hits": concepts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
