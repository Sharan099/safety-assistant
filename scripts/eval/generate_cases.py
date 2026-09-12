"""Grow the gold set from the ingested corpus — grounded, verified, honestly labelled.

    uv run python scripts/eval/generate_cases.py --per-regulation 60 --seed 7 --out evals/datasets/generated_v2.yaml

For each sampled normative section (CLAUSE/DEFINITION of an ACTIVE regulation) the LLM proposes up
to two engineer-style questions with `key_facts`. A case is kept only if every key fact is a
verbatim span of the section text (whitespace-normalised) — the model cannot invent facts, and
section paths/regulation keys come from the database, never from the model. Nothing is filtered
by whether retrieval finds it (that would tune the benchmark to the system).

Generated cases carry `review_status: AUTO_GROUNDED` and a `notes` provenance line. They are
merged with the human-reviewed `regulatory_v1` cases by `scripts/eval/build_dataset.py`.
LLM calls are cached per section content hash under evals/cache/gen/ (gitignored) so re-runs
are free and deterministic.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from safety_assistant.persistence import get_engine
from safety_assistant.providers.llm import LLMError, LLMMessage, LLMSchemaError
from safety_assistant.providers.llm.factory import get_llm_provider

QUERY_TYPES = (
    "numeric_threshold",
    "paraphrase",
    "definition",
    "exception_condition",
    "exact_clause_lookup",
    "units_operators",
)

SYSTEM = """You write evaluation questions for a regulatory retrieval system used by passive-safety
engineers. You are given ONE clause of a UN vehicle regulation. Produce questions an engineer would
realistically ask whose answer is contained in this clause alone.

Rules:
- Return JSON: {"questions": [{"query": str, "query_type": str, "difficulty": "easy"|"medium"|"hard",
  "key_facts": [str, ...]}]} with at most 2 questions.
- query_type is one of: numeric_threshold (asks for a limit/value), paraphrase (asks in different
  words than the clause), definition (asks what a term means), exception_condition (asks when a
  requirement does/does not apply), exact_clause_lookup (asks what paragraph X.Y.Z requires — only
  then may the query contain the paragraph number), units_operators (asks about a value with its
  unit/comparison operator).
- Do NOT put the paragraph number in the query unless query_type is exact_clause_lookup.
- Each key_fact must be an EXACT, contiguous quote copied from the clause text (max 120 characters),
  containing the decisive number/term. 1–3 key_facts per question.
- Prefer clauses with numbers, units, thresholds, conditions. If the clause has no answerable
  technical content (pure cross-reference, administrative text), return {"questions": []}.
"""


class Proposal(BaseModel):
    query: str = Field(min_length=15, max_length=300)
    query_type: str
    difficulty: str = "medium"
    key_facts: list[str] = Field(min_length=1, max_length=3)


class Proposals(BaseModel):
    questions: list[Proposal] = Field(max_length=2)
    model: str | None = None  # the model the gateway actually routed to (provenance)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _sample_sections(session: Session, per_regulation: int, seed: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT r.regulation_key, r.title AS reg_title, v.version_label, s.path, s.kind, s.title, s.content,
                   s.content_sha256, s.normative
            FROM sections s
            JOIN regulation_versions v ON v.id = s.version_id
            JOIN regulations r ON r.id = v.regulation_id
            WHERE v.status = 'ACTIVE' AND r.kind = 'REGULATION' AND r.scope = 'AUTHORITATIVE_ORG'
              AND s.kind IN ('CLAUSE', 'DEFINITION')
              AND s.content ~ '(shall|must|means|not exceed|at least|minimum|maximum)'
              AND length(s.content) BETWEEN 150 AND 1500
            """
        )
    ).mappings()
    by_reg: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_reg.setdefault(r["regulation_key"], []).append(dict(r))
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for key in sorted(by_reg):
        pool = by_reg[key]
        # stratify: definitions, numeric clauses, other clauses
        defs = [s for s in pool if s["kind"] == "DEFINITION"]
        numeric = [s for s in pool if s["kind"] == "CLAUSE" and re.search(r"\d", s["content"])]
        other = [s for s in pool if s["kind"] == "CLAUSE" and s not in numeric]
        for bucket, share in ((defs, 0.2), (numeric, 0.6), (other, 0.2)):
            rng.shuffle(bucket)
            out.extend(bucket[: max(1, int(per_regulation * share))])
    return out


def _propose(llm: Any, section: dict[str, Any], cache_dir: pathlib.Path) -> Proposals | None:
    cache = cache_dir / f"{section['content_sha256']}.json"
    if cache.exists():
        return Proposals.model_validate(json.loads(cache.read_text(encoding="utf-8")))
    user = (
        f"Regulation: {section['regulation_key']} — {section['reg_title']} ({section['version_label']})\n"
        f"Paragraph {section['path']}{' — ' + section['title'] if section['title'] else ''} ({section['kind']})\n\n"
        f"<clause>\n{section['content']}\n</clause>"
    )
    for attempt in range(6):
        try:
            resp = llm.generate(
                [LLMMessage(role="system", content=SYSTEM), LLMMessage(role="user", content=user)],
                schema=Proposals,
                temperature=0.2,
                max_tokens=1200,
            )
            assert isinstance(resp.parsed, Proposals)
            payload = {**resp.parsed.model_dump(), "model": resp.model}
            cache.write_text(json.dumps(payload), encoding="utf-8")
            return Proposals.model_validate(payload)
        except LLMSchemaError:
            return None  # malformed proposal for this section — skip, do not retry into the budget
        except LLMError as exc:
            wait = min(60, 5 * 2**attempt)
            print(f"  llm error ({exc}); sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def _verify(section: dict[str, Any], p: Proposal) -> list[str]:
    content = _norm(section["content"])
    facts = [f.strip() for f in p.key_facts if 3 <= len(f.strip()) <= 120 and _norm(f) in content]
    if p.query_type not in QUERY_TYPES:
        return []
    if p.query_type != "exact_clause_lookup" and section["path"] in p.query:
        return []
    return facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-regulation", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evals/datasets/generated_v2.yaml")
    ap.add_argument("--cache", default="evals/cache/gen")
    ap.add_argument("--limit", type=int, default=None, help="stop after N sections (smoke runs)")
    args = ap.parse_args(argv)

    llm = get_llm_provider()
    if llm is None:
        print("LLM_PROVIDER is not configured", file=sys.stderr)
        return 2
    cache_dir = pathlib.Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Session(get_engine()) as session:
        sections = _sample_sections(session, args.per_regulation, args.seed)
    if args.limit:
        sections = sections[: args.limit]
    print(f"{len(sections)} sections sampled; model={llm.model}", file=sys.stderr)

    cases: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for i, sec in enumerate(sections, 1):
        proposals = _propose(llm, sec, cache_dir)
        if proposals is None:
            continue
        for j, p in enumerate(proposals.questions):
            facts = _verify(sec, p)
            if not facts or _norm(p.query) in seen_queries:
                continue
            seen_queries.add(_norm(p.query))
            slug = sec["regulation_key"].lower().replace("un-", "")
            path_slug = sec["path"].replace(".", "-")
            cases.append(
                {
                    "case_id": f"gen-{slug}-{path_slug}-{j}",
                    "query": p.query.strip(),
                    "query_type": p.query_type,
                    "difficulty": p.difficulty if p.difficulty in ("easy", "medium", "hard") else "medium",
                    "expected_regulation_key": sec["regulation_key"],
                    "expected_version_label": sec["version_label"],
                    "expected_section_paths": [sec["path"]],
                    "key_facts": facts,
                    "acceptable_citations": [],
                    "answerability": "answerable",
                    "review_status": "AUTO_GROUNDED",
                    "notes": (
                        f"generated by {proposals.model or llm.model} from section text "
                        f"sha256:{sec['content_sha256'][:12]}; key_facts verified verbatim against the section"
                    ),
                }
            )
        if i % 10 == 0:
            print(f"  {i}/{len(sections)} sections → {len(cases)} cases", file=sys.stderr)
            _write(args.out, cases, args)
    _write(args.out, cases, args)
    print(f"wrote {len(cases)} cases to {args.out}", file=sys.stderr)
    return 0


def _write(out: str, cases: list[dict[str, Any]], args: argparse.Namespace) -> None:
    header = {
        "dataset_version": "generated_v2",
        "generation": {"per_regulation": args.per_regulation, "seed": args.seed, "review_status": "AUTO_GROUNDED"},
        "cases": cases,
    }
    pathlib.Path(out).write_text(
        yaml.safe_dump(header, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
