"""Generates OKF concept files for every ingested document —
ENVIRONMENT_SETUP.md §10-12, TRD_LEVEL3.md §18, Instructions §14.

"Do not manually rewrite entire manuals into OKF... OKF contains curated
concepts/knowledge, each concept links to exact source/page/section." One
concept per ingested `DocumentRevision` — its longest real extracted
section, verbatim (truncated, never rewritten or summarized by an LLM;
this is a deterministic, offline pipeline stage) — not one file per
section, which would approach "rewriting the entire manual" rather than
curating a concept.

Usage:
    uv run python scripts/generate_okf_concepts.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from packages.domain.db import get_engine  # noqa: E402
from packages.domain.knowledge import (  # noqa: E402
    Document,
    DocumentRevision,
    DocumentSection,
    KnowledgeSource,
)

OKF_ROOT = ROOT / "knowledge" / "07_okf"

# KnowledgeSource.category (packages/ingestion/pipeline.py) -> OKF category
# folder (ENVIRONMENT_SETUP.md §11's directory design).
CATEGORY_FOLDER = {
    "REGULATION": "regulations",
    "OFFICIAL_DOCUMENTATION": "solver",
    "REFERENCE": "reference",
    "NHTSA_DUMMY_MODEL": "historical",
    "NHTSA_VEHICLE_MODEL": "historical",
    "NHTSA_TEST_BUCK": "historical",
    "NHTSA_RESTRAINT_MODEL": "historical",
}

MAX_CONTENT_CHARS = 2000


def _slugify(document_key: str) -> str:
    return document_key.lower().replace("_", "-")


def _best_section(session: Session, revision_id: object) -> DocumentSection | None:
    sections = session.query(DocumentSection).filter_by(document_revision_id=revision_id).all()
    real = [s for s in sections if s.content and s.content.strip()]
    if not real:
        return None
    return max(real, key=lambda s: len(s.content or ""))


def main() -> None:
    with Session(get_engine()) as session:
        revisions = session.query(DocumentRevision).filter_by(status="READY").all()
        written = 0

        for revision in revisions:
            document = session.get(Document, revision.document_id)
            if document is None:
                continue
            knowledge_source = session.get(KnowledgeSource, document.knowledge_source_id)
            if knowledge_source is None:
                continue

            section = _best_section(session, revision.id)
            content = (section.content or "").strip() if section else ""
            truncated = content[:MAX_CONTENT_CHARS]

            publisher = (knowledge_source.publisher or "").upper()
            if "NHTSA" in publisher:
                folder = "historical"
            else:
                folder = CATEGORY_FOLDER.get(knowledge_source.category or "", "reference")
            slug = _slugify(document.document_key)
            out_dir = OKF_ROOT / folder / slug
            out_dir.mkdir(parents=True, exist_ok=True)

            frontmatter = {
                "type": "Concept",
                "title": document.title,
                "description": f"Curated excerpt from {document.title} ({revision.revision_label}).",
                "tags": [folder, knowledge_source.source_type.lower()],
                "sources": [
                    {
                        "document_id": document.document_key,
                        "locator": {
                            "section": section.title if section else None,
                            "page": section.start_page if section else None,
                        },
                    }
                ],
                "authority": (knowledge_source.authority_level or "UNKNOWN").lower(),
                "status": "draft",
                "created_by": "scripts/generate_okf_concepts.py",
            }

            body = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True) + "---\n\n"
            body += truncated if truncated else "_No section content was extracted for this revision._\n"

            out_path = out_dir / "index.md"
            out_path.write_text(body, encoding="utf-8")
            written += 1
            print(f"wrote {out_path.relative_to(ROOT)}")

        print(f"\n{written} OKF concept file(s) written.")


if __name__ == "__main__":
    main()
