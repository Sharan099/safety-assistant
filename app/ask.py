"""Phase 1 CLI: ask a regulation question and print the cited answer + sources.

Usage::

    python -m app.ask "What is the HIC15 limit in R94?"
    python -m app.ask "..." --regulation-id UN-ECE-R94 --top-k 8 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from generation.answer import answer_question


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.ask",
        description="Phase 1: retrieve + grounded answer over UNECE regulations.",
    )
    p.add_argument("question", help="Natural-language question")
    p.add_argument("--regulation-id", default=None, help="Optional regulation filter")
    p.add_argument("--top-k", type=int, default=None, help="Dense retrieval depth")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full structured JSON (answer + sources with page/bbox)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_human(result) -> None:
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.answer)
    print()
    print("=" * 60)
    print(f"SOURCES ({len(result.sources)})")
    print("=" * 60)
    if not result.sources:
        print("(none)")
        return
    for i, src in enumerate(result.sources, 1):
        page = src.page_number if src.page_number is not None else "?"
        bbox = src.bounding_box if src.bounding_box else []
        print(
            f"{i}. {src.citation}  score={src.score:.3f}  "
            f"type={src.content_type}  chunk_id={src.chunk_id}"
        )
        print(f"   title: {src.section_title or '(none)'}")
        print(f"   page={page}  bbox={bbox}")
        preview = " ".join((src.text or "").split())[:160]
        if preview:
            print(f"   text: {preview}…")
        print()
    print(f"[provider={result.provider} model={result.model} cached={result.cached}]")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = answer_question(
            args.question,
            top_k=args.top_k,
            regulation_id=args.regulation_id,
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Ask failed: %s", exc)
        sys.exit(1)

    if args.json:
        print(result.to_json())
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
