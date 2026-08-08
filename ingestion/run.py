"""CLI: parse → validate → LightOnOCR fallback → chunk → enrich → embed → upsert.

Usage::

    python -m ingestion.run --pdf path/to/file.pdf \\
        --regulation-id "UN-ECE-R94" --revision "Rev.3"

The extraction validator always runs after Docling. Pages flagged for
clause-in-caption, density anomalies, figures/tables, or section discontinuity
are automatically re-processed with LightOnOCR (pypdfium2 renders). Ambiguous
disagreements land in ``data/ocr_review_queue/`` for human review.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Must be set before Docling/torch load (see ingestion.parse).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from dotenv import load_dotenv

from ingestion.chunk import chunk_document
from ingestion.embed_upsert import upsert_chunks
from ingestion.enrich import enrich_chunks
from ingestion.parse import parse_pdf, print_parse_summary

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_DIR = Path("./data/ocr_review_queue")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ingestion.run",
        description="Ingest a UNECE regulation PDF into Qdrant (idempotent by regulation_id).",
    )
    p.add_argument("--pdf", type=Path, required=True, help="Path to regulation PDF")
    p.add_argument("--regulation-id", required=True, help='e.g. "UN-ECE-R94"')
    p.add_argument("--revision", required=True, help='e.g. "Rev.3"')
    p.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Write Docling JSON here (default: DOCLING_EXPORT_DIR or ./data/docling)",
    )
    p.add_argument("--ocr", action="store_true", help="Enable OCR (scanned PDFs)")
    p.add_argument("--skip-upsert", action="store_true", help="Parse/chunk/enrich only")
    p.add_argument(
        "--vlm-figure-pass",
        action="store_true",
        help=(
            "Force LightOnOCR on all figure/table-adjacent pages even when the "
            "extraction validator would not otherwise expand the set. Also "
            "enabled when VLM_FIGURE_PASS=1. Validator-flagged pages always run."
        ),
    )
    p.add_argument(
        "--vlm-audit-json",
        type=Path,
        default=None,
        help="Optional extra page list (audit JSON) merged into LightOnOCR set",
    )
    p.add_argument(
        "--vlm-pages",
        type=str,
        default=None,
        help="Comma-separated 1-indexed pages for VLM pass (merged with validator)",
    )
    p.add_argument(
        "--vlm-work-dir",
        type=Path,
        default=None,
        help="Cache/renders/logs for VLM pass (default: data/vlm_figure_pass)",
    )
    p.add_argument(
        "--ocr-review-dir",
        type=Path,
        default=None,
        help="Side-by-side review queue (default: data/ocr_review_queue)",
    )
    p.add_argument(
        "--skip-auto-lighton",
        action="store_true",
        help="Disable automatic LightOnOCR fallback (validator still runs/logs)",
    )
    p.add_argument(
        "--layout-score-threshold",
        type=float,
        default=None,
        help="Docling layout OD confidence floor (default 0.3)",
    )
    p.add_argument(
        "--layout-preset",
        type=str,
        default=None,
        help="Docling layout preset (e.g. layout_egret_large)",
    )
    p.add_argument(
        "--caption-guard-strict",
        action="store_true",
        help="Fail ingest if clause-as-caption remains after remediation",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(
    pdf: Path,
    regulation_id: str,
    revision: str,
    *,
    export_dir: Path | None = None,
    do_ocr: bool = False,
    skip_upsert: bool = False,
    vlm_figure_pass: bool = False,
    vlm_audit_json: Path | None = None,
    vlm_pages: list[int] | None = None,
    vlm_work_dir: Path | None = None,
    ocr_review_dir: Path | None = None,
    skip_auto_lighton: bool = False,
    layout_score_threshold: float | None = None,
    layout_preset: str | None = None,
    caption_guard_strict: bool = False,
) -> int:
    load_dotenv()
    export_dir = export_dir or Path(os.getenv("DOCLING_EXPORT_DIR", "./data/docling"))
    review_dir = ocr_review_dir or Path(
        os.getenv("OCR_REVIEW_DIR", str(DEFAULT_REVIEW_DIR))
    )

    doc = parse_pdf(
        pdf,
        export_dir=export_dir,
        do_ocr=do_ocr,
        layout_score_threshold=layout_score_threshold,
        layout_preset=layout_preset,
    )
    print_parse_summary(doc)

    # --- Stage 1 unified element extract (schema contract) -------------------
    from ingestion.extract import extract_from_docling, remediate_clause_as_caption

    # Deterministic caption→paragraph fix runs up-front so chunking never drops
    # clause text swallowed as a caption (tibia-force / Figure 3 class). LightOnOCR
    # below still re-processes flagged pages for fuller layout recovery.
    caption_fix_actions = remediate_clause_as_caption(doc)
    if caption_fix_actions:
        print(f"Caption-clause remediation: {len(caption_fix_actions)} action(s)")

    stage1_extract = extract_from_docling(
        doc,
        document_id=regulation_id,
        regulation_id=regulation_id,
        revision=revision,
        source_path=str(pdf),
        pdf_path=pdf,
    )
    extract_path = export_dir / f"{pdf.stem}.extract.json"
    extract_path.write_text(
        stage1_extract.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(
        f"Stage 1 extract: pages={len(stage1_extract.pages)} "
        f"elements={len(stage1_extract.elements)} → {extract_path}"
    )

    # --- Stage 1 automatic failure detector (always on) ---------------------
    from ingestion.extraction_validator import (
        pages_needing_lighton,
        trigger_resolved_after_vlm,
        validate_extraction,
        write_review_queue_entry,
    )
    from ingestion.vlm_figure_pass import (
        apply_vlm_figure_pass,
        collect_figure_adjacent_pages,
        export_docling_page_markdown,
        load_audit_page_list,
    )

    validation = validate_extraction(doc)
    flags_path = export_dir / f"{pdf.stem}.extraction_flags.json"
    validation.save(flags_path)
    flagged = validation.flagged_pages
    print(
        f"Extraction validator: flagged_pages={flagged} "
        f"flags={len(validation.flags)} → {flags_path}"
    )
    for page in flagged:
        triggers = validation.triggers_for_page(page)
        print(f"  page {page}: {', '.join(triggers)}")

    strict = caption_guard_strict or os.getenv(
        "CAPTION_CLAUSE_GUARD_STRICT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    env_vlm = os.getenv("VLM_FIGURE_PASS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auto_lighton = (not skip_auto_lighton) and os.getenv(
        "EXTRACTION_AUTO_LIGHTON", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}

    # Build LightOnOCR page set: validator flags ∪ optional broad/manual sets.
    pages_for_vlm: set[int] = set(pages_needing_lighton(validation)) if auto_lighton else set()
    if vlm_pages:
        pages_for_vlm.update(int(p) for p in vlm_pages)
    if vlm_audit_json is not None:
        pages_for_vlm.update(load_audit_page_list(vlm_audit_json))
    if vlm_figure_pass or env_vlm:
        pages_for_vlm.update(collect_figure_adjacent_pages(doc))

    vlm_result = None
    if pages_for_vlm:
        print(
            f"LightOnOCR fallback: processing {len(pages_for_vlm)} page(s) "
            f"{sorted(pages_for_vlm)}"
        )
        vlm_result = apply_vlm_figure_pass(
            doc,
            pdf,
            pages=sorted(pages_for_vlm),
            audit_json=None,
            work_dir=vlm_work_dir,
        )
        print(
            f"VLM figure pass: processed={len(vlm_result.pages_processed)} "
            f"corrected={len(vlm_result.pages_corrected)} "
            f"discrepancies={len(vlm_result.discrepancies)}"
        )
        if vlm_result.log_path:
            print(f"  log: {vlm_result.log_path}")

        # Reconciliation: keep VLM when it resolves the specific trigger(s);
        # otherwise queue both versions for human review.
        review_count = 0
        auto_resolved = 0
        for page in sorted(pages_for_vlm):
            page_flags = validation.flags_for_page(page)
            triggers = [f.trigger for f in page_flags] or ["manual_or_broad_vlm"]
            vlm_page = (vlm_result.page_results or {}).get(page)
            vlm_md = vlm_page.markdown if vlm_page else ""
            unresolved: list[dict] = []
            resolved_any = False
            for fl in page_flags:
                ok, note = trigger_resolved_after_vlm(
                    fl.trigger,
                    doc=doc,
                    page_number=page,
                    vlm_markdown=vlm_md,
                    original_length=validation.page_text_lengths.get(page),
                    baseline=validation.density_baseline.get(page),
                )
                if ok:
                    resolved_any = True
                else:
                    unresolved.append(
                        {
                            "trigger": fl.trigger,
                            "detail": fl.detail,
                            "note": note,
                            "evidence": fl.evidence,
                        }
                    )

            # Prefer VLM when apply_vlm_figure_pass already corrected the page,
            # or when at least one validator trigger resolved.
            if page in vlm_result.pages_corrected or resolved_any:
                auto_resolved += 1

            # Queue when triggers remain unresolved OR VLM introduced a new
            # disagreement that was not auto-preferred.
            new_disagreement = any(
                d.page_number == page and d.preferred != "vlm"
                for d in vlm_result.discrepancies
            )
            if unresolved or new_disagreement:
                write_review_queue_entry(
                    review_dir=review_dir,
                    regulation_id=regulation_id,
                    pdf_stem=pdf.stem,
                    page_number=page,
                    triggers=triggers,
                    unresolved=unresolved
                    or [{"trigger": "new_disagreement", "note": "vlm not preferred"}],
                    docling_markdown=export_docling_page_markdown(doc, page),
                    vlm_markdown=vlm_md,
                    extra={
                        "corrected": page in vlm_result.pages_corrected,
                        "resolved_any": resolved_any,
                    },
                )
                review_count += 1

        print(
            f"LightOnOCR reconciliation: auto_resolved_pages={auto_resolved} "
            f"review_queue={review_count} dir={review_dir}"
        )

        # Re-export Docling JSON so corrected labels persist for audits.
        if hasattr(doc, "save_as_json"):
            out_path = export_dir / f"{pdf.stem}.docling.json"
            export_dir.mkdir(parents=True, exist_ok=True)
            doc.save_as_json(out_path)
            logger.info("Re-wrote Docling JSON after VLM pass → %s", out_path)

        # Re-validate clause-in-caption after remediation.
        post = validate_extraction(
            doc,
            flag_density=False,
            flag_figures_tables=False,
            flag_discontinuity=False,
        )
        clause_left = [f for f in post.flags if f.trigger == "clause_in_caption"]
        if clause_left:
            left_pages = sorted({f.page_number for f in clause_left})
            print(f"Caption clause guard AFTER VLM: still flagged pages={left_pages}")
            if strict:
                from ingestion.caption_guard import validate_no_clause_as_caption

                validate_no_clause_as_caption(doc, raise_on_violation=True)
        else:
            print("Caption clause guard AFTER VLM: clear")
    elif flagged and not auto_lighton:
        print(
            "Extraction validator: LightOnOCR fallback skipped "
            "(--skip-auto-lighton or EXTRACTION_AUTO_LIGHTON=0)"
        )
        if strict:
            from ingestion.caption_guard import validate_no_clause_as_caption

            validate_no_clause_as_caption(doc, raise_on_violation=True)

    chunks = chunk_document(doc, regulation_id=regulation_id, revision=revision)
    # Stage 2 — figure descriptions (VLM when enabled; caption+context fallback).
    from ingestion.describe_figures import describe_and_chunk_figures

    fig_chunks = describe_and_chunk_figures(
        regulation_id=regulation_id,
        revision=revision,
        extract=stage1_extract,
        vlm_result=vlm_result,
    )
    if fig_chunks:
        chunks.extend(fig_chunks)
        print(f"Figure chunks added: {len(fig_chunks)}")
    elif vlm_result is not None:
        # Backward-compatible path if Stage 2 collector found nothing.
        from ingestion.vlm_figure_pass import build_figure_chunks_from_vlm

        fig_chunks = build_figure_chunks_from_vlm(
            vlm_result,
            doc,
            regulation_id=regulation_id,
            revision=revision,
        )
        chunks.extend(fig_chunks)
        print(f"Figure chunks added (legacy VLM builder): {len(fig_chunks)}")

    print(
        f"Chunks: {len(chunks)} "
        f"(tables={sum(1 for c in chunks if c.content_type == 'table')}, "
        f"figures={sum(1 for c in chunks if c.content_type == 'figure')})"
    )

    enriched = enrich_chunks(chunks)
    if enriched:
        print("Enrichment sample:")
        sample = enriched[0].enriched_text.splitlines()[0]
        print(f"  {sample}")

    if skip_upsert:
        logger.info("Skipping upsert (--skip-upsert)")
        return len(enriched)

    n = upsert_chunks(enriched)
    print(
        f"Upserted {n} points into Qdrant collection 'regulations' "
        f"(regulation_id={regulation_id})"
    )

    # Optional one-time limits extraction (LLM) after ingest.
    if os.getenv("EXTRACT_LIMITS_ON_INGEST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            from ingestion.extract_limits import extract_limits_for_regulation

            table = extract_limits_for_regulation(regulation_id)
            print(f"Extracted {len(table.limits)} limit rows for {regulation_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_limits_on_ingest failed: %s", exc)

    return n


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    pages = None
    if args.vlm_pages:
        pages = [int(x.strip()) for x in args.vlm_pages.split(",") if x.strip()]
    try:
        run(
            args.pdf,
            args.regulation_id,
            args.revision,
            export_dir=args.export_dir,
            do_ocr=args.ocr,
            skip_upsert=args.skip_upsert,
            vlm_figure_pass=args.vlm_figure_pass,
            vlm_audit_json=args.vlm_audit_json,
            vlm_pages=pages,
            vlm_work_dir=args.vlm_work_dir,
            ocr_review_dir=args.ocr_review_dir,
            skip_auto_lighton=args.skip_auto_lighton,
            layout_score_threshold=args.layout_score_threshold,
            layout_preset=args.layout_preset,
            caption_guard_strict=args.caption_guard_strict,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
