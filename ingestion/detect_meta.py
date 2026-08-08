"""Cheap cover-page metadata detection for uploaded regulation PDFs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_REG_PATTERNS = [
    (re.compile(r"\bRegulation\s+No\.?\s*94\b", re.I), "UN-ECE-R94"),
    (re.compile(r"\bUN\s*[- ]?\s*ECE\s*[- ]?\s*R?\s*94\b", re.I), "UN-ECE-R94"),
    (re.compile(r"\bR\s*[- ]?\s*94\b", re.I), "UN-ECE-R94"),
    (re.compile(r"\bRegulation\s+No\.?\s*95\b", re.I), "UN-ECE-R95"),
    (re.compile(r"\bUN\s*[- ]?\s*ECE\s*[- ]?\s*R?\s*95\b", re.I), "UN-ECE-R95"),
    (re.compile(r"\bR\s*[- ]?\s*95\b", re.I), "UN-ECE-R95"),
    (re.compile(r"\bRegulation\s+No\.?\s*16\b", re.I), "UN-ECE-R16"),
    (re.compile(r"\bR\s*[- ]?\s*16\b", re.I), "UN-ECE-R16"),
    (re.compile(r"\bRegulation\s+No\.?\s*129\b", re.I), "UN-ECE-R129"),
    (re.compile(r"\bR\s*[- ]?\s*129\b", re.I), "UN-ECE-R129"),
]

_REV_RE = re.compile(r"\b(?:Revision|Rev\.?)\s*([0-9]+(?:\.[0-9]+)?)\b", re.I)


def extract_page1_text(pdf_path: Path, *, max_chars: int = 4000) -> str:
    """Extract text from page 1 via pypdfium2 (no Docling)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 unavailable — cannot peek cover page")
        return ""

    path = Path(pdf_path)
    try:
        doc = pdfium.PdfDocument(str(path))
        if len(doc) < 1:
            return ""
        page = doc[0]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded() or ""
        textpage.close()
        page.close()
        doc.close()
        return " ".join(text.split())[:max_chars]
    except Exception as exc:  # noqa: BLE001
        logger.warning("page-1 extract failed: %s", exc)
        return ""


def detect_from_text(text: str) -> tuple[str, str]:
    """Heuristic regulation_id + revision from cover text."""
    regulation_id = ""
    for pattern, rid in _REG_PATTERNS:
        if pattern.search(text or ""):
            regulation_id = rid
            break
    revision = ""
    m = _REV_RE.search(text or "")
    if m:
        revision = f"Rev.{m.group(1)}"
    return regulation_id, revision


def detect_with_llm(text: str) -> tuple[str, str]:
    """Optional small-model parse when heuristics miss; never raises."""
    if not (text or "").strip():
        return "", ""
    try:
        from generation.llm_client import LLMClient

        client = LLMClient()
        if client.provider != "groq":
            return "", ""
        result = client.rewrite(
            (
                "From this UNECE regulation cover-page text, extract JSON only:\n"
                '{"regulation_id":"UN-ECE-R94|UN-ECE-R95|UN-ECE-R16|UN-ECE-R129|",'
                '"revision":"Rev.N or empty"}\n\n'
                f"Text:\n{text[:2500]}"
            ),
            system=(
                "You extract regulation metadata. Reply with a single JSON object. "
                "Use UN-ECE-R## ids. If unknown, use empty strings."
            ),
        )
        import json
        import re as _re

        raw = (result.text or "").strip()
        if raw.startswith("```"):
            raw = _re.sub(r"^```(?:json)?\s*", "", raw)
            raw = _re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        rid = str(data.get("regulation_id") or "").strip()
        rev = str(data.get("revision") or "").strip()
        if rid and not rid.upper().startswith("UN-ECE-"):
            m = _re.search(r"R?\s*(\d+)", rid, _re.I)
            if m:
                rid = f"UN-ECE-R{m.group(1)}"
        return rid, rev
    except Exception as exc:  # noqa: BLE001
        logger.info("LLM cover detect skipped: %s", exc)
        return "", ""


def detect_regulation_meta(pdf_path: Path) -> dict[str, str]:
    """Return ``regulation_id``, ``revision``, ``source`` (heuristic|llm|none)."""
    text = extract_page1_text(pdf_path)
    rid, rev = detect_from_text(text)
    source = "heuristic" if rid or rev else "none"
    if not rid or not rev:
        llm_rid, llm_rev = detect_with_llm(text)
        if not rid and llm_rid:
            rid = llm_rid
            source = "llm"
        if not rev and llm_rev:
            rev = llm_rev
            source = "llm" if source == "none" else source
    return {
        "regulation_id": rid,
        "revision": rev or "Rev.unknown",
        "source": source,
        "page1_preview": (text or "")[:240],
    }
