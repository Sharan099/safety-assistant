import uuid

from safety_assistant.ingestion.chunk import CitationContext, chunk_document, estimate_tokens
from safety_assistant.ingestion.normalize import normalize_regulation
from safety_assistant.ingestion.parse.contract import ParsedPage, ParsedTable

LONG = " ".join(f"Requirement sentence number {i} shall apply." for i in range(180))
BODY = f"""E/ECE/324/Rev.1/Add.93/Rev.4
12
1.
Specifications
1.1.
Short clause one shall apply.
1.2.
Short clause two shall apply.
1.3.
{LONG}
"""


def _pages() -> list[ParsedPage]:
    return [ParsedPage(page_number=12, text=BODY, char_count=len(BODY), text_quality=1.0, needs_ocr=False)]


def test_structural_chunks_merge_tiny_siblings_and_split_long_clauses() -> None:
    nd = normalize_regulation(_pages())
    chunks = chunk_document(nd, [], CitationContext("UN-R94", "Rev.4 (04 series)"))
    labels = [c.citation_label for c in chunks]
    assert "UN R94 Rev.4 §1.1–1.2 (p. 12)" in labels  # merged tiny siblings, exact range
    merged = next(c for c in chunks if c.citation_label.startswith("UN R94 Rev.4 §1.1"))
    assert "1.1." in merged.content and "1.2." in merged.content  # clause numbers kept inline
    assert merged.metadata["merged_paths"] == ["1.1", "1.2"]
    parts = [c for c in chunks if "§1.3" in c.citation_label]
    assert len(parts) >= 2 and all("part" in c.citation_label for c in parts)
    assert all(c.token_count <= 600 for c in chunks)
    assert all(c.content.startswith("UN R94 › ") for c in chunks)  # version-independent header


def test_chunk_ids_are_deterministic_per_version() -> None:
    nd = normalize_regulation(_pages())
    ctx = CitationContext("UN-R94", "Rev.4 (04 series)")
    a, b = chunk_document(nd, [], ctx), chunk_document(nd, [], ctx)
    v = uuid.uuid4()
    assert [c.deterministic_id(v) for c in a] == [c.deterministic_id(v) for c in b]
    assert a[0].deterministic_id(v) != a[0].deterministic_id(uuid.uuid4())


def test_table_chunks_carry_headers_and_skip_degenerate_tables() -> None:
    nd = normalize_regulation(_pages())
    good = ParsedTable(
        12, 0, (0, 0, 1, 1), ["Criterion", "Limit"], [["HPC", "1,000"], ["ThCC", "42 mm"]], "pymupdf", 1.0
    )
    junk = ParsedTable(12, 1, (0, 0, 1, 1), None, [["a", ""], ["", None]], "pymupdf", 0.5)
    chunks = chunk_document(nd, [good, junk], CitationContext("UN-R94", "Rev.4 (04 series)"))
    tables = [c for c in chunks if c.chunk_type == "TABLE"]
    assert len(tables) == 1
    assert "Criterion | Limit" in tables[0].content and "ThCC | 42 mm" in tables[0].content
    assert tables[0].citation_label.endswith("Table 1 (p. 12)")


def test_token_estimate_monotonic() -> None:
    assert estimate_tokens("") == 1 and estimate_tokens("a b c") < estimate_tokens("a b c d e f")


def test_uploaded_documents_are_cited_by_title_not_generated_key() -> None:
    from safety_assistant.ingestion.chunk.structural import CitationContext

    assert CitationContext("UN-R94", "Rev.4 (04 series)", "UN Regulation No. 94").prefix == "UN R94 Rev.4"
    assert CitationContext("DOC-FFD57DADD19E", "v1", "ACME Z4 frontal ODB test report TR-2026-0417").prefix == (
        "ACME Z4 frontal ODB test report TR-2026-0417 v1"
    )
    assert CitationContext("DOC-FFD57DADD19E", "v1").prefix == "DOC FFD57DADD19E v1"
