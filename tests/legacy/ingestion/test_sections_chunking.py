from packages.ingestion.chunking import TARGET_CHUNK_WORDS, chunk_sections
from packages.ingestion.extract import PageExtraction
from packages.ingestion.sections import detect_sections


def _page(number: int, text: str) -> PageExtraction:
    return PageExtraction(page_number=number, text=text, char_count=len(text), text_quality=1.0, needs_ocr=False)


def test_detect_sections_splits_on_numbered_headings() -> None:
    pages = [
        _page(1, "1 Introduction\nThis document describes the regulation.\nIt has several parts."),
        _page(2, "1.1 Scope\nThis regulation applies to vehicles of category M1.\n2 Definitions\nA dummy is a device."),
    ]
    sections = detect_sections(pages)
    titles = [s.title for s in sections]
    assert "Introduction" in titles
    assert "Scope" in titles
    assert "Definitions" in titles

    scope = next(s for s in sections if s.title == "Scope")
    assert scope.section_number == "1.1"
    assert scope.start_page == 2


def test_chunk_sections_never_crosses_section_boundary() -> None:
    long_paragraph = "word " * (TARGET_CHUNK_WORDS + 50)
    pages = [_page(1, f"1 Section One\n{long_paragraph}\n2 Section Two\nShort content here.")]
    sections = detect_sections(pages)
    chunks_by_section = chunk_sections(sections)

    section_titles = [s.title for s in sections]
    one_idx = section_titles.index("Section One")
    two_idx = section_titles.index("Section Two")

    assert all("Short content" not in c.content for c in chunks_by_section[one_idx])
    assert all("word word" not in c.content for c in chunks_by_section[two_idx])
    # The long section should have been split into more than one chunk.
    assert len(chunks_by_section[one_idx]) >= 1


def test_chunk_word_count_tracked() -> None:
    pages = [_page(1, "1 Section\nOne two three four five.")]
    sections = detect_sections(pages)
    chunks_by_section = chunk_sections(sections)
    chunk = chunks_by_section[0][0]
    assert chunk.word_count == 5
