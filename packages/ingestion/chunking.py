"""Pack a section's paragraphs into chunks — TRD.md §18.

Structure-aware in the sense required by TRD.md §18 ("do not blindly split
every PDF into fixed-size chunks"): chunk boundaries never cross a detected
section boundary. Within a section, paragraphs are packed up to a target
word count — a real token-aware splitter is a straightforward upgrade once
an embedding model is chosen (TRD.md §20) and its tokenizer is known.
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.ingestion.sections import DetectedSection

TARGET_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 50


@dataclass
class Chunk:
    section_title: str
    section_number: str | None
    start_page: int
    end_page: int
    content: str
    word_count: int


def chunk_section(section: DetectedSection) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0
    buffer_start_page: int | None = None
    buffer_end_page: int | None = None

    def flush() -> None:
        nonlocal buffer, buffer_words, buffer_start_page, buffer_end_page
        if not buffer:
            return
        chunks.append(
            Chunk(
                section_title=section.title,
                section_number=section.section_number,
                start_page=buffer_start_page or section.start_page,
                end_page=buffer_end_page or section.end_page,
                content="\n".join(buffer),
                word_count=buffer_words,
            )
        )
        buffer = []
        buffer_words = 0
        buffer_start_page = None
        buffer_end_page = None

    for page_number, paragraph in section.paragraphs:
        if buffer_start_page is None:
            buffer_start_page = page_number
        buffer_end_page = page_number
        buffer.append(paragraph)
        buffer_words += len(paragraph.split())
        if buffer_words >= TARGET_CHUNK_WORDS:
            flush()
    flush()

    if len(chunks) >= 2 and chunks[-1].word_count < MIN_CHUNK_WORDS:
        last = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = Chunk(
            section_title=prev.section_title,
            section_number=prev.section_number,
            start_page=prev.start_page,
            end_page=last.end_page,
            content=prev.content + "\n" + last.content,
            word_count=prev.word_count + last.word_count,
        )
    return chunks


def chunk_sections(sections: list[DetectedSection]) -> dict[int, list[Chunk]]:
    """Section index -> its chunks, so the caller can map back to the
    `DocumentSection` row it persists for that index."""
    return {i: chunk_section(s) for i, s in enumerate(sections)}
