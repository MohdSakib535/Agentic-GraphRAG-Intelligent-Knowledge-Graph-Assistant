"""Token-aware, sentence-preserving chunker with configurable size and overlap."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.parser import TextBlock
from app.utils.text import split_sentences
from app.utils.tokens import count_tokens


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int
    page_number: int | None
    section: str | None
    page_end: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int | None
    section: str | None


def _split_long(sentence: str, max_tokens: int) -> list[str]:
    """Split a single over-long sentence on word boundaries."""
    words = sentence.split()
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= max_tokens:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts


class TokenChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, min_section_tokens: int = 60) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_tokens = min_section_tokens

    def _units(self, blocks: list[TextBlock]) -> list[_Unit]:
        units: list[_Unit] = []
        for block in blocks:
            for sentence in split_sentences(block.text):
                tokens = count_tokens(sentence)
                if tokens > self.chunk_size:
                    for part in _split_long(sentence, self.chunk_size):
                        units.append(_Unit(part, count_tokens(part), block.page_number, block.section))
                else:
                    units.append(_Unit(sentence, tokens, block.page_number, block.section))
        return units

    def chunk(self, blocks: list[TextBlock]) -> list[Chunk]:
        units = self._units(blocks)
        chunks: list[Chunk] = []
        current: list[_Unit] = []
        current_tokens = 0
        fresh = 0  # units in ``current`` that are not overlap carried from the previous chunk

        def emit() -> None:
            nonlocal current, current_tokens, fresh
            fresh = 0
            if not current:
                return
            text = " ".join(u.text for u in current)
            pages = [u.page for u in current if u.page is not None]
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    token_count=count_tokens(text),
                    page_number=pages[0] if pages else None,
                    page_end=pages[-1] if pages else None,
                    section=current[0].section,
                )
            )
            # Carry trailing sentences forward as overlap (never crossing a page boundary).
            overlap: list[_Unit] = []
            overlap_tokens = 0
            last_page = current[-1].page
            for unit in reversed(current):
                if overlap_tokens + unit.tokens > self.chunk_overlap or unit.page != last_page:
                    break
                overlap.insert(0, unit)
                overlap_tokens += unit.tokens
            # Never carry the whole chunk (would loop forever / duplicate content).
            if len(overlap) == len(current):
                overlap = overlap[1:]
                overlap_tokens = sum(u.tokens for u in overlap)
            current = overlap
            current_tokens = overlap_tokens

        for unit in units:
            page_changed = bool(current) and unit.page is not None and current[-1].page not in (None, unit.page)
            section_changed = (
                bool(current)
                and unit.section != current[-1].section
                and current_tokens >= self.min_section_tokens
            )
            if current and (current_tokens + unit.tokens > self.chunk_size or page_changed or section_changed):
                emit()
                if page_changed or section_changed:
                    # Do not leak overlap across a page/section boundary.
                    current, current_tokens = [], 0
            current.append(unit)
            current_tokens += unit.tokens
            fresh += 1
        if current and fresh:
            emit()
        return chunks
