"""Document parsers: PyMuPDF (PDF), python-docx (DOCX), plain Python (TXT/MD).

Each parser yields :class:`TextBlock` objects carrying page and section
information so that chunks - and therefore citations - keep precise provenance.
"""

from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import DocumentProcessingError, InvalidFileError
from app.utils.text import clean_text


@dataclass
class TextBlock:
    text: str
    page_number: int | None = None
    section: str | None = None


@dataclass
class ParsedDocument:
    blocks: list[TextBlock]
    title: str | None
    page_count: int | None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s+[A-Z][^.!?]{1,80}$")


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90 or line.endswith((".", ",", ";", ":", "?", "!")):
        return bool(_NUMBERED_HEADING.match(line)) if line else False
    if _NUMBERED_HEADING.match(line):
        return True
    words = line.split()
    if len(words) > 10:
        return False
    capitalised = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha())
    return capitalised / len(words) >= 0.75 or line.isupper()


def parse_pdf(data: bytes) -> ParsedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidFileError("Unable to open PDF") from exc
    if doc.needs_pass:
        raise InvalidFileError("Encrypted PDFs are not supported")
    blocks: list[TextBlock] = []
    meta_title = (doc.metadata or {}).get("title") or None
    first_heading: str | None = None
    section: str | None = None
    try:
        # Body font size across the document drives heading detection.
        sizes: list[float] = []
        pages = []
        for page in doc:
            info = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
            pages.append(info)
            for blk in info.get("blocks", []):
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(round(float(span.get("size", 0)), 1))
        body_size = statistics.median(sizes) if sizes else 0.0
        for page_index, info in enumerate(pages, start=1):
            paragraph: list[str] = []

            def flush(page_no: int = page_index) -> None:
                if paragraph:
                    text = clean_text(" ".join(paragraph))
                    if text:
                        blocks.append(TextBlock(text=text, page_number=page_no, section=section))
                    paragraph.clear()

            for blk in info.get("blocks", []):
                for line in blk.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    line_text = "".join(s["text"] for s in spans).strip()
                    max_size = max(float(s.get("size", 0)) for s in spans)
                    bold = any(int(s.get("flags", 0)) & 16 for s in spans)
                    is_heading = (
                        len(line_text) <= 90
                        and (max_size >= body_size * 1.15 or (bold and _looks_like_heading(line_text)))
                        and not line_text.endswith((".", ","))
                    )
                    if is_heading:
                        flush()
                        section = clean_text(line_text)
                        first_heading = first_heading or section
                    else:
                        paragraph.append(line_text)
                flush()
    finally:
        page_count = doc.page_count
        doc.close()
    if not blocks:
        raise DocumentProcessingError("No extractable text found in PDF (scanned PDFs require OCR)")
    return ParsedDocument(blocks, meta_title or first_heading, page_count, {"parser": "pymupdf"})


def parse_docx(data: bytes) -> ParsedDocument:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise InvalidFileError("Unable to open DOCX document") from exc
    blocks: list[TextBlock] = []
    section: str | None = None
    title = document.core_properties.title or None
    first_heading: str | None = None
    for para in document.paragraphs:
        text = clean_text(para.text)
        if not text:
            continue
        style = (para.style.name if para.style is not None else "") or ""
        if style.startswith("Heading") or style == "Title":
            section = text
            first_heading = first_heading or text
            if style == "Title" and not title:
                title = text
            continue
        blocks.append(TextBlock(text=text, section=section))
    for table in document.tables:
        for row in table.rows:
            cells = [clean_text(c.text) for c in row.cells]
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                blocks.append(TextBlock(text=row_text, section=section or "Table"))
    if not blocks:
        raise DocumentProcessingError("No text found in DOCX document")
    return ParsedDocument(blocks, title or first_heading, None, {"parser": "python-docx"})


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"`{1,3}([^`]*)`{1,3}"), r"\1"),
    (re.compile(r"(\*\*|__)(.*?)\1"), r"\2"),
    (re.compile(r"(?<!\w)[*_](.*?)[*_](?!\w)"), r"\1"),
]


def _strip_md(text: str) -> str:
    for pattern, repl in _MD_INLINE:
        text = pattern.sub(repl, text)
    return text


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_markdown(data: bytes) -> ParsedDocument:
    text = _decode(data)
    blocks: list[TextBlock] = []
    section: str | None = None
    title: str | None = None
    paragraph: list[str] = []
    in_code = False

    def flush() -> None:
        if paragraph:
            cleaned = clean_text(_strip_md("\n".join(paragraph)))
            if cleaned:
                blocks.append(TextBlock(text=cleaned, section=section))
            paragraph.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        match = _MD_HEADING.match(line) if not in_code else None
        if match:
            flush()
            section = _strip_md(match.group(2)).strip()
            if len(match.group(1)) == 1 and title is None:
                title = section
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.rstrip())
    flush()
    if not blocks:
        raise DocumentProcessingError("Markdown document contains no text")
    return ParsedDocument(blocks, title, None, {"parser": "markdown"})


def parse_text(data: bytes) -> ParsedDocument:
    text = clean_text(_decode(data))
    blocks: list[TextBlock] = []
    section: str | None = None
    title: str | None = None
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        lines = para.split("\n")
        if len(lines) == 1 and _looks_like_heading(lines[0]) and len(lines[0].split()) <= 8:
            section = lines[0].strip()
            title = title or section
            continue
        if _looks_like_heading(lines[0]) and len(lines) > 1 and len(lines[0].split()) <= 8:
            section = lines[0].strip()
            title = title or section
            para = "\n".join(lines[1:])
        blocks.append(TextBlock(text=" ".join(para.split()), section=section))
    if not blocks:
        raise DocumentProcessingError("Text document is empty")
    return ParsedDocument(blocks, title, None, {"parser": "text"})


PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "md": parse_markdown, "txt": parse_text}


def parse_document(data: bytes, file_type: str, filename: str) -> ParsedDocument:
    parser = PARSERS.get(file_type)
    if parser is None:
        raise InvalidFileError(f"No parser for file type {file_type}")
    parsed = parser(data)
    if not parsed.title:
        parsed.title = Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename
    return parsed
