"""OCR for scanned PDFs and table-aware parsing/chunking/extraction."""

from __future__ import annotations

import pymupdf
import pytest

from app.core.errors import UnsupportedFileError
from app.ingestion.chunker import TokenChunker
from app.ingestion.loader import validate_upload
from app.ingestion.parser import ParseOptions, TextBlock, markdown_table, ocr_available, parse_document
from app.ingestion.relationship_extractor import HeuristicGraphExtractor

ROWS = [["Project", "Manager", "Technology"], ["Project Alpha", "Rahul", "Kafka"], ["Project Beta", "Priya", "Django"]]


def ruled_table_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Project roster below.", fontsize=11)
    x0, y0, cw, rh = 72, 100, 150, 24
    for i, row in enumerate(ROWS):
        for j, cell in enumerate(row):
            page.insert_text((x0 + j * cw + 4, y0 + i * rh + 16), cell, fontsize=10)
    for i in range(len(ROWS) + 1):
        page.draw_line((x0, y0 + i * rh), (x0 + 3 * cw, y0 + i * rh))
    for j in range(4):
        page.draw_line((x0 + j * cw, y0), (x0 + j * cw, y0 + len(ROWS) * rh))
    page.insert_text((72, 230), "End of roster.", fontsize=11)
    return doc.tobytes()


def scanned_pdf(text: str) -> bytes:
    src = pymupdf.open()
    page = src.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 520, 400), text, fontsize=14)
    pix = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    scan.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, pixmap=pix)
    return scan.tobytes()


def test_pdf_tables_become_atomic_markdown_blocks() -> None:
    parsed = parse_document(ruled_table_pdf(), "pdf", "roster.pdf")
    kinds = [b.kind for b in parsed.blocks]
    assert kinds == ["text", "table", "text"] and parsed.metadata["tables"] == 1
    assert "| Project Alpha | Rahul | Kafka |" in parsed.blocks[1].text
    assert all("Rahul" not in b.text for b in parsed.blocks if b.kind == "text")  # cells are not duplicated as prose


@pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")
def test_scanned_pdf_is_ocrd() -> None:
    data = scanned_pdf("Rahul manages Project Alpha. Project Alpha uses Kafka for event streaming.")
    parsed = parse_document(data, "pdf", "scan.pdf")
    text = " ".join(b.text for b in parsed.blocks)
    assert parsed.metadata["ocr_pages"] == 1
    assert "Rahul manages Project Alpha" in text and "Kafka" in text
    with pytest.raises(Exception, match="No extractable text"):
        parse_document(data, "pdf", "scan.pdf", ParseOptions(ocr_enabled=False))


def test_table_chunks_never_split_rows_and_repeat_header() -> None:
    rows = [["Name", "Team"]] + [[f"Person {i}", f"Team {i % 5}"] for i in range(80)]
    blocks = [TextBlock("Intro.", 1, "S"), TextBlock(markdown_table(rows), 1, "S", kind="table"), TextBlock("Outro.", 1, "S")]
    chunks = TokenChunker(150, 20).chunk(blocks)
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) > 1 and chunks[0].content_type == chunks[-1].content_type == "text"
    assert all(c.token_count <= 150 for c in chunks)
    assert all(c.text.split("\n")[1] == "| Name | Team |" for c in tables)
    body = [line for c in tables for line in c.text.split("\n")[3:]]
    assert len(body) == 80 and len(set(body)) == 80  # every row exactly once


def test_docx_and_markdown_tables_and_table_relationships() -> None:
    from pathlib import Path

    docx = parse_document((Path(__file__).resolve().parents[2] / "data/samples/project-overview.docx").read_bytes(),
                          "docx", "project-overview.docx")
    table = next(b for b in docx.blocks if b.kind == "table")
    assert table.text.startswith("| Project | Manager | Status |")
    extractor = HeuristicGraphExtractor()
    result = extractor.extract(f"Table:\n{table.text}", "c1")
    rels = {(r.source, r.relationship.value, r.target) for r in result.relationships}
    assert {("Rahul", "MANAGES", "Project Alpha"), ("Priya", "MANAGES", "Project Gamma")} <= rels
    md = parse_document(b"# T\n\nIntro.\n\n| Name | Role |\n|---|---|\n| Amit | Developer |\n\nAfter.", "md", "x.md")
    assert [b.kind for b in md.blocks] == ["text", "table", "text"]


def test_csv_is_redirected_to_datasets() -> None:
    with pytest.raises(UnsupportedFileError) as exc:
        validate_upload("sales.csv", "text/csv", b"a,b\n1,2\n", 1024)
    assert exc.value.code == "USE_DATASETS_FOR_TABULAR_DATA"
