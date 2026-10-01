from __future__ import annotations

import io
from pathlib import Path

import pytest

from app.core.errors import FileTooLargeError, InvalidFileError, UnsupportedFileError
from app.ingestion.chunker import TokenChunker
from app.ingestion.loader import read_limited, validate_upload
from app.ingestion.metadata import chunk_records
from app.ingestion.parser import TextBlock, parse_document
from app.utils.tokens import count_tokens

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
MB = 1024 * 1024


@pytest.mark.parametrize(("name", "ftype"), [("architecture.pdf", "pdf"), ("project-overview.docx", "docx"),
                                             ("team-directory.md", "md"), ("technology-glossary.txt", "txt")])
def test_parsers_extract_text_with_provenance(name: str, ftype: str) -> None:
    parsed = parse_document((SAMPLES / name).read_bytes(), ftype, name)
    assert parsed.title
    assert parsed.blocks and all(b.text for b in parsed.blocks)
    assert any(b.section for b in parsed.blocks)
    if ftype == "pdf":
        assert parsed.page_count == 4
        assert {b.page_number for b in parsed.blocks} == {1, 2, 3, 4}
        kafka = next(b for b in parsed.blocks if "distributed event streaming" in b.text)
        assert kafka.page_number == 2 and kafka.section == "Event Streaming with Kafka"
    if ftype == "docx":
        assert any("Project Beta | Rahul | Active" in b.text for b in parsed.blocks)  # tables are extracted


def test_upload_validation_rejects_bad_files() -> None:
    with pytest.raises(UnsupportedFileError):
        validate_upload("malware.exe", None, b"MZ...", MB)
    with pytest.raises(InvalidFileError):
        validate_upload("fake.pdf", "application/pdf", b"not a pdf", MB)
    with pytest.raises(InvalidFileError):
        validate_upload("fake.docx", None, b"PK\x03\x04garbage", MB)
    with pytest.raises(InvalidFileError):
        validate_upload("bin.txt", "text/plain", b"\x00\x01\x02", MB)
    with pytest.raises(InvalidFileError):
        validate_upload("empty.md", None, b"", MB)
    with pytest.raises(UnsupportedFileError):
        validate_upload("doc.pdf", "image/png", (SAMPLES / "architecture.pdf").read_bytes(), MB)
    with pytest.raises(FileTooLargeError):
        read_limited(io.BytesIO(b"x" * (2 * MB)), MB)
    ok = validate_upload("../../etc/passwd/../notes.md", "text/markdown", b"# Hi\n\nhello", MB)
    assert ok.filename == "notes.md" and ok.file_type == "md" and len(ok.checksum) == 64


def test_chunker_respects_size_overlap_and_metadata() -> None:
    sentence = "Kafka streams events between services reliably. "
    blocks = [TextBlock(text=sentence * 120, page_number=1, section="Intro"),
              TextBlock(text="Rahul manages Project Alpha. " * 5, page_number=2, section="Projects")]
    chunks = TokenChunker(chunk_size=100, chunk_overlap=20).chunk(blocks)
    assert len(chunks) > 5
    assert all(c.token_count <= 100 for c in chunks)
    # Overlap: consecutive chunks on the same page share their boundary sentence(s).
    assert chunks[1].text.split(". ")[0] in chunks[0].text
    # Page boundaries are never crossed and metadata is preserved.
    assert all(c.page_number == c.page_end for c in chunks)
    assert chunks[-1].page_number == 2 and chunks[-1].section == "Projects"
    records = chunk_records(chunks, tenant_id="t", document_id="00000000-0000-0000-0000-000000000001",
                            source_filename="a.pdf", title="A")
    assert {"id", "tenant_id", "document_id", "page_number", "section", "source_filename"} <= set(records[0])
    assert len({r["id"] for r in records}) == len(records)


def test_chunker_default_config_and_validation() -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=100, chunk_overlap=100)
    chunker = TokenChunker()
    assert (chunker.chunk_size, chunker.chunk_overlap) == (800, 100)
    long_sentence = "word " * 2000
    chunks = TokenChunker(200, 20).chunk([TextBlock(text=long_sentence)])
    assert all(count_tokens(c.text) <= 210 for c in chunks)
