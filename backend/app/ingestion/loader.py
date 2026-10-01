"""Upload validation and tenant-scoped file storage."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.errors import FileTooLargeError, InvalidFileError, UnsupportedFileError
from app.utils.text import safe_filename

SUPPORTED_TYPES: dict[str, set[str]] = {
    "pdf": {"application/pdf", "application/x-pdf", "application/octet-stream"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    "txt": {"text/plain", "application/octet-stream"},
    "md": {"text/markdown", "text/x-markdown", "text/plain", "application/octet-stream"},
}
_EXTENSION_ALIASES = {"markdown": "md", "text": "txt"}


@dataclass(frozen=True)
class ValidatedFile:
    filename: str
    file_type: str
    content_type: str | None
    data: bytes
    checksum: str

    @property
    def size(self) -> int:
        return len(self.data)


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    ext = _EXTENSION_ALIASES.get(ext, ext)
    if ext not in SUPPORTED_TYPES:
        raise UnsupportedFileError(
            f"Unsupported file type '.{ext or '?'}'. Supported: PDF, DOCX, TXT, MD", code="UNSUPPORTED_FILE_TYPE"
        )
    return ext


def read_limited(stream: BinaryIO, max_bytes: int, chunk_size: int = 1024 * 1024) -> bytes:
    """Read at most ``max_bytes``; raise as soon as the limit is exceeded (no unbounded buffering)."""
    buf = io.BytesIO()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        if buf.tell() + len(chunk) > max_bytes:
            raise FileTooLargeError(f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB")
        buf.write(chunk)
    return buf.getvalue()


def _check_signature(file_type: str, data: bytes) -> None:
    if not data:
        raise InvalidFileError("The uploaded file is empty")
    if file_type == "pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise InvalidFileError("File content is not a valid PDF")
    elif file_type == "docx":
        if not data.startswith(b"PK\x03\x04"):
            raise InvalidFileError("File content is not a valid DOCX document")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
                if "word/document.xml" not in names:
                    raise InvalidFileError("DOCX archive is missing word/document.xml")
                # Zip-bomb guard: total uncompressed size must stay bounded.
                total = sum(info.file_size for info in zf.infolist())
                if total > 200 * 1024 * 1024:
                    raise InvalidFileError("DOCX archive expands to an unsafe size")
        except zipfile.BadZipFile as exc:
            raise InvalidFileError("Corrupted DOCX archive") from exc
    else:
        if b"\x00" in data[:8192]:
            raise InvalidFileError("Text file appears to be binary")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                data.decode("latin-1")
            except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes everything
                raise InvalidFileError("Text file must be UTF-8 encoded") from exc


def validate_upload(filename: str | None, content_type: str | None, data: bytes, max_bytes: int) -> ValidatedFile:
    if not filename:
        raise InvalidFileError("A filename is required")
    clean_name = safe_filename(filename)
    file_type = detect_file_type(clean_name)
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB")
    base_ct = (content_type or "").split(";")[0].strip().lower() or None
    if base_ct and base_ct not in SUPPORTED_TYPES[file_type]:
        raise UnsupportedFileError(f"Content type '{base_ct}' does not match a .{file_type} file")
    _check_signature(file_type, data)
    return ValidatedFile(clean_name, file_type, base_ct, data, hashlib.sha256(data).hexdigest())


class FileStorage:
    """Stores uploads under ``{root}/{tenant_id}/{document_id}.{ext}``; paths never derive from user input."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def path_for(self, tenant_id: str, document_id: str, file_type: str) -> Path:
        return self.root / str(tenant_id) / f"{document_id}.{file_type}"

    def save(self, tenant_id: str, document_id: str, file_type: str, data: bytes) -> str:
        path = self.path_for(tenant_id, document_id, file_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return str(path)

    def load(self, storage_path: str) -> bytes:
        path = Path(storage_path).resolve()
        if self.root.resolve() not in path.parents:
            raise InvalidFileError("Stored file path is outside the upload directory")
        return path.read_bytes()

    def delete(self, storage_path: str) -> None:
        path = Path(storage_path).resolve()
        if self.root.resolve() in path.parents and path.exists():
            path.unlink()
