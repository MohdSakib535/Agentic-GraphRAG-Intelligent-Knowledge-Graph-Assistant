"""CSV ingestion: validation, type inference, column profiling and Parquet storage."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from app.core.errors import FileTooLargeError, InvalidFileError
from app.utils.text import safe_filename

_NUMERIC = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "DECIMAL", "UTINYINT",
            "USMALLINT", "UINTEGER", "UBIGINT")
_TEMPORAL = ("DATE", "TIMESTAMP", "TIME")


@dataclass
class ProfiledDataset:
    name: str
    filename: str
    checksum: str
    row_count: int
    columns: list[dict[str, Any]]
    parquet_bytes: bytes


def column_kind(duck_type: str) -> str:
    t = duck_type.upper()
    if t.startswith(_NUMERIC):
        return "number"
    if t.startswith(_TEMPORAL):
        return "date"
    if t == "BOOLEAN":
        return "boolean"
    return "text"


def sanitize_columns(names: list[str]) -> list[str]:
    """snake_case, SQL-safe, unique column names (originals are kept in the profile)."""
    out: list[str] = []
    for i, raw in enumerate(names):
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw).strip()).strip("_").lower() or f"column_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        base, n = name, 2
        while name in out:
            name, n = f"{base}_{n}", n + 1
        out.append(name)
    return out


def profile_csv(filename: str, data: bytes, max_bytes: int, max_rows: int) -> ProfiledDataset:
    clean_name = safe_filename(filename)
    if Path(clean_name).suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise InvalidFileError("Upload a .csv or .tsv file")
    if not data.strip():
        raise InvalidFileError("The CSV file is empty")
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB")
    if b"\x00" in data[:8192]:
        raise InvalidFileError("The file looks binary, not CSV")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.csv")
        out = os.path.join(tmp, "data.parquet")
        with open(src, "wb") as fh:
            fh.write(data)
        conn = duckdb.connect(":memory:")
        try:
            try:
                conn.execute("CREATE TABLE raw AS SELECT * FROM read_csv(?, header=true, sample_size=20000)", [src])
            except duckdb.Error as exc:
                raise InvalidFileError(f"Could not parse the CSV: {str(exc).splitlines()[0][:160]}") from exc
            described = conn.execute("DESCRIBE raw").fetchall()
            if not described:
                raise InvalidFileError("The CSV has no columns")
            originals = [row[0] for row in described]
            safe = sanitize_columns(originals)
            select = ", ".join(f'"{o.replace(chr(34), chr(34) * 2)}" AS "{s}"' for o, s in zip(originals, safe, strict=True))
            conn.execute(f"CREATE TABLE data AS SELECT {select} FROM raw")
            row_count = conn.execute("SELECT count(*) FROM data").fetchone()[0]
            if row_count == 0:
                raise InvalidFileError("The CSV has a header but no rows")
            if row_count > max_rows:
                raise InvalidFileError(f"The CSV has {row_count:,} rows; the limit is {max_rows:,}")
            columns = [_profile_column(conn, name, original, dtype)
                       for (original, dtype, *_), name in zip(described, safe, strict=True)]
            conn.execute(f"COPY data TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
        finally:
            conn.close()
        parquet = Path(out).read_bytes()
    return ProfiledDataset(
        name=Path(clean_name).stem.replace("_", " ").replace("-", " ").strip() or clean_name,
        filename=clean_name, checksum=hashlib.sha256(data).hexdigest(), row_count=int(row_count),
        columns=columns, parquet_bytes=parquet,
    )


def _profile_column(conn: duckdb.DuckDBPyConnection, name: str, original: str, dtype: str) -> dict[str, Any]:
    kind = column_kind(dtype)
    q = f'"{name}"'
    nulls, distinct = conn.execute(f"SELECT count(*) - count({q}), approx_count_distinct({q}) FROM data").fetchone()
    profile: dict[str, Any] = {"name": name, "original_name": original, "type": dtype, "kind": kind,
                               "nulls": int(nulls), "distinct": int(distinct)}
    if kind in {"number", "date"}:
        lo, hi = conn.execute(f"SELECT min({q}), max({q}) FROM data").fetchone()
        profile.update(min=_plain(lo), max=_plain(hi))
        if kind == "number":
            profile["mean"] = _plain(conn.execute(f"SELECT avg({q}) FROM data").fetchone()[0])
    if kind in {"text", "boolean"}:
        top = conn.execute(
            f"SELECT {q}, count(*) AS n FROM data WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY n DESC, 1 LIMIT 15"
        ).fetchall()
        profile["top_values"] = [str(v) for v, _ in top]
        profile["categorical"] = distinct <= 50
    return profile


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        return round(value, 4)
    return str(value)


class DatasetStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def path_for(self, tenant_id: str, dataset_id: str) -> Path:
        return self.root / str(tenant_id) / "datasets" / f"{dataset_id}.parquet"

    def save(self, tenant_id: str, dataset_id: str, data: bytes) -> str:
        path = self.path_for(tenant_id, dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return str(path)

    def delete(self, storage_path: str) -> None:
        path = Path(storage_path).resolve()
        if self.root.resolve() in path.parents and path.exists():
            path.unlink()
