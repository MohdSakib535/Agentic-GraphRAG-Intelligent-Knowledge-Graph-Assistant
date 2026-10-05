"""Validation and sandboxed execution of SQL against a single dataset.

Generated SQL (LLM or planner) and user-edited SQL are untrusted:

1. DuckDB's own parser turns the query into an AST (``json_serialize_sql``); only a single
   SELECT statement is accepted.
2. The AST is walked: every table reference must be ``data`` (or a CTE defined in the query);
   table functions (``read_csv``, ``glob``...), string-literal file paths and file/system
   functions are rejected.
3. Execution happens in a fresh in-memory DuckDB holding only the ``data`` table, with
   external access disabled and configuration locked, a row limit and a timeout.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import duckdb

from app.core.errors import ValidationFailed

TABLE = "data"
_FORBIDDEN_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json", "read_json_auto", "read_ndjson",
    "read_text", "read_blob", "glob", "sniff_csv", "parquet_metadata", "parquet_schema", "parquet_file_metadata",
    "getenv", "current_setting", "duckdb_settings", "duckdb_extensions", "query", "query_table", "iceberg_scan",
    "delta_scan", "sqlite_scan", "postgres_scan", "mysql_scan",
}


class SQLRejected(ValidationFailed):
    code = "SQL_REJECTED"
    message = "The SQL query was rejected"


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: int


def _walk(node: Any, ctes: set[str]) -> None:
    if isinstance(node, dict):
        cte_map = node.get("cte_map")
        if isinstance(cte_map, dict):
            for entry in cte_map.get("map", []) or []:
                if isinstance(entry, dict) and entry.get("key"):
                    ctes.add(str(entry["key"]).lower())
        kind = node.get("type")
        if kind == "TABLE_FUNCTION":
            raise SQLRejected("Table functions are not allowed")
        if kind == "BASE_TABLE":
            name = str(node.get("table_name", "")).lower()
            if node.get("schema_name") or node.get("catalog_name") or (name != TABLE and name not in ctes):
                raise SQLRejected(f"Only the '{TABLE}' table can be queried")
        if node.get("class") == "FUNCTION" and str(node.get("function_name", "")).lower() in _FORBIDDEN_FUNCTIONS:
            raise SQLRejected(f"Function {node['function_name']} is not allowed")
        for value in node.values():
            _walk(value, ctes)
    elif isinstance(node, list):
        for item in node:
            _walk(item, ctes)


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if not sql or len(sql) > 5000:
        raise SQLRejected("Empty or overly long SQL")
    conn = duckdb.connect(":memory:")
    try:
        parsed = json.loads(conn.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    finally:
        conn.close()
    if parsed.get("error"):
        raise SQLRejected(f"Only a single SELECT query is allowed ({parsed.get('error_message', 'parse error')[:120]})")
    statements = parsed.get("statements") or []
    if len(statements) != 1:
        raise SQLRejected("Exactly one SELECT statement is allowed")
    ctes: set[str] = set()
    # CTE names must be known before table references are checked; walk twice (collect, then verify).
    _collect_ctes(statements[0], ctes)
    _walk(statements[0], ctes)
    return sql


def _collect_ctes(node: Any, ctes: set[str]) -> None:
    if isinstance(node, dict):
        cte_map = node.get("cte_map")
        if isinstance(cte_map, dict):
            for entry in cte_map.get("map", []) or []:
                if isinstance(entry, dict) and entry.get("key"):
                    ctes.add(str(entry["key"]).lower())
        for value in node.values():
            _collect_ctes(value, ctes)
    elif isinstance(node, list):
        for item in node:
            _collect_ctes(item, ctes)


def run_sql(parquet_path: str, sql: str, *, limit: int, timeout_seconds: float) -> QueryResult:
    """Execute validated SQL in an isolated, locked-down in-memory DuckDB."""
    sql = validate_sql(sql)
    conn = duckdb.connect(":memory:", config={"threads": 2, "memory_limit": "512MB"})
    timer = threading.Timer(timeout_seconds, conn.interrupt)
    started = time.perf_counter()
    try:
        conn.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM read_parquet(?)", [parquet_path])
        conn.execute("SET enable_external_access = false")
        conn.execute("SET lock_configuration = true")
        timer.start()
        cursor = conn.execute(f"SELECT * FROM ({sql}) AS q LIMIT {int(limit) + 1}")
        columns = [d[0] for d in cursor.description]
        rows = [list(r) for r in cursor.fetchall()]
    except duckdb.InterruptException as exc:
        raise SQLRejected(f"Query exceeded the {timeout_seconds:.0f}s time limit") from exc
    except duckdb.Error as exc:
        raise SQLRejected(f"Query failed: {str(exc).splitlines()[0][:200]}") from exc
    finally:
        timer.cancel()
        conn.close()
    truncated = len(rows) > limit
    return QueryResult(columns, [[_jsonable(v) for v in r] for r in rows[:limit]], truncated,
                       int((time.perf_counter() - started) * 1000))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return str(value)
