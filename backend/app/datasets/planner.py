"""Natural-language question -> SQL over a profiled dataset, and result -> answer.

* ``LLMPlanner`` - schema-aware text-to-SQL (with one self-repair attempt on rejected/failed SQL)
  and a grounded natural-language summary of the actual result rows.
* ``RulePlanner`` - deterministic offline planner covering the common analytical shapes:
  counts, sum/avg/min/max, group-by ("by department", "which team has the highest ..."),
  equality filters on known values, numeric comparisons, top-N, distinct lists and row lookups.

Every SQL string - whatever its origin - is validated and sandboxed by ``sql_safety``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.core.errors import AppError
from app.datasets.sql_safety import QueryResult, SQLRejected, validate_sql
from app.llm.client import LLMClient, to_messages
from app.utils.text import STOPWORDS, content_terms


class PlanError(AppError):
    code, status_code, message = "QUESTION_NOT_UNDERSTOOD", 422, "Could not translate the question into a query"


@dataclass
class Plan:
    sql: str
    explanation: str
    kind: str = "table"  # scalar | grouped | list | rows | table
    meta: dict[str, Any] = field(default_factory=dict)


class Runner(Protocol):
    def __call__(self, sql: str) -> QueryResult: ...


SYNONYMS = {
    "salary": {"pay", "compensation", "wage", "earn", "earning", "income"},
    "revenue": {"sales", "income", "turnover"},
    "department": {"dept", "team", "division"},
    "name": {"employee", "person", "people", "who", "staff"},
    "quantity": {"qty", "units", "count"},
    "price": {"cost", "amount"},
    "city": {"location", "office"},
    "date": {"day", "when"},
    "age": {"old"},
}
AGGREGATES = [
    ("avg", re.compile(r"\b(?:average|mean|avg)\b")),
    ("sum", re.compile(r"\b(?:total|sum|overall|combined)\b")),
    ("max", re.compile(r"\b(?:max|maximum|highest|largest|biggest|most|top|greatest)\b")),
    ("min", re.compile(r"\b(?:min|minimum|lowest|smallest|least|bottom|fewest)\b")),
    ("count", re.compile(r"\b(?:how many|number of|count)\b")),
]
_COMPARATORS = [
    (">=", r"(?:>=|at least|no less than)"), ("<=", r"(?:<=|at most|no more than)"),
    (">", r"(?:>|greater than|more than|over|above|exceeds?|higher than)"),
    ("<", r"(?:<|less than|under|below|lower than|fewer than)"),
]
_NUMBER = r"(-?\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?"


def _num(text: str, suffix: str | None) -> float:
    value = float(text.replace(",", ""))
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get((suffix or "").lower(), 1)
    return value * mult


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _lit(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


class RulePlanner:
    name = "rules"

    def __init__(self, columns: list[dict[str, Any]], distinct_values: Any = None, row_count: int | None = None) -> None:
        self.columns = columns
        self.distinct_values = distinct_values  # callable(col) -> list[str] for value matching
        self.row_count = row_count or max((c.get("distinct", 0) + c.get("nulls", 0) for c in columns), default=0)

    def _identifier_column(self) -> dict[str, Any] | None:
        """The column naming a row ("Employee Name"), used for "who ..." questions."""
        people = [c for c in self.columns if c["kind"] == "text"
                  and self._col_terms(c) & {"name", "employee", "person", "people", "staff", "customer", "user"}]
        return (people or [c for c in self.columns if c["kind"] == "text"] or [None])[0]

    def _is_unique(self, col: dict[str, Any]) -> bool:
        return bool(self.row_count) and col.get("distinct", 0) >= 0.9 * self.row_count

    # ------------------------------------------------------------- matching
    def _col_terms(self, col: dict[str, Any]) -> set[str]:
        terms = set(content_terms(col["name"].replace("_", " "))) | set(content_terms(str(col.get("original_name", ""))))
        for key, syns in SYNONYMS.items():
            if key in terms or any(t.startswith(key) for t in terms):
                terms |= syns
        return terms

    def mentioned(self, question: str) -> list[dict[str, Any]]:
        q_terms = set(content_terms(question))
        scored = []
        for col in self.columns:
            terms = self._col_terms(col)
            hit = q_terms & terms
            fuzzy = max((fuzz.ratio(a, b) for a in q_terms for b in terms), default=0)
            if hit or fuzzy >= 88:
                pos = min((question.lower().find(t) for t in (hit or terms) if question.lower().find(t) >= 0), default=999)
                scored.append((pos, col))
        return [c for _, c in sorted(scored, key=lambda x: x[0])]

    def value_filters(self, question: str) -> list[tuple[dict[str, Any], str]]:
        found: list[tuple[dict[str, Any], str]] = []
        lowered = question.lower()
        for col in self.columns:
            if col["kind"] not in {"text", "boolean"}:
                continue
            values = col.get("top_values") or []
            if self.distinct_values is not None and col.get("distinct", 0) <= 2000:
                values = self.distinct_values(col["name"])
            for value in sorted(values, key=lambda v: -len(str(v))):
                v = str(value)
                if len(v) < 2 or v.lower() in STOPWORDS:
                    continue
                if re.search(rf"(?<![\w]){re.escape(v.lower())}(?![\w])", lowered):
                    found.append((col, v))
                    lowered = lowered.replace(v.lower(), " ")
                    break
        return found

    def numeric_filters(self, question: str, numeric_cols: list[dict[str, Any]]) -> list[str]:
        clauses = []
        lowered = question.lower()
        between = re.search(rf"\bbetween\s+{_NUMBER}\s+and\s+{_NUMBER}", lowered)
        target = numeric_cols[0] if numeric_cols else None
        if between and target:
            lo, hi = _num(between.group(1), between.group(2)), _num(between.group(3), between.group(4))
            clauses.append(f"{_q(target['name'])} BETWEEN {lo:g} AND {hi:g}")
        for op, pattern in _COMPARATORS:
            for m in re.finditer(rf"{pattern}\s*\$?\s*{_NUMBER}", lowered):
                if between and between.start() <= m.start() <= between.end():
                    continue
                before = lowered[: m.start()]
                col = next((c for c in reversed(numeric_cols)
                            if any(before.rfind(t) >= 0 for t in self._col_terms(c))), target)
                if col is not None:
                    clauses.append(f"{_q(col['name'])} {op} {_num(m.group(1), m.group(2)):g}")
        return clauses

    # ------------------------------------------------------------------ plan
    def plan(self, question: str) -> Plan:
        q = question.lower().strip()
        mentioned = self.mentioned(question)
        filters = self.value_filters(question)
        filter_cols = {c["name"] for c, _ in filters}
        numeric = [c for c in mentioned if c["kind"] == "number"]
        dims = [c for c in mentioned if c["kind"] in {"text", "boolean", "date"} and c["name"] not in filter_cols]
        ident = self._identifier_column()
        if re.search(r"\bwho(?:m|se)?\b", q) and ident is not None and ident not in dims and ident["name"] not in filter_cols:
            dims.insert(0, ident)
        where = [f"{_q(c['name'])} = {_lit(v)}" for c, v in filters] + self.numeric_filters(question, numeric)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        filter_text = " and ".join(f"{c['original_name']} = {v}" for c, v in filters)
        agg = next((name for name, rx in AGGREGATES if rx.search(q)), None)
        top_n = re.search(r"\b(top|bottom|first|last)\s+(\d+)\b", q)
        group_hint = re.search(r"\b(?:by|per|for each|each|across|which|what)\s+(\w+)", q)
        group_col = None
        if group_hint:
            word = group_hint.group(1)
            group_col = next((c for c in dims if word in self._col_terms(c) or fuzz.ratio(word, c["name"]) >= 85), None)
        if group_col is None and re.search(r"\b(?:by|per|for each|breakdown)\b", q) and dims:
            group_col = dims[0]
        date_bucket = None
        if re.search(r"\b(?:by|per|each)\s+(month|year|day|week)\b|\b(monthly|yearly|daily|weekly)\b", q):
            unit = re.search(r"(month|year|day|week)", q).group(1)  # type: ignore[union-attr]
            date_col = next((c for c in self.columns if c["kind"] == "date"), None)
            if date_col:
                date_bucket = (date_col, unit)

        # "how many rows/records" / "how many employees in Sales"
        if agg == "count" and not numeric and not group_col and not date_bucket:
            return Plan(f"SELECT count(*) AS count FROM data{where_sql}", f"Count rows{' where ' + filter_text if filter_text else ''}",
                        "scalar", {"agg": "count", "filters": filter_text})

        # Distinct listing: "list the departments", "what are the unique cities"
        if re.search(r"\b(?:list|unique|distinct|what are the|which are the|show all)\b", q) and dims and not numeric \
                and agg is None:
            col = group_col or dims[0]
            return Plan(f"SELECT DISTINCT {_q(col['name'])} FROM data{where_sql} ORDER BY 1",
                        f"Distinct values of {col['original_name']}", "list", {"column": col["original_name"]})

        metric = numeric[0] if numeric else None
        if group_col is not None and self._is_unique(group_col) and (top_n or agg in {"max", "min"}) and metric:
            group_col = None  # "top 3 employees by salary": rank rows, don't group a unique column
        if agg in {"avg", "sum", "max", "min", "count"} and (
                metric or agg == "count" or (agg in {"max", "min"} and group_col is not None)):
            func = {"avg": "avg", "sum": "sum", "max": "max", "min": "min", "count": "count"}[agg]
            expr = f"{func}({_q(metric['name'])})" if metric and agg != "count" else "count(*)"
            alias = f"{agg}_{metric['name']}" if metric and agg != "count" else "count"
            if date_bucket:
                col, unit = date_bucket
                fmt = {"month": "%Y-%m", "year": "%Y", "day": "%Y-%m-%d", "week": "%G-W%V"}[unit]
                return Plan(f"SELECT strftime({_q(col['name'])}, '{fmt}') AS {unit}, {expr} AS {alias} FROM data"
                            f"{where_sql} GROUP BY 1 ORDER BY 1", f"{agg} of {metric['original_name'] if metric else 'rows'} per {unit}",
                            "grouped", {"agg": agg, "metric": metric, "group": unit, "ordered": False})
            if group_col:
                superlative = re.search(r"\b(?:which|what)\b", q) and agg in {"max", "min"} or (
                    re.search(r"\b(?:highest|lowest|most|least|top|bottom)\b", q) and re.search(r"\bwhich|what\b", q))
                inner = agg if agg not in {"max", "min"} or not superlative else _inner_agg(q)
                inner_expr = f"{inner}({_q(metric['name'])})" if metric and inner != "count" else "count(*)"
                inner_alias = f"{inner}_{metric['name']}" if metric and inner != "count" else "count"
                order = "ASC" if agg == "min" or (top_n and top_n.group(1) in {"bottom", "last"}) else "DESC"
                limit = f" LIMIT {int(top_n.group(2))}" if top_n else (" LIMIT 1" if superlative else "")
                sql = (f"SELECT {_q(group_col['name'])}, {inner_expr} AS {inner_alias} FROM data{where_sql} "
                       f"GROUP BY 1 ORDER BY 2 {order}{limit}")
                return Plan(sql, f"{inner} of {metric['original_name'] if metric else 'rows'} by {group_col['original_name']}",
                            "grouped", {"agg": inner, "metric": metric, "group": group_col["original_name"],
                                        "ordered": bool(superlative or top_n), "direction": order, "filters": filter_text})
            if agg in {"max", "min"} and (dims or top_n):
                # "who has the highest salary" -> the row(s) holding the extreme value
                show = dims[0] if dims else next((c for c in self.columns if c["kind"] == "text"), None)
                cols = ", ".join(_q(c["name"]) for c in [show, metric] if c)
                order = "DESC" if agg == "max" else "ASC"
                limit = int(top_n.group(2)) if top_n else 1
                return Plan(f"SELECT {cols} FROM data{where_sql} ORDER BY {_q(metric['name'])} {order} LIMIT {limit}",
                            f"Rows with the {agg} {metric['original_name']}", "rows",
                            {"agg": agg, "metric": metric, "filters": filter_text})
            return Plan(f"SELECT {expr} AS {alias} FROM data{where_sql}",
                        f"{agg} of {metric['original_name'] if metric else 'rows'}", "scalar",
                        {"agg": agg, "metric": metric, "filters": filter_text})

        if top_n and metric:
            order = "ASC" if top_n.group(1) in {"bottom", "last"} else "DESC"
            show = [c for c in self.columns if c["kind"] == "text"][:1] + [metric]
            return Plan(f"SELECT {', '.join(_q(c['name']) for c in show)} FROM data{where_sql} "
                        f"ORDER BY {_q(metric['name'])} {order} LIMIT {int(top_n.group(2))}",
                        f"Top {top_n.group(2)} by {metric['original_name']}", "rows", {"metric": metric})
        if where and (numeric or dims):
            cols = numeric + [c for c in dims if c not in numeric]
            key = next((c for c in self.columns if c["kind"] == "text" and c["name"] not in {x["name"] for x in cols}), None)
            select = ", ".join(_q(c["name"]) for c in ([key] if key else []) + cols)
            return Plan(f"SELECT {select} FROM data{where_sql} LIMIT 50",
                        f"{', '.join(c['original_name'] for c in cols)} where {filter_text or 'conditions match'}",
                        "rows", {"filters": filter_text})
        if where:
            return Plan(f"SELECT * FROM data{where_sql} LIMIT 50", f"Rows where {filter_text or 'conditions match'}",
                        "rows", {"filters": filter_text})
        raise PlanError("I couldn't map that question to the columns of this dataset. Try naming a column, e.g. "
                        + ", ".join(c["original_name"] for c in self.columns[:5]))


def _inner_agg(q: str) -> str:
    for name, rx in AGGREGATES:
        if name in {"avg", "sum", "count"} and rx.search(q):
            return name
    return "sum" if re.search(r"\b(?:sales|revenue|amount|spend|cost)\b", q) else "count"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1e15 else f"{value:.3g}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def summarize(question: str, plan: Plan, result: QueryResult, columns: list[dict[str, Any]] | None = None) -> str:
    """Deterministic, result-grounded answer text."""
    names = {c["name"]: c.get("original_name", c["name"]) for c in columns or []}
    if not result.rows:
        return "No rows match that question."
    meta = plan.meta
    where = f" (where {meta['filters']})" if meta.get("filters") else ""
    if plan.kind == "scalar":
        value = result.rows[0][0]
        if meta.get("agg") == "count":
            return f"There are {_fmt(value)} matching rows{where}."
        metric = meta.get("metric") or {}
        label = {"avg": "average", "sum": "total", "max": "maximum", "min": "minimum"}.get(meta.get("agg", ""), "")
        return f"The {label} {metric.get('original_name', 'value')}{where} is {_fmt(value)}."
    if plan.kind == "list":
        values = [str(r[0]) for r in result.rows]
        shown = ", ".join(values[:20]) + (" …" if len(values) > 20 else "")
        return f"There are {len(values)}{'+' if result.truncated else ''} distinct {meta.get('column')} values: {shown}."
    if plan.kind == "grouped":
        first = result.rows[0]
        if meta.get("ordered") and len(result.rows) == 1:
            word = "highest" if meta.get("direction") == "DESC" else "lowest"
            return f"{first[0]} has the {word} {_label(meta)}{where}: {_fmt(first[1])}."
        if meta.get("ordered"):
            listed = "; ".join(f"{r[0]}: {_fmt(r[1])}" for r in result.rows[:10])
            return f"{_label(meta).capitalize()} by {meta.get('group')}{where} — {listed}."
        return f"{_label(meta).capitalize()} by {meta.get('group')}{where} for {len(result.rows)} groups (table below)."
    if plan.kind == "rows" and len(result.rows) == 1:
        return "; ".join(f"{names.get(c, c)}: {_fmt(v)}" for c, v in zip(result.columns, result.rows[0], strict=False)) + "."
    return f"{len(result.rows)}{'+' if result.truncated else ''} matching rows (table below)."


def _label(meta: dict[str, Any]) -> str:
    metric = (meta.get("metric") or {}).get("original_name")
    agg = {"avg": "average", "sum": "total", "count": "count", "max": "maximum", "min": "minimum"}.get(meta.get("agg", ""), "")
    return f"{agg} {metric}" if metric and agg != "count" else "number of records"


def chart_for(result: QueryResult, columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(result.columns) != 2 or not 1 < len(result.rows) <= 60:
        return None
    if not all(isinstance(r[1], (int, float)) and not isinstance(r[1], bool) for r in result.rows if r[1] is not None):
        return None
    x = result.columns[0]
    temporal = x in {"month", "year", "day", "week"} or any(c["name"] == x and c["kind"] == "date" for c in columns)
    return {"type": "line" if temporal else "bar", "x": x, "y": result.columns[1]}


# ----------------------------------------------------------------- LLM planner
class _SQLPlan(BaseModel):
    sql: str = Field(max_length=4000)
    explanation: str = Field(default="", max_length=500)


class _Answer(BaseModel):
    answer: str = Field(max_length=2000)


_SQL_SYSTEM = """You write a single DuckDB SQL SELECT query over one table named `data` to answer a question.
Rules: use only the listed columns (exact names, double-quoted if needed); never use other tables, files or
table functions; prefer aggregates/GROUP BY for analytical questions; add ORDER BY for rankings; LIMIT row
listings to 50. Return the SQL and a one-sentence explanation."""

_ANSWER_SYSTEM = """Answer the user's question using ONLY the query result rows provided. State numbers exactly as
given (you may round sensibly). If the result is empty, say no rows matched. Do not invent values."""


def schema_text(columns: list[dict[str, Any]], row_count: int) -> str:
    lines = [f"Table `data` ({row_count} rows):"]
    for c in columns:
        extra = []
        if c.get("top_values"):
            extra.append("e.g. " + ", ".join(repr(v) for v in c["top_values"][:6]))
        if c.get("min") is not None:
            extra.append(f"range {c['min']} .. {c['max']}")
        lines.append(f'- "{c["name"]}" {c["type"]} (from "{c["original_name"]}")' + (f"; {'; '.join(extra)}" if extra else ""))
    return "\n".join(lines)


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: LLMClient, columns: list[dict[str, Any]], row_count: int) -> None:
        self.llm = llm
        self.columns = columns
        self.schema = schema_text(columns, row_count)

    async def plan(self, question: str, run: Runner) -> tuple[Plan, QueryResult]:
        prompt = f"{self.schema}\n\nQuestion: {question}"
        last_error = ""
        for _ in range(2):  # one self-repair attempt
            generated = await self.llm.astructured(
                _SQLPlan, to_messages(_SQL_SYSTEM, prompt + (f"\n\nYour previous SQL failed: {last_error}" if last_error else "")),
                task="text2sql")
            try:
                sql = validate_sql(generated.sql)
                return Plan(sql, generated.explanation, "table"), run(sql)
            except SQLRejected as exc:
                last_error = exc.message
        raise SQLRejected(f"Generated SQL was rejected: {last_error}")

    async def answer(self, question: str, result: QueryResult) -> str:
        head = "\n".join(" | ".join(str(v) for v in row) for row in result.rows[:50])
        prompt = (f"Question: {question}\nColumns: {' | '.join(result.columns)}\nRows ({len(result.rows)}"
                  f"{'+, truncated' if result.truncated else ''}):\n{head or '(no rows)'}")
        out = await self.llm.astructured(_Answer, to_messages(_ANSWER_SYSTEM, prompt), task="dataset_answer")
        return out.answer
