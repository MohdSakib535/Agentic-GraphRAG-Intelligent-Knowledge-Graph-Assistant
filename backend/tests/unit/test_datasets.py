"""Chat with CSV: profiling, SQL safety + sandbox, offline planner and the LLM text-to-SQL path."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import InvalidFileError
from app.datasets.planner import PlanError, RulePlanner, chart_for, summarize
from app.datasets.sql_safety import SQLRejected, run_sql, validate_sql
from app.datasets.store import profile_csv, sanitize_columns

CSV = (Path(__file__).resolve().parents[2] / "data" / "datasets" / "employees.csv").read_bytes()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    profiled = profile_csv("employees.csv", CSV, 10**7, 10**6)
    path = tmp_path_factory.mktemp("ds") / "employees.parquet"
    path.write_bytes(profiled.parquet_bytes)
    return profiled, str(path)


def ask(dataset, question: str):
    profiled, path = dataset

    def distinct(col: str) -> list[str]:
        return [str(r[0]) for r in run_sql(path, f'SELECT DISTINCT "{col}" FROM data', limit=2000, timeout_seconds=5).rows]

    plan = RulePlanner(profiled.columns, distinct, profiled.row_count).plan(question)
    result = run_sql(path, plan.sql, limit=200, timeout_seconds=5)
    return plan, result, summarize(question, plan, result, profiled.columns)


def test_profiling(dataset) -> None:
    profiled, _ = dataset
    assert profiled.row_count == 12
    cols = {c["name"]: c for c in profiled.columns}
    assert cols["annual_salary"]["kind"] == "number" and cols["annual_salary"]["max"] == 185000
    assert cols["hire_date"]["kind"] == "date" and cols["department"]["categorical"]
    assert set(cols["city"]["top_values"]) == {"Bangalore", "Pune", "Mumbai", "Delhi", "Hyderabad"}
    assert sanitize_columns(["Name", "name", "1st Score", "", "Revenue ($)"]) == ["name", "name_2", "c_1st_score", "column_4", "revenue"]
    with pytest.raises(InvalidFileError):
        profile_csv("x.csv", b"", 10**6, 100)
    with pytest.raises(InvalidFileError):
        profile_csv("x.csv", b"a,b\n", 10**6, 100)
    with pytest.raises(InvalidFileError):
        profile_csv("x.pdf", b"a,b\n1,2\n", 10**6, 100)


@pytest.mark.parametrize(("question", "expected"), [
    ("How many employees are there?", "There are 12"),
    ("How many employees work in Sales?", "There are 3"),
    ("What is the average salary?", "109,916.67"),
    ("Which department has the highest average salary?", "Engineering has the highest average Annual Salary: 138,000"),
    ("Who has the highest salary?", "Employee Name: Rahul; Annual Salary: 185,000"),
    ("Who earns the least in Sales?", "Vikram"),
    ("Total salary in Bangalore", "445,000"),
    ("Average salary of Engineering in Bangalore", "152,500"),
    ("Which city has the most employees?", "Bangalore"),
    ("List the cities", "Bangalore, Delhi, Hyderabad, Mumbai, Pune"),
    ("What is Amit's salary?", "120,000"),
])
def test_offline_planner_answers(dataset, question: str, expected: str) -> None:
    _, _, answer = ask(dataset, question)
    assert expected in answer


def test_grouped_results_suggest_charts(dataset) -> None:
    profiled, _ = dataset
    _, result, _ = ask(dataset, "What is the average salary by department?")
    assert len(result.rows) == 4 and chart_for(result, profiled.columns) == {
        "type": "bar", "x": "department", "y": "avg_annual_salary"}
    _, by_year, _ = ask(dataset, "Number of hires by year")
    assert chart_for(by_year, profiled.columns)["type"] == "line"
    _, over, _ = ask(dataset, "How many employees have a salary over 100k?")
    assert over.rows[0][0] == 6


def test_unmappable_question(dataset) -> None:
    with pytest.raises(PlanError):
        ask(dataset, "What is the weather today?")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM '/etc/passwd'", "DROP TABLE data", "SELECT 1; SELECT 2",
    "SELECT * FROM other_table", "SELECT * FROM main.data", "SELECT getenv('HOME')", "COPY data TO '/tmp/x.csv'",
    "ATTACH '/tmp/x.db' AS x", "INSTALL httpfs", "SELECT * FROM data, glob('*')", "PRAGMA database_list",
    "UPDATE data SET city = 'x'",
])
def test_unsafe_sql_is_rejected(sql: str) -> None:
    with pytest.raises(SQLRejected):
        validate_sql(sql)


def test_sandbox_limits_rows_and_blocks_file_access(dataset) -> None:
    _, path = dataset
    result = run_sql(path, "WITH t AS (SELECT * FROM data) SELECT * FROM t", limit=5, timeout_seconds=5)
    assert len(result.rows) == 5 and result.truncated
    with pytest.raises(SQLRejected):
        run_sql(path, "SELECT * FROM data WHERE no_such_column = 1", limit=5, timeout_seconds=5)


async def test_llm_text_to_sql_self_repairs_unsafe_sql(dataset, stub_llm) -> None:
    from app.datasets.planner import LLMPlanner

    profiled, path = dataset
    planner = LLMPlanner(stub_llm, profiled.columns, profiled.row_count)
    plan, result = await planner.plan("Average salary per department?",
                                      lambda sql: run_sql(path, sql, limit=200, timeout_seconds=5))
    assert "read_csv" not in plan.sql and result.rows[0][0] == "Engineering"
    answer = await planner.answer("Which department pays most?", result)
    assert answer.startswith("Engineering has the highest average salary")
