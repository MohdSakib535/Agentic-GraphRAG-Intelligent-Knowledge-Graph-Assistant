"""Validation and tenant-scoping of LLM-generated Cypher.

LLM-generated Cypher is treated as hostile input:

1. String literals are masked so keywords hidden inside strings cannot confuse
   the analysis (and real keywords cannot hide as strings).
2. Comments, multiple statements, backticks, procedure calls, subqueries,
   admin commands and every write/DDL keyword are rejected.
3. Only the clause whitelist (MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN,
   ORDER BY, LIMIT, UNWIND, SKIP, DISTINCT, AS ...) is accepted.
4. Labels / relationship types must exist in the graph schema; variable-length
   patterns must be bounded.
5. **Every node pattern is rewritten to include ``tenant_id: $tenant_id``**, so
   the query can only ever touch the caller's tenant.
6. A LIMIT is enforced (appended or clamped).

The rewritten query is then executed in a READ transaction with a timeout, and
returned rows are post-filtered for foreign-tenant nodes (see ``GraphReader``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.errors import CypherValidationError
from app.graph.schema import ALL_RELATION_TYPES, NODE_LABELS

MAX_QUERY_LENGTH = 2000
MAX_LIMIT = 100
MAX_VAR_LENGTH = 4

FORBIDDEN_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "LOAD", "CALL", "FOREACH", "USE",
    "SHOW", "GRANT", "DENY", "REVOKE", "ALTER", "RENAME", "START", "STOP", "TERMINATE", "IMPORT", "FINISH",
    "INSERT", "YIELD", "EXISTS", "COLLECT", "COUNT",
)
# EXISTS/COLLECT/COUNT are forbidden only as subquery openers ("EXISTS {"), handled below.
_SUBQUERY_OPENERS = ("EXISTS", "COLLECT", "COUNT")
ALLOWED_CLAUSES = ("MATCH", "OPTIONAL", "WHERE", "RETURN", "WITH", "ORDER", "BY", "LIMIT", "UNWIND", "SKIP", "DISTINCT")

_STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_COMMENT_RE = re.compile(r"//|/\*|\*/")
_DANGEROUS_FUNC_RE = re.compile(r"\b(apoc|dbms|db|gds|genai|file|io)\s*\.", re.IGNORECASE)
_LABEL_RE = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_]*)")
_MAP_LITERAL_RE = re.compile(r"\{[^{}]*\}")
_NODE_PATTERN_RE = re.compile(
    r"(?<![\w$])\(\s*([A-Za-z_][A-Za-z0-9_]*)?\s*((?::\s*[A-Za-z_][A-Za-z0-9_]*\s*)*)(\{[^{}()]*\})?\s*\)"
)
_REL_PATTERN_RE = re.compile(r"\[([^\[\]]*)\]")
_VAR_LENGTH_RE = re.compile(r"\*\s*(\d*)\s*(?:\.\.\s*(\d*))?")
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+|\$\w+)\s*$", re.IGNORECASE)
_TENANT_REF_RE = re.compile(r"tenant_id", re.IGNORECASE)
# Keyword tokens, ignoring property accesses such as ``p.role`` or ``c.start``.
_KEYWORD_TOKEN_RE = re.compile(r"(?<![.\w$])[A-Za-z_][A-Za-z0-9_]*")
_NODE_OPEN_RE = re.compile(r"(?<![\w$])\(\s*[A-Za-z_]?[A-Za-z0-9_]*\s*(?=[:{)])")


@dataclass(frozen=True)
class ValidatedCypher:
    original: str
    query: str


def _mask_strings(query: str) -> tuple[str, list[str]]:
    literals: list[str] = []

    def repl(match: re.Match[str]) -> str:
        literals.append(match.group(0))
        return f"__STR{len(literals) - 1}__"

    return _STRING_RE.sub(repl, query), literals


def _unmask(query: str, literals: list[str]) -> str:
    for i, lit in enumerate(literals):
        query = query.replace(f"__STR{i}__", lit)
    return query


def _reject(reason: str) -> CypherValidationError:
    return CypherValidationError(f"Generated Cypher rejected: {reason}")


def validate_cypher(query: str, *, max_limit: int = MAX_LIMIT) -> ValidatedCypher:
    if not query or not query.strip():
        raise _reject("empty query")
    original = query.strip()
    if len(original) > MAX_QUERY_LENGTH:
        raise _reject("query too long")
    if original.startswith("```"):
        original = re.sub(r"^```(?:cypher)?|```$", "", original, flags=re.IGNORECASE).strip()
    masked, literals = _mask_strings(original)
    if "'" in masked or '"' in masked:
        raise _reject("unbalanced string literal")
    if _COMMENT_RE.search(masked):
        raise _reject("comments are not allowed")
    if "`" in masked:
        raise _reject("backtick identifiers are not allowed")
    stripped = masked.rstrip().rstrip(";")
    if ";" in stripped:
        raise _reject("multiple statements are not allowed")
    if _DANGEROUS_FUNC_RE.search(stripped):
        raise _reject("procedures and system functions are not allowed")
    upper_tokens = [t.upper() for t in _KEYWORD_TOKEN_RE.findall(stripped)]
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in upper_tokens:
            if keyword in _SUBQUERY_OPENERS:
                if re.search(rf"\b{keyword}\s*\{{", stripped, re.IGNORECASE):
                    raise _reject(f"{keyword} subqueries are not allowed")
                continue
            raise _reject(f"forbidden keyword {keyword}")
    if "{" in re.sub(_NODE_PATTERN_RE, "", stripped) and re.search(r"\{\s*\(", stripped):
        raise _reject("pattern subqueries are not allowed")
    first = upper_tokens[0] if upper_tokens else ""
    if first not in {"MATCH", "OPTIONAL", "WITH", "UNWIND"}:
        raise _reject("query must start with MATCH, OPTIONAL MATCH, WITH or UNWIND")
    if "RETURN" not in upper_tokens:
        raise _reject("query must RETURN results")
    if _TENANT_REF_RE.search(stripped):
        raise _reject("tenant_id must not be referenced by generated queries")
    if "$" in stripped.replace("$tenant_id", ""):
        raise _reject("parameters other than $tenant_id are not allowed")

    # Schema check for labels and relationship types.
    # With map literals removed, every remaining ":Name" is a label or relationship type.
    for label in _LABEL_RE.findall(_MAP_LITERAL_RE.sub("", stripped)):
        if label not in NODE_LABELS and label not in ALL_RELATION_TYPES:
            raise _reject(f"unknown label or relationship type '{label}'")
    for rel_body in _REL_PATTERN_RE.findall(stripped):
        if "|" in rel_body.replace(":", ""):
            types = [t.strip(" :") for t in rel_body.split(":", 1)[-1].split("|")]
            for t in types:
                t = re.split(r"[\s*{]", t)[0]
                if t and t not in ALL_RELATION_TYPES:
                    raise _reject(f"unknown relationship type '{t}'")
        for lo, hi in _VAR_LENGTH_RE.findall(rel_body):
            if "*" in rel_body and (not hi or int(hi) > MAX_VAR_LENGTH):
                raise _reject(f"variable-length patterns must be bounded (max {MAX_VAR_LENGTH} hops)")
            if lo and int(lo) > MAX_VAR_LENGTH:
                raise _reject("variable-length lower bound too large")

    scoped = _inject_tenant(stripped)
    _assert_all_nodes_scoped(scoped)
    scoped = _enforce_limit(scoped, max_limit)
    return ValidatedCypher(original=original, query=_unmask(scoped, literals))


def _inject_tenant(query: str) -> str:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        var, labels, props = match.group(1) or "", match.group(2) or "", match.group(3)
        labels = labels.strip()
        if props:
            inner = props.strip()[1:-1].strip()
            new_props = "{" + (inner + ", " if inner else "") + "tenant_id: $tenant_id}"
        else:
            new_props = "{tenant_id: $tenant_id}"
        head = f"{var}{labels}".strip()
        return f"({head} {new_props})" if head else f"({new_props})"

    rewritten = _NODE_PATTERN_RE.sub(repl, query)
    if count == 0:
        raise _reject("query contains no node patterns")
    return rewritten


def _matching_paren(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise _reject("unbalanced parentheses")


def _assert_all_nodes_scoped(query: str) -> None:
    """Every node-shaped pattern must carry the injected tenant constraint.

    Catches node patterns the rewrite could not handle (nested maps, inline
    ``WHERE`` inside a node pattern, ...) - those are rejected rather than run unscoped.
    """
    for match in _NODE_OPEN_RE.finditer(query):
        end = _matching_paren(query, match.start())
        if "tenant_id: $tenant_id" not in query[match.start() : end + 1]:
            raise _reject("unsupported node pattern syntax")


def _enforce_limit(query: str, max_limit: int) -> str:
    match = _LIMIT_RE.search(query)
    if match:
        value = match.group(1)
        if value.startswith("$") or int(value) > max_limit:
            return query[: match.start()] + f"LIMIT {max_limit}"
        return query
    return f"{query}\nLIMIT {max_limit}"
