"""Schema-aware Text2Cypher with strict validation.

Question -> schema-aware prompt -> generated Cypher -> validate -> inject tenant
filter -> reject dangerous operations -> execute read-only -> results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.access import current_scope
from app.core.errors import AppError, CypherValidationError
from app.core.logging import get_logger
from app.graph.cypher_validator import ValidatedCypher, validate_cypher
from app.graph.repository import GraphReader
from app.graph.schema import schema_prompt
from app.llm.client import LLMClient, to_messages

logger = get_logger(__name__)

_SYSTEM = """You translate questions into a single READ-ONLY Neo4j Cypher query.

Graph schema:
{schema}

Rules:
- Use only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, UNWIND.
- Never write data, never call procedures, never use subqueries, comments or backticks.
- Never reference tenant_id or query parameters; tenant scoping is added automatically.
- Match names case-insensitively, e.g. WHERE toLower(p.name) CONTAINS toLower('rahul').
- Bound variable-length paths, e.g. -[*1..3]-.
- RETURN readable scalar values (names, types) and include relationship types when useful.
- Always end with LIMIT 50 or less."""


class _GeneratedCypher(BaseModel):
    cypher: str = Field(max_length=2000)
    explanation: str = ""


class Text2Cypher:
    def __init__(self, llm: LLMClient, reader: GraphReader, max_rows: int = 50) -> None:
        self.llm = llm
        self.reader = reader
        self.max_rows = max_rows
        self.system = _SYSTEM.format(schema=schema_prompt())

    async def generate(self, question: str, hints: str = "") -> ValidatedCypher:
        user = f"Question: {question}\n{hints}".strip()
        generated = await self.llm.astructured(_GeneratedCypher, to_messages(self.system, user), task="text2cypher")
        return validate_cypher(generated.cypher, max_limit=self.max_rows)

    async def run(self, question: str, tenant_id: str, hints: str = "") -> tuple[ValidatedCypher | None, list[dict[str, Any]]]:
        scope = current_scope()
        if scope is None or scope.restricted:
            return None, []  # see GraphReader.run_validated_readonly
        try:
            validated = await self.generate(question, hints)
        except CypherValidationError as exc:
            logger.warning("text2cypher_rejected", extra={"reason": exc.message})
            return None, []
        except AppError as exc:
            logger.warning("text2cypher_generation_failed", extra={"error": exc.code})
            return None, []
        rows = await self.reader.run_validated_readonly(validated.query, {}, tenant_id, self.max_rows)
        logger.info("text2cypher_executed", extra={"rows": len(rows)})
        return validated, rows
