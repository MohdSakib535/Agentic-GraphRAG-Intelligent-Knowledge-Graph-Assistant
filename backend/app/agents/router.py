"""Query analysis and retrieval routing.

The analyzer produces a structured :class:`QueryAnalysis` (intent, entities,
relationships, temporal constraints, answer type, strategy).

* With an LLM configured, a structured classifier decides - except for trivially
  classifiable questions (small talk, plain definitions), which skip the LLM call.
* The decision is grounded in the tenant graph: candidate entities are linked
  against Neo4j, so "who manages X" only routes to GRAPH when X actually exists.
* Coreference ("that project", "he") is resolved against conversation memory.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.core.errors import AppError
from app.core.logging import get_logger
from app.graph.schema import ENTITY_TYPES, RELATION_DESCRIPTIONS, RelationType
from app.ingestion.entity_extractor import TECHNOLOGIES, HeuristicEntityExtractor
from app.llm.client import LLMClient, to_messages
from app.retrieval.graph import GraphRetriever
from app.retrieval.query_parsing import (
    expected_answer_type,
    filename_mentions,
    is_definition_question,
    is_smalltalk,
    relation_hints,
    temporal_constraints,
)
from app.retrieval.types import LinkedEntity
from app.utils.text import STOPWORDS

logger = get_logger(__name__)

StrategyName = Literal["VECTOR", "GRAPH", "HYBRID", "DIRECT"]
Intent = Literal[
    "definition", "factual_lookup", "relationship_lookup", "multi_hop", "comparison", "aggregation",
    "document_lookup", "summary", "smalltalk", "other",
]


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(max_length=2000)
    intent: Intent = "other"
    entities: list[str] = Field(default_factory=list, max_length=15)
    relationships: list[str] = Field(default_factory=list, max_length=8)
    temporal_constraints: list[str] = Field(default_factory=list, max_length=5)
    answer_type: str | None = None
    retrieval_strategy: StrategyName = "HYBRID"
    reasoning: str = Field(default="", max_length=600)

    @field_validator("relationships")
    @classmethod
    def _whitelist(cls, value: list[str]) -> list[str]:
        allowed = {r.value for r in RelationType}
        return [v.strip().upper() for v in value if v.strip().upper() in allowed]

    @field_validator("answer_type")
    @classmethod
    def _entity_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return next((t for t in ENTITY_TYPES if t.lower() == value.strip().lower()), None)

    @field_validator("entities")
    @classmethod
    def _clean_entities(cls, value: list[str]) -> list[str]:
        return [" ".join(v.split())[:120] for v in value if v and v.strip()]


# ------------------------------------------------------------- coreference
_ANAPHORA = {
    "Project": re.compile(r"\b(?:that|this|the same|the|said)\s+project\b", re.I),
    "Company": re.compile(r"\b(?:that|this|the same|the)\s+company\b", re.I),
    "Technology": re.compile(r"\b(?:that|this|the same)\s+(?:technology|tool|framework|database)\b", re.I),
    "Department": re.compile(r"\b(?:that|this|the same)\s+(?:team|department)\b", re.I),
    "Person": re.compile(r"\b(?:he|she|him|her|that person|this person)\b", re.I),
}
_POSSESSIVE = re.compile(r"\b(?:his|her)\b", re.I)
_IT = re.compile(r"\b(?:it|its)\b", re.I)
_THEY = re.compile(r"\b(?:they|them|their|those projects|these projects)\b", re.I)


def resolve_coreferences(question: str, focus: list[dict[str, str]]) -> str:
    """Replace anaphora with the most recent compatible entity from conversation memory."""
    if not focus:
        return question
    resolved = question
    for etype, pattern in _ANAPHORA.items():
        candidate = next((f["name"] for f in focus if f.get("type") == etype), None)
        if candidate and pattern.search(resolved):
            resolved = pattern.sub(candidate, resolved, count=1)
    person = next((f["name"] for f in focus if f.get("type") == "Person"), None)
    if person and _POSSESSIVE.search(resolved):
        resolved = _POSSESSIVE.sub(f"{person}'s", resolved, count=1)
    if _IT.search(resolved):
        primary = next((f["name"] for f in focus if f.get("type") != "Person"), None)
        if primary:
            resolved = _IT.sub(primary, resolved, count=1)
    if _THEY.search(resolved):
        names = [f["name"] for f in focus[:3]]
        if len(names) > 1:
            resolved = _THEY.sub(" and ".join(names), resolved, count=1)
    return resolved


_CAP_SPAN = re.compile(r"\b[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*")
_QUOTED = re.compile(r"[\"']([^\"']{2,60})[\"']")
_QUESTION_CAPS = {"Who", "What", "Which", "Where", "When", "Why", "How", "Does", "Do", "Is", "Are", "Can", "Tell", "List",
                  "Show", "Give", "Explain", "Describe", "I", "The", "A", "An", "Please", "In", "Of"}


def candidate_entities(question: str) -> list[str]:
    """Cheap entity candidates from a question (known patterns, capitalised spans, quotes)."""
    extractor = HeuristicEntityExtractor()
    found = [m.name for m in extractor.mentions(question)]
    for match in _CAP_SPAN.finditer(question):
        tokens = [t for t in match.group(0).split() if t not in _QUESTION_CAPS]
        span = " ".join(tokens)
        if span and span.lower() not in STOPWORDS and len(span) > 1:
            found.append(span)
    found.extend(m.group(1) for m in _QUOTED.finditer(question))
    for surface, canonical in TECHNOLOGIES.items():
        if re.search(rf"(?<![\w]){re.escape(surface)}(?![\w])", question, re.IGNORECASE):
            found.append(canonical)
    # De-duplicate (case-insensitive), dropping spans contained in longer ones.
    unique: list[str] = []
    for name in sorted(set(found), key=len, reverse=True):
        if not any(name.lower() in u.lower() for u in unique):
            unique.append(name)
    return unique


def choose_strategy(question: str, hints: list[str], linked: list[LinkedEntity], answer_type: str | None) -> tuple[StrategyName, Intent, str]:
    """Deterministic routing policy (also used to sanity-check LLM decisions)."""
    if is_smalltalk(question):
        return "DIRECT", "smalltalk", "small talk / meta question - no retrieval needed"
    semantic_hints = [h for h in hints if h != RelationType.RELATED_TO.value]
    asks_for_sources = bool(re.search(r"\b(?:documents?|sources?|evidence|cite|citations?|support)\b", question, re.I))
    n_linked = len(linked)
    if asks_for_sources and n_linked:
        return "HYBRID", "document_lookup", "asks for supporting documents of graph facts"
    if is_definition_question(question) and not semantic_hints:
        return "VECTOR", "definition", "definition/explanation question - semantic passage retrieval"
    if is_definition_question(question) and semantic_hints and n_linked:
        return "HYBRID", "summary", "explanation about related entities - needs passages and graph facts"
    if semantic_hints and n_linked:
        multi_hop = len(set(semantic_hints)) >= 2 or (n_linked >= 2 and answer_type is not None)
        if multi_hop:
            return "HYBRID", "multi_hop", f"multi-hop relational question over {n_linked} linked entities"
        if re.search(r"\b(?:shared|common|both|compare|difference)\b", question, re.I):
            return "HYBRID", "comparison", "comparison across entities"
        return "GRAPH", "relationship_lookup", "single-relation lookup on linked graph entities"
    if semantic_hints and not n_linked:
        return "HYBRID", "relationship_lookup", "relational cues but no linked entity - combine graph and text"
    if n_linked and answer_type:
        return "HYBRID", "factual_lookup", "entity-centric question - combine graph neighbourhood and text"
    return "VECTOR", "factual_lookup" if n_linked else "other", "semantic search over document chunks"


_ANALYZER_SYSTEM = """You analyse questions for an enterprise knowledge-graph assistant and choose a retrieval strategy.

Knowledge graph relationships:
{relations}

Strategies:
- VECTOR: definitions, explanations, descriptive or open-ended questions answered by passages ("What is Kafka?").
- GRAPH: direct relationship lookups about named entities ("Who manages Project Alpha?", "Which projects use Kafka?").
- HYBRID: multi-hop or constrained relational questions, or questions needing both facts and passages
  ("Which developers work on Kafka projects managed by Rahul?").
- DIRECT: greetings or questions about the assistant itself. Never use DIRECT for factual questions.

Rewrite the question into a standalone question using the conversation history (resolve "that project", "he", "it").
List the named entities exactly as written. Use only the relationship names listed above."""


class QueryAnalyzer:
    def __init__(self, graph: GraphRetriever, llm: LLMClient | None = None) -> None:
        self.graph = graph
        self.llm = llm
        relations = "\n".join(f"- {d}" for d in RELATION_DESCRIPTIONS.values())
        self.system = _ANALYZER_SYSTEM.format(relations=relations)

    async def analyze(
        self, question: str, tenant_id: str, history: list[dict[str, str]] | None = None,
        focus: list[dict[str, str]] | None = None,
    ) -> tuple[QueryAnalysis, list[LinkedEntity]]:
        standalone = resolve_coreferences(question, focus or [])
        heuristic = await self._heuristic(standalone, tenant_id)
        analysis, linked = heuristic
        # Avoid an LLM call when the deterministic policy is already unambiguous.
        trivially_classified = analysis.intent in {"smalltalk", "definition"}
        if self.llm is None or trivially_classified:
            return analysis, linked
        try:
            llm_analysis = await self._llm(question, history or [], linked)
        except AppError as exc:
            logger.warning("llm_analysis_failed_using_heuristic", extra={"error": exc.code})
            return analysis, linked
        # Re-link with LLM-extracted entities (may differ after coreference resolution).
        linked = await self.graph.link_entities(tenant_id, llm_analysis.standalone_question, llm_analysis.entities)
        llm_analysis = self._sanity_check(llm_analysis, linked)
        return llm_analysis, linked

    async def _heuristic(self, question: str, tenant_id: str) -> tuple[QueryAnalysis, list[LinkedEntity]]:
        hints = relation_hints(question)
        answer_type = expected_answer_type(question)
        entities = candidate_entities(question)
        linked = await self.graph.link_entities(tenant_id, question, entities) if not is_smalltalk(question) else []
        strategy, intent, reasoning = choose_strategy(question, hints, linked, answer_type)
        analysis = QueryAnalysis(
            standalone_question=question,
            intent=intent,
            entities=[e.name for e in linked] or entities,
            relationships=[h for h in hints if h != RelationType.RELATED_TO.value] or hints,
            temporal_constraints=temporal_constraints(question),
            answer_type=answer_type,
            retrieval_strategy=strategy,
            reasoning=reasoning,
        )
        return analysis, linked

    async def _llm(self, question: str, history: list[dict[str, str]], linked: list[LinkedEntity]) -> QueryAnalysis:
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:])
        known = ", ".join(f"{e.name} ({e.type})" for e in linked) or "none"
        user = (
            f"Conversation history:\n{convo or '(none)'}\n\n"
            f"Entities found in the knowledge graph for this question: {known}\n\n"
            f"Question: {question}"
        )
        return await self.llm.astructured(QueryAnalysis, to_messages(self.system, user), task="analyze_query")  # type: ignore[union-attr]

    @staticmethod
    def _sanity_check(analysis: QueryAnalysis, linked: list[LinkedEntity]) -> QueryAnalysis:
        strategy = analysis.retrieval_strategy
        if strategy == "GRAPH" and not linked:
            strategy = "HYBRID"  # graph-only with nothing to anchor on would retrieve nothing
        if strategy == "DIRECT" and not is_smalltalk(analysis.standalone_question):
            strategy = "HYBRID"  # never answer factual questions without retrieval
        if strategy != analysis.retrieval_strategy:
            analysis = analysis.model_copy(update={
                "retrieval_strategy": strategy,
                "reasoning": f"{analysis.reasoning} (adjusted to {strategy}: grounded routing policy)",
            })
        return analysis


def detect_metadata_filters(question: str) -> dict[str, Any]:
    files = filename_mentions(question)
    return {"filenames": files} if files else {}
