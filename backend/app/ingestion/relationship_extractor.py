"""Relationship extraction and the combined per-chunk graph extractors.

Only the predefined relationship types in :mod:`app.graph.schema` are accepted,
and every relationship must satisfy the (source type, target type) constraints.
Relationships whose endpoints are not among the chunk's validated entities are
discarded - the LLM cannot invent nodes through a relationship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.graph.schema import RELATION_DESCRIPTIONS, EntityType, RelationType, is_valid_relation
from app.ingestion.entity_extractor import (
    MAX_NAME_LENGTH,
    EntityMention,
    ExtractedEntity,
    HeuristicEntityExtractor,
    validate_entities,
)
from app.llm.client import LLMClient, to_messages
from app.utils.text import split_sentences

logger = get_logger(__name__)


class ExtractedRelationship(BaseModel):
    source: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    relationship: RelationType
    target: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    evidence: str = Field(default="", max_length=600)
    source_chunk: str | None = None

    @field_validator("relationship", mode="before")
    @classmethod
    def _normalise_rel(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper().replace(" ", "_").replace("-", "_")
        return value

    @field_validator("source", "target")
    @classmethod
    def _clean(cls, value: str) -> str:
        return " ".join(value.split()).strip(" .,;:'\"`")


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


def validate_relationships(
    raw_items: list[Any], entities: list[ExtractedEntity], source_chunk: str | None = None
) -> list[ExtractedRelationship]:
    """Validate untrusted relationship dicts against the whitelist and the chunk's entities."""
    by_name = {e.name.lower(): e for e in entities}
    valid: list[ExtractedRelationship] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items or []:
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        try:
            rel = ExtractedRelationship.model_validate({**item, "source_chunk": source_chunk})
        except ValidationError:
            logger.debug("dropped_invalid_relationship")
            continue
        src, tgt = by_name.get(rel.source.lower()), by_name.get(rel.target.lower())
        if src is None or tgt is None or src.name.lower() == tgt.name.lower():
            continue
        if not is_valid_relation(rel.relationship.value, src.type.value, tgt.type.value):
            continue
        key = (src.name.lower(), rel.relationship.value, tgt.name.lower())
        if key in seen:
            continue
        seen.add(key)
        valid.append(rel.model_copy(update={"source": src.name, "target": tgt.name}))
    return valid


# ===================================================== heuristic relation extraction
@dataclass(frozen=True)
class _RelPattern:
    rel: RelationType
    regex: re.Pattern[str]
    reverse: bool = False


def _p(rel: RelationType, pattern: str, reverse: bool = False) -> _RelPattern:
    return _RelPattern(rel, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), reverse)


_ROLE = r"(?:developer|engineer|member|contributor|architect|analyst|designer|lead|tester)"
REL_PATTERNS: list[_RelPattern] = [
    _p(RelationType.MANAGES, r"(?:is|are|was|were)\s+(?:managed|led|headed|overseen|owned|run)\s+by", reverse=True),
    _p(RelationType.WORKS_ON, r"(?:is|are|was|were)\s+(?:developed|maintained)\s+by", reverse=True),
    _p(RelationType.USES, r"(?:is|are|was|were)\s+used\s+(?:by|in)", reverse=True),
    _p(RelationType.USES, r"(?:is|are|was|were)?\s*(?:built|implemented|developed|written)\s+(?:on|with|using|in)"
       r"|uses|used|use|using|is\s+using|are\s+using|relies\s+on|rely\s+on|leverages?|is\s+powered\s+by"
       r"|runs\s+on|utili[sz]es|adopted|stores\s+(?:data|events)\s+in|streams\s+events\s+(?:through|via|with)"),
    _p(RelationType.MANAGES, r"manages|managed|manage|is\s+managing|leads|led|lead|is\s+leading|heads|oversees"
       r"|owns|runs|is\s+(?:the\s+)?(?:\w+\s+)?(?:manager|lead|head|owner)\s+(?:of|for)"),
    _p(RelationType.WORKS_ON, rf"works\s+on|worked\s+on|work\s+on|is\s+working\s+on|are\s+working\s+on"
       rf"|contributes\s+to|contributed\s+to|contribute\s+to|(?:is|are)\s+assigned\s+to"
       rf"|(?:is|are)\s+(?:a|an|the)?\s*(?:\w+\s+){{0,2}}{_ROLE}s?\s+(?:on|for|in)"),
    _p(RelationType.WORKS_FOR, r"works\s+for|worked\s+for|work\s+for|works\s+at|worked\s+at|work\s+at"
       r"|is\s+employed\s+by|(?:is\s+)?(?:an\s+)?employee\s+of|joined|is\s+(?:a|an|the)\s+(?:\w+\s+){0,2}"
       r"(?:manager|engineer|developer|architect|director|analyst)\s+at"),
    _p(RelationType.REPORTS_TO, r"reports\s+(?:directly\s+)?to|reported\s+to|report\s+to"),
    _p(RelationType.DEPENDS_ON, r"depends\s+on|depend\s+on|is\s+dependent\s+on|requires|require"),
    _p(RelationType.BELONGS_TO, r"belongs\s+to|is\s+(?:a\s+)?part\s+of|are\s+part\s+of|(?:is|are)\s+located\s+in"
       r"|located\s+in|(?:is|are)\s+based\s+in|based\s+in|(?:is|are)\s+(?:a\s+)?(?:team|unit)\s+(?:in|within)"),
    _p(RelationType.RELATED_TO, r"is\s+related\s+to|relates\s+to|is\s+associated\s+with|integrates\s+with"
       r"|collaborates\s+with|works\s+(?:closely\s+)?with"),
]
_SEPARATOR_RE = re.compile(r"^\s*(?:,|and|or|as well as|&|plus|,\s*and)?\s*(?:,\s*)?(?:and\s+)?$", re.IGNORECASE)
_MAX_GAP_WORDS = 10
_PRONOUN_RE = re.compile(r"^\s*(He|She|They)\b")


def _coordinated(mentions: list[EntityMention], index: int, text: str, direction: int) -> list[EntityMention]:
    """Return the mention at ``index`` plus siblings joined by separators (``A, B and C``)."""
    group = [mentions[index]]
    i = index
    while 0 <= i + direction < len(mentions):
        nxt = mentions[i + direction]
        a, b = (mentions[i], nxt) if direction > 0 else (nxt, mentions[i])
        if not _SEPARATOR_RE.match(text[a.end : b.start]) or nxt.type != mentions[index].type:
            break
        group.append(nxt)
        i += direction
    return group


class HeuristicRelationshipExtractor:
    def extract(self, text: str, mentions_by_sentence: list[tuple[str, list[EntityMention]]],
                source_chunk: str | None = None) -> list[dict[str, str]]:
        relations: list[dict[str, str]] = []
        last_person: EntityMention | None = None
        for sentence, mentions in mentions_by_sentence:
            if not mentions:
                continue
            sentence_mentions = list(mentions)
            pronoun = _PRONOUN_RE.match(sentence)
            if pronoun and last_person is not None:
                sentence_mentions.insert(
                    0, EntityMention(last_person.name, last_person.type, pronoun.start(1), pronoun.end(1))
                )
            used_spans: list[tuple[int, int]] = []
            for pattern in REL_PATTERNS:
                for match in pattern.regex.finditer(sentence):
                    if any(match.start() < e and match.end() > s for s, e in used_spans):
                        continue
                    before = [i for i, m in enumerate(sentence_mentions) if m.end <= match.start()]
                    after = [i for i, m in enumerate(sentence_mentions) if m.start >= match.end()]
                    if not before or not after:
                        continue
                    found = self._pair(sentence, sentence_mentions, before, after, match, pattern)
                    if found:
                        used_spans.append((match.start(), match.end()))
                        relations.extend(found)
            persons = [m for m in mentions if m.type == EntityType.PERSON.value]
            if persons:
                last_person = persons[0]
        return relations

    def _pair(self, sentence: str, mentions: list[EntityMention], before: list[int], after: list[int],
              match: re.Match[str], pattern: _RelPattern) -> list[dict[str, str]]:
        rel = pattern.rel.value
        left_role, right_role = ("target", "source") if pattern.reverse else ("source", "target")

        def valid(m: EntityMention, role: str, other_type: str | None = None) -> bool:
            if role == "source":
                return is_valid_relation(rel, m.type, other_type) if other_type else any(
                    is_valid_relation(rel, m.type, t.value) for t in EntityType
                )
            return is_valid_relation(rel, other_type, m.type) if other_type else any(
                is_valid_relation(rel, t.value, m.type) for t in EntityType
            )

        left_idx = next((i for i in reversed(before) if valid(mentions[i], left_role)), None)
        right_idx = next((i for i in after if valid(mentions[i], right_role)), None)
        if left_idx is None or right_idx is None:
            return []
        # Keep relations local: limit the words between each endpoint and the phrase.
        gap_left = sentence[mentions[left_idx].end : match.start()]
        gap_right = sentence[match.end() : mentions[right_idx].start]
        if len(gap_left.split()) > _MAX_GAP_WORDS or len(gap_right.split()) > _MAX_GAP_WORDS:
            return []
        lefts = _coordinated(mentions, left_idx, sentence, -1)
        rights = _coordinated(mentions, right_idx, sentence, +1)
        out: list[dict[str, str]] = []
        for lm in lefts:
            for rm in rights:
                src, tgt = (rm, lm) if pattern.reverse else (lm, rm)
                if src.name != tgt.name and is_valid_relation(rel, src.type, tgt.type):
                    out.append({"source": src.name, "relationship": rel, "target": tgt.name, "evidence": sentence})
        return out


# =========================================================== combined extractors
class GraphExtractor(Protocol):
    name: str

    def prime(self, texts: list[str]) -> None: ...

    def extract(self, text: str, source_chunk: str | None = None) -> ChunkExtraction: ...


class HeuristicGraphExtractor:
    name = "heuristic"

    def __init__(self) -> None:
        self.entities = HeuristicEntityExtractor()
        self.relations = HeuristicRelationshipExtractor()

    def prime(self, texts: list[str]) -> None:
        self.entities.prime(texts)

    def extract(self, text: str, source_chunk: str | None = None) -> ChunkExtraction:
        entities = self.entities.extract(text, source_chunk)
        per_sentence = [(s, self.entities.mentions(s)) for s in split_sentences(text)]
        raw_rels = self.relations.extract(text, per_sentence, source_chunk)
        return ChunkExtraction(
            entities=entities, relationships=validate_relationships(raw_rels, entities, source_chunk)
        )


_EXTRACTION_SYSTEM = """You extract a knowledge graph from enterprise documents.
Return ONLY entities and relationships explicitly stated in the text. Do not infer or invent facts.

Allowed entity types: Person, Company, Project, Technology, Department, Product, Location, Concept.
Allowed relationship types (use exactly these names and directions):
{relations}

Rules:
- Use the most complete name that appears in the text (e.g. "Project Alpha", "Apache Kafka").
- Every relationship's source and target MUST be the exact name of an entity you listed.
- description: one short sentence from the text describing the entity.
- evidence: the sentence from the text that states the relationship.
- If nothing relevant is present, return empty lists."""


class _RawEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class _RawRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    evidence: str = ""


class _RawExtraction(BaseModel):
    entities: list[_RawEntity] = Field(default_factory=list)
    relationships: list[_RawRelationship] = Field(default_factory=list)


class LLMGraphExtractor:
    """Joint entity+relationship extraction in a single structured LLM call per chunk."""

    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        relations = "\n".join(f"- {d}" for d in RELATION_DESCRIPTIONS.values())
        self.system = _EXTRACTION_SYSTEM.format(relations=relations)

    def prime(self, texts: list[str]) -> None:
        return None

    def extract(self, text: str, source_chunk: str | None = None) -> ChunkExtraction:
        try:
            raw = self.llm.structured(
                _RawExtraction, to_messages(self.system, f"Text:\n\"\"\"\n{text}\n\"\"\""), task="extract_graph"
            )
        except AppError:
            logger.warning("llm_extraction_failed_falling_back", extra={"chunk": source_chunk})
            fallback = HeuristicGraphExtractor()
            fallback.prime([text])
            return fallback.extract(text, source_chunk)
        entities = validate_entities([e.model_dump() for e in raw.entities], source_chunk)
        rels = validate_relationships([r.model_dump() for r in raw.relationships], entities, source_chunk)
        return ChunkExtraction(entities=entities, relationships=rels)


def build_graph_extractor(settings: Settings, llm: LLMClient | None) -> GraphExtractor:
    if llm is not None and settings.resolved_llm_provider == "openai":
        return LLMGraphExtractor(llm)
    return HeuristicGraphExtractor()
