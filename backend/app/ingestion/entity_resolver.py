"""Entity resolution: collapse duplicates within a document and against the tenant graph.

Resolution ladder (cheapest first, LLM only for genuinely ambiguous pairs):

1. **Normalisation** - casefold, strip punctuation, alias table and type-specific
   affixes ("Apache Kafka", "kafka platform" -> ``kafka``).
2. **Exact match** on the canonical key (within the document, then the graph).
3. **Fuzzy/semantic similarity** - max(string similarity, embedding cosine)
   against same-type candidates; merged above ``entity_similarity_threshold``.
4. **LLM-assisted** yes/no decision only inside the ambiguity band.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from rapidfuzz import fuzz

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.graph.schema import EntityType
from app.ingestion.embedding import Embedder, cosine
from app.ingestion.entity_extractor import ExtractedEntity
from app.ingestion.relationship_extractor import ExtractedRelationship
from app.llm.client import LLMClient, to_messages
from app.utils.ids import entity_id
from app.utils.text import normalize_name

logger = get_logger(__name__)

ALIASES = {
    "postgres": "postgresql", "postgre sql": "postgresql", "psql": "postgresql", "k8s": "kubernetes",
    "golang": "go", "js": "javascript", "ts": "typescript", "nodejs": "node js", "node": "node js",
    "elastic search": "elasticsearch", "elastic": "elasticsearch", "mongo": "mongodb", "gcp": "google cloud",
    "google cloud platform": "google cloud", "amazon web services": "aws", "rabbit mq": "rabbitmq",
    "sql alchemy": "sqlalchemy", "fast api": "fastapi", "neo4j database": "neo4j", "redis cache": "redis",
}
_TECH_PREFIXES = ("apache ", "the ", "amazon ", "aws ", "microsoft ", "google ")
_TECH_SUFFIXES = (
    " platform", " framework", " library", " database", " db", " cluster", " server", " service", " broker",
    " engine", " cache", " streams platform", " messaging", " queue",
)
_COMPANY_SUFFIXES = (" inc", " corp", " corporation", " ltd", " limited", " llc", " gmbh", " co", " pvt ltd", " pvt")
_DEPT_SUFFIXES = (" department", " team", " division", " dept")


def canonical_key(name: str, entity_type: str) -> str:
    key = normalize_name(name)
    key = ALIASES.get(key, key)
    if entity_type == EntityType.TECHNOLOGY.value:
        for prefix in _TECH_PREFIXES:
            if key.startswith(prefix) and len(key) > len(prefix) + 1:
                stripped = key[len(prefix) :]
                key = ALIASES.get(stripped, stripped)
        for suffix in _TECH_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix) + 1:
                key = key[: -len(suffix)]
        key = ALIASES.get(key, key)
    elif entity_type == EntityType.COMPANY.value:
        for suffix in _COMPANY_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix) + 1:
                key = key[: -len(suffix)]
    elif entity_type == EntityType.PROJECT.value:
        key = re.sub(r"^(?:the )?(.+?) project$", r"project \1", key)
        if not key.startswith("project "):
            key = f"project {key}"
    elif entity_type == EntityType.DEPARTMENT.value:
        for suffix in _DEPT_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix) + 1:
                key = key[: -len(suffix)]
    elif entity_type == EntityType.PERSON.value:
        key = re.sub(r"^(?:mr|mrs|ms|dr|prof) ", "", key)
    return key.strip()


@dataclass
class ResolvedEntity:
    id: str
    name: str
    type: str
    normalized_name: str
    description: str = ""
    aliases: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    existing: bool = False
    redirect: ResolvedEntity | None = None  # set when merged into another document entity

    @property
    def final_id(self) -> str:
        return self.redirect.final_id if self.redirect is not None else self.id

    def as_record(self, tenant_id: str, document_id: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": tenant_id,
            "name": self.name,
            "type": self.type,
            "normalized_name": self.normalized_name,
            "description": self.description[:600],
            "aliases": sorted(self.aliases - {self.name})[:25],
            "document_id": document_id,
        }


@dataclass
class ResolvedRelationship:
    source_id: str
    target_id: str
    type: str
    evidence: str
    chunk_ids: set[str] = field(default_factory=set)


@dataclass
class ResolutionResult:
    entities: list[ResolvedEntity]
    relationships: list[ResolvedRelationship]
    mentions: list[tuple[str, str]]  # (chunk_id, entity_id)
    merges: int = 0


class EntityLookup(Protocol):
    def find_by_keys(self, tenant_id: str, keys: list[str]) -> list[dict[str, Any]]: ...

    def find_candidates(self, tenant_id: str, entity_type: str, tokens: list[str], limit: int) -> list[dict[str, Any]]: ...


class _SameEntity(BaseModel):
    same_entity: bool
    reason: str = ""


class EntityResolver:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        lookup: EntityLookup | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.lookup = lookup
        self.llm = llm

    # ---------------------------------------------------------------- public
    def resolve(
        self,
        tenant_id: str,
        extractions: list[tuple[str, list[ExtractedEntity], list[ExtractedRelationship]]],
    ) -> ResolutionResult:
        groups = self._group_mentions(extractions)
        merges = sum(len(g["names"]) - 1 for g in groups.values())
        entities = self._build_entities(tenant_id, groups)
        merges += self._merge_person_first_names(entities)
        merges += self._match_existing(tenant_id, entities)
        return self._finalise(entities, extractions, merges)

    # ------------------------------------------------------------- internals
    def _group_mentions(self, extractions: list[Any]) -> dict[tuple[str, str], dict[str, Any]]:
        # A key seen with several types (e.g. Kafka as Technology and Product) takes the majority type.
        type_votes: dict[str, Counter[str]] = defaultdict(Counter)
        for _, entities, _ in extractions:
            for ent in entities:
                type_votes[normalize_name(ent.name)][ent.type.value] += 1
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for chunk_id, entities, _ in extractions:
            for ent in entities:
                etype = type_votes[normalize_name(ent.name)].most_common(1)[0][0]
                key = canonical_key(ent.name, etype)
                if not key:
                    continue
                group = groups.setdefault(
                    (etype, key), {"names": Counter(), "descriptions": [], "chunks": set()}
                )
                group["names"][ent.name] += 1
                if ent.description:
                    group["descriptions"].append(ent.description)
                group["chunks"].add(chunk_id)
        return groups

    def _build_entities(self, tenant_id: str, groups: dict[tuple[str, str], dict[str, Any]]) -> list[ResolvedEntity]:
        entities = []
        for (etype, key), group in groups.items():
            # Display name: most frequent surface form, ties broken by length (more complete).
            name = max(group["names"].items(), key=lambda kv: (kv[1], len(kv[0])))[0]
            descriptions = sorted(group["descriptions"], key=len, reverse=True)
            entities.append(
                ResolvedEntity(
                    id=entity_id(tenant_id, etype, key),
                    name=name,
                    type=etype,
                    normalized_name=key,
                    description=descriptions[0] if descriptions else "",
                    aliases=set(group["names"]),
                    chunk_ids=set(group["chunks"]),
                )
            )
        return entities

    @staticmethod
    def _merge_person_first_names(entities: list[ResolvedEntity]) -> int:
        """Merge "Rahul" into "Rahul Sharma" when the first name is unambiguous in the document."""
        persons = [e for e in entities if e.type == EntityType.PERSON.value]
        merges = 0
        for short in [p for p in persons if " " not in p.normalized_name]:
            full = [p for p in persons if " " in p.normalized_name and p.normalized_name.split()[0] == short.normalized_name]
            if len(full) == 1:
                target = full[0]
                target.aliases |= short.aliases
                target.chunk_ids |= short.chunk_ids
                target.description = target.description or short.description
                short.redirect = target
                short.normalized_name = ""
                merges += 1
        return merges

    def _match_existing(self, tenant_id: str, entities: list[ResolvedEntity]) -> int:
        if self.lookup is None:
            return 0
        live = [e for e in entities if e.redirect is None]
        merges = 0
        existing = self.lookup.find_by_keys(tenant_id, sorted({e.normalized_name for e in live}))
        by_key = {(row["type"], row["normalized_name"]): row for row in existing}
        unmatched: list[ResolvedEntity] = []
        for ent in live:
            row = by_key.get((ent.type, ent.normalized_name))
            if row:
                self._adopt(ent, row)
            else:
                unmatched.append(ent)
        for ent in unmatched:
            tokens = [t for t in ent.normalized_name.split() if len(t) > 2 and t != "project"][:5]
            if not tokens:
                continue
            candidates = self.lookup.find_candidates(tenant_id, ent.type, tokens, 25)
            best, best_score = self._best_candidate(ent, candidates)
            if best is None:
                continue
            if best_score >= self.settings.entity_similarity_threshold or (
                best_score >= self.settings.entity_llm_resolution_band and self._llm_same(ent, best)
            ):
                self._adopt(ent, best)
                merges += 1
        return merges

    def _best_candidate(self, ent: ResolvedEntity, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
        best, best_score = None, 0.0
        if not candidates:
            return None, 0.0
        vectors: list[list[float]] | None = None
        if self.embedder is not None and self.settings.resolved_embedding_provider == "openai":
            try:
                vectors = self.embedder.embed_documents([ent.name] + [c["name"] for c in candidates])
            except AppError:
                vectors = None
        for i, cand in enumerate(candidates):
            if cand.get("type") != ent.type:
                continue
            string_score = fuzz.token_sort_ratio(ent.normalized_name, cand["normalized_name"]) / 100.0
            semantic = cosine(vectors[0], vectors[i + 1]) if vectors else 0.0
            # Person names need near-exact agreement; embeddings conflate different people.
            score = string_score if ent.type == EntityType.PERSON.value else max(string_score, semantic)
            if score > best_score:
                best, best_score = cand, score
        return best, best_score

    def _llm_same(self, ent: ResolvedEntity, cand: dict[str, Any]) -> bool:
        if self.llm is None:
            return False
        prompt = (
            f"Entity A: {ent.name} ({ent.type}) - {ent.description[:200]}\n"
            f"Entity B: {cand['name']} ({cand['type']}) - {str(cand.get('description') or '')[:200]}\n"
            "Do A and B refer to the same real-world entity?"
        )
        try:
            verdict = self.llm.structured(
                _SameEntity,
                to_messages("You resolve duplicate entities in a knowledge graph. Be conservative.", prompt),
                task="entity_resolution",
            )
        except AppError:
            return False
        return verdict.same_entity

    @staticmethod
    def _adopt(ent: ResolvedEntity, row: dict[str, Any]) -> None:
        ent.aliases.add(ent.name)
        ent.id = row["id"]
        ent.name = row["name"]
        ent.normalized_name = row["normalized_name"]
        ent.existing = True

    def _finalise(self, entities: list[ResolvedEntity], extractions: list[Any], merges: int) -> ResolutionResult:
        # Map every surface form -> final entity id.
        by_surface: dict[tuple[str, str], str] = {}
        by_id: dict[str, ResolvedEntity] = {}
        for ent in entities:
            for alias in ent.aliases | {ent.name}:
                by_surface[(alias.lower(), ent.type)] = ent.final_id
            if ent.redirect is None:
                target = by_id.get(ent.id)
                if target is None:
                    by_id[ent.id] = ent
                else:  # two document groups resolved to the same graph entity
                    target.aliases |= ent.aliases
                    target.chunk_ids |= ent.chunk_ids
        for ent in entities:  # redirected first-name entities
            if ent.redirect is not None and ent.final_id in by_id:
                by_id[ent.final_id].chunk_ids |= ent.chunk_ids
        name_type: dict[str, str] = {}
        for _, ents, _ in extractions:
            for e in ents:
                name_type.setdefault(e.name.lower(), e.type.value)

        def lookup(name: str) -> str | None:
            etype = name_type.get(name.lower())
            if etype and (name.lower(), etype) in by_surface:
                return by_surface[(name.lower(), etype)]
            return next((v for (n, _), v in by_surface.items() if n == name.lower()), None)

        rels: dict[tuple[str, str, str], ResolvedRelationship] = {}
        mentions: set[tuple[str, str]] = set()
        for chunk_id, ents, rel_list in extractions:
            for e in ents:
                eid = lookup(e.name)
                if eid:
                    mentions.add((chunk_id, eid))
            for r in rel_list:
                sid, tid = lookup(r.source), lookup(r.target)
                if not sid or not tid or sid == tid:
                    continue
                key = (sid, r.relationship.value, tid)
                rel = rels.setdefault(key, ResolvedRelationship(sid, tid, r.relationship.value, r.evidence))
                rel.chunk_ids.add(chunk_id)
        return ResolutionResult(list(by_id.values()), list(rels.values()), sorted(mentions), merges)
