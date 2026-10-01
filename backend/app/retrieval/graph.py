"""Graph retrieval: entity linking + templated multi-hop traversal (+ optional Text2Cypher).

For "Who works on projects managed by Rahul that use Kafka?":

1. Link anchors: ``Rahul`` (Person), ``Kafka`` (Technology).
2. Relation hints from the question: MANAGES, WORKS_ON, USES.
3. Bridges: entities adjacent to *all* anchors -> ``Project Alpha``.
4. Hint-guided expansion (bounded hops) collects facts such as
   ``Amit -WORKS_ON-> Project Alpha``.
5. Answer candidates: entities of the expected answer type (Person) attached to bridges.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rapidfuzz import fuzz

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.repository import GraphReader
from app.graph.schema import ENTITY_TYPES, RELATION_CONSTRAINTS
from app.graph.text2cypher import Text2Cypher
from app.ingestion.entity_resolver import canonical_key
from app.retrieval.query_parsing import expected_answer_type, primary_relation, relation_hints
from app.retrieval.types import LinkedEntity, RetrievalResult
from app.schemas.search import GraphFact
from app.utils.text import STOPWORDS, normalize_name

logger = get_logger(__name__)

_GENERIC_TOKENS = {"project", "projects", "team", "department", "company", "technology", "the", "and", "of"}


def _fact_from_row(row: dict[str, Any], score: float) -> GraphFact:
    return GraphFact(
        source_id=row.get("source_id"),
        target_id=row.get("target_id"),
        source=row["source"],
        source_type=row["source_type"],
        relationship=row["relationship"],
        target=row["target"],
        target_type=row["target_type"],
        evidence=row.get("evidence"),
        chunk_ids=list(row.get("chunk_ids") or []),
        document_ids=list(row.get("document_ids") or []),
        score=score,
        hops=int(row.get("hops") or 1),
    )


def anchor_relations(entity_type: str, hints: list[str]) -> list[str] | None:
    """Relationship types an anchor may use to reach a bridge: the hinted relations compatible with
    the anchor's type, or ``None`` (any relation) when no hint applies to that type."""
    compatible = [
        h for h in hints
        if h in RELATION_CONSTRAINTS and (entity_type in RELATION_CONSTRAINTS[h][0] or entity_type in RELATION_CONSTRAINTS[h][1])
        and h != "RELATED_TO"
    ]
    return compatible or None


class GraphRetriever:
    def __init__(self, reader: GraphReader, settings: Settings, text2cypher: Text2Cypher | None = None) -> None:
        self.reader = reader
        self.settings = settings
        self.text2cypher = text2cypher

    # ------------------------------------------------------------ linking
    async def link_entities(self, tenant_id: str, question: str, names: list[str] | None = None) -> list[LinkedEntity]:
        names = [n for n in (names or []) if n and n.strip()]
        queries = []
        for name in names:
            keys = {canonical_key(name, t) for t in ENTITY_TYPES} | {normalize_name(name)}
            for key in keys:
                queries.append({"raw": name, "key": key, "lower": name.lower().strip()})
        exact_rows, text_rows = await asyncio.gather(
            self.reader.link_entities_exact(tenant_id, queries),
            self.reader.entities_in_text(tenant_id, normalize_name(question)),
        )
        linked: dict[str, LinkedEntity] = {}
        for row in [*exact_rows, *text_rows]:
            linked.setdefault(row["id"], LinkedEntity(**{k: row.get(k) for k in ("id", "name", "type", "description", "query", "score")}))
        resolved_queries = {row["query"] for row in exact_rows if row.get("query")}
        resolved_names = {e.name.lower() for e in linked.values()}
        missing = [n for n in names if n not in resolved_queries and n.lower() not in resolved_names]
        if missing:
            fuzzy_q = []
            for name in missing:
                tokens = [t for t in normalize_name(name).split() if t not in _GENERIC_TOKENS and t not in STOPWORDS and len(t) > 2]
                if tokens:
                    fuzzy_q.append({"raw": name, "tokens": tokens})
            for row in await self.reader.link_entities_fuzzy(tenant_id, fuzzy_q):
                ratio = fuzz.token_set_ratio(normalize_name(row["query"]), normalize_name(row["name"]))
                if row["score"] >= 0.5 and ratio >= 80 and row["id"] not in linked:
                    linked[row["id"]] = LinkedEntity(**{k: row.get(k) for k in ("id", "name", "type", "description", "query")}, score=ratio / 100)
        # Drop entities subsumed by a longer linked name ("Alpha" inside "Project Alpha").
        values = list(linked.values())
        return [
            e for e in values
            if not any(o.id != e.id and len(o.name) > len(e.name) and e.name.lower() in o.name.lower() for o in values)
        ]

    # ------------------------------------------------------------- search
    async def search(
        self,
        question: str,
        tenant_id: str,
        *,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
        answer_type: str | None = None,
        use_text2cypher: bool | None = None,
    ) -> RetrievalResult:
        hints = [r for r in (relations or relation_hints(question)) if r]
        answer_type = answer_type or expected_answer_type(question)
        linked = await self.link_entities(tenant_id, question, entities)
        result = RetrievalResult(strategy="GRAPH", query=question, linked_entities=linked)
        facts: dict[tuple[str, str, str], GraphFact] = {}

        def add(rows: list[dict[str, Any]], score: float) -> None:
            for row in rows:
                key = (row["source"], row["relationship"], row["target"])
                fact = _fact_from_row(row, score / max(1, int(row.get("hops") or 1)))
                if key not in facts or facts[key].score < fact.score:
                    facts[key] = fact

        anchor_ids = [e.id for e in linked]
        limit = self.settings.graph_max_facts
        if anchor_ids:
            # 1-hop context around anchors (hint-filtered when hints exist, else everything).
            hop1 = await self.reader.neighborhood(tenant_id, anchor_ids, hints or None, limit)
            add(hop1, 1.0)
            bridges: list[dict[str, Any]] = []
            if len(anchor_ids) >= 2:
                anchors = [{"id": e.id, "rels": anchor_relations(e.type, hints)} for e in linked]
                bridges = await self.reader.common_neighbors(tenant_id, anchors, min_anchors=len(anchor_ids))
                if not bridges and len(anchor_ids) > 2:
                    bridges = await self.reader.common_neighbors(tenant_id, anchors, min_anchors=2)
                add(await self.reader.paths_between(tenant_id, anchor_ids, self.settings.graph_max_hops), 0.9)
            bridge_ids = {b["id"] for b in bridges}
            result.bridges = [LinkedEntity(id=b["id"], name=b["name"], type=b["type"], score=float(b["anchors"])) for b in bridges]
            if bridge_ids:
                add(await self.reader.neighborhood(tenant_id, list(bridge_ids), hints or None, limit), 1.2)
            # Hint-guided multi-hop expansion, bounded by graph_max_hops.
            if hints:
                seen = set(anchor_ids) | bridge_ids
                frontier = ({r["source_id"] for r in hop1} | {r["target_id"] for r in hop1}) - seen
                for hop in range(2, self.settings.graph_max_hops + 1):
                    if not frontier:
                        break
                    seen |= frontier
                    expansion = await self.reader.neighborhood(tenant_id, sorted(frontier), hints, limit)
                    for row in expansion:
                        row["hops"] = hop
                    near_bridge = [r for r in expansion if r["source_id"] in bridge_ids or r["target_id"] in bridge_ids]
                    others = [r for r in expansion if not (r["source_id"] in bridge_ids or r["target_id"] in bridge_ids)]
                    # hop-decayed: score / hops (see add()); bridge-adjacent facts are favoured
                    add(near_bridge, 1.2 * hop)
                    add(others, (0.5 if bridge_ids else 1.0) * hop)
                    frontier = ({r["source_id"] for r in expansion} | {r["target_id"] for r in expansion}) - seen
            candidates, intermediates = self._answer_candidates(
                list(facts.values()), answer_type, linked, result.bridges, primary_relation(question), hints
            )
            result.answer_candidates = candidates
            result.bridges = result.bridges + intermediates

        # Text2Cypher for complex questions the templates could not answer.
        allow_t2c = self.settings.enable_text2cypher if use_text2cypher is None else use_text2cypher
        if self.text2cypher is not None and allow_t2c and (not facts or (len(anchor_ids) >= 2 and not result.answer_candidates)):
            hint_text = f"Known entities: {', '.join(f'{e.name} ({e.type})' for e in linked)}" if linked else ""
            validated, rows = await self.text2cypher.run(question, tenant_id, hint_text)
            if validated is not None:
                result.cypher = validated.query
                result.cypher_rows = rows

        ranked = sorted(facts.values(), key=lambda f: (-f.score, f.hops, f.source, f.target))
        result.facts = ranked[:limit]
        return result

    @staticmethod
    def _answer_candidates(
        facts: list[GraphFact], answer_type: str | None, anchors: list[LinkedEntity], bridges: list[LinkedEntity],
        answer_rel: str | None = None, hints: list[str] | None = None,
    ) -> tuple[list[LinkedEntity], list[LinkedEntity]]:
        """Return (answer candidates, intermediate entities used to reach them)."""
        if not answer_type:
            return [], []
        hints = hints or []
        anchor_names = {a.name for a in anchors}
        # A bridge of the requested type is itself the answer (e.g. technologies shared by two projects).
        typed_bridges = [b for b in bridges if b.type == answer_type]
        if typed_bridges:
            return [LinkedEntity(id=b.id, name=b.name, type=b.type, score=1.0) for b in typed_bridges], []
        focus = {b.name for b in bridges} or anchor_names

        def adjacent(nodes: set[str], rels: set[str] | None, want: str | None = None,
                     exclude: str | None = None) -> dict[str, tuple[float, str]]:
            out: dict[str, tuple[float, str]] = {}
            for f in facts:
                if rels and f.relationship not in rels:
                    continue
                for name, etype, other in ((f.source, f.source_type, f.target), (f.target, f.target_type, f.source)):
                    if other not in nodes or name in anchor_names or name in nodes:
                        continue
                    if (want and etype != want) or (exclude and etype == exclude):
                        continue
                    if name not in out or out[name][0] < f.score:
                        out[name] = (f.score, etype)
            return out

        via: dict[str, tuple[float, str]] = {}
        rels = {answer_rel} if answer_rel else (set(hints) or None)
        scores = adjacent(focus, rels, want=answer_type)
        if not scores and answer_rel and set(hints) - {answer_rel}:
            # Two hops: anchor -(constraint relation)-> intermediate -(answer relation)-> candidate.
            via = adjacent(focus, set(hints) - {answer_rel}, exclude=answer_type)
            if via:
                scores = adjacent(set(via), {answer_rel}, want=answer_type)
                if not scores:
                    via = {}
        if not scores and hints:
            scores = adjacent(focus, set(hints), want=answer_type)
        candidates = [LinkedEntity(id=name, name=name, type=answer_type, score=score)
                      for name, (score, _) in sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0]))]
        intermediates = [LinkedEntity(id=name, name=name, type=etype, score=score) for name, (score, etype) in via.items()]
        return candidates, intermediates
