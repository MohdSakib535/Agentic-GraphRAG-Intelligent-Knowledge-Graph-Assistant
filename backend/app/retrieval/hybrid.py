"""Hybrid retrieval = vector search + keyword search + graph search + metadata filtering.

Chunk lists are fused with weighted Reciprocal Rank Fusion. Chunks cited as
evidence by graph facts, or mentioning linked entities, enter the fusion as
their own ranked lists so graph knowledge directly promotes the right passages.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.repository import GraphReader
from app.retrieval.graph import GraphRetriever
from app.retrieval.types import RetrievalResult
from app.retrieval.vector import VectorRetriever, row_to_hit
from app.schemas.search import ChunkHit

logger = get_logger(__name__)

RRF_K = 60
WEIGHTS = {"vector": 1.0, "keyword": 0.7, "graph_evidence": 1.2, "entity_mention": 0.6}


def reciprocal_rank_fusion(ranked_lists: dict[str, list[ChunkHit]], top_k: int) -> list[ChunkHit]:
    fused: dict[str, ChunkHit] = {}
    scores: dict[str, float] = {}
    for name, hits in ranked_lists.items():
        weight = WEIGHTS.get(name, 1.0)
        for rank, hit in enumerate(hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (RRF_K + rank + 1)
            if hit.chunk_id in fused:
                existing = fused[hit.chunk_id]
                existing.retrievers = sorted(set(existing.retrievers) | set(hit.retrievers))
                existing.metadata.update({k: v for k, v in hit.metadata.items() if k.endswith("_score")})
            else:
                fused[hit.chunk_id] = hit.model_copy(deep=True)
    if not scores:
        return []
    best_possible = sum(WEIGHTS.get(n, 1.0) for n in ranked_lists if ranked_lists[n]) / (RRF_K + 1)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for chunk_id, score in ordered:
        hit = fused[chunk_id]
        # Relevance in [0,1]: blend normalised RRF with the strongest raw similarity signal.
        raw = max(
            float(hit.metadata.get("vector_score", 0.0)),
            float(hit.metadata.get("keyword_score", 0.0)) * 0.8,
            float(hit.metadata.get("graph_evidence_score", 0.0)),
        )
        hit.metadata["rrf_score"] = round(score, 5)
        hit.score = round(0.5 * min(1.0, score / best_possible) + 0.5 * raw, 4)
        out.append(hit)
    return out


class HybridRetriever:
    def __init__(self, vector: VectorRetriever, graph: GraphRetriever, reader: GraphReader, settings: Settings) -> None:
        self.vector = vector
        self.graph = graph
        self.reader = reader
        self.settings = settings

    async def search(
        self,
        question: str,
        tenant_id: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
        answer_type: str | None = None,
    ) -> RetrievalResult:
        pool = max(top_k * 2, 10)
        vector_task = self.vector.similarity_search(question, tenant_id, pool, filters)
        keyword_task = self.vector.keyword_search(question, tenant_id, pool, filters)
        graph_task = self.graph.search(question, tenant_id, entities=entities, relations=relations, answer_type=answer_type)
        vector_hits, keyword_hits, graph_result = await asyncio.gather(
            vector_task, keyword_task, graph_task, return_exceptions=True
        )
        errors: list[str] = []
        if isinstance(vector_hits, BaseException):
            errors.append(f"vector:{type(vector_hits).__name__}")
            vector_hits = []
        if isinstance(keyword_hits, BaseException):
            errors.append(f"keyword:{type(keyword_hits).__name__}")
            keyword_hits = []
        if isinstance(graph_result, BaseException):
            errors.append(f"graph:{type(graph_result).__name__}")
            graph_result = RetrievalResult(strategy="GRAPH", query=question)
        if len(errors) == 3:
            raise RuntimeError("All hybrid retrievers failed: " + ", ".join(errors))

        # Graph-derived chunk lists.
        evidence_ids: list[str] = []
        for fact in graph_result.facts[:20]:
            for cid in fact.chunk_ids:
                if cid not in evidence_ids:
                    evidence_ids.append(cid)
        focus_ids = [e.id for e in graph_result.linked_entities] + [b.id for b in graph_result.bridges]
        evidence_rows, mention_rows = await asyncio.gather(
            self.reader.chunks_by_ids(tenant_id, evidence_ids[:pool]),
            self.reader.chunks_mentioning(tenant_id, focus_ids, pool),
        )
        order = {cid: i for i, cid in enumerate(evidence_ids)}
        evidence_hits = sorted(
            (row_to_hit(r, "graph_evidence", 1.0 - 0.02 * order.get(r["chunk_id"], 0)) for r in evidence_rows),
            key=lambda h: order.get(h.chunk_id, 0),
        )
        mention_hits = [row_to_hit(r, "entity_mention") for r in mention_rows]
        if filters:
            evidence_hits = [h for h in evidence_hits if _passes(h, filters)]
            mention_hits = [h for h in mention_hits if _passes(h, filters)]
        fused = reciprocal_rank_fusion(
            {"vector": vector_hits, "keyword": keyword_hits, "graph_evidence": evidence_hits, "entity_mention": mention_hits},
            top_k,
        )
        graph_result.strategy = "HYBRID"
        graph_result.chunks = fused
        graph_result.errors = errors
        return graph_result


def _passes(hit: ChunkHit, filters: dict[str, Any]) -> bool:
    if filters.get("document_ids") and hit.document_id not in filters["document_ids"]:
        return False
    if filters.get("filenames") and hit.source_filename not in filters["filenames"]:
        return False
    page = hit.page_number
    if filters.get("page_from") and (page is None or page < filters["page_from"]):
        return False
    if filters.get("page_to") and (page is None or page > filters["page_to"]):
        return False
    section = filters.get("section")
    return not (section and section.lower() not in (hit.section or "").lower())
