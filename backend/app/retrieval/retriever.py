"""Retrieval facade: strategy dispatch, tenant-scoped caching, timeouts, reranking."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.access import require_scope
from app.core.config import Settings
from app.core.errors import AppError, RetrievalError
from app.core.logging import get_logger
from app.db.redis import TenantCache, query_hash
from app.graph.repository import GraphReader
from app.retrieval.graph import GraphRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankContext, Reranker
from app.retrieval.types import RetrievalResult
from app.retrieval.vector import VectorRetriever

logger = get_logger(__name__)

STRATEGIES = ("VECTOR", "GRAPH", "HYBRID")
# Bump when retrieval logic changes so cached results from older code are never served.
RETRIEVAL_VERSION = "6"


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        reader: GraphReader,
        vector: VectorRetriever,
        graph: GraphRetriever,
        hybrid: HybridRetriever,
        reranker: Reranker,
        cache: TenantCache | None = None,
    ) -> None:
        self.settings = settings
        self.reader = reader
        self.vector = vector
        self.graph = graph
        self.hybrid = hybrid
        self.reranker = reranker
        self.cache = cache

    async def retrieve(
        self,
        strategy: str,
        question: str,
        tenant_id: str,
        *,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
        answer_type: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        use_cache: bool = True,
    ) -> RetrievalResult:
        strategy = strategy.upper()
        if strategy not in STRATEGIES:
            raise RetrievalError(f"Unknown retrieval strategy {strategy}")
        top_k = top_k or self.settings.top_k
        scope = require_scope()
        digest = query_hash(RETRIEVAL_VERSION, scope.fingerprint, strategy, question.strip().lower(), sorted(entities or []), sorted(relations or []),
                            answer_type, top_k, filters or {}, rerank, self.reranker.name)
        if use_cache and self.cache is not None:
            cached = await self.cache.get(tenant_id, "retrieval", digest)
            if cached:
                result = RetrievalResult.model_validate(cached)
                result.cached = True
                return result
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._dispatch(strategy, question, tenant_id, entities, relations, answer_type, top_k, filters),
                timeout=self.settings.tool_timeout_seconds,
            )
        except TimeoutError as exc:
            logger.warning("retrieval_timeout", extra={"strategy": strategy})
            raise RetrievalError("Retrieval timed out") from exc
        except AppError:
            raise
        except Exception as exc:
            logger.exception("retrieval_failed", extra={"strategy": strategy})
            raise RetrievalError() from exc
        if rerank and result.chunks:
            names = [e.name for e in result.linked_entities] + [b.name for b in result.bridges]
            ctx = RerankContext(entity_names=names, filenames=list((filters or {}).get("filenames") or []))
            result.chunks = self.reranker.rerank(question, result.chunks, ctx)[:top_k]
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        if use_cache and self.cache is not None and not result.errors:
            await self.cache.set(tenant_id, "retrieval", digest, result.model_dump(mode="json"))
        return result

    async def _dispatch(
        self, strategy: str, question: str, tenant_id: str, entities: list[str] | None, relations: list[str] | None,
        answer_type: str | None, top_k: int, filters: dict[str, Any] | None,
    ) -> RetrievalResult:
        if strategy == "VECTOR":
            hits = await self.vector.similarity_search(question, tenant_id, top_k, filters)
            return RetrievalResult(strategy="VECTOR", query=question, chunks=hits)
        if strategy == "GRAPH":
            result = await self.graph.search(question, tenant_id, entities=entities, relations=relations, answer_type=answer_type)
            # Attach the evidence chunks backing the top facts (for citations).
            chunk_ids: list[str] = []
            for fact in result.facts[:15]:
                chunk_ids.extend(c for c in fact.chunk_ids if c not in chunk_ids)
            from app.retrieval.vector import row_to_hit

            rows = await self.reader.chunks_by_ids(tenant_id, chunk_ids[: top_k * 2])
            order = {c: i for i, c in enumerate(chunk_ids)}
            result.chunks = sorted(
                (row_to_hit(r, "graph_evidence", 1.0) for r in rows), key=lambda h: order.get(h.chunk_id, 0)
            )[:top_k]
            return result
        return await self.hybrid.search(
            question, tenant_id, top_k=top_k, filters=filters, entities=entities, relations=relations, answer_type=answer_type
        )
