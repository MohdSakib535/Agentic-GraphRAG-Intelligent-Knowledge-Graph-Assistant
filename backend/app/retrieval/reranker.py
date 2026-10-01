"""Configurable reranking.

* ``none``          - keep fused order.
* ``score``         - cheap feature-based reranker: retrieval score + term overlap
                      + graph relevance (linked entities in chunk) + metadata relevance.
* ``cross_encoder`` - optional sentence-transformers cross-encoder (install
                      ``sentence-transformers``); falls back to ``score`` if unavailable.
Expensive reranking is never mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.search import ChunkHit
from app.utils.text import overlap_ratio, term_set

logger = get_logger(__name__)


@dataclass
class RerankContext:
    entity_names: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[ChunkHit], context: RerankContext | None = None) -> list[ChunkHit]: ...


class NoopReranker:
    name = "none"

    def rerank(self, query: str, hits: list[ChunkHit], context: RerankContext | None = None) -> list[ChunkHit]:
        return hits


class ScoreReranker:
    name = "score"

    def __init__(self, weights: tuple[float, float, float, float] = (0.45, 0.30, 0.17, 0.08)) -> None:
        self.weights = weights

    def features(self, query: str, hit: ChunkHit, context: RerankContext) -> dict[str, float]:
        q_terms = term_set(query)
        text_lower = hit.text.lower()
        graph_rel = (
            sum(1 for n in context.entity_names if n.lower() in text_lower) / len(context.entity_names)
            if context.entity_names
            else 0.0
        )
        meta_text = " ".join(str(x) for x in (hit.source_filename, hit.section, hit.metadata.get("document_title")) if x)
        meta_rel = overlap_ratio(q_terms, meta_text)
        if context.filenames and hit.source_filename in context.filenames:
            meta_rel = 1.0
        return {
            "retrieval": min(1.0, max(0.0, hit.score)),
            "term_overlap": overlap_ratio(q_terms, hit.text),
            "graph_relevance": graph_rel,
            "metadata_relevance": meta_rel,
        }

    def rerank(self, query: str, hits: list[ChunkHit], context: RerankContext | None = None) -> list[ChunkHit]:
        context = context or RerankContext()
        w = self.weights
        out = []
        for hit in hits:
            f = self.features(query, hit, context)
            score = w[0] * f["retrieval"] + w[1] * f["term_overlap"] + w[2] * f["graph_relevance"] + w[3] * f["metadata_relevance"]
            out.append(hit.model_copy(update={"score": round(score, 4), "metadata": {**hit.metadata, "rerank": f}}))
        return sorted(out, key=lambda h: -h.score)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any | None:
    try:
        from sentence_transformers import CrossEncoder  # optional dependency
    except ImportError:
        logger.warning("cross_encoder_unavailable_falling_back_to_score_reranker")
        return None
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.fallback = ScoreReranker()

    def rerank(self, query: str, hits: list[ChunkHit], context: RerankContext | None = None) -> list[ChunkHit]:
        model = _load_cross_encoder(self.model_name)
        if model is None or not hits:
            return self.fallback.rerank(query, hits, context)
        import math

        raw = model.predict([(query, h.text) for h in hits])
        out = [
            h.model_copy(update={"score": round(1 / (1 + math.exp(-float(s))), 4), "metadata": {**h.metadata, "cross_encoder": float(s)}})
            for h, s in zip(hits, raw, strict=True)
        ]
        return sorted(out, key=lambda h: -h.score)


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "none":
        return NoopReranker()
    if settings.reranker == "cross_encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)
    return ScoreReranker()
