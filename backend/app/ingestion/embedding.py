"""Embedding providers.

* ``OpenAIEmbedder`` - any OpenAI-compatible embeddings endpoint (batched).
* ``HashingEmbedder`` - deterministic, offline feature-hashing embeddings
  (signed hashing of stemmed terms, bigrams and character n-grams, sublinear TF,
  L2-normalised). Lexical rather than semantic, but stable and dependency-free,
  which makes the whole stack runnable without an API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import Counter
from typing import Protocol

from app.core.config import Settings
from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.utils.text import content_terms

logger = get_logger(__name__)


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.name = f"feature-hashing-{dimensions}"

    def _features(self, text: str) -> Counter[str]:
        terms = content_terms(text)
        feats: Counter[str] = Counter()
        for term in terms:
            feats[f"w:{term}"] += 2
            padded = f"<{term}>"
            if len(term) > 3:
                for i in range(len(padded) - 2):
                    feats[f"c:{padded[i : i + 3]}"] += 0.3  # type: ignore[assignment]
        for a, b in zip(terms, terms[1:], strict=False):
            feats[f"b:{a}_{b}"] += 1
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for feat, tf in self._features(text).items():
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # Neo4j cosine indexes reject zero vectors; use a tiny constant direction.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.dimensions = settings.embedding_dimensions
        self.name = settings.embedding_model
        self._batch = settings.embedding_batch_size
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        kwargs: dict[str, object] = {
            "model": settings.embedding_model,
            "api_key": api_key,
            "base_url": settings.openai_base_url,
            "chunk_size": settings.embedding_batch_size,
            "max_retries": settings.llm_max_retries,
            "request_timeout": settings.llm_timeout_seconds,
        }
        # text-embedding-3-* support shortened output dimensions.
        if settings.embedding_model.startswith("text-embedding-3"):
            kwargs["dimensions"] = settings.embedding_dimensions
        self._client = OpenAIEmbeddings(**kwargs)

    def _check(self, vectors: list[list[float]]) -> list[list[float]]:
        for v in vectors:
            if len(v) != self.dimensions:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: got {len(v)}, expected {self.dimensions}"
                )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            for attempt in range(3):
                try:
                    out.extend(self._client.embed_documents(batch))
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise EmbeddingError() from exc
                    time.sleep(2**attempt)
        return self._check(out)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._check(await self._client.aembed_documents(texts))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError() from exc

    async def aembed_query(self, text: str) -> list[float]:
        try:
            vec = await asyncio.wait_for(self._client.aembed_query(text), timeout=30)
        except Exception as exc:
            raise EmbeddingError() from exc
        return self._check([vec])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.resolved_embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return HashingEmbedder(settings.embedding_dimensions)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
