from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import RateLimitExceeded
from app.db.redis import RateLimiter, TenantCache
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import NoopReranker, RerankContext, ScoreReranker, build_reranker
from app.schemas.search import ChunkHit


def hit(cid: str, text: str, retriever: str, score: float) -> ChunkHit:
    return ChunkHit(chunk_id=cid, document_id="d", text=text, score=score, retrievers=[retriever],
                    metadata={f"{retriever}_score": score})


def test_rrf_fusion_rewards_agreement_across_retrievers() -> None:
    fused = reciprocal_rank_fusion({
        "vector": [hit("a", "Kafka", "vector", 0.9), hit("b", "Redis", "vector", 0.8)],
        "keyword": [hit("b", "Redis", "keyword", 1.0)],
        "graph_evidence": [hit("b", "Redis", "graph_evidence", 1.0), hit("c", "x", "graph_evidence", 0.9)],
    }, top_k=3)
    assert [h.chunk_id for h in fused][0] == "b"
    assert set(fused[0].retrievers) == {"vector", "keyword", "graph_evidence"}
    assert all(0.0 <= h.score <= 1.0 for h in fused)


def test_score_reranker_uses_graph_and_term_relevance() -> None:
    hits = [hit("a", "Unrelated text about lunch menus", "vector", 0.7),
            hit("b", "Rahul manages Project Alpha which uses Kafka", "vector", 0.6)]
    ranked = ScoreReranker().rerank("Who manages Project Alpha?", hits, RerankContext(entity_names=["Project Alpha"]))
    assert ranked[0].chunk_id == "b" and "rerank" in ranked[0].metadata
    assert NoopReranker().rerank("q", hits) == hits
    assert build_reranker(Settings(reranker="none")).name == "none"
    assert build_reranker(Settings(reranker="cross_encoder")).name == "cross_encoder"  # optional dependency


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def get(self, key: str):
        return self.data.get(key)

    async def set(self, key: str, value: object, ex: int | None = None) -> None:
        self.data[key] = value

    async def incr(self, key: str) -> int:
        self.data[key] = str(int(self.data.get(key, "0")) + 1)
        return int(self.data[key])

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    async def zrem(self, key: str, member: str) -> None:
        self.zsets.get(key, {}).pop(member, None)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.r, self.ops = redis, []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def zremrangebyscore(self, key: str, lo: float, hi: float) -> None:
        self.ops.append(("rem", key, hi))

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self.ops.append(("add", key, mapping))

    def zcard(self, key: str) -> None:
        self.ops.append(("card", key))

    def zrange(self, key: str, start: int, end: int, withscores: bool = False) -> None:
        self.ops.append(("range", key))

    def expire(self, key: str, ttl: int) -> None:
        self.ops.append(("expire", key))

    async def execute(self) -> list[object]:
        out: list[object] = []
        for op in self.ops:
            z = self.r.zsets.setdefault(op[1], {})
            if op[0] == "rem":
                for m in [m for m, s in z.items() if s <= op[2]]:
                    del z[m]
                out.append(None)
            elif op[0] == "add":
                z.update(op[2])
                out.append(1)
            elif op[0] == "card":
                out.append(len(z))
            elif op[0] == "range":
                out.append(sorted(((m, s) for m, s in z.items()), key=lambda x: x[1])[:1])
            else:
                out.append(True)
        return out


async def test_tenant_cache_keys_never_collide_and_invalidate() -> None:
    cache = TenantCache(FakeRedis(), ttl_seconds=60)  # type: ignore[arg-type]
    await cache.set("tenant-a", "retrieval", "samehash", {"answer": "A"})
    await cache.set("tenant-b", "retrieval", "samehash", {"answer": "B"})
    assert (await cache.get("tenant-a", "retrieval", "samehash")) == {"answer": "A"}
    assert (await cache.get("tenant-b", "retrieval", "samehash")) == {"answer": "B"}
    assert (await cache.key("tenant-a", "retrieval", "h")).startswith("tenant:tenant-a:")
    await cache.invalidate_tenant("tenant-a")
    assert await cache.get("tenant-a", "retrieval", "samehash") is None
    assert (await cache.get("tenant-b", "retrieval", "samehash")) == {"answer": "B"}


async def test_rate_limiter_blocks_after_limit() -> None:
    limiter = RateLimiter(FakeRedis(), fail_open=False)  # type: ignore[arg-type]
    for _ in range(3):
        await limiter.hit("chat:user:1", limit=3, window_seconds=60)
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.hit("chat:user:1", limit=3, window_seconds=60)
    assert exc.value.retry_after >= 1
    await limiter.hit("chat:user:2", limit=3, window_seconds=60)  # other identities unaffected
