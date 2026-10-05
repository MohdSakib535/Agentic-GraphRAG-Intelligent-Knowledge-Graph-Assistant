"""Redis client, tenant-scoped cache and rate limiter."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings
from app.core.errors import RateLimitExceeded, RedisUnavailable
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: aioredis.Redis | None = None


def init_redis(settings: Settings | None = None) -> aioredis.Redis:
    global _client
    settings = settings or get_settings()
    _client = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
        health_check_interval=30,
        max_connections=100,
    )
    return _client


def get_redis() -> aioredis.Redis:
    if _client is None:
        return init_redis()
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


async def ping_redis() -> bool:
    return bool(await get_redis().ping())


def query_hash(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class TenantCache:
    """Cache whose keys are always namespaced by tenant.

    Keys look like ``tenant:{tenant_id}:v{version}:query:{namespace}:{hash}``.
    The per-tenant version is bumped whenever the tenant's knowledge base changes
    (ingestion completed, document deleted), invalidating every cached retrieval
    for that tenant without a SCAN. Because the tenant id is part of every key,
    two tenants asking the same question can never collide.
    """

    def __init__(self, client: aioredis.Redis | None, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    @staticmethod
    def _version_key(tenant_id: str) -> str:
        return f"tenant:{tenant_id}:cache_version"

    async def _version(self, tenant_id: str) -> str:
        assert self._client is not None
        return (await self._client.get(self._version_key(tenant_id))) or "0"

    async def key(self, tenant_id: str, namespace: str, digest: str) -> str:
        version = await self._version(tenant_id)
        return f"tenant:{tenant_id}:v{version}:query:{namespace}:{digest}"

    async def get(self, tenant_id: str, namespace: str, digest: str) -> Any | None:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(await self.key(tenant_id, namespace, digest))
        except RedisError as exc:
            logger.warning("cache_get_failed", extra={"error": type(exc).__name__})
            return None
        return json.loads(raw) if raw else None

    async def set(self, tenant_id: str, namespace: str, digest: str, value: Any) -> None:
        if self._client is None:
            return
        try:
            key = await self.key(tenant_id, namespace, digest)
            await self._client.set(key, json.dumps(value, default=str), ex=self._ttl)
        except RedisError as exc:
            logger.warning("cache_set_failed", extra={"error": type(exc).__name__})

    async def invalidate_tenant(self, tenant_id: str) -> None:
        if self._client is None:
            return
        try:
            await self._client.incr(self._version_key(tenant_id))
        except RedisError as exc:
            logger.warning("cache_invalidate_failed", extra={"error": type(exc).__name__})


    # ---- unversioned entries (content that does not depend on the knowledge base, e.g. query embeddings)
    async def get_static(self, tenant_id: str, namespace: str, digest: str) -> str | None:
        if self._client is None:
            return None
        try:
            return await self._client.get(f"tenant:{tenant_id}:static:{namespace}:{digest}")
        except RedisError as exc:
            logger.warning("cache_get_failed", extra={"error": type(exc).__name__})
            return None

    async def set_static(self, tenant_id: str, namespace: str, digest: str, value: str, ttl: int) -> None:
        if self._client is None:
            return
        try:
            await self._client.set(f"tenant:{tenant_id}:static:{namespace}:{digest}", value, ex=ttl)
        except RedisError as exc:
            logger.warning("cache_set_failed", extra={"error": type(exc).__name__})

    async def set_with_ttl(self, tenant_id: str, namespace: str, digest: str, value: Any, ttl: int) -> None:
        if self._client is None:
            return
        try:
            key = await self.key(tenant_id, namespace, digest)
            await self._client.set(key, json.dumps(value, default=str), ex=ttl)
        except RedisError as exc:
            logger.warning("cache_set_failed", extra={"error": type(exc).__name__})


def invalidate_tenant_cache_sync(redis_url: str, tenant_id: str) -> None:
    """Synchronous invalidation used from Celery workers."""
    import redis as sync_redis

    try:
        client = sync_redis.Redis.from_url(redis_url, socket_timeout=2.0)
        client.incr(TenantCache._version_key(tenant_id))
        client.close()
    except RedisError as exc:
        logger.warning("cache_invalidate_failed", extra={"error": type(exc).__name__})


class RateLimiter:
    """Sliding-window rate limiter backed by a Redis sorted set (atomic pipeline)."""

    def __init__(self, client: aioredis.Redis, fail_open: bool = True) -> None:
        self._client = client
        self._fail_open = fail_open

    async def hit(self, key: str, limit: int, window_seconds: int) -> tuple[int, int]:
        """Record a hit. Returns ``(remaining, retry_after)``; raises when exceeded."""
        now = time.time()
        member = f"{now:.6f}-{time.perf_counter_ns()}"
        redis_key = f"ratelimit:{key}"
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(redis_key, 0, now - window_seconds)
                pipe.zadd(redis_key, {member: now})
                pipe.zcard(redis_key)
                pipe.zrange(redis_key, 0, 0, withscores=True)
                pipe.expire(redis_key, window_seconds + 1)
                _, _, count, oldest, _ = await pipe.execute()
        except RedisError as exc:
            if self._fail_open:
                logger.warning("rate_limiter_unavailable_fail_open", extra={"error": type(exc).__name__})
                return limit, 0
            raise RedisUnavailable() from exc
        if count > limit:
            oldest_ts = oldest[0][1] if oldest else now
            retry_after = max(1, int(oldest_ts + window_seconds - now) + 1)
            try:
                await self._client.zrem(redis_key, member)  # rejected hits don't consume quota
            except RedisError:
                logger.debug("rate_limiter_cleanup_failed")
            raise RateLimitExceeded(retry_after=retry_after)
        return limit - count, 0


# Access-token revocation list (populated on logout, expires with the token).
async def revoke_access_token(jti: str, ttl_seconds: int) -> None:
    if ttl_seconds > 0:
        await get_redis().set(f"revoked:access:{jti}", "1", ex=ttl_seconds)


async def is_access_token_revoked(jti: str) -> bool:
    try:
        return bool(await get_redis().exists(f"revoked:access:{jti}"))
    except RedisError:
        logger.warning("revocation_check_failed")
        return False
