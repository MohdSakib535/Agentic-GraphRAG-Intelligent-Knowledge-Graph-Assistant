"""Integration fixtures: a real FastAPI app against running PostgreSQL, Neo4j and Redis.

Run the dependencies with ``docker compose up -d postgres neo4j redis`` (or the full stack)
and point POSTGRES_HOST / NEO4J_URI / REDIS_URL at them. Tests are skipped when unreachable.
Celery runs in eager mode so ingestion happens synchronously inside the test.
"""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Iterator
from urllib.parse import urlparse

import pytest

os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp(prefix="graphrag-it-"))
os.environ["RATE_LIMIT_ENABLED"] = "false"


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def _services_up() -> bool:
    from app.core.config import get_settings

    s = get_settings()
    neo = urlparse(s.neo4j_uri)
    redis = urlparse(s.redis_url)
    return (_reachable(s.postgres_host, s.postgres_port) and _reachable(neo.hostname or "localhost", neo.port or 7687)
            and _reachable(redis.hostname or "localhost", redis.port or 6379))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = os.path.dirname(__file__)
    integration = [i for i in items if str(i.fspath).startswith(here)]
    for item in integration:
        item.add_marker(pytest.mark.integration)
    if integration and not _services_up():
        skip = pytest.mark.skip(reason="PostgreSQL / Neo4j / Redis not reachable")
        for item in integration:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client() -> Iterator:
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient

    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "..", "alembic.ini"))
    command.upgrade(cfg, "head")

    from app.main import create_app
    from app.workers.celery_app import celery_app

    celery_app.conf.task_always_eager = True
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
