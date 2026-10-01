"""Deterministic identifiers for graph objects (idempotent re-ingestion)."""

from __future__ import annotations

import hashlib
import uuid


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def entity_id(tenant_id: str, entity_type: str, normalized_name: str) -> str:
    return "ent_" + _digest(tenant_id, entity_type, normalized_name)[:32]


def chunk_id(document_id: str, index: int) -> str:
    return f"chk_{uuid.UUID(document_id).hex}_{index:05d}"


def new_request_id() -> str:
    return uuid.uuid4().hex


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError):
        return False
    return True
