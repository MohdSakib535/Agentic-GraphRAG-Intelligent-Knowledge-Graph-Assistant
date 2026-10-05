"""Document-level access control inside a tenant.

A document with an empty ``access_groups`` list is visible to everyone in the tenant;
otherwise only to members of at least one listed group. Admins see everything.

Per request we compute an :class:`AccessScope` - the set of restricted documents the caller
may NOT see - and make it the ambient scope for all knowledge-graph reads. Reads without
an active scope fail closed (see ``require_scope``).
"""

from __future__ import annotations

import contextvars
import hashlib
import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.errors import AuthorizationError, ValidationFailed

GROUP_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,39}$")


def normalize_groups(groups: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for raw in groups or []:
        g = str(raw).strip().lower()
        if not g:
            continue
        if not GROUP_RE.match(g):
            raise ValidationFailed(f"Invalid group name {raw!r} (use a-z, 0-9, '-', '_'; max 40 chars)")
        if g not in out:
            out.append(g)
    return sorted(out)


def can_access(access_groups: Sequence[str] | None, user_groups: Sequence[str] | None, is_admin: bool) -> bool:
    return is_admin or not access_groups or bool(set(access_groups) & set(user_groups or []))


@dataclass(frozen=True)
class AccessScope:
    denied_document_ids: tuple[str, ...] = ()

    @property
    def restricted(self) -> bool:
        return bool(self.denied_document_ids)

    @property
    def fingerprint(self) -> str:
        """Stable id of the visibility set; part of every cache key so users never share results across scopes."""
        if not self.denied_document_ids:
            return "all"
        return hashlib.sha256(",".join(sorted(self.denied_document_ids)).encode()).hexdigest()[:16]

    @property
    def denied_chunk_prefixes(self) -> list[str]:
        return [f"chk_{uuid.UUID(d).hex}_" for d in self.denied_document_ids]

    def params(self) -> dict[str, Any]:
        return {"denied": list(self.denied_document_ids), "denied_prefixes": self.denied_chunk_prefixes}

    def to_config(self) -> list[str]:
        return list(self.denied_document_ids)

    @classmethod
    def from_config(cls, value: Any) -> AccessScope:
        if value is None:
            raise AuthorizationError("Missing access scope", code="ACCESS_SCOPE_MISSING")
        return cls(tuple(sorted(str(v) for v in value)))


UNRESTRICTED = AccessScope()
_scope: contextvars.ContextVar[AccessScope | None] = contextvars.ContextVar("access_scope", default=None)


@contextmanager
def use_scope(scope: AccessScope) -> Iterator[AccessScope]:
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def set_scope(scope: AccessScope) -> None:
    _scope.set(scope)


def current_scope() -> AccessScope | None:
    return _scope.get()


def require_scope() -> AccessScope:
    scope = _scope.get()
    if scope is None:
        # Fail closed: a read without an explicit scope is a programming error, never "show everything".
        raise AuthorizationError("Knowledge-base read attempted without an access scope", code="ACCESS_SCOPE_MISSING")
    return scope


def _denied_query(tenant_id: uuid.UUID) -> Any:
    from app.models.document import Document

    return select(Document.id, Document.access_groups).where(Document.tenant_id == tenant_id)


def _scope_from_rows(rows: Sequence[Any], user_groups: Sequence[str], is_admin: bool) -> AccessScope:
    if is_admin:
        return UNRESTRICTED
    denied = [str(doc_id) for doc_id, groups in rows if not can_access(groups, user_groups, False)]
    return AccessScope(tuple(sorted(denied)))


async def compute_scope(db: AsyncSession, tenant_id: uuid.UUID, user_groups: Sequence[str], is_admin: bool) -> AccessScope:
    if is_admin:
        return UNRESTRICTED
    rows = (await db.execute(_denied_query(tenant_id))).all()
    return _scope_from_rows(rows, user_groups, is_admin)


def compute_scope_sync(db: Session, tenant_id: uuid.UUID, user_groups: Sequence[str], is_admin: bool) -> AccessScope:
    if is_admin:
        return UNRESTRICTED
    rows = db.execute(_denied_query(tenant_id)).all()
    return _scope_from_rows(rows, user_groups, is_admin)
