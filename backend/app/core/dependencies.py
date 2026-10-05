"""FastAPI dependencies: authentication, tenant context, rate limiting, container access."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import AccessScope, compute_scope, set_scope
from app.core.config import Settings, get_settings, parse_rate
from app.core.container import Container
from app.core.errors import AuthenticationError, AuthorizationError
from app.core.logging import tenant_id_ctx, user_id_ctx
from app.core.security import decode_token
from app.db.postgres import get_db_session
from app.db.redis import RateLimiter, get_redis, is_access_token_revoked
from app.models.user import User

bearer = HTTPBearer(auto_error=False, description="JWT access token")

DBSession = Annotated[AsyncSession, Depends(get_db_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@dataclass(frozen=True)
class CurrentUser:
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    role: str
    token_jti: str
    token_expires_at: datetime
    groups: tuple[str, ...] = ()

    @property
    def tenant(self) -> str:
        return str(self.tenant_id)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_current_user(
    db: DBSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Missing bearer token", code="NOT_AUTHENTICATED")
    claims = decode_token(credentials.credentials, "access")
    if await is_access_token_revoked(claims.jti):
        raise AuthenticationError("Token has been revoked", code="TOKEN_REVOKED")
    user = (await db.execute(select(User).where(User.id == uuid.UUID(claims.subject)))).scalar_one_or_none()
    # The tenant in the token must still match the user's tenant (defends against stale/forged claims).
    if user is None or not user.is_active or str(user.tenant_id) != claims.tenant_id:
        raise AuthenticationError("User is inactive or no longer exists", code="INVALID_TOKEN")
    tenant_id_ctx.set(str(user.tenant_id))
    user_id_ctx.set(str(user.id))
    return CurrentUser(user.id, user.tenant_id, user.email, user.role, claims.jti, claims.expires_at,
                       tuple(user.groups or []))


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]


async def get_access_scope(user: CurrentUserDep, db: DBSession) -> AccessScope:
    """The caller's document visibility, made ambient for knowledge-graph reads in this request."""
    scope = await compute_scope(db, user.tenant_id, user.groups, user.is_admin)
    set_scope(scope)
    return scope


AccessScopeDep = Annotated[AccessScope, Depends(get_access_scope)]


async def require_admin(user: CurrentUserDep) -> CurrentUser:
    if not user.is_admin:
        raise AuthorizationError()
    return user


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RateLimit:
    """Dependency enforcing a configurable limit per user (or per IP when anonymous)."""

    def __init__(self, scope: str, per: str = "user") -> None:
        self.scope = scope
        self.per = per

    def _rate(self, settings: Settings) -> str:
        return {
            "auth": settings.rate_limit_auth,
            "chat": settings.rate_limit_chat,
            "upload": settings.rate_limit_upload,
        }.get(self.scope, settings.rate_limit_default)

    async def __call__(
        self,
        request: Request,
        settings: SettingsDep,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if not settings.rate_limit_enabled:
            return
        limit, window = parse_rate(self._rate(settings))
        identity = f"ip:{_client_ip(request)}"
        if self.per == "user" and credentials is not None:
            try:
                claims = decode_token(credentials.credentials, "access")
                identity = f"tenant:{claims.tenant_id}:user:{claims.subject}"
            except AuthenticationError:
                pass  # authentication dependency reports the real error
        limiter = RateLimiter(get_redis(), fail_open=settings.rate_limit_fail_open)
        remaining, _ = await limiter.hit(f"{self.scope}:{identity}", limit, window)
        request.state.rate_limit = {"limit": limit, "remaining": remaining, "window": window}
