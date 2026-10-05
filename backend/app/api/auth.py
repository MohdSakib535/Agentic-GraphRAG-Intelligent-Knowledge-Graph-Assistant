"""Authentication endpoints (JWT access + rotating refresh tokens)."""

from __future__ import annotations

import uuid
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.core.dependencies import CurrentUser, CurrentUserDep, DBSession, RateLimit, SettingsDep, require_admin
from app.core.errors import AppError
from app.core.logging import get_logger
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.auth import (
    AuthResponse,
    CreateUserRequest,
    LoginRequest,
    LogoutRequest,
    MeResponse,
    RefreshRequest,
    RegisterRequest,
    TenantOut,
    TokenResponse,
    UpdateUserRequest,
    UserOut,
)
from app.schemas.common import ERROR_RESPONSES
from app.services.auth_service import AuthService
from app.services.google_login import GoogleLogin

router = APIRouter(prefix="/auth", tags=["Authentication"], responses=ERROR_RESPONSES)
auth_rate_limit = Depends(RateLimit("auth", per="ip"))
logger = get_logger(__name__)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED,
             dependencies=[auth_rate_limit], summary="Register a user and create their tenant")
async def register(body: RegisterRequest, db: DBSession, settings: SettingsDep) -> AuthResponse:
    """Creates a new tenant (organisation) with the registering user as its admin and returns tokens."""
    user, tokens = await AuthService(db, settings).register(body)
    return AuthResponse(**tokens.model_dump(), user=UserOut.model_validate(user))


@router.post("/login", response_model=AuthResponse, dependencies=[auth_rate_limit], summary="Log in")
async def login(body: LoginRequest, db: DBSession, settings: SettingsDep) -> AuthResponse:
    user, tokens = await AuthService(db, settings).login(body.email, body.password)
    return AuthResponse(**tokens.model_dump(), user=UserOut.model_validate(user))


@router.post("/refresh", response_model=TokenResponse, dependencies=[auth_rate_limit],
             summary="Rotate tokens using a refresh token")
async def refresh(body: RefreshRequest, db: DBSession, settings: SettingsDep) -> TokenResponse:
    """Refresh tokens are single-use: the old token is revoked, and re-use revokes the whole token family."""
    return await AuthService(db, settings).refresh(body.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Log out")
async def logout(body: LogoutRequest, user: CurrentUserDep, db: DBSession, settings: SettingsDep) -> None:
    """Revokes the current access token (until it expires) and the supplied refresh token."""
    await AuthService(db, settings).logout(user.id, user.tenant_id, user.token_jti, user.token_expires_at,
                                           body.refresh_token)


@router.get("/me", response_model=MeResponse, summary="Current user and tenant")
async def me(user: CurrentUserDep, db: DBSession) -> MeResponse:
    db_user = await db.get(User, user.id)
    tenant = await db.get(Tenant, user.tenant_id)
    return MeResponse(user=UserOut.model_validate(db_user), tenant=TenantOut.model_validate(tenant))


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED,
             summary="Create a user inside the admin's tenant")
async def create_user(body: CreateUserRequest, admin: Annotated[CurrentUser, Depends(require_admin)], db: DBSession,
                      settings: SettingsDep) -> UserOut:
    admin_user = await db.get(User, admin.id)
    user = await AuthService(db, settings).create_user(admin_user, admin.tenant_id, body)
    return UserOut.model_validate(user)


@router.get("/users", response_model=list[UserOut], summary="List users in my tenant (admin)")
async def list_users(admin: Annotated[CurrentUser, Depends(require_admin)], db: DBSession,
                     settings: SettingsDep) -> list[UserOut]:
    return [UserOut.model_validate(u) for u in await AuthService(db, settings).list_users(admin.tenant_id)]


@router.patch("/users/{user_id}", response_model=UserOut, summary="Update a user's role, groups or status (admin)")
async def update_user(user_id: uuid.UUID, body: UpdateUserRequest, admin: Annotated[CurrentUser, Depends(require_admin)],
                      db: DBSession, settings: SettingsDep) -> UserOut:
    """Groups drive document-level permissions; deactivating a user revokes their refresh tokens."""
    user = await AuthService(db, settings).update_user(admin.id, admin.tenant_id, user_id, body)
    return UserOut.model_validate(user)


# ------------------------------------------------------------------ Google sign-in (OIDC)
class SSOExchangeRequest(BaseModel):
    code: str = Field(min_length=20, max_length=200)


@router.get("/google/login", dependencies=[auth_rate_limit], summary="Start Sign in with Google (browser redirect)")
async def google_login(db: DBSession, settings: SettingsDep) -> RedirectResponse:
    return RedirectResponse(await GoogleLogin(db, settings).authorization_url(), status_code=status.HTTP_302_FOUND)


@router.get("/google/callback", include_in_schema=False)
async def google_callback(db: DBSession, settings: SettingsDep, code: Annotated[str | None, Query(max_length=2048)] = None,
                          state: Annotated[str | None, Query(max_length=200)] = None,
                          error: Annotated[str | None, Query(max_length=100)] = None) -> RedirectResponse:
    """Google redirects the browser here; we redirect to the frontend with a single-use code (never tokens)."""
    target = settings.frontend_url.rstrip("/") + "/"
    try:
        if error or not code or not state:
            raise AppError("Google sign-in was cancelled", code="GOOGLE_LOGIN_CANCELLED")
        one_time = await GoogleLogin(db, settings).callback(code, state)
        query = {"sso_code": one_time}
    except AppError as exc:
        logger.info("google_login_failed", extra={"code": exc.code})
        query = {"sso_error": exc.code}
    return RedirectResponse(f"{target}?{urlencode(query)}", status_code=status.HTTP_302_FOUND)


@router.post("/google/exchange", response_model=AuthResponse, dependencies=[auth_rate_limit],
             summary="Trade the single-use sign-in code for tokens")
async def google_exchange(body: SSOExchangeRequest, db: DBSession, settings: SettingsDep) -> AuthResponse:
    user, tokens = await GoogleLogin(db, settings).exchange(body.code)
    return AuthResponse(**tokens.model_dump(), user=UserOut.model_validate(user))
