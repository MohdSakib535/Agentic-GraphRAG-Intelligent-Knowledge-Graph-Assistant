"""Authentication endpoints (JWT access + rotating refresh tokens)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.dependencies import CurrentUser, CurrentUserDep, DBSession, RateLimit, SettingsDep, require_admin
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

router = APIRouter(prefix="/auth", tags=["Authentication"], responses=ERROR_RESPONSES)
auth_rate_limit = Depends(RateLimit("auth", per="ip"))


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
