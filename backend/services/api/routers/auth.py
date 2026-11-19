"""
Authentication API Routes
"""

import ipaddress
import os
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

from backend.services.api.user_app.middleware.auth import get_current_user
from backend.services.api.user_app.schemas.user import (
    AdminLogin,
    CheckAvailabilityRequest,
    ForgotPasswordRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
    PhoneLoginRequest,
    PhoneRegisterRequest,
    PhoneResetPasswordRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    VerificationRequest,
)
from backend.services.api.user_app.services import (
    AuthService,
    PasswordResetService,
    VerificationService,
)
from backend.services.api.user_app.services.user_service import UserService
from backend.shared.database_manager_v2 import get_session
from config.settings import settings

class RefreshTokenRequest(BaseModel):
    refresh_token: str


router = APIRouter(prefix="/auth")
is_oss = settings.edition == "oss"


def _local_single_user_mode_enabled() -> bool:
    return (
        os.getenv("LOCAL_SINGLE_USER_MODE", "").strip().lower() == "true"
        and os.getenv("LOCAL_SINGLE_USER_BIND_ADDRESS", "").strip() == "127.0.0.1"
    )


def _is_loopback_hostname(value: str | None) -> bool:
    if not value:
        return False
    try:
        parsed_host = urlsplit(f"//{value}")
        hostname = parsed_host.hostname
        if parsed_host.username is not None or parsed_host.password is not None:
            return False
        port = parsed_host.port  # Accessing port rejects malformed values.
        if port is not None and not 0 <= port <= 65535:
            return False
    except ValueError:
        return False
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_local_single_user_request(request: Request) -> bool:
    if not _is_loopback_hostname(request.headers.get("host")):
        return False

    origin = request.headers.get("origin")
    if not origin or origin.lower() in {"null", "file://"}:
        return True
    try:
        parsed_origin = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed_origin.scheme in {"http", "https"}
        and _is_loopback_hostname(parsed_origin.netloc)
    )


def get_auth_service() -> AuthService:
    """获取认证服务"""
    return AuthService()


@router.post("/local-session", response_model=TokenResponse)
async def create_local_single_user_session(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Issue a real admin session only for an explicitly loopback-bound deployment."""
    if not _local_single_user_mode_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    if not _is_local_single_user_request(request):
        raise HTTPException(status_code=403, detail="Local access required")

    try:
        return await auth_service.login_local_single_user(
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, auth_service: AuthService = Depends(get_auth_service)):
    """
    用户注册
    """
    try:
        return await auth_service.register(user_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    用户登录
    """
    try:
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")
        return await auth_service.login(credentials, ip_address, user_agent)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    token: str | None = None,
    request: Request = None,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    用户登出
    """
    if not token and request:
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if token:
        await auth_service.logout(token)
    return None


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshTokenRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    令牌刷新
    """
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    try:
        return await auth_service.refresh_tokens(payload.refresh_token, ip_address, user_agent)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

# 仅在非 OSS 模式下保留高级功能
if not is_oss:
    @router.post("/send-verification")
    async def send_verification(verification: VerificationRequest, request: Request):
        # ... (implementation kept for other editions)
        pass

    @router.post("/forgot-password")
    async def forgot_password(request_body: ForgotPasswordRequest):
        pass

    @router.post("/register/phone")
    async def register_by_phone(request: PhoneRegisterRequest):
        pass

    @router.post("/login/phone")
    async def login_by_phone(request: PhoneLoginRequest):
        pass

@router.post("/check-availability")
async def check_availability(
    request: CheckAvailabilityRequest,
    user_service: UserService = Depends(lambda: UserService()),
):
    """检查可用性"""
    exists = False
    if request.type == "username":
        user = await user_service.get_user_by_username(request.value, request.tenant_id)
        exists = user is not None
    elif request.type == "email":
        user = await user_service.get_user_by_email(request.value, request.tenant_id)
        exists = user is not None

    return {"code": 200, "message": "success", "data": {"available": not exists}}



@router.post("/change-password")
async def change_password(
    request_body: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    修改当前用户密码
    """
    try:
        await auth_service.change_password(
            user_id=current_user["user_id"],
            tenant_id=current_user["tenant_id"],
            old_password=request_body.old_password,
            new_password=request_body.new_password,
        )
        return {"code": 200, "message": "密码修改成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"修改密码失败: {str(e)}")
