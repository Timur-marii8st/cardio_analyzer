from fastapi import APIRouter, Depends, Request
from ..auth import UserCreate, UserLogin, Token, UserInfo, TokenData, get_current_user
from pydantic import BaseModel
from ..deps import get_auth_service  # импорт из deps

router = APIRouter()

class RefreshTokenRequest(BaseModel):
    refresh_token: str

# Эти эндпоинты будут использовать auth_service из main.py
def get_auth_service_helper(request: Request):
    """Helper для получения auth_service (костыль, не обязателен)"""
    return request.app.state.auth_service

@router.post("/register", response_model=UserInfo)
async def register(user: UserCreate, auth_service = Depends(get_auth_service)):
    return auth_service.register_user(user)

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, auth_service = Depends(get_auth_service)):
    """Вход в систему"""
    return auth_service.login(credentials)

@router.post("/refresh", response_model=Token)
async def refresh_token(
    payload: RefreshTokenRequest,
    auth_service = Depends(get_auth_service)
):
    """Обновление access токена"""
    return auth_service.refresh_access_token(payload.refresh_token)

@router.post("/logout")
async def logout(
    payload: RefreshTokenRequest,
    current_user: TokenData = Depends(get_current_user), 
    auth_service = Depends(get_auth_service)
):
    """Выход из системы"""
    auth_service.logout(payload.refresh_token)
    return {"message": "Successfully logged out"}
