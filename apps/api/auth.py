from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from fastapi import Depends, HTTPException, status, Security, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Table, Column, String, DateTime, Boolean, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
import secrets
import logging

from .settings import api_settings

logger = logging.getLogger(__name__)

# Используем настройки из конфига
SECRET_KEY = api_settings.secret_key
ALGORITHM = api_settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = api_settings.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = api_settings.refresh_token_expire_days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class UserCreate(BaseModel):
    email: str # Потом вернуть на EmailStr
    password: str = Field(..., min_length=8)
    full_name: str
    role: str = "doctor"
    department: Optional[str] = None

class UserLogin(BaseModel):
    email: str # Потом вернуть на EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None

class UserInfo(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: str
    department: Optional[str]
    is_active: bool
    created_at: datetime

class AuthService:
    """Сервис аутентификации и авторизации"""
    
    def __init__(self, storage):
        self.storage = storage
        self._init_tables()
    
    def _init_tables(self):
        """Инициализация таблицы пользователей"""
        self.users_table = Table(
            "users", self.storage.meta,
            Column("user_id", String, primary_key=True),
            Column("email", String, unique=True, nullable=False),
            Column("password_hash", String, nullable=False),
            Column("full_name", String, nullable=False),
            Column("role", String, nullable=False),
            Column("department", String, nullable=True),
            Column("is_active", Boolean, default=True),
            Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
            Column("last_login", DateTime(timezone=True), nullable=True)
        )
        
        self.refresh_tokens_table = Table(
            "refresh_tokens", self.storage.meta,
            Column("token", String, primary_key=True),
            Column("user_id", String, nullable=False),
            Column("expires_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=text("NOW()"))
        )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        with self.storage.engine.begin() as conn:
            conn.execute(
                pg_insert(self.refresh_tokens_table).values(
                    token=token,
                    user_id=user_id,
                    expires_at=expires_at
                ).on_conflict_do_nothing()
            )
        
        return token
    
    def verify_token(self, token: str) -> TokenData:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("email")
            user_id: str = payload.get("user_id")
            role: str = payload.get("role")
            
            if email is None or user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return TokenData(email=email, user_id=user_id, role=role)
        
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def register_user(self, user: UserCreate) -> UserInfo:
        user_id = secrets.token_urlsafe(16)
        password_hash = self.get_password_hash(user.password)
        
        with self.storage.engine.begin() as conn:
            result = conn.execute(
                text("SELECT user_id FROM users WHERE email = :email"),
                {"email": user.email}
            )
            
            if result.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this email already exists"
                )
            
            conn.execute(
                pg_insert(self.users_table).values(
                    user_id=user_id,
                    email=user.email,
                    password_hash=password_hash,
                    full_name=user.full_name,
                    role=user.role,
                    department=user.department,
                    is_active=True
                )
            )
        
        return UserInfo(
            user_id=user_id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            department=user.department,
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        with self.storage.engine.begin() as conn:
            result = conn.execute(
                text("""
                    SELECT user_id, email, password_hash, full_name, role, 
                           department, is_active 
                    FROM users 
                    WHERE email = :email
                """),
                {"email": email}
            )
            
            user = result.fetchone()
            
            if not user:
                return None
            
            if not self.verify_password(password, user.password_hash):
                return None
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is disabled"
                )
            
            conn.execute(
                text("UPDATE users SET last_login = NOW() WHERE user_id = :user_id"),
                {"user_id": user.user_id}
            )
            
            return {
                "user_id": user.user_id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "department": user.department
            }
    
    def login(self, credentials: UserLogin) -> Token:
        user = self.authenticate_user(credentials.email, credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_data = {
            "email": user["email"],
            "user_id": user["user_id"],
            "role": user["role"]
        }
        
        access_token = self.create_access_token(data=access_token_data)
        refresh_token = self.create_refresh_token(user["user_id"])
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token
        )
    
    def refresh_access_token(self, refresh_token: str) -> Token:
        with self.storage.engine.begin() as conn:
            result = conn.execute(
                text("""
                    SELECT rt.user_id, rt.expires_at, u.email, u.role 
                    FROM refresh_tokens rt 
                    JOIN users u ON rt.user_id = u.user_id 
                    WHERE rt.token = :token AND u.is_active = true
                """),
                {"token": refresh_token}
            )
            
            row = result.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            if row.expires_at < datetime.utcnow():
                conn.execute(
                    text("DELETE FROM refresh_tokens WHERE token = :token"),
                    {"token": refresh_token}
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token expired"
                )
            
            access_token_data = {
                "email": row.email,
                "user_id": row.user_id,
                "role": row.role
            }
            
            new_access_token = self.create_access_token(data=access_token_data)
            
            return Token(
                access_token=new_access_token,
                refresh_token=refresh_token
            )
    
    def logout(self, refresh_token: str):
        with self.storage.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM refresh_tokens WHERE token = :token"),
                {"token": refresh_token}
            )

# Dependency с правильной инъекцией
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenData:
    """Dependency для получения текущего пользователя"""
    token = credentials.credentials
    auth_service = request.app.state.auth_service
    return auth_service.verify_token(token)

async def require_role(required_role: str):
    """Dependency factory для проверки роли"""
    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        if current_user.role not in [required_role, "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker