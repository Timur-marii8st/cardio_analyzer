from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routers import ingest, stream, adapters, history, reports
from .routers.auth import router as auth_router
from .routers.patients import router as patients_router
from .services.storage import Storage, DbConfig
from .services.inference import InferenceService
from .services.streaming import StreamingServiceRedis
from .settings import api_settings
from .auth import AuthService, get_current_user, TokenData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("Starting up CTG Realtime Service...")
    
    # Инициализация storage
    storage = Storage(DbConfig(url=api_settings.database_url))
    app.state.storage = storage
    logger.info("Database storage service configured")
    
    # Инициализация auth service
    auth_service = AuthService(storage)
    app.state.auth_service = auth_service
    logger.info("Auth service initialized")
    
    # Инициализация ML сервисов
    inference_service = InferenceService(api_settings.artifacts_path)
    app.state.inference_service = inference_service
    
    streaming_service = StreamingServiceRedis(
        inference_service, 
        redis_url=api_settings.redis_url
    )
    app.state.streaming_service = streaming_service
    logger.info("ML services initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")

app = FastAPI(
    title="CTG Realtime Service", 
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/api/docs" if api_settings.debug else None,
    redoc_url="/api/redoc" if api_settings.debug else None
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(
    patients_router,
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    ingest.router, 
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    stream.router
)
app.include_router(
    adapters.router,
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    history.router,
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    reports.router,
    dependencies=[Depends(get_current_user)]
)

@app.get("/healthz")
def health():
    """Health check endpoint"""
    return {"status": "ok", "version": "0.3.0"}

@app.get("/api/v1/me")
def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Получить информацию о текущем пользователе"""
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "role": current_user.role
    }