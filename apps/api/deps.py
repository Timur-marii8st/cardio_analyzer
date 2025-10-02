from fastapi import Request, HTTPException, status
from typing import Annotated
from fastapi import Depends

from .services.storage import Storage
from .auth import AuthService
from .services.inference import InferenceService
from .services.streaming import StreamingServiceRedis

def get_storage(request: Request) -> Storage:
    """Get storage instance from app state"""
    storage = getattr(request.app.state, "storage", None)
    if storage is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage service not initialized"
        )
    return storage

def get_auth_service(request: Request) -> AuthService:
    """Get auth service instance from app state"""
    auth_service = getattr(request.app.state, "auth_service", None)
    if auth_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth service not initialized"
        )
    return auth_service

def get_inference_service(request: Request) -> InferenceService:
    """Get inference service instance from app state"""
    inference_service = getattr(request.app.state, "inference_service", None)
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference service not initialized"
        )
    return inference_service

def get_streaming_service(request: Request) -> StreamingServiceRedis:
    """Get streaming service instance from app state"""
    streaming_service = getattr(request.app.state, "streaming_service", None)
    if streaming_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Streaming service not initialized"
        )
    return streaming_service

# Type annotations для удобства
StorageDep = Annotated[Storage, Depends(get_storage)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]
StreamingServiceDep = Annotated[StreamingServiceRedis, Depends(get_streaming_service)]