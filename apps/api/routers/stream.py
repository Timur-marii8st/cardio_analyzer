from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
import asyncio, json, logging
from datetime import datetime, timezone
from typing import Optional

from ..deps import get_storage, get_streaming_service
from ..auth import AuthService # Импортируем сам сервис

router = APIRouter(prefix="/v1", tags=["stream"])
logger = logging.getLogger(__name__)

@router.websocket("/stream/{session_id}")
async def ws_stream(
    websocket: WebSocket, 
    session_id: str,
    token: Optional[str] = Query(None) # Получаем токен как обычный параметр
):
    # 1. Сначала принимаем соединение. Рукопожатие отправлено.
    await websocket.accept()
    
    # 2. Теперь, внутри установленного соединения, проверяем аутентификацию.
    auth_service: AuthService = websocket.app.state.auth_service
    try:
        if not token:
            # Если токен не предоставлен, закрываем соединение с ошибкой.
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
            return

        # Проверяем токен. Если он невалиден, verify_token выбросит HTTPException.
        token_data = auth_service.verify_token(token)
        logger.info(f"WebSocket connected for user {token_data.email} on session {session_id}")

    except Exception as e:
        # Если аутентификация провалилась, закрываем соединение.
        reason = getattr(e, "detail", "Authentication failed")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=reason)
        return

    # 3. Если аутентификация прошла успешно, начинаем основной цикл работы.
    try:
        storage = websocket.app.state.storage
        streaming = websocket.app.state.streaming_service
        
        while True:
            result = streaming.tick(session_id)
            if result:
                await websocket.send_text(json.dumps(result, ensure_ascii=False, default=str))
                if result.get("risk"):
                    storage.save_risk(
                        session_id, 
                        datetime.now(timezone.utc), 
                        result["risk"]["hypoxia_prob"], 
                        result["risk"]["band"]
                    )
                    storage.save_new_decel_events(
                        session_id, 
                        result.get("decel_events", [])
                    )
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket for session {session_id}", exc_info=True)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)