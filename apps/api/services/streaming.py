from __future__ import annotations
from packages.ctg_core.realtime_redis import RealtimeProcessorRedis
from ..settings import api_settings
import logging

logger = logging.getLogger(__name__)

class StreamingServiceRedis:
    """Обновленный сервис стриминга с использованием Redis"""
    
    def __init__(self, inference_service, redis_url: str = None):
        """
        Args:
            inference_service: Сервис для ML инференса
            redis_url: URL подключения к Redis
        """
        self.rt = RealtimeProcessorRedis(
            predict_fn=inference_service.predict,
            redis_url=redis_url or api_settings.redis_url
        )
        logger.info("Streaming service initialized with Redis backend")
    
    def ingest(self, session_id: str, samples: list[dict]):
        """Принять и сохранить сэмплы"""
        try:
            self.rt.ingest_samples(session_id, samples)
            logger.debug(f"Ingested {len(samples)} samples for session {session_id}")
        except Exception as e:
            logger.error(f"Error ingesting samples for session {session_id}: {e}")
            raise
    
    def tick(self, session_id: str):
        """Выполнить шаг обработки для сессии"""
        try:
            result = self.rt.step(session_id)
            if result:
                logger.debug(f"Processing step completed for session {session_id}")
            return result
        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}")
            raise
    
    def get_status(self, session_id: str):
        """Получить статус сессии"""
        return self.rt.get_session_status(session_id)
    
    def cleanup(self):
        """Очистка старых сессий"""
        try:
            self.rt.cleanup_old_sessions()
            logger.info("Old sessions cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")