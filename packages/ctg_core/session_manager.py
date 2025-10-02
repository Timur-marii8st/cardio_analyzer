# packages/ctg_core/session_manager.py
from __future__ import annotations
import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    """Структура данных сессии для сериализации"""
    session_id: str
    bpm_buffer: List[Dict[str, Any]]  # [{"ts": iso_string, "value": float}, ...]
    ua_buffer: List[Dict[str, Any]]
    last_risk_prob: Optional[float] = None
    last_risk_band: Optional[str] = None
    last_risk_ts: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class RedisSessionManager:
    """
    Менеджер сессий, использующий Redis для хранения состояния.
    Это решает проблему масштабирования при использовании нескольких воркеров.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 session_ttl: int = 3600, buffer_max_size: int = 10000):
        """
        Args:
            redis_url: URL подключения к Redis
            session_ttl: TTL сессии в секундах (по умолчанию 1 час)
            buffer_max_size: Максимальный размер буфера для каждого канала
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.session_ttl = session_ttl
        self.buffer_max_size = buffer_max_size
        
        # Проверка подключения
        try:
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Получить данные сессии из Redis"""
        key = f"session:{session_id}"
        data = self.redis_client.get(key)
        
        if not data:
            return None
            
        try:
            session_dict = json.loads(data)
            return SessionData(**session_dict)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error deserializing session {session_id}: {e}")
            return None
    
    def create_or_update_session(self, session_id: str, 
                                session_data: Optional[SessionData] = None) -> SessionData:
        """Создать или обновить сессию"""
        if session_data is None:
            # Создаем новую сессию
            session_data = SessionData(
                session_id=session_id,
                bpm_buffer=[],
                ua_buffer=[],
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
        else:
            session_data.updated_at = datetime.utcnow().isoformat()
        
        key = f"session:{session_id}"
        value = json.dumps(asdict(session_data))
        
        # Сохраняем с TTL
        self.redis_client.setex(key, self.session_ttl, value)
        
        return session_data
    
    def add_samples(self, session_id: str, channel: str, 
                   samples: List[Dict[str, Any]]) -> SessionData:
        """
        Добавить сэмплы в буфер сессии
        
        Args:
            session_id: ID сессии
            channel: "bpm" или "uterus"
            samples: Список словарей с полями ts и value
        """
        session = self.get_session(session_id)
        
        if session is None:
            session = self.create_or_update_session(session_id)
        
        # Выбираем нужный буфер
        buffer = session.bpm_buffer if channel == "bpm" else session.ua_buffer
        
        # Добавляем новые сэмплы
        for sample in samples:
            buffer.append({
                "ts": sample.get("ts"),
                "value": float(sample.get("value"))
            })
        
        # Обрезаем буфер если превышен максимальный размер
        if len(buffer) > self.buffer_max_size:
            buffer = buffer[-self.buffer_max_size:]
        
        # Обновляем буфер в объекте сессии
        if channel == "bpm":
            session.bpm_buffer = buffer
        else:
            session.ua_buffer = buffer
        
        # Сохраняем обратно в Redis
        self.create_or_update_session(session_id, session)
        
        return session
    
    def update_risk(self, session_id: str, prob: float, 
                   band: str, ts: Optional[datetime] = None) -> SessionData:
        """Обновить информацию о риске в сессии"""
        session = self.get_session(session_id)
        
        if session is None:
            session = self.create_or_update_session(session_id)
        
        session.last_risk_prob = prob
        session.last_risk_band = band
        session.last_risk_ts = (ts or datetime.utcnow()).isoformat()
        
        self.create_or_update_session(session_id, session)
        
        return session
    
    def trim_old_samples(self, session_id: str, window_seconds: int = 3600) -> SessionData:
        """
        Удалить старые сэмплы из буферов
        
        Args:
            session_id: ID сессии
            window_seconds: Оставить только сэмплы за последние N секунд
        """
        session = self.get_session(session_id)
        
        if session is None:
            return None
        
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        cutoff_str = cutoff.isoformat()
        
        # Фильтруем буферы
        session.bpm_buffer = [
            s for s in session.bpm_buffer 
            if s.get("ts", "") >= cutoff_str
        ]
        
        session.ua_buffer = [
            s for s in session.ua_buffer 
            if s.get("ts", "") >= cutoff_str
        ]
        
        self.create_or_update_session(session_id, session)
        
        return session
    
    def get_all_sessions(self) -> List[str]:
        """Получить список всех активных сессий"""
        pattern = "session:*"
        keys = self.redis_client.keys(pattern)
        
        # Извлекаем session_id из ключей
        session_ids = [key.replace("session:", "") for key in keys]
        
        return session_ids
    
    def delete_session(self, session_id: str) -> bool:
        """Удалить сессию"""
        key = f"session:{session_id}"
        return bool(self.redis_client.delete(key))
    
    def extend_ttl(self, session_id: str) -> bool:
        """Продлить TTL сессии"""
        key = f"session:{session_id}"
        return self.redis_client.expire(key, self.session_ttl)
    
    def get_session_buffers_as_dataframes(self, session_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Получить буферы сессии как pandas DataFrames
        Удобно для передачи в функции обработки
        """
        session = self.get_session(session_id)
        
        if session is None:
            return pd.DataFrame(), pd.DataFrame()
        
        # Конвертируем в DataFrames
        df_bpm = pd.DataFrame(session.bpm_buffer) if session.bpm_buffer else pd.DataFrame()
        df_ua = pd.DataFrame(session.ua_buffer) if session.ua_buffer else pd.DataFrame()
        
        # Преобразуем ts в datetime если есть данные
        if not df_bpm.empty and "ts" in df_bpm.columns:
            df_bpm["ts"] = pd.to_datetime(df_bpm["ts"])
            df_bpm = df_bpm.rename(columns={"value": "value"})
        
        if not df_ua.empty and "ts" in df_ua.columns:
            df_ua["ts"] = pd.to_datetime(df_ua["ts"])
            df_ua = df_ua.rename(columns={"value": "value"})
        
        return df_bpm, df_ua