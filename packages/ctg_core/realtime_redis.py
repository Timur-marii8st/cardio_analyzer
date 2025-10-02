from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from apps.api.settings import api_settings
from .config import ml_settings
from .processing import resample_uniform, basic_qc, rolling_baseline
from .features import compute_ctg_features_window
from .anomalies import detect_anomalies
from .session_manager import RedisSessionManager

class RealtimeProcessorRedis:
    """
    Обновленная версия RealtimeProcessor, использующая Redis для хранения состояния.
    Это решает проблему масштабирования при использовании нескольких воркеров.
    """
    
    def __init__(self, predict_fn, redis_url: str = None):
        """
        Args:
            predict_fn: Функция предсказания (features_dict) -> prob
            redis_url: URL подключения к Redis
        """
        self.predict_fn = predict_fn
        self.session_manager = RedisSessionManager(
            redis_url=redis_url or api_settings.redis_url or "redis://localhost:6379",
            session_ttl=3600,  # 1 час
            buffer_max_size=ml_settings.feature_window_sec * int(ml_settings.target_freq_hz) * 2
        )
    
    def ingest_samples(self, session_id: str, samples: list[dict]):
        """
        Принимает сэмплы и сохраняет их в Redis
        
        Args:
            session_id: Идентификатор сессии
            samples: Список словарей с полями ts, channel, value
        """
        # Группируем сэмплы по каналам
        bpm_samples = []
        ua_samples = []
        
        for s in samples:
            ts = pd.to_datetime(s["ts"], utc=True).isoformat()
            ch = s["channel"]
            v = float(s["value"])
            
            sample_dict = {"ts": ts, "value": v}
            
            if ch == "bpm":
                bpm_samples.append(sample_dict)
            elif ch == "uterus":
                ua_samples.append(sample_dict)
        
        # Сохраняем в Redis через менеджер сессий
        if bpm_samples:
            self.session_manager.add_samples(session_id, "bpm", bpm_samples)
        
        if ua_samples:
            self.session_manager.add_samples(session_id, "uterus", ua_samples)
        
        # Обрезаем старые сэмплы
        window_sec = ml_settings.feature_window_sec + 300  # Окно + запас 5 минут
        self.session_manager.trim_old_samples(session_id, window_sec)
    
    def step(self, session_id: str) -> Optional[dict]:
        """
        Выполняет шаг обработки для сессии
        
        Returns:
            Словарь с результатами или None если недостаточно данных
        """
        # Получаем буферы из Redis
        df_bpm, df_ua = self.session_manager.get_session_buffers_as_dataframes(session_id)
        
        if df_bpm.empty:
            return None
        
        # Фильтруем по временному окну
        now = datetime.now(timezone.utc)
        start = now - timedelta(seconds=ml_settings.feature_window_sec)
        
        if not df_bpm.empty:
            df_bpm = df_bpm[df_bpm["ts"] >= start]
        
        if not df_ua.empty:
            df_ua = df_ua[df_ua["ts"] >= start]
        else:
            # Создаем пустой DataFrame с теми же временными метками
            df_ua = pd.DataFrame({
                "ts": df_bpm["ts"] if not df_bpm.empty else [],
                "value": np.nan
            })
        
        if df_bpm.empty:
            return None
        
        # Ресэмплинг и выравнивание
        uni = resample_uniform(df_bpm, df_ua)
        uni = basic_qc(uni)
        
        if len(uni) < ml_settings.min_window_points:
            return None
        
        # Вычисление признаков
        feats = compute_ctg_features_window(uni[["t_sec", "bpm", "ua"]])
        
        # Предсказание риска
        prob = float(self.predict_fn(feats))
        band = "normal" if prob < 0.2 else ("elevated" if prob < 0.5 else "high")
        
        # Сохраняем результат в Redis
        self.session_manager.update_risk(session_id, prob, band, now)
        
        # Детекция аномалий
        anoms = detect_anomalies(uni[["t_sec", "bpm", "ua"]])
        
        # Baseline для графика
        bl = rolling_baseline(uni["bpm"].to_numpy(float), ml_settings.baseline_roll_sec).tolist()
        
        # События с таймстампами
        ts_series = pd.to_datetime(uni["ts"])
        decel_events = []
        
        for ev in anoms.get("decel_events", []):
            s = int(ev["start_idx"])
            e = int(ev["end_idx"])
            
            if 0 <= s < len(ts_series) and 0 <= e < len(ts_series):
                decel_events.append({
                    "start_ts": ts_series.iloc[s].isoformat(),
                    "end_ts": ts_series.iloc[e].isoformat(),
                    "dur_s": ev.get("dur_s"),
                    "min_bpm": ev.get("min_bpm"),
                    "max_drop": ev.get("max_drop"),
                })
        
        return {
            "ts": now.isoformat(),
            "features": feats,
            "risk": {"hypoxia_prob": prob, "band": band},
            "anomalies": {k: v for k, v in anoms.items() if k not in ("decel_events",)},
            "decel_events": decel_events,
            "series": {
                "ts": ts_series.astype(str).tolist(),
                "bpm": uni["bpm"].astype(float).tolist(),
                "ua": uni["ua"].astype(float).tolist(),
                "baseline_60s": bl,
            }
        }
    
    def get_session_status(self, session_id: str) -> Optional[dict]:
        """
        Получить статус сессии
        
        Returns:
            Словарь с информацией о сессии или None
        """
        session = self.session_manager.get_session(session_id)
        
        if session is None:
            return None
        
        return {
            "session_id": session.session_id,
            "bpm_samples": len(session.bpm_buffer),
            "ua_samples": len(session.ua_buffer),
            "last_risk": {
                "prob": session.last_risk_prob,
                "band": session.last_risk_band,
                "timestamp": session.last_risk_ts
            } if session.last_risk_prob is not None else None,
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }
    
    def cleanup_old_sessions(self, inactive_hours: int = 24):
        """
        Очистка неактивных сессий
        
        Args:
            inactive_hours: Количество часов неактивности для удаления
        """
        cutoff = datetime.utcnow() - timedelta(hours=inactive_hours)
        
        for session_id in self.session_manager.get_all_sessions():
            session = self.session_manager.get_session(session_id)
            
            if session and session.updated_at:
                updated = datetime.fromisoformat(session.updated_at)
                if updated < cutoff:
                    self.session_manager.delete_session(session_id)